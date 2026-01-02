"""Training script for PufferLib Drone Swarm using PPO.

A clean, standalone PPO implementation that works with PufferLib environments.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pufferlib
import pufferlib.vector
import pufferlib.pytorch

from puffer_drone_swarm import PufferDroneSwarm
from env import EnvConfig

# ============================================================================
# TRAINING STRATEGY FOR RELAY BEHAVIOR (Stage C3+)
# ============================================================================
# Problem: Policy confirms victims at far distances but fails to deliver 
# because it doesn't maintain connectivity chains back to base.
#
# Solution: Add relay-focused reward shaping:
#   - r_relay_bonus: Reward drones that serve as relay nodes for disconnected owners
#   - r_chain_progress: Potential-based reward for getting owner closer to base
#   - r_owner_connected: Existing reward, keep it
#
# Curriculum (incremental, warm-start from stage_c2_best.pt):
#
# Stage C3a: Train at 20-45 range with relay shaping (moderate difficulty)
#   - Keep r_owner_connected=0.02 
#   - Add r_relay_bonus=0.03, r_chain_progress=0.01
#   - This teaches: "if you're near a disconnected owner, help relay"
#   - Train until ~90%+ success on 20-45
#
# Stage C3b: Extend to 25-55 range 
#   - Same rewards, harder victim distances
#   - Train until ~85%+ success on 25-55
#
# Stage C4: Domain randomization for robustness
#   - Mix victim distances: 50% near (10-35), 50% far (25-55)
#   - Slight comm drop noise (p_comm_drop=0.02)
# ============================================================================

# Edit these configs directly instead of passing CLI flags.
ENV_CONFIG = EnvConfig(
    n_drones=4,
    n_victims=6,
    r_comm=18.0,
    r_comm_min=0.0,
    r_comm_max=0.0,
    r_confirm_radius=8.0,
    t_confirm=1,
    t_confirm_values=(),
    m_deliver=120,
    m_deliver_values=(),
    r_found=1.0,
    r_found_divide_by_n=False,
    r_confirm_reward=0.0,
    r_explore=0.01,
    r_scan_near_victim=0.01,
    r_connectivity=0.0,
    r_dispersion=0.0,
    r_owner_connected=0.02,
    # NEW: Relay shaping rewards - INCREASED for stronger signal
    r_relay_bonus=0.1,        # Reward drones serving as relay nodes (was 0.03)
    r_chain_progress=0.05,    # Potential shaping for chain completion (was 0.01)
    detect_prob_scale=2.0,
    detect_noise_std=0.0,
    false_positive_rate=0.0,
    false_positive_confidence=0.3,
    p_comm_drop=0.0,
    p_comm_drop_min=0.0,
    p_comm_drop_max=0.0,
    c_time=0.01,
    c_energy=0.0,
    c_scan=0.0,
    spawn_near_base=True,
    # Stage C4: Push toward far range - primary 20-45, mix in 25-55
    victim_min_dist_from_base=20.0,
    victim_max_dist_from_base=45.0,
    # Mix in far cases 30% of the time
    victim_mix_prob=0.3,
    victim_min_dist_from_base_alt=25.0,
    victim_max_dist_from_base_alt=55.0,
    obs_n_nearest=3,
    r_sense=80.0,
    spawn_radius=5.0,

)

TRAINING_CONFIG = {
    "total_timesteps": 100_000_000,  # Continue from ~58M
    "num_envs": 64,
    "num_workers": 1,
    "num_steps": 256,
    "num_minibatches": 4,
    "log_interval": 32,
    "print_interval": 10,
    "learning_rate": 1e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "clip_vloss": True,
    "ent_coef": 0.002,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "update_epochs": 4,
    "norm_adv": True,
    "anneal_lr": True,
    "hidden_size": 256,
    "device": "cpu",
    "seed": 42,
    "checkpoint_dir": "checkpoints_new",
    "checkpoint_interval": 10,
    # Resume from checkpoint_890 (92% on 20-45, 40% on 25-55)
    "resume": "checkpoints_new/checkpoint_890.pt",
    "reset_optimizer": True,
}


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer weights using orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DroneSwarmPolicy(nn.Module):
    """Actor-Critic policy for continuous action drone swarm."""
    
    def __init__(self, obs_size: int, act_size: int, hidden_size: int = 256):
        super().__init__()
        
        # Deeper shared feature extractor
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        
        # Actor head - outputs mean of action distribution
        self.actor_mean = layer_init(nn.Linear(hidden_size, act_size), std=0.01)
        # Log std as learnable parameter - start with lower std for less random actions
        self.actor_logstd = nn.Parameter(torch.full((1, act_size), -0.5))
        
        # Critic head
        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)
    
    def get_value(self, x):
        """Get value estimate only."""
        return self.critic(self.encoder(x))
    
    def get_action_and_value(self, x, action=None):
        """Get action, log prob, entropy, and value for PPO (tanh-squashed Gaussian)."""
        hidden = self.encoder(x)
        
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        
        # Create normal distribution
        probs = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            pre_tanh = probs.sample()
            action = torch.tanh(pre_tanh)
        else:
            eps = 1e-6
            action = torch.clamp(action, -1 + eps, 1 - eps)
            pre_tanh = 0.5 * torch.log((1 + action) / (1 - action))

        # Tanh-squash correction
        eps = 1e-6
        logprob = probs.log_prob(pre_tanh) - torch.log(1 - action * action + eps)
        logprob = logprob.sum(1)
        base_entropy = probs.entropy().sum(1)     # entropy of pre-tanh normal
        entropy = base_entropy

        return (
            action,
            logprob,
            entropy,
            self.critic(hidden),
        )


def make_env_creator(config: EnvConfig = None, log_interval: int = 32):
    """Create an environment factory function."""
    if config is None:
        config = EnvConfig()
    
    def create_env(buf=None, seed=0):
        return PufferDroneSwarm(config=config, buf=buf, seed=seed, log_interval=log_interval)
    
    return create_env


def train(env_config: EnvConfig = ENV_CONFIG, cfg: dict | None = None):
    """Main training loop using PPO."""
    if cfg is None:
        cfg = TRAINING_CONFIG

    # Seeding
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    if cfg["device"] == 'mps':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device(cfg["device"] if torch.cuda.is_available() or cfg["device"] == 'cpu' else 'cpu')
    print(f"Using device: {device}")

    num_envs = cfg["num_envs"]
    num_workers = cfg["num_workers"]
    num_steps = cfg["num_steps"]
    num_minibatches = cfg["num_minibatches"]
    total_timesteps = cfg["total_timesteps"]
    log_interval = cfg["log_interval"]
    
    # Create vectorized environment
    env_creator = make_env_creator(env_config, log_interval=log_interval)
    
    if num_workers > 1:
        backend = pufferlib.vector.Multiprocessing
        envs = pufferlib.vector.make(
            env_creator,
            num_envs=num_envs,
            num_workers=num_workers,
            batch_size=num_envs,
            backend=backend,
        )
    else:
        backend = pufferlib.vector.Serial
        envs = pufferlib.vector.make(
            env_creator,
            num_envs=num_envs,
            backend=backend,
        )
    
    # Get environment info
    obs_size = envs.single_observation_space.shape[0]
    act_size = envs.single_action_space.shape[0]
    num_agents = envs.num_agents  # Total agents across all envs
    
    print(f"Observation size: {obs_size}")
    print(f"Action size: {act_size}")
    print(f"Total agents per step: {num_agents}")
    
    # Create policy
    agent = DroneSwarmPolicy(obs_size, act_size, hidden_size=cfg["hidden_size"]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg["learning_rate"], eps=1e-5)

    # Resume from checkpoint if provided
    global_step = 0
    if cfg["resume"]:
        checkpoint = torch.load(cfg["resume"], map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        if not cfg["reset_optimizer"] and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = int(checkpoint.get('global_step', 0))
        print(f"Resumed from {cfg['resume']} at global_step={global_step:,}")
    
    # Calculate batch sizes
    batch_size = int(num_agents * num_steps)
    minibatch_size = int(batch_size // num_minibatches)
    num_iterations = total_timesteps // batch_size
    start_iteration = global_step // batch_size + 1
    
    print(f"\n{'='*60}")
    print(f"Training Drone Swarm with {env_config.n_drones} drones, {env_config.n_victims} victims")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Batch size: {batch_size}, Minibatch size: {minibatch_size}")
    print(f"Num iterations: {num_iterations}")
    print(f"{'='*60}\n")
    
    # Storage
    obs = torch.zeros((num_steps, num_agents) + (obs_size,)).to(device)
    actions = torch.zeros((num_steps, num_agents) + (act_size,)).to(device)
    logprobs = torch.zeros((num_steps, num_agents)).to(device)
    rewards = torch.zeros((num_steps, num_agents)).to(device)
    dones = torch.zeros((num_steps, num_agents)).to(device)
    values = torch.zeros((num_steps, num_agents)).to(device)
    
    # Logging
    start_time = time.time()
    episode_returns = []
    episode_lengths = []
    episode_delivered = []
    episode_confirmed = []
    episode_success = []
    
    # Initialize environment
    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(num_agents).to(device)
    
    if start_iteration > num_iterations:
        print("Total timesteps already reached for this run; increase TRAINING_CONFIG['total_timesteps'] to continue.")
        envs.close()
        return agent

    for iteration in range(start_iteration, num_iterations + 1):
        # Annealing the learning rate
        if cfg["anneal_lr"]:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * cfg["learning_rate"]
            optimizer.param_groups[0]["lr"] = lrnow
        
        # Rollout
        for step in range(0, num_steps):
            global_step += num_agents
            obs[step] = next_obs
            dones[step] = next_done
            
            # Get action from policy
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob
            
            # Execute action
            # Clamp actions to valid range before stepping
            action_np = action.cpu().numpy()
            
            next_obs_np, reward, terminations, truncations, infos = envs.step(action_np)
            done = np.logical_or(terminations, truncations)
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done).to(device)
            
            # Log episode info
            for info in infos:
                if 'episode_return' in info:
                    episode_returns.append(info['episode_return'])
                    episode_lengths.append(info.get('episode_length', 0))
                    episode_delivered.append(info.get('delivered', 0))
                    episode_confirmed.append(info.get('confirmed', 0))
                    episode_success.append(info.get('success', False))
        
        # Compute GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + cfg["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + cfg["gamma"] * cfg["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values
        
        # Flatten batch
        b_obs = obs.reshape((-1,) + (obs_size,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (act_size,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        
        # PPO update
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(cfg["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg["clip_coef"]).float().mean().item())
                
                mb_advantages = b_advantages[mb_inds]
                if cfg["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg["clip_coef"], 1 + cfg["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if cfg["clip_vloss"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -cfg["clip_coef"], cfg["clip_coef"]
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg["ent_coef"] * entropy_loss + v_loss * cfg["vf_coef"]
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg["max_grad_norm"])
                optimizer.step()
        
        # Logging
        if iteration % cfg["print_interval"] == 0 or iteration == 1:
            sps = int(global_step / (time.time() - start_time))
            
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{num_iterations}")
            print(f"Global Step: {global_step:,}")
            print(f"SPS: {sps}")
            
            if episode_returns:
                print(f"Mean Episode Return: {np.mean(episode_returns[-100:]):.3f}")
                print(f"Mean Episode Length: {np.mean(episode_lengths[-100:]):.1f}")
                print(f"Mean Delivered: {np.mean(episode_delivered[-100:]):.2f}")
                print(f"Mean Confirmed: {np.mean(episode_confirmed[-100:]):.2f}")
                print(f"Success Rate: {100*np.mean(episode_success[-100:]):.1f}%")
            
            print(f"Value Loss: {v_loss.item():.4f}")
            print(f"Policy Loss: {pg_loss.item():.4f}")
            print(f"Entropy: {entropy_loss.item():.4f}")
            print(f"Approx KL: {approx_kl.item():.4f}")
            print(f"Clipfrac: {np.mean(clipfracs):.4f}")
            print(f"{'='*60}")
        
        # Save checkpoint
        if cfg["checkpoint_dir"] and iteration % cfg["checkpoint_interval"] == 0:
            os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
            checkpoint_path = os.path.join(cfg["checkpoint_dir"], f"checkpoint_{iteration}.pt")
            torch.save({
                'iteration': iteration,
                'global_step': global_step,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'env_config': vars(env_config),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    if cfg["checkpoint_dir"]:
        os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
        final_path = os.path.join(cfg["checkpoint_dir"], "policy_final.pt")
        torch.save({
            'iteration': iteration,
            'global_step': global_step,
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'env_config': vars(env_config),
        }, final_path)
        print(f"\nSaved final model to {final_path}")
    
    envs.close()
    
    return agent


if __name__ == '__main__':
    train()
