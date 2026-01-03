"""Training script for PufferLib Drone Swarm using PPO.

A clean, standalone PPO implementation that works with PufferLib environments.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pufferlib
import pufferlib.vector
import pufferlib.pytorch

from puffer_drone_swarm import PufferDroneSwarm
from env import EnvConfig
from policy import DroneSwarmPolicy
from configs import get_env_config

# ============================================================================
# ENVIRONMENT V2.1: CHALLENGING BUT FEASIBLE
# ============================================================================
# V2.0 was physically impossible - victims at 70 units need 5 relay hops
# but 4 drones only provide 3 hops (45 unit max reach with r_comm=15).
#
# V2.1 fixes this with more drones OR closer victims:
# - Option A: 6 drones (5 hops × 15 = 75 unit reach) ✅
# - Option B: Shorter max victim distance (45 units) ✅
#
# We choose Option A (more drones) for more impressive swarm behavior.
#
# Key differences from v1:
# 1. LARGER WORLD: 120x120 (1.44x area vs 100x100)
# 2. MORE DRONES: 6 (was 4) - enables longer relay chains
# 3. SMALLER SENSE: r_sense=30 (was 80) - must explore
# 4. MORE VICTIMS: 8 (was 6)
# 5. FEASIBLE DISTANCES: max 60 units (6 drones × 12 r_comm = 60 reach)
#
# Expected performance:
# - Random walk: ~5-15% (coverage too hard)
# - RL target: 70%+ with good coordination
# ============================================================================
# 
# WILDERNESS SAR (REALISTIC, HARD-ONLY, FROM SCRATCH)
# ============================================================================
# Assumes no infrastructure (mesh-only comms).
# Goal: 90%+ on hard variant by keeping it feasible and relay-friendly.
# - Fixed comm range (15) to reduce variance
# - Hard victim distances (35-55) only
# - Stronger relay shaping, lower dispersion/explore pressure
# ============================================================================

PRESET_NAME = "wilderness_hard_obstacles_stage2"
ENV_CONFIG = get_env_config(PRESET_NAME)

TRAINING_CONFIG = {
    "total_timesteps": 120_000_000,
    "num_envs": 32,
    "num_workers": 1,
    "num_steps": 128,
    "num_minibatches": 4,
    "log_interval": 32,
    "print_interval": 10,
    "learning_rate": 1.5e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_coef": 0.1,
    "clip_vloss": True,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "update_epochs": 4,
    "norm_adv": True,
    "anneal_lr": True,
    "hidden_size": 192,
    "device": "cpu",
    "seed": 42,
    "checkpoint_dir": "checkpoints_v2.7_wilderness_gru",
    "checkpoint_interval": 80,
    # Warm-start from stage-1 checkpoint and reset optimizer
    "resume": "checkpoints_v2.7_wilderness_gru/checkpoint_320.pt",
    "reset_optimizer": True,
}

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
        old_state_dict = checkpoint['model_state_dict']
        new_state_dict = agent.state_dict()
        
        # Check if observation dimension changed (warm-weight transfer)
        old_input_dim = old_state_dict['encoder.0.weight'].shape[1]
        new_input_dim = new_state_dict['encoder.0.weight'].shape[1]
        
        if old_input_dim != new_input_dim:
            print(f"Observation dimension changed: {old_input_dim} -> {new_input_dim}")
            print("Performing warm-weight transfer for first layer...")
            
            # Expand the first layer weights to accommodate new input dimension
            old_weight = old_state_dict['encoder.0.weight']  # [hidden, old_dim]
            new_weight = new_state_dict['encoder.0.weight']  # [hidden, new_dim]
            
            # Copy old weights and initialize new columns to small values
            min_dim = min(old_input_dim, new_input_dim)
            new_weight[:, :min_dim] = old_weight[:, :min_dim]
            if new_input_dim > old_input_dim:
                # Initialize new input dimensions with small random values
                nn.init.orthogonal_(new_weight[:, old_input_dim:], gain=0.1)
            
            old_state_dict['encoder.0.weight'] = new_weight
            print(f"  First layer weight shape: {old_weight.shape} -> {new_weight.shape}")
        
        incompatible = agent.load_state_dict(old_state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print("Warning: checkpoint keys did not fully match current model.")
            if incompatible.missing_keys:
                print(f"  Missing keys: {len(incompatible.missing_keys)}")
            if incompatible.unexpected_keys:
                print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")
        
        if not cfg["reset_optimizer"] and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Don't resume global_step when changing observation space
        if old_input_dim != new_input_dim:
            global_step = 0
            print("Starting fresh step count due to observation space change.")
        else:
            global_step = int(checkpoint.get('global_step', 0))
        
        print(f"Resumed from {cfg['resume']} at global_step={global_step:,}")
    
    # Calculate batch sizes
    batch_size = int(num_agents * num_steps)
    minibatch_agents = max(1, int(num_agents // num_minibatches))
    minibatch_size = int(minibatch_agents * num_steps)
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
    next_h = agent.init_hidden(num_agents, device)
    
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

        h0 = next_h.detach()
        
        # Rollout
        for step in range(0, num_steps):
            global_step += num_agents
            obs[step] = next_obs
            dones[step] = next_done
            
            # Get action from policy
            with torch.no_grad():
                action, logprob, _, value, next_h = agent.get_action_and_value(
                    next_obs, next_h, next_done
                )
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
            next_value, _ = agent.get_value(next_obs, next_h, next_done)
            next_value = next_value.reshape(1, -1)
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
        
        # PPO update (sequence-aware for GRU)
        agent_inds = np.arange(num_agents)
        clipfracs = []
        for epoch in range(cfg["update_epochs"]):
            np.random.shuffle(agent_inds)
            for start in range(0, num_agents, minibatch_agents):
                end = start + minibatch_agents
                mb_inds = agent_inds[start:end]

                obs_seq = obs[:, mb_inds]
                action_seq = actions[:, mb_inds]
                done_seq = dones[:, mb_inds]
                old_logprobs = logprobs[:, mb_inds]
                old_values = values[:, mb_inds]
                mb_advantages = advantages[:, mb_inds]
                mb_returns = returns[:, mb_inds]
                h0_mb = h0[:, mb_inds]

                newlogprob, entropy, newvalue = agent.get_action_and_value_sequence(
                    obs_seq, action_seq, h0_mb, done_seq
                )

                # Flatten time and batch dims
                newlogprob = newlogprob.reshape(-1)
                entropy = entropy.reshape(-1)
                newvalue = newvalue.reshape(-1)
                old_logprobs = old_logprobs.reshape(-1)
                old_values = old_values.reshape(-1)
                mb_advantages = mb_advantages.reshape(-1)
                mb_returns = mb_returns.reshape(-1)

                logratio = newlogprob - old_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg["clip_coef"]).float().mean().item())

                if cfg["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg["clip_coef"], 1 + cfg["clip_coef"]
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if cfg["clip_vloss"]:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = old_values + torch.clamp(
                        newvalue - old_values, -cfg["clip_coef"], cfg["clip_coef"]
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

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
