"""PufferLib Drone Swarm SAR - Main entry point.

A high-throughput multi-agent SAR benchmark where drones must explore efficiently
and deliver confirmations to base through a limited-range comm network.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description='PufferLib Drone Swarm SAR Environment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test the environment
  python examples/main.py test
  
  # Train with default settings
  python examples/main.py train
  
  # Train with custom settings
  python examples/main.py train --n-drones 12 --total-timesteps 1000000
  
  # Evaluate a trained model
  python examples/main.py eval --checkpoint checkpoints/policy_final.pt
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the environment')
    test_parser.add_argument('--steps', type=int, default=100, help='Number of steps to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a policy')
    train_parser.add_argument('--n-drones', type=int, default=8)
    train_parser.add_argument('--n-victims', type=int, default=10)
    train_parser.add_argument('--total-timesteps', type=int, default=10_000_000)
    train_parser.add_argument('--num-envs', type=int, default=4)
    train_parser.add_argument('--num-workers', type=int, default=1)
    train_parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    train_parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a trained policy')
    eval_parser.add_argument('--checkpoint', type=str, required=True)
    eval_parser.add_argument('--episodes', type=int, default=10)
    
    args = parser.parse_args()
    
    if args.command == 'test':
        test_env(args)
    elif args.command == 'train':
        train_policy(args)
    elif args.command == 'eval':
        eval_policy(args)
    else:
        parser.print_help()


def test_env(args):
    """Test the environment with random actions."""
    from puffer_drone_swarm import PufferDroneSwarm
    from env import EnvConfig
    import numpy as np
    
    print("Testing PufferDroneSwarm environment...")
    
    config = EnvConfig()
    env = PufferDroneSwarm(config=config)
    
    print(f"\nEnvironment Config:")
    print(f"  Drones: {config.n_drones}")
    print(f"  Victims: {config.n_victims}")
    print(f"  World size: {config.world_size}")
    print(f"  Comm radius: {config.r_comm}")
    print(f"  Max steps: {config.max_steps}")
    
    print(f"\nSpaces:")
    print(f"  Observation: {env.single_observation_space}")
    print(f"  Action: {env.single_action_space}")
    
    obs, _ = env.reset(seed=42)
    total_reward = 0
    episode_count = 0
    
    for step in range(args.steps):
        actions = env.action_space.sample()
        obs, rewards, terminals, truncations, infos = env.step(actions)
        total_reward += rewards.sum()
        
        if infos:
            for info in infos:
                if 'episode_return' in info:
                    episode_count += 1
                    print(f"Episode {episode_count}: return={info['episode_return']:.2f}, "
                          f"delivered={info.get('delivered', 0)}, "
                          f"success={info.get('success', False)}")
    
    print(f"\nRan {args.steps} steps, {episode_count} episodes completed")
    print(f"Total reward: {total_reward:.2f}")
    env.close()


def train_policy(args):
    """Train a policy using the examples/train.py script."""
    import subprocess
    
    train_path = os.path.join(os.path.dirname(__file__), 'train.py')
    cmd = [
        sys.executable, train_path,
        '--n-drones', str(args.n_drones),
        '--n-victims', str(args.n_victims),
        '--total-timesteps', str(args.total_timesteps),
        '--num-envs', str(args.num_envs),
        '--num-workers', str(args.num_workers),
        '--device', args.device,
        '--checkpoint-dir', args.checkpoint_dir,
    ]
    
    subprocess.run(cmd)


def eval_policy(args):
    """Evaluate a trained policy."""
    import torch
    import numpy as np
    from puffer_drone_swarm import PufferDroneSwarm
    from env import EnvConfig
    from policy import DroneSwarmPolicy
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Recreate environment with saved config
    env_config_dict = checkpoint.get('env_config', {})
    env_config = EnvConfig.from_dict(env_config_dict) if env_config_dict else EnvConfig()
    
    env = PufferDroneSwarm(config=env_config)
    
    # Recreate policy
    obs_size = env.single_observation_space.shape[0]
    act_size = env.single_action_space.shape[0]
    hidden_size = checkpoint.get("model_state_dict", {}).get("encoder.0.weight", None)
    if hidden_size is not None:
        hidden_size = int(hidden_size.shape[0])
    agent = DroneSwarmPolicy(obs_size, act_size, hidden_size=hidden_size or 256)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    print(f"\nEvaluating for {args.episodes} episodes...")
    
    episode_returns = []
    episode_lengths = []
    delivered_counts = []
    success_count = 0
    
    for ep in range(args.episodes):
        obs, _ = env.reset()
        obs = torch.FloatTensor(obs)
        done = False
        h = agent.init_hidden(env.num_agents, obs.device)
        done_mask = torch.zeros(env.num_agents, device=obs.device)
        ep_reward = 0
        steps = 0
        
        while not done:
            with torch.no_grad():
                action, _, _, _, h = agent.get_action_and_value(obs, h, done_mask)
            action = action.numpy()
            action = np.clip(action, -1.0, 1.0)
            
            obs, rewards, terminals, truncations, infos = env.step(action)
            obs = torch.FloatTensor(obs)
            ep_reward += rewards.sum()
            steps += 1
            done = terminals.any() or truncations.any()
            done_mask = torch.tensor(terminals | truncations, device=obs.device, dtype=torch.float32)
            
            if infos:
                for info in infos:
                    if 'delivered' in info:
                        delivered_counts.append(info['delivered'])
                        if info.get('success', False):
                            success_count += 1
        
        episode_returns.append(ep_reward)
        episode_lengths.append(steps)
        print(f"Episode {ep+1}: return={ep_reward:.2f}, steps={steps}, "
              f"delivered={delivered_counts[-1] if delivered_counts else 0}")
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({args.episodes} episodes):")
    print(f"  Mean Return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"  Mean Length: {np.mean(episode_lengths):.1f}")
    if delivered_counts:
        print(f"  Mean Delivered: {np.mean(delivered_counts):.2f}")
    print(f"  Success Rate: {success_count/args.episodes*100:.1f}%")
    print(f"{'='*50}")
    
    env.close()


if __name__ == "__main__":
    main()
