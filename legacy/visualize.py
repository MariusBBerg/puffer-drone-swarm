"""Legacy pygame visualization for the Drone Swarm environment.

Raylib is the preferred renderer (see c_src/drone_swarm_demo.c).
"""

import numpy as np
import pygame
import sys
import time
import os
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from legacy.legacy_env import DroneSwarmEnv
from env_config import EnvConfig


# Colors
BACKGROUND = (6, 24, 24)
GRID_COLOR = (18, 72, 72)
BASE_COLOR = (255, 215, 0)  # Gold
DRONE_COLORS = [
    (0, 187, 187),   # Cyan
    (187, 0, 187),   # Magenta
    (0, 187, 0),     # Green
    (187, 187, 0),   # Yellow
    (187, 0, 0),     # Red
    (0, 0, 187),     # Blue
    (187, 127, 0),   # Orange
    (127, 0, 187),   # Purple
]
VICTIM_UNKNOWN = (100, 100, 100)
VICTIM_CONFIRMED = (255, 165, 0)  # Orange
VICTIM_DELIVERED = (0, 255, 0)    # Green
COMM_LINE_COLOR = (0, 100, 100, 128)
SCAN_COLOR = (255, 255, 0, 64)
OBSTACLE_COLOR = (40, 70, 40)
OBSTACLE_OUTLINE = (70, 110, 70)


class DroneSwarmVisualizer:
    """Pygame-based visualizer for the drone swarm environment."""
    
    def __init__(self, env: DroneSwarmEnv, scale: float = 6.0, fps: int = 30):
        self.env = env
        self.scale = scale
        self.fps = fps
        
        # Calculate window size
        self.width = int(env.cfg.world_size * scale)
        self.height = int(env.cfg.world_size * scale)
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Drone Swarm SAR")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # For comm line rendering
        self.comm_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
    def world_to_screen(self, x: float, y: float) -> tuple:
        """Convert world coordinates to screen coordinates."""
        return (int(x * self.scale), int(self.height - y * self.scale))
    
    def draw_grid(self):
        """Draw a subtle grid."""
        grid_spacing = int(10 * self.scale)
        for x in range(0, self.width, grid_spacing):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, grid_spacing):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.width, y))
    
    def draw_comm_range(self, pos: np.ndarray, connected: bool):
        """Draw communication range circle."""
        screen_pos = self.world_to_screen(pos[0], pos[1])
        radius = int(self.env.cfg.r_comm * self.scale)
        color = (0, 100, 100, 30) if connected else (100, 50, 50, 30)
        pygame.draw.circle(self.comm_surface, color, screen_pos, radius, 1)
    
    def draw_comm_links(self):
        """Draw communication links between connected drones."""
        self.comm_surface.fill((0, 0, 0, 0))
        
        positions = self.env.positions
        n = len(positions)
        
        # Draw links between drones
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= self.env.cfg.r_comm:
                    p1 = self.world_to_screen(positions[i][0], positions[i][1])
                    p2 = self.world_to_screen(positions[j][0], positions[j][1])
                    pygame.draw.line(self.comm_surface, (0, 150, 150, 100), p1, p2, 1)
        
        # Draw links to base
        base_screen = self.world_to_screen(self.env.base_pos[0], self.env.base_pos[1])
        for i in range(n):
            dist = np.linalg.norm(positions[i] - self.env.base_pos)
            if dist <= self.env.cfg.r_comm:
                p1 = self.world_to_screen(positions[i][0], positions[i][1])
                pygame.draw.line(self.comm_surface, (255, 215, 0, 100), p1, base_screen, 2)
        
        self.screen.blit(self.comm_surface, (0, 0))

    def draw_obstacles(self):
        """Draw axis-aligned obstacle rectangles."""
        if self.env.obstacle_count == 0:
            return
        for rect in self.env.obstacles:
            x0, y0, x1, y1 = rect
            w = (x1 - x0) * self.scale
            h = (y1 - y0) * self.scale
            screen_x = x0 * self.scale
            screen_y = self.height - y1 * self.scale
            pygame.draw.rect(self.screen, OBSTACLE_COLOR, pygame.Rect(screen_x, screen_y, w, h))
            pygame.draw.rect(self.screen, OBSTACLE_OUTLINE, pygame.Rect(screen_x, screen_y, w, h), 1)
    
    def draw_base(self):
        """Draw the base station."""
        screen_pos = self.world_to_screen(self.env.base_pos[0], self.env.base_pos[1])
        
        # Base station
        pygame.draw.circle(self.screen, BASE_COLOR, screen_pos, int(5 * self.scale / 2))
        pygame.draw.circle(self.screen, (255, 255, 255), screen_pos, int(5 * self.scale / 2), 2)
        
        # Comm range
        comm_radius = int(self.env.cfg.r_comm * self.scale)
        pygame.draw.circle(self.screen, (255, 215, 0, 50), screen_pos, comm_radius, 1)
    
    def draw_victims(self):
        """Draw victims with status colors."""
        for i in range(self.env.cfg.n_victims):
            pos = self.env.victim_pos[i]
            status = self.env.victim_status[i]
            screen_pos = self.world_to_screen(pos[0], pos[1])
            
            if status == 0:  # Unknown
                color = VICTIM_UNKNOWN
                radius = 4
            elif status == 1:  # Confirmed
                color = VICTIM_CONFIRMED
                radius = 6
            else:  # Delivered
                color = VICTIM_DELIVERED
                radius = 5
            
            pygame.draw.circle(self.screen, color, screen_pos, int(radius * self.scale / 4))
            
            # Show confirm progress
            if status == 0 and self.env.confirm_progress[i] > 0:
                progress = self.env.confirm_progress[i] / self.env.cfg.t_confirm
                pygame.draw.arc(self.screen, VICTIM_CONFIRMED, 
                               (screen_pos[0] - 8, screen_pos[1] - 8, 16, 16),
                               0, progress * 2 * np.pi, 2)
    
    def draw_drones(self, actions: Optional[np.ndarray] = None):
        """Draw drones with status indicators."""
        for i in range(self.env.cfg.n_drones):
            pos = self.env.positions[i]
            screen_pos = self.world_to_screen(pos[0], pos[1])
            color = DRONE_COLORS[i % len(DRONE_COLORS)]
            
            # Draw drone body
            connected = self.env.connected[i]
            radius = int(4 * self.scale / 3)
            pygame.draw.circle(self.screen, color, screen_pos, radius)
            
            # Connection indicator
            if connected:
                pygame.draw.circle(self.screen, (0, 255, 0), screen_pos, radius + 2, 2)
            else:
                pygame.draw.circle(self.screen, (255, 0, 0), screen_pos, radius + 2, 2)
            
            # Battery indicator
            battery = self.env.battery[i]
            bar_width = int(20 * self.scale / 6)
            bar_height = 3
            bar_x = screen_pos[0] - bar_width // 2
            bar_y = screen_pos[1] - radius - 8
            pygame.draw.rect(self.screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
            battery_color = (0, 255, 0) if battery > 0.3 else (255, 165, 0) if battery > 0.1 else (255, 0, 0)
            pygame.draw.rect(self.screen, battery_color, (bar_x, bar_y, int(bar_width * battery), bar_height))
            
            # Scan indicator
            if actions is not None and actions[i, 2] > 0:
                scan_radius = int(self.env.cfg.r_confirm_radius * self.scale)
                scan_surface = pygame.Surface((scan_radius * 2, scan_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(scan_surface, (255, 255, 0, 50), (scan_radius, scan_radius), scan_radius)
                self.screen.blit(scan_surface, (screen_pos[0] - scan_radius, screen_pos[1] - scan_radius))
            
            # Velocity indicator
            if actions is not None:
                vel = actions[i, :2] * self.env.cfg.v_max
                end_pos = (
                    screen_pos[0] + int(vel[0] * self.scale * 2),
                    screen_pos[1] - int(vel[1] * self.scale * 2)
                )
                pygame.draw.line(self.screen, (255, 255, 255), screen_pos, end_pos, 2)
    
    def draw_stats(self):
        """Draw statistics overlay."""
        stats = [
            f"Step: {self.env.step_count}/{self.env.cfg.max_steps}",
            f"Delivered: {(self.env.victim_status == 2).sum()}/{self.env.cfg.n_victims}",
            f"Confirmed: {(self.env.victim_status == 1).sum()}",
            f"Connected: {self.env.connected.sum()}/{self.env.cfg.n_drones}",
            f"Avg Battery: {self.env.battery.mean():.2f}",
        ]
        
        y = 10
        for stat in stats:
            text = self.font.render(stat, True, (200, 200, 200))
            self.screen.blit(text, (10, y))
            y += 25
    
    def draw_legend(self):
        """Draw legend."""
        legend_items = [
            ("Base", BASE_COLOR),
            ("Obstacle", OBSTACLE_COLOR),
            ("Unknown", VICTIM_UNKNOWN),
            ("Confirmed", VICTIM_CONFIRMED),
            ("Delivered", VICTIM_DELIVERED),
        ]
        
        x = self.width - 120
        y = 10
        for label, color in legend_items:
            pygame.draw.circle(self.screen, color, (x, y + 8), 6)
            text = self.font.render(label, True, (200, 200, 200))
            self.screen.blit(text, (x + 15, y))
            y += 25
    
    def render(self, actions: Optional[np.ndarray] = None) -> bool:
        """Render one frame. Returns False if window closed."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        
        self.screen.fill(BACKGROUND)
        self.draw_grid()
        self.draw_obstacles()
        self.draw_comm_links()
        self.draw_base()
        self.draw_victims()
        self.draw_drones(actions)
        self.draw_stats()
        self.draw_legend()
        
        pygame.display.flip()
        self.clock.tick(self.fps)
        return True
    
    def close(self):
        pygame.quit()


def run_random_demo():
    """Run a demo with random actions."""
    config = EnvConfig(n_drones=8, n_victims=10, max_steps=600)
    env = DroneSwarmEnv(config)
    # Adjust scale for world size (aim for ~600-900 pixel window)
    scale = min(6.0, 800.0 / config.world_size)
    viz = DroneSwarmVisualizer(env, scale=scale, fps=30)
    
    env.reset(seed=42)
    running = True
    
    while running:
        actions = env.rng.uniform(-1.0, 1.0, size=(config.n_drones, 3))
        # Make scan action binary-ish
        actions[:, 2] = (actions[:, 2] > 0.5).astype(np.float32)
        
        obs, rewards, done, info = env.step(actions)
        
        if not viz.render(actions):
            break
        
        if done:
            print(f"Episode done! Delivered: {info['delivered']}/{config.n_victims}")
            env.reset()
    
    viz.close()


def run_policy_demo(checkpoint_path: str):
    """Run a demo with a trained policy."""
    import torch
    from policy import DroneSwarmPolicy
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    env_config_dict = checkpoint.get('env_config', {})
    env_config = EnvConfig.from_dict(env_config_dict) if env_config_dict else EnvConfig()
    
    # Create environment and visualizer
    env = DroneSwarmEnv(env_config)
    # Adjust scale for world size (aim for ~600-900 pixel window)
    scale = min(6.0, 800.0 / env_config.world_size)
    viz = DroneSwarmVisualizer(env, scale=scale, fps=30)
    
    # Get obs_size from checkpoint or compute from env config
    obs_size = 10 + 3 * env_config.obs_n_nearest + 3 * env_config.obs_n_obstacles
    checkpoint_obs = checkpoint.get("model_state_dict", {}).get("encoder.0.weight")
    if checkpoint_obs is not None:
        obs_size = int(checkpoint_obs.shape[1])
    hidden_size = checkpoint.get("model_state_dict", {}).get("encoder.0.weight")
    if hidden_size is not None:
        hidden_size = int(hidden_size.shape[0])
    act_size = 3
    agent = DroneSwarmPolicy(obs_size, act_size, hidden_size=hidden_size or 256)
    agent.load_state_dict(checkpoint['model_state_dict'])
    agent.eval()
    
    env.reset(seed=42)
    running = True
    
    h = agent.init_hidden(env_config.n_drones, device=torch.device("cpu"))
    done_mask = torch.zeros(env_config.n_drones, device=torch.device("cpu"))

    while running:
        obs = env._get_obs()
        obs_tensor = torch.FloatTensor(obs)
        
        with torch.no_grad():
            actions, _, _, _, h = agent.get_action_and_value(obs_tensor, h, done_mask)
        
        actions = actions.numpy()
        actions = np.clip(actions, -1.0, 1.0)
        
        obs, rewards, done, info = env.step(actions)
        
        if not viz.render(actions):
            break
        
        if done:
            print(f"Episode done! Delivered: {info['delivered']}/{env_config.n_victims}")
            env.reset()
            h = agent.init_hidden(env_config.n_drones, device=torch.device("cpu"))
            done_mask = torch.zeros(env_config.n_drones, device=torch.device("cpu"))
    
    viz.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to policy checkpoint')
    args = parser.parse_args()
    
    if args.checkpoint:
        run_policy_demo(args.checkpoint)
    else:
        run_random_demo()
