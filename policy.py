"""Policy network definition for DroneSwarm."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer weights using orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class DroneSwarmPolicy(nn.Module):
    """Actor-Critic policy with a GRU memory for partial observability."""

    def __init__(self, obs_size: int, act_size: int, hidden_size: int = 256):
        super().__init__()

        self.hidden_size = hidden_size

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

        self.gru = nn.GRU(hidden_size, hidden_size)

        # Actor head - outputs mean of action distribution
        self.actor_mean = layer_init(nn.Linear(hidden_size, act_size), std=0.01)
        # Log std as learnable parameter - start with lower std for less random actions
        self.actor_logstd = nn.Parameter(torch.full((1, act_size), -0.5))

        # Critic head - outputs value estimate
        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def _apply_gru(self, encoded: torch.Tensor, h: torch.Tensor, done: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Reset hidden state for terminated agents
        if done is not None:
            done = done.view(1, -1, 1)
            h = h * (1.0 - done)
        out, h_next = self.gru(encoded.unsqueeze(0), h)
        return out.squeeze(0), h_next

    def _dist_from_hidden(self, hidden: torch.Tensor):
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return torch.distributions.Normal(action_mean, action_std)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        done: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        encoded = self.encoder(obs)
        hidden, h_next = self._apply_gru(encoded, h, done)
        probs = self._dist_from_hidden(hidden)

        if action is None:
            action = probs.sample()
        action_tanh = torch.tanh(action)

        eps = 1e-6
        logprob = probs.log_prob(action)
        logprob = logprob - torch.log(1 - action_tanh * action_tanh + eps)
        logprob = logprob.sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(hidden).view(-1)

        return action_tanh, logprob, entropy, value, h_next

    def get_value(self, obs: torch.Tensor, h: torch.Tensor, done: torch.Tensor):
        encoded = self.encoder(obs)
        hidden, h_next = self._apply_gru(encoded, h, done)
        return self.critic(hidden), h_next

    def get_action_and_value_sequence(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        h0: torch.Tensor,
        done_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_steps, batch_size = obs_seq.shape[0], obs_seq.shape[1]
        logprobs = torch.zeros((t_steps, batch_size), device=obs_seq.device)
        entropies = torch.zeros((t_steps, batch_size), device=obs_seq.device)
        values = torch.zeros((t_steps, batch_size), device=obs_seq.device)

        h = h0
        for t in range(t_steps):
            encoded = self.encoder(obs_seq[t])
            hidden, h = self._apply_gru(encoded, h, done_seq[t])
            probs = self._dist_from_hidden(hidden)

            action = action_seq[t]
            eps = 1e-6
            action = torch.clamp(action, -1 + eps, 1 - eps)
            pre_tanh = 0.5 * torch.log((1 + action) / (1 - action))
            logprob = probs.log_prob(pre_tanh) - torch.log(1 - action * action + eps)
            logprobs[t] = logprob.sum(1)
            entropies[t] = probs.entropy().sum(1)
            values[t] = self.critic(hidden).view(-1)

        return logprobs, entropies, values
