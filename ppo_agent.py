import math
from typing import Tuple, Union

import numpy as np
from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.data import ListStorage, ReplayBuffer
from torchrl.objectives.value.functional import generalized_advantage_estimate

from reduction_env import ReductionEnvConfig, ReductionEnvironment


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1)]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 128, num_heads: int = 8, num_layers: int = 6, dropout_p: int = 0.1, max_len: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        self.pos_encoding = PositionalEncoding(embedding_dim, max_len=max_len)
        self.input_projection = nn.Linear(input_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4*embedding_dim,
            dropout=dropout_p,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, seq_lengths):
        batch_size = x.size(0)
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        padding_mask = torch.zeros(
            batch_size, self.max_len, dtype=torch.bool, device=x.device)
        for i, length in enumerate(seq_lengths):
            padding_mask[i, :length] = True

        attn_mask = ~padding_mask

        encoder_output = self.transformer_encoder(
            x, src_key_padding_mask=attn_mask)
        return encoder_output


class ActorCritic(nn.Module):
    def __init__(self, max_basis_dim: int, action_dim: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.max_basis_dim = max_basis_dim

        self.gs_norms_features_hidden_dim = 128
        self.action_embedding_dim = 8
        self.dropout_p = dropout_p

        self.gs_norms_encoder = TransformerEncoder(
            input_dim=1,
            embedding_dim=self.gs_norms_features_hidden_dim,
            num_heads=4,
            num_layers=3,
            dropout_p=self.dropout_p,
            max_len=self.max_basis_dim
        )

        self.action_embedding = nn.Sequential(
            nn.Linear(2, self.action_embedding_dim),
            nn.LeakyReLU()
        )

        self.combined_feature_dim = self.gs_norms_features_hidden_dim + \
            self.action_embedding_dim
        self.actor_hidden_dim = 128

        self.actor = nn.Sequential(
            nn.Linear(self.combined_feature_dim, self.actor_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.actor_hidden_dim, self.actor_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.actor_hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.combined_feature_dim, self.actor_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.actor_hidden_dim, self.actor_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.actor_hidden_dim, 1),
        )

    def forward(self, tensordict: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        tensordict = self.preprocess_inputs(tensordict)

        gs_norms = tensordict["gs_norms"]  # [batch_size, basis_dim]
        basis_dim = tensordict["basis_dim"]  # [batch_size]
        previous_action = tensordict["last_action"]

        gs_norms_reshaped = gs_norms.unsqueeze(-1)
        gs_norms_features = self.gs_norms_encoder(gs_norms_reshaped, basis_dim)
        gs_norms_features = gs_norms_features.mean(dim=1)

        indices = torch.arange(self.action_dim, device=previous_action.device).unsqueeze(0)
        # block size \(b\) corresponds to action id \(b - 1\)
        previous_effective_block_size = torch.min(previous_action.expand(-1, self.max_basis_dim), basis_dim - indices) + 1
        previous_relative_block_size = previous_effective_block_size / basis_dim
        prev_action_embedding = torch.stack([previous_effective_block_size, previous_relative_block_size], dim=1)
        prev_action_embedding = self.action_embedding(prev_action_embedding.transpose(dim0=-2, dim1=-1)).squeeze(1)

        # Combine all features
        combined = torch.cat([
            gs_norms_features,
            prev_action_embedding.mean(dim=1)
        ], dim=1)

        # Forward through actor and critic heads
        return self.actor(combined), self.critic(combined).squeeze(-1)

    def preprocess_inputs(self, tensordict: TensorDict) -> TensorDict:
        basis = tensordict["basis"]  # [batch_size, max_basis_dim, max_basis_dim]
        basis_dim = tensordict["basis_dim"]  # [batch_size]
        _, max_basis_dim, _ = basis.shape

        mask = torch.arange(max_basis_dim, device=basis.device) < basis_dim.view(-1, 1, 1)
        masked_basis = basis * mask

        _, R = torch.linalg.qr(masked_basis)
        diag = torch.diagonal(R, dim1=-2, dim2=-1).abs()

        gs_norms = diag * mask.squeeze(1)

        return TensorDict({
            "gs_norms": gs_norms,
            "last_action": tensordict["last_action"],
            "basis_dim": basis_dim
        }, batch_size=[])


class PPOConfig:
    def __init__(self, env_config: ReductionEnvConfig = None,
                 lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_epsilon: float = 0.2,
                 clip_grad_norm: float = 0.5, epochs: int = 4,
                 dropout_p: float = 0.2):
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.clip_grad_norm = clip_grad_norm
        self.epochs = epochs
        self.dropout_p = dropout_p
        self.env_config = env_config if env_config is not None else ReductionEnvConfig()


class PPOAgent(nn.Module):
    def __init__(self, ppo_config: PPOConfig, device: Union[torch.device, str]) -> None:
        super().__init__()
        self.ppo_config = ppo_config
        self.device = device
        self.action_dim = self.ppo_config.env_config.actions_n

        self.actor_critic = ActorCritic(
            self.ppo_config.env_config.max_basis_dim,
            self.action_dim,
            ppo_config.dropout_p
        )

        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.actor_critic.parameters(), lr=ppo_config.lr)

        def sample_transform(tensordict: TensorDict) -> TensorDict:
            tensordict = TensorDict(
                {key: tensordict[key].flatten(0, 1) for key in tensordict.keys()})
            return tensordict

        self.replay_buffer = ReplayBuffer(
            storage=ListStorage(),
            transform=sample_transform
        )

        self.env = ReductionEnvironment(self.ppo_config.env_config)

    def store_transition(self, state, action, log_prob, reward, done, next_state):
        td = TensorDict({
            "state": {
                "basis": state["basis"],
                "last_action": state["last_action"],
                "basis_dim": state["basis_dim"]
            },
            "action": action,
            "log_prob": log_prob,
            "reward": reward,
            "done": done,
            "next_state": {
                "basis": next_state["basis"],
                "last_action": next_state["last_action"],
                "basis_dim": next_state["basis_dim"]
            }
        }, batch_size=[action.size(0)])
        self.replay_buffer.add(td)

    def _mask_logits(self, logits, basis_dim, last_action):
        indices = torch.arange(self.action_dim, device=logits.device).unsqueeze(0)
        thresholds = last_action.unsqueeze(1)
        # mask entries which are False will be masked out
        # indices >= thresholds are entries not smaller than previous block size
        # indices <= basis_dim are entries with block size smaller than dim
        # indices == 0 is the termination action
        mask = ((indices >= thresholds) & (indices <= basis_dim)) | (indices == 0)
        return logits.masked_fill(~mask, float('-inf'))

    def get_action(self, state: TensorDict) -> Tuple[int, float, float]:
        with torch.no_grad():
            logits, value = self.actor_critic(state)

        masked_logits = self._mask_logits(logits, state["basis_dim"], state["last_action"])

        dist = torch.distributions.Categorical(logits=masked_logits)
        action = dist.sample()

        return action, dist.log_prob(action), value

    def update(self) -> None:
        if len(self.replay_buffer) < self.ppo_config.env_config.batch_size:
            return None

        self.train()

        batch = self.replay_buffer.sample(len(self.replay_buffer)).to(self.device)

        states = batch["state"]
        next_states = batch["next_state"]
        actions = batch["action"]
        old_log_probs = batch["log_prob"]
        rewards = batch["reward"]
        dones = batch["done"]

        # Calculate advantages by Generalized Advantage Estimation (GAE) using current value function
        with torch.no_grad():
            """
            This block computes the value estimates for the current states and the next states 
            using the critic part of the actor-critic model.

            Using `torch.no_grad()` means we do not track gradients during this computation because 
            these estimates serve as targets for calculating the advantage; they are not directly 
            updated in this backward pass.

            The computed `values` and `next_values` are used to calculate the temporal-difference 
            errors (deltas), which are the building blocks for GAE.
            """
            _, values = self.actor_critic(states)
            _, next_values = self.actor_critic(next_states)

        advantages, returns = generalized_advantage_estimate(
            gamma=self.ppo_config.gamma,
            lmbda=self.ppo_config.gae_lambda,
            state_value=values.unsqueeze(1),
            next_state_value=next_values.unsqueeze(1),
            reward=rewards.unsqueeze(1),
            done=dones.unsqueeze(1)
        )

        actor_losses, critic_losses, entropy_losses, total_losses = [], [], [], []
        clip_fractions = []
        approx_kls = []

        for _ in range(self.ppo_config.epochs):
            logits, values = self.actor_critic(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)

            r"""
            The probability ratio is 
            \[ r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \]
            This ratio measures how much the new policy differs from the old one.
            """
            ratios = (new_log_probs - old_log_probs).exp()

            r"""
            PPO modifies the standard policy gradient update using a clipped surrogate objective:
            \[ L(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right) \right] \]
            """
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.ppo_config.clip_epsilon,1 + self.ppo_config.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(values, returns.squeeze(1))
            entropy_loss = -dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.ppo_config.clip_grad_norm)
            self.optimizer.step()

            # Logging metrics
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())
            total_losses.append(loss.item())

            clipped = (ratios < 1 - self.ppo_config.clip_epsilon) | (ratios > 1 + self.ppo_config.clip_epsilon)
            clip_fractions.append(clipped.float().mean().item())

            approx_kl = (old_log_probs - new_log_probs).mean().item()
            approx_kls.append(approx_kl)

        self.replay_buffer.empty()
        return {
            "update/avg_actor_loss": np.mean(actor_losses),
            "update/avg_critic_loss": np.mean(critic_losses),
            "update/avg_entropy": np.mean(entropy_losses),
            "update/total_loss": np.mean(total_losses),
            "update/approx_kl": np.mean(approx_kl),
            "update/advantages_mean": advantages.mean().item(),
            "update/advantages_std": advantages.std().item(),
        }

    def collect_experiences(self) -> float:
        total_reward = 0.0
        steps = 0
        total_log_prob = 0.0
        total_value = 0.0

        state, _ = self.env.reset()
        done = False

        state = TensorDict({k: v.unsqueeze(0).to(self.device)
                           for k, v in state.items()}, batch_size=[])
        while not done:
            action, log_prob, value = self.get_action(state.to(self.device))
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            next_state = TensorDict({k: v.unsqueeze(0).to(self.device)
                                    for k, v in next_state.items()}, batch_size=[])
            self.store_transition(
                state,
                action,
                log_prob,
                torch.tensor([reward], dtype=torch.float32).to(self.device),
                torch.tensor([done], dtype=torch.bool).to(self.device),
                next_state
            )
            state = next_state

            total_reward += reward.item()
            steps += 1
            total_log_prob += log_prob.item()
            total_value += value.item()
        return {
            "episode/reward": total_reward,
            "episode/steps": steps,
            "episode/avg_action_log_prob": total_log_prob / steps,
            "episode/avg_value_estimate": total_value / steps,
        }
