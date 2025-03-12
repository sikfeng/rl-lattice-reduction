import math
from typing import Dict, Tuple, Union

from einops import repeat
from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchrl.data import ListStorage, ReplayBuffer
from torchrl.objectives.value.functional import generalized_advantage_estimate
from tqdm import tqdm

from reduction_env import ReductionEnvConfig, VectorizedReductionEnvironment


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x


class BasisEncoder(nn.Module):
    def __init__(self, basis_dim, embedding_dim=512, dropout_p=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_embedding = PositionalEncoding(
            embedding_dim, max_len=basis_dim)

        self.projection_layer = nn.Linear(basis_dim, embedding_dim)

        self.dropout = nn.Dropout(dropout_p)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=6)

    def forward(self, basis):
        x = self.projection_layer(basis)
        x = self.pos_embedding(x)
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=basis.size(0))
        x = torch.cat([cls_token, x], dim=1)
        x = self.dropout(x)
        x = self.transformer_encoder(x)

        return x[:, 0]  # use embedding of cls_token


class ActorCritic(nn.Module):
    def __init__(self, basis_dim: int, action_history_size: int, action_dim: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_history_size = action_history_size
        self.basis_dim = basis_dim

        self.basis_features_hidden_dim = 256
        self.gs_norms_features_hidden_dim = 128
        self.action_embedding_dim = 8
        self.action_embedding_hidden_dim = 64

        # Transformer for processing basis vectors
        self.basis_encoder = BasisEncoder(
            basis_dim=basis_dim, embedding_dim=self.basis_features_hidden_dim)

        self.gs_norms_encoder = nn.Sequential(
            nn.Linear(basis_dim, self.gs_norms_features_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.gs_norms_features_hidden_dim,
                      self.gs_norms_features_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.gs_norms_features_hidden_dim,
                      self.gs_norms_features_hidden_dim)
        )

        # Action history processor
        self.action_processor = nn.Sequential(
            nn.Embedding(action_dim + 1,
                         self.action_embedding_dim, action_dim),
            nn.Flatten(-2, -1),
            nn.Linear(action_history_size * self.action_embedding_dim,
                      self.action_embedding_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.action_embedding_hidden_dim,
                      self.action_embedding_hidden_dim)
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(self.basis_features_hidden_dim +
                      self.gs_norms_features_hidden_dim + self.action_embedding_hidden_dim, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(self.basis_features_hidden_dim +
                      self.gs_norms_features_hidden_dim + self.action_embedding_hidden_dim, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 1)
        )

    def forward(self, tensordict: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        tensordict = self.preprocess_inputs(tensordict)

        basis = tensordict["basis"]  # [batch_size, basis_dim, basis_dim]
        gs_norms = tensordict["gs_norms"]  # [batch_size, basis_dim]

        basis_features = self.basis_encoder(basis)
        gs_norms_features = self.gs_norms_encoder(gs_norms)

        # Process action history
        action_history = tensordict["action_history"].to(torch.long)
        action_history[(action_history == -1)] = self.action_dim
        action_embedding = self.action_processor(action_history)

        # Combine all features
        combined = torch.cat([
            basis_features,
            gs_norms_features,
            action_embedding
        ], dim=1)

        # Forward through actor and critic heads
        return self.actor(combined), self.critic(combined).squeeze(-1)

    @staticmethod
    def preprocess_inputs(tensordict: TensorDict) -> TensorDict:
        basis = tensordict["basis"]

        # Q has orthonormal rows, hence diagonal elements of R are the GS norms
        _, R = torch.linalg.qr(basis)
        gs_norms = torch.diagonal(R, dim1=-2, dim2=-1)

        # Create and return TensorDict with all features
        return TensorDict({
            "basis": basis,
            "gs_norms": gs_norms,
            "action_history": tensordict["action_history"]
        }, batch_size=[])


class PPOConfig:
    def __init__(self, env_config: ReductionEnvConfig = None, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, epochs=4, batch_size=64, dropout_p=0.2):
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.env_config = env_config if env_config is not None else ReductionEnvConfig()


class PPOAgent(nn.Module):
    def __init__(self, ppo_config: PPOConfig) -> None:
        super().__init__()
        self.ppo_config = ppo_config
        self.action_dim = self.ppo_config.env_config.actions_n

        self.actor_critic = ActorCritic(
            self.ppo_config.env_config.basis_dim,
            self.ppo_config.env_config.action_history_size,
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

        self.env = VectorizedReductionEnvironment(self.ppo_config.env_config)

    def __del__(self):
        self.env.close()

    def store_transition(self, state, action, log_prob, reward, done, next_state):
        td = TensorDict({
            "state": {
                "basis": state["basis"],
                "action_history": state["action_history"]
            },
            "action": action,
            "log_prob": log_prob,
            "reward": reward,
            "done": done,
            "next_state": {
                "basis": next_state["basis"],
                "action_history": next_state["action_history"]
            }
        }, batch_size=[action.size(0)])
        self.replay_buffer.add(td)

    def get_action(self, state: TensorDict) -> Tuple[int, float, float]:
        with torch.no_grad():
            logits, value = self.actor_critic(state)

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()

        return action, dist.log_prob(action), value

    def update(self, device: Union[torch.device, str]) -> None:
        if len(self.replay_buffer) < (self.ppo_config.batch_size / self.ppo_config.env_config.batch_size):
            return None

        batch = self.replay_buffer.sample(len(self.replay_buffer)).to(device)

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

        # Policy updates
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
            surr2 = torch.clamp(ratios, 1 - self.ppo_config.clip_epsilon,
                                1 + self.ppo_config.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            critic_loss = self.mse_loss(values, returns.squeeze(1))

            # Entropy bonus
            entropy_loss = -dist.entropy().mean()

            # Total loss
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1e3)
            self.optimizer.step()

        avg_reward = rewards.mean().item()
        self.replay_buffer.empty()
        return avg_reward

    def train_step(self, batch: TensorDict, device: Union[torch.device, str]) -> float:
        self.train()

        # Reset environment
        states, _ = self.env.reset(options=batch)
        dones = torch.zeros(
            self.ppo_config.env_config.batch_size, dtype=torch.bool)

        # Run episode
        while not torch.all(dones):
            actions, log_probs, _ = self.get_action(states.to(device))
            next_states, rewards, terminateds, truncateds, _ = self.env.step(
                actions)
            dones = torch.logical_or(terminateds, truncateds)

            self.store_transition(
                states,
                actions,
                log_probs,
                rewards,
                dones,
                next_states
            )
            states = next_states

        # Update agent
        avg_reward = self.update(device)
        return avg_reward

    def evaluate(self, dataloader: DataLoader, device: Union[torch.device, str]) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            total_reward = 0.0
            total_steps = 0
            success_count = 0
            num_samples = len(dataloader.dataset)

            for batch in tqdm(dataloader, dynamic_ncols=True):
                states, infos = self.env.reset(options=batch)
                log_defect_history = [infos["log_defect"]]
                shortest_length_history = [infos["shortest_length"]]
                dones = torch.zeros(
                    (dataloader.batch_size, ), dtype=torch.bool)
                episode_reward = 0
                steps = 0

                while not torch.all(dones):
                    actions, _, _ = self.get_action(states.to(device))
                    next_states, rewards, terminateds, truncateds, infos = self.env.step(
                        actions)
                    log_defect_history.append(infos["log_defect"])
                    shortest_length_history.append(infos["shortest_length"])
                    dones = torch.logical_or(terminateds, truncateds)
                    episode_reward += rewards
                    steps += self.ppo_config.env_config.basis_dim
                    states = next_states

                total_reward += episode_reward.sum().item()
                total_steps += steps

                # Check success
                final_shortest_length = shortest_length_history[-1]

                successes = final_shortest_length - \
                    batch["shortest_vector_length"] < 1e-3
                success_count += torch.count_nonzero(successes)

            return {
                'avg_reward': total_reward / num_samples,
                'avg_steps': total_steps / num_samples,
                'success_rate': success_count / num_samples
            }
