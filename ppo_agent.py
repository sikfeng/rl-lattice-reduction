import math
from typing import Dict, Tuple, Union

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchrl.data import ListStorage, ReplayBuffer
from torchrl.objectives.value.functional import generalized_advantage_estimate
from tqdm import tqdm

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
        x = x + self.pe[:, :x.size(1), :]
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

        self.dropout_p = dropout_p

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

        self.action_embedding = nn.Embedding(
            action_dim + 1, self.action_embedding_dim)

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

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.gs_norms_features_hidden_dim + self.action_embedding_dim,
            nhead=4,
            dim_feedforward=4*self.gs_norms_features_hidden_dim,
            dropout=self.dropout_p,
            batch_first=True
        )
        self.gs_norm_simulator = nn.TransformerDecoder(
            decoder_layer,
            num_layers=3
        )
        self.time_simulator = nn.Sequential(
            nn.Linear(self.combined_feature_dim +
                      self.action_embedding_dim, self.combined_feature_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.combined_feature_dim, 1)
        )

    def forward(self, tensordict: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        tensordict = self.preprocess_inputs(tensordict)

        gs_norms = tensordict["gs_norms"]  # [batch_size, basis_dim]
        basis_dim = tensordict["basis_dim"]  # [batch_size]

        gs_norms_reshaped = gs_norms.unsqueeze(-1)
        gs_norms_features = self.gs_norms_encoder(gs_norms_reshaped, basis_dim)
        gs_norms_features = gs_norms_features.mean(dim=1)

        action_embedding = self.action_embedding(
            tensordict["last_action"].long()).squeeze(1)
        # Combine all features
        combined = torch.cat([
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
        gs_norms = torch.abs(torch.diagonal(R, dim1=-2, dim2=-1))

        # Create and return TensorDict with all features
        return TensorDict({
            "gs_norms": gs_norms,
            "last_action": tensordict["last_action"]
        }, batch_size=[])

    def simulate(self, current_gs_norms: torch.Tensor, previous_action: torch.Tensor, current_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate the Gram-Schmidt norms after applying BKZ with a specific block size.

        Args:
            gs_norms (torch.Tensor): Current Gram-Schmidt norms [batch_size, basis_dim]
            previous_block_size (torch.Tensor): Previously used block size [batch_size]
            block_size (torch.Tensor): Block size to use for simulation [batch_size]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (simulated_gs_norms, estimated_time)
        """
        batch_size, basis_dim = current_gs_norms.shape
        device = current_gs_norms.device

        # [batch_size, basis_dim, 1]
        gs_norms_3d = current_gs_norms.unsqueeze(-1)
        encoded_gs_norms = self.gs_norms_encoder(
            gs_norms_3d,
            torch.full((batch_size,), basis_dim, device=device)
        )  # [batch_size, basis_dim, hidden_dim]
        prev_action_embedding = self.action_embedding(
            previous_action.long()).squeeze(1)
        current_action_embedding = self.action_embedding(
            current_action.long()).squeeze(1)

        time_sim_context = torch.cat([
            encoded_gs_norms.mean(dim=1),
            prev_action_embedding,
            current_action_embedding
        ], dim=1)
        simulated_time = self.time_simulator(time_sim_context)

        gs_norm_sim_context = torch.cat([
            encoded_gs_norms,
            current_action_embedding.unsqueeze(1).expand(-1, basis_dim, -1)
        ], dim=2)
        simulated_gs_norms = torch.zeros(batch_size, basis_dim, device=device)
        generated_sequence = torch.zeros(batch_size, 1, 1, device=device)

        for i in range(basis_dim):
            tgt = self.gs_norms_encoder.input_projection(generated_sequence)
            tgt = self.gs_norms_encoder.pos_encoding(tgt)

            # Create causal mask for autoregressive generation
            tgt_mask = None
            if i > 0:
                tgt_mask = torch.triu(torch.ones(
                    i+1, i+1, device=device) * float('-inf'), diagonal=1)

            tgt = torch.cat([tgt, current_action_embedding.unsqueeze(
                1).expand(-1, tgt.size(1), -1)], dim=2)
            decoder_output = self.gs_norm_simulator(
                tgt=tgt,
                memory=gs_norm_sim_context,
                tgt_mask=tgt_mask
            )
            # [batch_size, 1, hidden_dim]
            current_prediction = decoder_output[:, -
                                                1:, :self.gs_norms_features_hidden_dim]

            # Project to get the norm value, tie with input projection weights
            predicted_norm = torch.nn.functional.linear(
                current_prediction,
                self.gs_norms_encoder.input_projection.weight.t(),
                # self.gs_norms_encoder.input_projection.bias
                bias=None
            )
            predicted_norm = torch.abs(predicted_norm)  # Enforce positivity

            simulated_gs_norms[:, i] = predicted_norm.squeeze(-1).squeeze(-1)

            if i < basis_dim - 1:
                generated_sequence = torch.cat([
                    generated_sequence,
                    predicted_norm.detach()
                ], dim=1)

        return simulated_gs_norms, simulated_time.squeeze(-1)

    def preprocess_inputs(self, tensordict: TensorDict) -> TensorDict:
        basis = tensordict["basis"]
        basis_dim = tensordict["basis_dim"]
        batch_size = basis.size(0)
        max_basis_dim = self.max_basis_dim
        device = basis.device

        gs_norms = torch.zeros(batch_size, max_basis_dim, device=device)

        for i in range(batch_size):
            actual_dim = basis_dim[i].item()
            actual_basis = basis[i, :actual_dim, :actual_dim]
            _, R = torch.linalg.qr(actual_basis)
            diag = torch.diag(R).abs()
            gs_norms[i, :actual_dim] = diag

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

    def store_transition(self, state, action, log_prob, reward, done, next_state, time_taken):
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
            },
            "time_taken": time_taken
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
            critic_loss = self.mse_loss(values, returns.squeeze(1))
            entropy_loss = -dist.entropy().mean()

            current_features = self.actor_critic.preprocess_inputs(states)
            next_features = self.actor_critic.preprocess_inputs(next_states)
            predicted_gs_norms, predicted_time = self.actor_critic.simulate(
                current_features["gs_norms"],
                current_features["last_action"],
                actions)

            # Calculate simulator losses
            gs_norm_sim_loss = torch.nn.functional.mse_loss(
                predicted_gs_norms, next_features["gs_norms"])
            time_sim_loss = torch.nn.functional.mse_loss(
                predicted_time, batch["time_taken"])

            loss = actor_loss + 0.5 * critic_loss + 0.01 * \
                entropy_loss + 0.1 * gs_norm_sim_loss + 0.1 * time_sim_loss

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.ppo_config.clip_grad_norm)
            self.optimizer.step()

        avg_reward = rewards.mean().item()
        self.replay_buffer.empty()
        return avg_reward

    def collect_experiences(self) -> float:
        state, info = self.env.reset()
        done = False

        state = TensorDict({k: v.unsqueeze(0).to(self.device)
                           for k, v in state.items()}, batch_size=[])
        while not done:
            action, log_prob, _ = self.get_action(state.to(self.device))
            next_state, reward, terminated, truncated, next_info = self.env.step(
                action)
            done = terminated or truncated

            next_state = TensorDict({k: v.unsqueeze(0).to(self.device)
                                    for k, v in next_state.items()}, batch_size=[])
            self.store_transition(
                state,
                action,
                log_prob,
                torch.tensor([reward], dtype=torch.float32).to(self.device),
                torch.tensor([done], dtype=torch.bool).to(self.device),
                next_state,
                torch.tensor([next_info["time"] - info["time"]],
                             dtype=torch.float32).to(self.device)
            )
            state = next_state
            info = next_info
        return

    def evaluate(self, dataloader: DataLoader, device: Union[torch.device, str]) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            total_reward = 0.0
            total_steps = 0
            success_count = 0
            shortness = 0.0
            length_improvement = 0.0
            time_taken = 0.0
            num_samples = len(dataloader.dataset)

            for batch in tqdm(dataloader, dynamic_ncols=True):
                state, info = self.env.reset(options=batch[0])
                log_defect_history = [info["log_defect"]]
                shortest_length_history = [info["shortest_length"]]
                time_history = [info["time"]]

                done = False
                episode_reward = 0
                steps = 0

                while not done:
                    state = TensorDict({k: v.unsqueeze(0).to(device)
                                       for k, v in state.items()}, batch_size=[])
                    action, _, _ = self.get_action(state.to(device))
                    next_state, reward, terminated, truncated, info = self.env.step(
                        action)
                    log_defect_history.append(info["log_defect"])
                    shortest_length_history.append(info["shortest_length"])
                    time_history.append(info["time"])
                    done = terminated or truncated
                    episode_reward += reward
                    steps += 1
                    state = next_state

                total_reward += episode_reward
                total_steps += steps

                shortness += min(shortest_length_history)
                success_count += min(shortest_length_history) < 1.05
                time_taken += time_history[-1] - time_history[0]
                length_improvement += shortest_length_history[-0] - \
                    shortest_length_history[-1]

            return {
                "avg_reward": total_reward.item() / num_samples,
                "avg_steps": total_steps / num_samples,
                "success_rate": success_count / num_samples,
                "avg_shortness": shortness / num_samples,
                "avg_length_improvement": length_improvement / num_samples,
                "avg_time": time_taken / num_samples
            }
