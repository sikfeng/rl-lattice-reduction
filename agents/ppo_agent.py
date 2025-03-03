from typing import Dict, Tuple, Union

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchrl.data import ListStorage, ReplayBuffer
from torchrl.objectives.value.functional import generalized_advantage_estimate
from tqdm import tqdm

from reduction_env import BKZEnvConfig, VectorizedReductionEnvironment


class LatticeTransformer(nn.Module):
    def __init__(
        self, basis_dim: int, feature_dim: int, nhead: int = 4,
        num_layers: int = 3, dim_feedforward: int = 512, dropout_p: float = 0.1
    ) -> None:
        super().__init__()

        # Initial embedding for each basis vector and its features
        self.basis_dim = basis_dim
        self.feature_dim = feature_dim

        # Transform each basis vector into a higher-dim representation
        self.vector_embedding = nn.Linear(feature_dim, dim_feedforward)

        # Positional encoding to maintain vector order information
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, basis_dim, dim_feedforward))

        # Transformer encoder layers
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward*2,
            dropout=dropout_p,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers)

        # Output projection
        self.output_projection = nn.Linear(dim_feedforward, dim_feedforward//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, basis_dim, feature_dim]
                containing features for each basis vector

        Returns:
            Tensor of shape [batch_size, basis_dim, output_dim]
        """
        # Initial embedding
        x = self.vector_embedding(x)

        # Add positional encoding
        x = x + self.pos_encoder

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Project output
        return self.output_projection(x)


class ActorCritic(nn.Module):
    def __init__(
        self, basis_dim: int, action_history_size: int, action_dim: int,
        dropout_p: float = 0.1, nhead: int = 4, transformer_layers: int = 3
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_history_size = action_history_size
        self.basis_dim = basis_dim

        # Per-vector feature dimension
        # basis vector, GS vector, norm, other features
        self.vector_feature_dim = basis_dim * 3 + 2

        # Transformer for processing basis vectors and their features
        self.lattice_transformer = LatticeTransformer(
            basis_dim=basis_dim,
            feature_dim=self.vector_feature_dim,
            nhead=nhead,
            num_layers=transformer_layers,
            dropout_p=dropout_p
        )

        # Global feature processor (for features that describe the whole basis)
        self.global_processor = nn.Sequential(
            nn.Linear(basis_dim*2 + 1 + (basis_dim*(basis_dim-1))//2, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 128)
        )

        # Action history processor
        self.action_processor = nn.Sequential(
            nn.Embedding(action_dim + 1, 32, action_dim),
            nn.Flatten(-2, -1),
            nn.Linear(action_history_size * 32, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64)
        )

        # Combine transformer outputs
        transformer_output_dim = (
            basis_dim * (self.lattice_transformer.output_projection.out_features))

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(transformer_output_dim + 128 + 64, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(transformer_output_dim + 128 + 64, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 1)
        )

    def forward(self, tensordict: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        tensordict = self.preprocess_inputs(tensordict)

        batch_size = tensordict["basis"].shape[0]

        # Prepare per-vector features
        # [batch_size, basis_dim, basis_dim]
        basis_vectors = tensordict["basis"]
        # [batch_size, basis_dim, basis_dim]
        gs_vectors = tensordict["gs_basis"]

        # Add basis norm as a feature for each vector
        # [batch_size, basis_dim, 1]
        basis_norms = torch.norm(basis_vectors, dim=2, keepdim=True)
        # [batch_size, basis_dim, 1]
        gs_norms = torch.norm(gs_vectors, dim=2, keepdim=True)

        # Create mu coefficient features (where mu[i,j] shows projection of b_i onto b*_j)
        mu_features = tensordict.get("mu_coefficients", torch.zeros(
            batch_size, self.basis_dim, self.basis_dim, device=basis_vectors.device))

        # For each vector, prepare its feature vector
        vector_features = []
        for i in range(self.basis_dim):
            # Get this vector's features
            basis_vector = basis_vectors[:, i, :]  # [batch_size, basis_dim]
            gs_vector = gs_vectors[:, i, :]  # [batch_size, basis_dim]
            basis_norm = basis_norms[:, i, :]  # [batch_size, 1]
            gs_norm = gs_norms[:, i, :]  # [batch_size, 1]

            # Get this vector's mu coefficients (projections onto previous GS vectors)
            mu_i = mu_features[:, i, :i]  # [batch_size, i]
            # Pad to full dimension
            padded_mu = torch.zeros(
                batch_size, self.basis_dim, device=basis_vectors.device)
            if i > 0:
                padded_mu[:, :i] = mu_i

            # Combine features for this vector
            this_vector_features = torch.cat([
                basis_vector,
                gs_vector,
                basis_norm,
                gs_norm,
                padded_mu
            ], dim=2 if padded_mu.dim() == 3 else 1)

            vector_features.append(this_vector_features)

        # Stack vector features [batch_size, basis_dim, feature_dim]
        vector_features = torch.stack(vector_features, dim=1)

        # Process through transformer
        transformed_features = self.lattice_transformer(vector_features)

        # Flatten transformer output
        flattened_features = transformed_features.reshape(batch_size, -1)

        # Process global features
        global_features = torch.cat([
            tensordict["orthogonality_defect"],
            tensordict["pairwise_angles"],
            torch.cat([basis_norms.squeeze(2), gs_norms.squeeze(2)], dim=1)
        ], dim=1)

        global_embedding = self.global_processor(global_features)

        # Process action history
        action_history = tensordict["action_history"].to(torch.long)
        action_history[(action_history == -1)] = self.action_dim
        action_embedding = self.action_processor(action_history)

        # Combine all features
        combined = torch.cat([
            flattened_features,
            global_embedding,
            action_embedding
        ], dim=1)

        # Forward through actor and critic heads
        return self.actor(combined), self.critic(combined).squeeze(-1)

    @staticmethod
    def preprocess_inputs(tensordict: TensorDict) -> TensorDict:
        basis = tensordict["basis"]

        batch_size, basis_dim, _ = basis.shape
        device = basis.device

        # Calculate Gram-Schmidt orthogonalization
        gs_basis = torch.zeros_like(basis)
        mu = torch.zeros(batch_size, basis_dim, basis_dim, device=device)

        for i in range(batch_size):
            b = basis[i]
            gs = torch.zeros_like(b)

            # GS orthogonalization
            for j in range(basis_dim):
                gs[j] = b[j].clone()
                for k in range(j):
                    # Calculate projection coefficient
                    mu_jk = torch.dot(b[j], gs[k]) / torch.dot(gs[k], gs[k])
                    mu[i, j, k] = mu_jk
                    # Subtract projection
                    gs[j] = gs[j] - mu_jk * gs[k]

            gs_basis[i] = gs

        # Calculate norms of basis vectors
        basis_norms = torch.norm(basis, dim=2)
        gs_norms = torch.norm(gs_basis, dim=2)

        # Calculate orthogonality defect
        prod_norms = torch.prod(basis_norms, dim=1, keepdim=True)
        det_basis = torch.abs(torch.linalg.det(basis))
        orthogonality_defect = prod_norms / (det_basis.unsqueeze(1) + 1e-10)

        # Calculate pairwise angles
        normalized_basis = basis / (basis_norms.unsqueeze(2) + 1e-10)
        cos_angles = torch.bmm(
            normalized_basis, normalized_basis.transpose(1, 2))

        # Extract upper triangular part (excluding diagonal)
        pairwise_angles = []
        for i in range(batch_size):
            angles = []
            for j in range(basis_dim):
                for k in range(j+1, basis_dim):
                    angles.append(cos_angles[i, j, k])
            pairwise_angles.append(torch.stack(angles))

        pairwise_angles = torch.stack(pairwise_angles)

        # Create and return TensorDict with all features
        return TensorDict({
            "basis": basis,
            "gs_basis": gs_basis,
            "basis_norms": torch.cat([basis_norms, gs_norms], dim=1),
            "orthogonality_defect": orthogonality_defect,
            "pairwise_angles": pairwise_angles,
            "mu_coefficients": mu,
            "action_history": tensordict["action_history"]
        }, batch_size=batch_size)


class PPOConfig:
    def __init__(self, env_config: BKZEnvConfig = None, lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, epochs=4, batch_size=64, dropout_p=0.2):
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.env_config = env_config if env_config is not None else BKZEnvConfig()


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
                dones = torch.zeros(
                    (dataloader.batch_size, ), dtype=torch.bool)
                episode_reward = 0
                steps = 0

                while not torch.all(dones):
                    actions, _, _ = self.get_action(states.to(device))
                    next_states, rewards, terminateds, truncateds, infos = self.env.step(
                        actions)
                    log_defect_history.append(infos["log_defect"])
                    dones = torch.logical_or(terminateds, truncateds)
                    episode_reward += rewards
                    steps += 1
                    states = next_states

                total_reward += episode_reward.sum().item()
                total_steps += steps

                # Check success
                final_log_defect = log_defect_history[-1]
                success_count += torch.count_nonzero(
                    final_log_defect - batch["lll_log_defect"] < 1e-3)

            return {
                'avg_reward': total_reward / num_samples,
                'avg_steps': total_steps / num_samples,
                'success_rate': success_count / num_samples
            }
