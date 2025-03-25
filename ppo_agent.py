import math
from pathlib import Path
from typing import Dict, Tuple, Union

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
    def __init__(self, max_basis_dim: int, action_dim: int, dropout_p: float = 0.1,
                 gs_norms_hidden_dim: int = 128, action_embedding_dim: int = 8,
                 simulator: bool = True) -> None:
        super().__init__()
        self.simulator = simulator
        self.action_dim = action_dim
        self.max_basis_dim = max_basis_dim

        self.gs_norms_hidden_dim = gs_norms_hidden_dim
        self.action_embedding_dim = action_embedding_dim
        self.dropout_p = dropout_p

        self.gs_norms_encoder = TransformerEncoder(
            input_dim=1,
            embedding_dim=self.gs_norms_hidden_dim,
            num_heads=4,
            num_layers=3,
            dropout_p=self.dropout_p,
            max_len=self.max_basis_dim
        )

        self.action_embedding = nn.Sequential(
            nn.Linear(2, self.action_embedding_dim),
            nn.LeakyReLU()
        )

        self.log_std = nn.Parameter(torch.zeros(1))

        self.combined_feature_dim = self.gs_norms_hidden_dim + self.action_embedding_dim
        self.actor_hidden_dim = 128

        # first output logit represents termination probability
        # second output logit represents block size
        self.actor = nn.Sequential(
            nn.Linear(self.combined_feature_dim, self.actor_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.actor_hidden_dim, self.actor_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.actor_hidden_dim, 2),
            nn.Sigmoid()
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

        if self.simulator:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.gs_norms_hidden_dim + self.action_embedding_dim,
                nhead=4,
                dim_feedforward=4*self.gs_norms_hidden_dim,
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

    def get_std(self) -> torch.Tensor:
        return torch.exp(self.log_std)

    def forward(self, tensordict: TensorDict, 
                cached_states: Dict[str, torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if cached_states is None:
            cached_states = dict()
        tensordict = self.preprocess_inputs(tensordict)

        gs_norms = tensordict["gs_norms"]  # [batch_size, basis_dim]
        basis_dim = tensordict["basis_dim"]  # [batch_size]
        previous_action = tensordict["last_action"]

        if "gs_norms_embedding" in cached_states:
            gs_norms_embedding = cached_states["gs_norms_embedding"]
        else:
            gs_norms_reshaped = gs_norms.unsqueeze(-1)
            gs_norms_embedding = self.gs_norms_encoder(gs_norms_reshaped, basis_dim)

        indices = torch.arange(self.action_dim, device=previous_action.device).unsqueeze(0)
        # block size \(b\) corresponds to action id \(b - 1\)
        if "prev_action_embedding" in cached_states:
            prev_action_embedding = cached_states["prev_action_embedding"]
        else:
            previous_effective_block_size = torch.min(previous_action.expand(-1, self.max_basis_dim), basis_dim - indices) + 1
            previous_relative_block_size = previous_effective_block_size / basis_dim
            prev_action_embedding = torch.stack([previous_effective_block_size, previous_relative_block_size], dim=1)
            prev_action_embedding = self.action_embedding(prev_action_embedding.transpose(dim0=-2, dim1=-1)).squeeze(1)

        # Combine all features
        combined = torch.cat([
            gs_norms_embedding.mean(dim=1),
            prev_action_embedding.mean(dim=1)
        ], dim=1)

        cached_states["gs_norms_embedding"] = gs_norms_embedding
        cached_states["prev_action_embedding"] = prev_action_embedding

        actor_output = self.actor(combined) # [terminate_prob, relative_size]
        termination_probs, relative_size = actor_output[:, 0], actor_output[:, 1]
        # scale sigmoid output in (0, 1) to allowed block sizes (prev_action + 1, basis_dim)
        scaled_block_size = (1 - relative_size) * (previous_action.squeeze(-1) + 1) + relative_size * basis_dim.squeeze(-1)
        return termination_probs, scaled_block_size, self.critic(combined).squeeze(-1), cached_states

    def simulate(self, current_gs_norms: torch.Tensor,
                 previous_action: torch.Tensor, current_action: torch.Tensor,
                 cached_states: Dict[str, torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.simulator:
            raise RuntimeError("ActorCritic model not initialized to simulate.")
        
        if cached_states is None:
            cached_states = dict()
        batch_size, basis_dim = current_gs_norms.shape
        device = current_gs_norms.device

        if "gs_norms_embedding" in cached_states:
            gs_norms_embedding = cached_states["gs_norms_embedding"]
        else:
            gs_norms_3d = current_gs_norms.unsqueeze(-1) # [batch_size, basis_dim, 1]
            gs_norms_embedding = self.gs_norms_encoder(
                gs_norms_3d,
                torch.full((batch_size,), basis_dim, device=device)
            )  # [batch_size, basis_dim, hidden_dim]

        indices = torch.arange(self.action_dim, device=previous_action.device).unsqueeze(0)
        # block size \(b\) corresponds to action id \(b - 1\)
        if "prev_action_embedding" in cached_states:
            prev_action_embedding = cached_states["prev_action_embedding"]
        else:
            previous_action_effective_block_size = torch.min(previous_action.expand(-1, basis_dim), basis_dim - indices) + 1
            previous_action_relative_block_size = previous_action_effective_block_size / basis_dim
            prev_action_embedding = torch.stack([previous_action_effective_block_size, previous_action_relative_block_size], dim=1)
            prev_action_embedding = self.action_embedding(prev_action_embedding.transpose(dim0=-2, dim1=-1)).squeeze(1)

        current_action_effective_block_size = torch.min(current_action.expand(-1, basis_dim), basis_dim - indices) + 1
        current_action_relative_block_size = current_action_effective_block_size / basis_dim
        current_action_embedding = torch.stack([current_action_effective_block_size, current_action_relative_block_size], dim=1)
        current_action_embedding = self.action_embedding(current_action_embedding.transpose(dim0=-2, dim1=-1)).squeeze(1)

        time_sim_context = torch.cat([
            gs_norms_embedding.mean(dim=1),
            prev_action_embedding.mean(dim=1),
            current_action_embedding.mean(dim=1)
        ], dim=1)
        simulated_time = self.time_simulator(time_sim_context)

        gs_norm_sim_context = torch.cat([
            gs_norms_embedding,
            current_action_embedding
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

            tgt = torch.cat([tgt, current_action_embedding[:, :i + 1, :]], dim=2)
            decoder_output = self.gs_norm_simulator(
                tgt=tgt,
                memory=gs_norm_sim_context,
                tgt_mask=tgt_mask
            )
            # [batch_size, 1, hidden_dim]
            current_prediction = decoder_output[:, -1:, :self.gs_norms_hidden_dim]

            # Project to get the norm value, tie with input projection weights
            current_prediction = current_prediction - self.gs_norms_encoder.input_projection.bias.unsqueeze(0).unsqueeze(0)
            predicted_norm = torch.nn.functional.linear(
                current_prediction,
                self.gs_norms_encoder.input_projection.weight.t(),
                bias=None
            )

            simulated_gs_norms[:, i] = predicted_norm.squeeze(-1).squeeze(-1)

            if i < basis_dim - 1:
                generated_sequence = torch.cat([
                    generated_sequence,
                    predicted_norm.detach()
                ], dim=1)

        cached_states["gs_norms_embedding"] = gs_norms_embedding
        cached_states["prev_action_embedding"] = prev_action_embedding

        return simulated_gs_norms, simulated_time.squeeze(-1), cached_states

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
                 dropout_p: float = 0.2, minibatch_size: int = 64,
                 simulator: bool = True):
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.clip_grad_norm = clip_grad_norm
        self.epochs = epochs
        self.dropout_p = dropout_p
        self.minibatch_size = minibatch_size
        self.simulator = simulator
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
            ppo_config.dropout_p,
            simulator=self.ppo_config.simulator
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

    def get_action(self, state: TensorDict) -> Tuple[int, float, float]:
        with torch.no_grad():
            termination_prob, block_size_float, value, _ = self.actor_critic(state)

            basis_dim = state["basis_dim"]
            last_action = state["last_action"]

            terminate = torch.bernoulli(termination_prob).bool()
            block_size = torch.round(block_size_float).int() # block sizes are integers
            block_size = torch.clamp(block_size, min=last_action + 1, max=basis_dim) # ensure block size is within valid range
            action = torch.where(terminate, torch.tensor(0, device=block_size.device), block_size - 1) # consolidate action ids

            termination_log_probs = torch.where(
                terminate,
                torch.log(termination_prob + 1e-8),
                torch.log(1 - termination_prob + 1e-8)
            )
            dist = torch.distributions.Normal(
                block_size_float,
                self.actor_critic.get_std().expand_as(block_size_float)
            )

            block_size_log_probs = dist.log_prob(block_size.float())
            log_probs = termination_log_probs + (1 - terminate.float()) * block_size_log_probs

        return action, log_probs, value

    def update(self) -> None:
        if len(self.replay_buffer) < self.ppo_config.minibatch_size:
            return dict()

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
            _, _, values, _ = self.actor_critic(states)
            _, _, next_values, _ = self.actor_critic(next_states)

        advantages, returns = generalized_advantage_estimate(
            gamma=self.ppo_config.gamma,
            lmbda=self.ppo_config.gae_lambda,
            state_value=values.unsqueeze(1),
            next_state_value=next_values.unsqueeze(1),
            reward=rewards.unsqueeze(1),
            done=dones.unsqueeze(1)
        )

        actor_losses, critic_losses, entropy_losses, total_losses = [], [], [], []
        term_losses, block_losses = [], []
        clip_fractions = []
        approx_kls = []

        if self.ppo_config.simulator:
            gs_norm_sim_losses = []
            time_sim_losses = []

        for _ in range(self.ppo_config.epochs):
            term_probs, block_preds, values, cached_states = self.actor_critic(states)
            # term_probs, block_preds, values [batch_size]

            # Create action masks
            terminate_mask = (actions == 0).squeeze(1) # [batch_size]
            continue_mask = ~terminate_mask # [batch_size]
            
            # Calculate termination log probs (Bernoulli distribution)
            term_dist = torch.distributions.Bernoulli(probs=term_probs)
            term_log_probs = term_dist.log_prob(terminate_mask.float()) # [batch_size]
            
            # Calculate block size log probs (Normal distribution)
            block_dist = torch.distributions.Normal(
                loc=block_preds[continue_mask],
                scale=self.actor_critic.get_std().expand_as(block_preds[continue_mask])
            )
            block_log_probs = block_dist.log_prob(actions[continue_mask].float().squeeze(1)) # [continue_size]
            
            # Combine log probabilities
            new_log_probs = torch.zeros_like(old_log_probs) # [batch_size, 1]
            new_log_probs[terminate_mask] = term_log_probs[terminate_mask].unsqueeze(1)
            if continue_mask.any():
                # Get log P(continue) for continue actions
                log_p_continue = term_dist.log_prob(0.0)  # [batch_size]
                log_p_continue = log_p_continue[continue_mask]  # [num_continue]
                
                # Get log P(block_size) for continue actions
                log_p_block = block_log_probs  # [num_continue]
                
                new_log_probs[continue_mask] = (log_p_continue + log_p_block).unsqueeze(1)

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
            term_entropy = term_dist.entropy().mean()
            block_entropy = block_dist.entropy().mean() if continue_mask.any() else 0.0
            entropy_loss = -(term_entropy + block_entropy)
            
            with torch.no_grad():
                term_loss = -torch.min(
                    surr1[terminate_mask],
                    surr2[terminate_mask]
                ).mean() if terminate_mask.any() else torch.tensor(0.0)
                
                # Block size component loss
                block_loss = -torch.min(
                    surr1[continue_mask],
                    surr2[continue_mask]
                ).mean() if continue_mask.any() else torch.tensor(0.0)

            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

            if self.ppo_config.simulator:
                current_features = self.actor_critic.preprocess_inputs(states)
                next_features = self.actor_critic.preprocess_inputs(next_states)
                predicted_gs_norms, predicted_time, _ = self.actor_critic.simulate(
                    current_features["gs_norms"],
                    current_features["last_action"],
                    actions.float(),
                    cached_states
                )

                # Calculate simulator losses
                gs_norm_sim_loss = torch.nn.functional.mse_loss(predicted_gs_norms, next_features["gs_norms"])
                time_sim_loss = torch.nn.functional.mse_loss(predicted_time, batch["time_taken"])

                loss = loss + 0.1 * gs_norm_sim_loss + 0.1 * time_sim_loss
                

            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.ppo_config.clip_grad_norm)
            self.optimizer.step()

            # Logging metrics
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())
            total_losses.append(loss.item())
            
            term_losses.append(term_loss.item())
            block_losses.append(block_loss.item())

            if self.ppo_config.simulator:
                gs_norm_sim_losses.append(gs_norm_sim_loss.item())
                time_sim_losses.append(time_sim_loss.item())

            clipped = (ratios < 1 - self.ppo_config.clip_epsilon) | (ratios > 1 + self.ppo_config.clip_epsilon)
            clip_fractions.append(clipped.float().mean().item())

            approx_kl = (old_log_probs - new_log_probs).mean().item()
            approx_kls.append(approx_kl)

        self.replay_buffer.empty()

        metrics = {
            "update/avg_actor_loss": np.mean(actor_losses),
            "update/avg_critic_loss": np.mean(critic_losses),
            "update/avg_term_loss": np.mean(term_losses),
            "update/avg_block_loss": np.mean(block_losses),
            "update/avg_entropy": np.mean(entropy_losses),
            "update/total_loss": np.mean(total_losses),
            "update/approx_kl": np.mean(approx_kl),
            "update/advantages_mean": advantages.mean().item(),
            "update/advantages_std": advantages.std().item(),
        }
        if self.ppo_config.simulator:
            metrics.update({
                "update/avg_gs_norm_sim_loss": np.mean(gs_norm_sim_losses),
                "update/avg_time_sim_loss": np.mean(time_sim_losses)
            })
        return metrics

    def collect_experiences(self) -> Dict[str, float]:
        total_reward = 0.0
        steps = 0
        total_log_prob = 0.0
        total_value = 0.0

        state, info = self.env.reset()
        done = False

        state = TensorDict({k: v.unsqueeze(0).to(self.device)
                           for k, v in state.items()}, batch_size=[])
        while not done:
            action, log_prob, value = self.get_action(state)
            next_state, reward, terminated, truncated, next_info = self.env.step(action)
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

    def evaluate(self, batch: TensorDict) -> Dict:
        self.eval()
        state, info = self.env.reset(options=batch[0])
        log_defect_history = [info["log_defect"]]
        shortest_length_history = [info["shortest_length"]]
        time_history = [info["time"]]

        done = False
        episode_reward = 0
        steps = 0

        while not done:
            state = TensorDict({k: v.unsqueeze(0).to(self.device)
                                for k, v in state.items()}, batch_size=[])
            action, _, _ = self.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            log_defect_history.append(info["log_defect"])
            shortest_length_history.append(info["shortest_length"])
            time_history.append(info["time"])
            done = terminated or truncated
            episode_reward += reward.item()
            steps += 1
            state = next_state

        return {
            "reward": episode_reward,
            "steps": steps,
            "shortest_length": min(shortest_length_history),
            "success": float(min(shortest_length_history) < 1.05),
            "time": time_history[-1] - time_history[0],
            "length_improvement": shortest_length_history[0] - min(shortest_length_history)
        }
    
    def save(self, path: Path):
        checkpoint = {
            'state_dict': self.state_dict(),
            'ppo_config': self.ppo_config,
        }
        torch.save(checkpoint, path)
        return
