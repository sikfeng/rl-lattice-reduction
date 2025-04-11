from collections import defaultdict
import math
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchrl.data import ListStorage, ReplayBuffer
from torchrl.objectives.value.functional import generalized_advantage_estimate

from reduction_env import ReductionEnvConfig, VectorizedReductionEnvironment


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        self.max_len = max_len

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1), :]
        return x


class GSNormEncoder(nn.Module):
    def __init__(self, dropout_p: float, max_basis_dim: int, hidden_dim: int):
        super().__init__()

        self.max_basis_dim = max_basis_dim
        self.dropout_p = dropout_p
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.pos_encoding = PositionalEncoding(
            self.hidden_dim,
            max_len=self.max_basis_dim,
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=4,
            dim_feedforward=4 * self.hidden_dim,
            dropout=dropout_p,
            batch_first=True,
        )
        self.basis_transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=3,
        )
        self.basis_encoder_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, basis, attn_mask):
        x = self.input_projection(basis)
        x = self.pos_encoding(x)
        x = self.basis_transformer_encoder(x, src_key_padding_mask=attn_mask)
        x = self.basis_encoder_projection(x)
        return x

    def _generate_attn_mask(self, seq_lengths):
        batch_size = seq_lengths.size(0)
        padding_mask = torch.zeros(
            batch_size,
            self.max_basis_dim,
            dtype=torch.bool,
            device=seq_lengths.device,
        )

        for i, length in enumerate(seq_lengths.int()):
            padding_mask[i, length:] = True

        return padding_mask


class ActionEncoder(nn.Module):
    def __init__(self, max_basis_dim, embedding_dim):
        super().__init__()

        self.max_basis_dim = max_basis_dim
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(2, self.embedding_dim),
            nn.LeakyReLU(),
        )

    def forward(self, action, basis_dim):
        indices = torch.arange(self.max_basis_dim, device=action.device).unsqueeze(0)
        # block size \(b\) corresponds to action id \(b - 1\)
        indices = F.pad(indices, (0, self.max_basis_dim - indices.size(1)), value=0)
        indices = indices.expand(-1, self.max_basis_dim)
        basis_dim_ = basis_dim.unsqueeze(-1).expand(-1, self.max_basis_dim)
        effective_block_size = (
            torch.min(
                action.unsqueeze(-1).expand(-1, self.max_basis_dim),
                basis_dim_ - indices,
            )
            + 1
        )
        relative_block_size = effective_block_size / basis_dim_
        action_embedding = torch.stack(
            [effective_block_size, relative_block_size], dim=1
        )
        action_embedding = self.encoder(action_embedding.transpose(dim0=-2, dim1=-1))

        return action_embedding


class ContinuousPolicyHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        dropout_p: float = 0.1,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.dropout_p = dropout_p
        self.hidden_dim = hidden_dim

        self.actor = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.hidden_dim, 3),
        )

    def forward(
        self,
        features: torch.Tensor,
        previous_action: torch.Tensor,
        basis_dim: torch.Tensor,
    ) -> torch.Tensor:
        actor_output = self.actor(features)
        absolute_size = ((previous_action + 1)
                         + (1 - F.sigmoid(actor_output[:, 1])) * (basis_dim - previous_action))
        actor_output = torch.stack([
            F.sigmoid(actor_output[:, 0]),
            absolute_size,
            F.softplus(actor_output[:, 2]),
        ], dim=1)

        return actor_output


class DiscretePolicyHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        dropout_p: float = 0.1,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.dropout_p = dropout_p
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Softmax(dim=-1),  # should be removed
        )

    def forward(
        self,
        features: torch.Tensor,
        previous_action: torch.Tensor,
        basis_dim: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.actor(features)

        # logits [batch_size, action_dim]
        # basis_dim [batch_size]
        # last_action [batch_size]
        indices = (
            torch.arange(self.action_dim, device=logits.device)
            .unsqueeze(0)
            .expand_as(logits)
        )
        thresholds = previous_action.unsqueeze(1).expand_as(logits)
        basis_dim_ = basis_dim.unsqueeze(1).expand_as(logits)
        # mask entries which are False will be masked out
        # indices >= thresholds are entries not smaller than previous block size
        # indices <= basis_dim are entries with block size smaller than dim
        # indices == 0 is the termination action
        valid_mask = ((indices >= thresholds) & (indices <= basis_dim_)) | (indices == 0)  # [batch_size, action_dim]
        masked_logits = logits.masked_fill(~valid_mask, float("-inf"))

        return masked_logits


class Simulator(nn.Module):
    def __init__(
        self,
        gs_norms_encoder: GSNormEncoder,
        action_encoder: ActionEncoder,
        dropout_p: float = 0.1,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.gs_norms_encoder = gs_norms_encoder
        self.action_encoder = action_encoder
        self.dropout_p = dropout_p
        self.hidden_dim = hidden_dim

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.gs_norms_encoder.hidden_dim
                    + 2 * self.action_encoder.embedding_dim,
            nhead=4,
            dim_feedforward=4 * self.hidden_dim,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.gs_norm_simulator = nn.TransformerDecoder(
            decoder_layer,
            num_layers=3,
        )
        self.gs_norm_projection = nn.Sequential(
            nn.Linear(
                self.gs_norms_encoder.hidden_dim
                + 2 * self.action_encoder.embedding_dim,
                self.hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, 1),
            nn.Softplus(),
        )
        self.time_simulator = nn.Sequential(
            nn.Linear(
                self.hidden_dim + 2 * self.action_encoder.embedding_dim,
                self.hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        current_gs_norms: torch.Tensor,
        previous_action: torch.Tensor,
        current_action: torch.Tensor,
        basis_dim: torch.Tensor,
        cached_states: Dict[str, torch.Tensor] = None,
        target_gs_norms: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if cached_states is None:
            cached_states = {}

        device = current_gs_norms.device

        if "gs_norms_embedding" in cached_states:
            gs_norms_embedding = cached_states["gs_norms_embedding"]
        else:
            gs_norms_reshaped = current_gs_norms.unsqueeze(-1)
            attn_mask = self.gs_norms_encoder._generate_attn_mask(basis_dim)
            gs_norms_embedding = self.gs_norms_encoder(gs_norms_reshaped, attn_mask)

        if "prev_action_embedding" in cached_states:
            prev_action_embedding = cached_states["prev_action_embedding"]
        else:
            prev_action_embedding = self.action_encoder(previous_action, basis_dim)

        current_action_embedding = self.action_encoder(current_action, basis_dim)

        time_sim_context = torch.cat(
            [
                gs_norms_embedding.mean(dim=1),
                prev_action_embedding.mean(dim=1),
                current_action_embedding.mean(dim=1),
            ],
            dim=1,
        )
        simulated_time = self.time_simulator(time_sim_context)

        if target_gs_norms is None:
            simulated_gs_norms = self._autoregressive_generation(
                gs_norms_embedding=gs_norms_embedding,
                prev_action_embedding=prev_action_embedding,
                current_action_embedding=current_action_embedding,
                basis_dim=basis_dim,
                device=device,
            )
        else:
            simulated_gs_norms = self._teacher_forced_generation(
                gs_norms_embedding=gs_norms_embedding,
                prev_action_embedding=prev_action_embedding,
                current_action_embedding=current_action_embedding,
                target_gs_norms=target_gs_norms,
                device=device,
            )

        cached_states["gs_norms_embedding"] = gs_norms_embedding
        cached_states["prev_action_embedding"] = prev_action_embedding

        return simulated_gs_norms, simulated_time.squeeze(-1), cached_states

    def _autoregressive_generation(
        self,
        gs_norms_embedding: torch.Tensor,
        prev_action_embedding: torch.Tensor,
        current_action_embedding: torch.Tensor,
        basis_dim: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        # buffers for storing generated output
        simulated_gs_norms = torch.zeros(
            (gs_norms_embedding.size(0), gs_norms_embedding.size(1)),
            device=device,
        )
        generated_sequence = torch.zeros(
            (gs_norms_embedding.size(0), 1, 1),
            device=device,
        )

        # encoder features
        gs_norm_sim_context = torch.cat(
            [gs_norms_embedding, prev_action_embedding, current_action_embedding], dim=2
        )

        for i in range(basis_dim.max()):
            # embedding features for sequence generated so far
            tgt = self.gs_norms_encoder.input_projection(generated_sequence)
            tgt = self.gs_norms_encoder.pos_encoding(tgt)
            tgt = torch.cat(
                [
                    tgt,
                    prev_action_embedding[:, : i + 1, :],
                    current_action_embedding[:, : i + 1, :],
                ],
                dim=2,
            )

            # causal mask for autoregressive generation
            tgt_mask = None
            if i > 0:
                tgt_mask = torch.triu(
                    torch.ones(i + 1, i + 1, device=device) * float("-inf"),
                    diagonal=1
                )

            decoder_output = self.gs_norm_simulator(
                tgt=tgt,
                memory=gs_norm_sim_context,
                tgt_mask=tgt_mask,
            )
            # [batch_size, 1, hidden_dim]
            current_hidden = decoder_output[:, -1:, :]
            predicted_gs_norm = self.gs_norm_projection(current_hidden)
            simulated_gs_norms[:, i] = predicted_gs_norm.squeeze(1).squeeze(1)

            if i < basis_dim.max() - 1:
                generated_sequence = torch.cat(
                    [
                        generated_sequence,
                        predicted_gs_norm.detach(),
                    ],
                    dim=1,
                )

        return predicted_gs_norm

    def _teacher_forced_generation(
        self,
        gs_norms_embedding: torch.Tensor,
        prev_action_embedding: torch.Tensor,
        current_action_embedding: torch.Tensor,
        target_gs_norms: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        tgt = self.gs_norms_encoder.input_projection(target_gs_norms.unsqueeze(-1))
        tgt = self.gs_norms_encoder.pos_encoding(tgt)
        seq_len = target_gs_norms.size(1)
        tgt = torch.cat(
            [
                tgt,
                prev_action_embedding[:, :seq_len, :],
                current_action_embedding[:, :seq_len, :],
            ],
            dim=2,
        )
        tgt_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1
        )

        gs_norm_sim_context = torch.cat([
            gs_norms_embedding,
            prev_action_embedding,
            current_action_embedding
        ], dim=2)

        decoder_output = self.gs_norm_simulator(
            tgt=tgt,
            memory=gs_norm_sim_context,
            tgt_mask=tgt_mask
        )

        simulated_gs_norms = self.gs_norm_projection(decoder_output).squeeze(-1)
        return simulated_gs_norms


class InverseModel(nn.Module):
    def __init__(
        self,
        input_embedding_dim: int,
        hidden_dim: int,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2 * input_embedding_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        current_embedding: torch.Tensor,
        next_embedding: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat(
            [
                current_embedding.flatten(start_dim=1),
                next_embedding.flatten(start_dim=1)
            ], dim=1)
        return self.model(combined)


class ActorCritic(nn.Module):
    def __init__(
        self,
        policy_type: str,
        max_basis_dim: int,
        dropout_p: float = 0.1,
        gs_norms_embedding_hidden_dim: int = 128,
        action_embedding_dim: int = 8,
        actor_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.policy_type = policy_type
        self.simulator = None
        self.max_basis_dim = max_basis_dim

        self.gs_norms_embedding_hidden_dim = gs_norms_embedding_hidden_dim
        self.action_embedding_dim = action_embedding_dim
        self.dropout_p = dropout_p

        self.gs_norms_encoder = GSNormEncoder(
            dropout_p=self.dropout_p,
            max_basis_dim=self.max_basis_dim,
            hidden_dim=self.gs_norms_embedding_hidden_dim,
        )
        self.action_encoder = ActionEncoder(
            max_basis_dim=self.max_basis_dim,
            embedding_dim=self.action_embedding_dim,
        )

        self.log_std = nn.Parameter(torch.ones(1))

        self.combined_feature_dim = (
            self.gs_norms_embedding_hidden_dim
            + self.action_embedding_dim
        )
        self.actor_hidden_dim = actor_hidden_dim

        if self.policy_type == "continuous":
            self.actor = ContinuousPolicyHead(
                feature_dim=self.combined_feature_dim,
                dropout_p=self.dropout_p,
                hidden_dim=self.actor_hidden_dim,
            )
        elif self.policy_type == "discrete":
            self.actor = DiscretePolicyHead(
                feature_dim=self.combined_feature_dim,
                action_dim=self.max_basis_dim,
                dropout_p=self.dropout_p,
                hidden_dim=self.actor_hidden_dim,
            )
        else:
            raise ValueError(
                "self.policy_type is of unknown type: %s", self.policy_type
            )
        self.critic = nn.Sequential(
            nn.Linear(self.combined_feature_dim, self.actor_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.actor_hidden_dim, 1),
        )

    def forward(
        self,
        tensordict: TensorDict,
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
            attn_mask = self.gs_norms_encoder._generate_attn_mask(basis_dim)
            gs_norms_embedding = self.gs_norms_encoder(gs_norms_reshaped, attn_mask)

        if "prev_action_embedding" in cached_states:
            prev_action_embedding = cached_states["prev_action_embedding"]
        else:
            prev_action_embedding = self.action_encoder(previous_action, basis_dim)

        combined = torch.cat([
            gs_norms_embedding.mean(dim=1),
            prev_action_embedding.mean(dim=1)
        ], dim=1)

        cached_states["gs_norms_embedding"] = gs_norms_embedding
        cached_states["prev_action_embedding"] = prev_action_embedding

        actor_output = self.actor(combined, previous_action, basis_dim)
        return actor_output, self.critic(combined).squeeze(-1), cached_states

    def preprocess_inputs(self, tensordict: TensorDict) -> TensorDict:
        basis = tensordict["basis"]  # [batch_size, max_basis_dim, max_basis_dim]
        basis_dim = tensordict["basis_dim"]  # [batch_size]
        _, max_basis_dim, _ = basis.shape

        mask = torch.arange(max_basis_dim, device=basis.device) < basis_dim.view(-1, 1, 1)
        masked_basis = basis * mask

        _, R = torch.linalg.qr(masked_basis)
        diag = torch.diagonal(R, dim1=-2, dim2=-1).abs()

        gs_norms = diag * mask.squeeze(1)

        return TensorDict(
            {
                "gs_norms": gs_norms,
                "last_action": tensordict["last_action"],
                "basis_dim": basis_dim,
            },
            batch_size=[],
        )


class PPOConfig:
    def __init__(
        self,
        lr: float = 1e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        clip_grad_norm: float = 0.5,
        epochs: int = 4,
        minibatch_size: int = 64,
    ) -> None:
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.clip_grad_norm = clip_grad_norm
        self.epochs = epochs
        self.minibatch_size = minibatch_size

    def __str__(self):
        self_dict = vars(self)
        return f"PPOConfig({', '.join(f'{k}={v}' for k, v in self_dict.items())})"

class SimulatorConfig:
    def __init__(
        self,
        lr: float = 1e-5,
        hidden_dim: int = 128, # TODO: must equal actor critic gs norm embedding hidden dim!
        gs_norm_weight: float = 1.0,
        time_weight: float = 1.0,
        inverse_weight: float = 0.05,
    ) -> None:
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.gs_norm_weight = gs_norm_weight
        self.time_weight = time_weight
        self.inverse_weight = inverse_weight

    def __str__(self):
        self_dict = vars(self)
        return f"SimulatorConfig({', '.join(f'{k}={v}' for k, v in self_dict.items())})"

class AgentConfig:
    def __init__(
        self,
        ppo_config: PPOConfig,
        device: Union[torch.device, str],
        batch_size: int = 1,
        dropout_p: float = 0.1,
        env_config: Optional[ReductionEnvConfig] = None,
        simulator: bool = False,
        pred_type: str = Union[Literal["continuous"], Literal["discrete"]],
        simulator_reward_weight: float = 0.1,
        simulator_config: Optional[SimulatorConfig] = None,
    ) -> None:
        self.ppo_config = ppo_config
        self.device = device
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.simulator = simulator
        self.pred_type = pred_type
        self.env_config = env_config if env_config is not None else ReductionEnvConfig()
        self.simulator_reward_weight = 0.0 if not simulator else simulator_reward_weight
        self.simulator_config = (
            None
            if not simulator
            else simulator_config if simulator_config is not None else SimulatorConfig()
        )

    def __str__(self):
        self_dict = vars(self)
        return f"AgentConfig({', '.join(f'{k}={v}' for k, v in self_dict.items())})"


class Agent(nn.Module):
    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__()
        self.agent_config = agent_config
        self.device = self.agent_config.device

        self.actor_critic = ActorCritic(
            policy_type=self.agent_config.pred_type,
            max_basis_dim=self.agent_config.env_config.net_dim,
            dropout_p=self.agent_config.dropout_p,
        )
        self.optimizer = optim.AdamW(
            self.actor_critic.parameters(),
            lr=self.agent_config.ppo_config.lr,
        )

        self.simulator = None
        if self.agent_config.simulator:
            self.simulator = Simulator(
                gs_norms_encoder=self.actor_critic.gs_norms_encoder,
                action_encoder=self.actor_critic.action_encoder,
                dropout_p=self.agent_config.dropout_p,
                hidden_dim=self.agent_config.simulator_config.hidden_dim,
            )
            self.inverse_model = InverseModel(
                input_embedding_dim=self.actor_critic.gs_norms_encoder.hidden_dim
                                    * self.agent_config.env_config.net_dim,
                hidden_dim=self.actor_critic.gs_norms_encoder.hidden_dim,
                dropout_p=self.agent_config.dropout_p,
            )
            self.sim_optimizer = optim.AdamW(
                [*self.simulator.parameters()] + [*self.inverse_model.parameters()],
                lr=self.agent_config.simulator_config.lr,
            )

        self.mse_loss = nn.MSELoss()

        def sample_transform(tensordict: TensorDict) -> TensorDict:
            tensordict = TensorDict(
                {key: tensordict[key].flatten(0, 1) for key in tensordict.keys()}
            )
            return tensordict

        self.replay_buffer = ReplayBuffer(
            storage=ListStorage(),
            transform=sample_transform,
        )

        self.batch_size = self.agent_config.batch_size
        self.agent_config.env_config.batch_size = self.batch_size
        self.env = VectorizedReductionEnvironment(self.agent_config.env_config)
        self.state, self.info = self.env.reset()
        self.state = TensorDict(self.state, batch_size=[]).to(self.device)

    def simulate(
        self,
        current_gs_norms: torch.Tensor,
        previous_action: torch.Tensor,
        basis_dim: torch.Tensor,
        current_action: torch.Tensor,
        cached_states: Dict[str, torch.Tensor] = None,
        target_gs_norms: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if self.simulator is None:
            raise RuntimeError("ActorCritic model not initialized to simulate.")

        return self.simulator(
            current_gs_norms=current_gs_norms,
            previous_action=previous_action,
            current_action=current_action,
            basis_dim=basis_dim,
            cached_states=cached_states,
            target_gs_norms=target_gs_norms,
        )

    def store_transition(
        self,
        state: TensorDict[str, torch.Tensor],
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_state: TensorDict[str, torch.Tensor],
        time_taken: torch.Tensor,
    ):
        td = TensorDict(
            {
                "state": {
                    "basis": state["basis"],
                    "last_action": state["last_action"],
                    "basis_dim": state["basis_dim"],
                },
                "action": action,
                "log_prob": log_prob,
                "reward": reward,
                "done": done,
                "next_state": {
                    "basis": next_state["basis"],
                    "last_action": next_state["last_action"],
                    "basis_dim": next_state["basis_dim"],
                },
                "time_taken": time_taken,
            },
            batch_size=[action.size(0)],
        )
        self.replay_buffer.extend(
            [*td.split(np.ones(td.batch_size[0], dtype=int).tolist(), dim=0)]
        )

    def get_action(self, state: TensorDict) -> Tuple[int, float, float, torch.Tensor]:
        with torch.no_grad():
            basis_dim = state["basis_dim"]
            last_action = state["last_action"]
            logits, value, _ = self.actor_critic(state)

            if self.agent_config.pred_type == "continuous":
                termination_prob, block_size_float, block_size_std = logits.unbind(dim=1)
                if self.training:
                    terminate_dist = torch.distributions.Bernoulli(termination_prob)
                    terminate = terminate_dist.sample()
                    termination_log_probs = terminate_dist.log_prob(terminate)

                    block_size_dist = torch.distributions.Normal(
                        block_size_float,
                        block_size_std,
                    )
                    cont_block = block_size_dist.rsample()

                    with torch.no_grad():
                        # discrete block size for environment
                        discrete_block = torch.clamp(
                            torch.round(cont_block),
                            min=last_action + 1,
                            max=basis_dim,
                        )
                    # straight through estimator to preserve gradients
                    # unnecessary for PPO, but necessary for SAC
                    block_size = (
                        cont_block + (discrete_block.float() - cont_block).detach()
                    )
                    block_size_log_probs = block_size_dist.log_prob(block_size)

                else:
                    terminate = termination_prob
                    termination_log_probs = torch.log(termination_prob)

                    block_size_dist = torch.distributions.Normal(
                        block_size_float,
                        block_size_std,
                    )
                    block_size = torch.round(block_size_float)
                    block_size_log_probs = block_size_dist.log_prob(block_size_float)

                log_probs = torch.where(
                    terminate > 0.5,
                    termination_log_probs,  # only termination log prob matters
                    termination_log_probs
                    + block_size_log_probs
                    - torch.log(
                        block_size_dist.cdf(basis_dim + 0.5)
                        - block_size_dist.cdf(last_action + 0.5)  # adjust for boundary
                    ),
                )

                action = torch.where(
                    terminate > 0.5,
                    torch.tensor(0, device=block_size.device),
                    block_size - 1,
                )  # consolidate action ids

            elif self.agent_config.pred_type == "discrete":
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_probs = dist.log_prob(action)

            return action, log_probs, value, logits

    def _update_continuous(self) -> Dict[str, float]:
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
            _, values, _ = self.actor_critic(states)
            _, next_values, _ = self.actor_critic(next_states)

        advantages, returns = generalized_advantage_estimate(
            gamma=self.agent_config.ppo_config.gamma,
            lmbda=self.agent_config.ppo_config.gae_lambda,
            state_value=values.unsqueeze(1),
            next_state_value=next_values.unsqueeze(1),
            reward=rewards.unsqueeze(1),
            done=dones.unsqueeze(1),
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses, critic_losses, entropy_losses, total_losses = [], [], [], []
        term_losses, block_losses = [], []
        term_entropies, block_entropies = [], []
        clip_fractions = []
        approx_kls = []

        if self.agent_config.simulator:
            gs_norm_sim_losses = []
            time_sim_losses = []
            inverse_losses = []

        for _ in range(self.agent_config.ppo_config.epochs):
            logits, values, cached_states = self.actor_critic(states)
            term_probs, block_mean_preds, block_pred_std = logits.unbind(dim=1)
            # term_probs, block_preds, values [batch_size]

            # Create action masks
            terminate_mask = actions == 0  # [batch_size]
            continue_mask = ~terminate_mask  # [batch_size]

            # Calculate termination log probs (Bernoulli distribution)
            term_dist = torch.distributions.Bernoulli(probs=term_probs)
            term_log_probs = term_dist.log_prob(terminate_mask.float())  # [batch_size]

            # Calculate block size log probs (Normal distribution)
            block_dist = torch.distributions.Normal(
                loc=block_mean_preds[continue_mask],
                scale=block_pred_std[continue_mask],
            )
            block_log_probs = block_dist.log_prob(
                actions[continue_mask].float()
            )  # [continue_size]

            # Combine log probabilities
            new_log_probs = term_log_probs
            new_log_probs[continue_mask] = (
                new_log_probs[continue_mask] + block_log_probs
            )

            r"""
            The probability ratio is
            \[
            r_t(\theta) = \frac{\pi_{\theta}(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
            \]
            This ratio measures how much the new policy differs from the old one.
            """
            ratios = (new_log_probs - old_log_probs).exp()

            r"""
            PPO modifies the standard policy gradient update using a clipped surrogate objective:
            \[
            L(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right) \right]
            \]
            """
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(
                    ratios,
                    1 - self.agent_config.ppo_config.clip_epsilon,
                    1 + self.agent_config.ppo_config.clip_epsilon,
                )
                * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(values, returns.squeeze(1))

            term_entropy = term_dist.entropy().mean()
            block_entropy_all = torch.zeros(actions.size(0), device=self.device)
            block_entropy_all[continue_mask] = block_dist.entropy()
            block_entropy = block_entropy_all.mean()
            # additivity property holds because termination and block size are independent
            entropy_loss = -(term_entropy + block_entropy)

            with torch.no_grad():
                term_loss = (
                    -torch.min(surr1[terminate_mask], surr2[terminate_mask]).mean()
                    if terminate_mask.any()
                    else torch.tensor(0.0)
                )

                # Block size component loss
                block_loss = (
                    -torch.min(surr1[continue_mask], surr2[continue_mask]).mean()
                    if continue_mask.any()
                    else torch.tensor(0.0)
                )

            actor_critic_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

            if self.agent_config.simulator:
                simulator_losses = self.get_sim_loss(
                    actions,
                    states,
                    next_states,
                    batch["time_taken"],
                )
                gs_norm_sim_loss = simulator_losses["gs_norm_loss"].nanmean()
                time_sim_loss = simulator_losses["time_loss"].nanmean()
                inverse_loss = simulator_losses["inverse_loss"].nanmean()
                simulator_loss = gs_norm_sim_loss + time_sim_loss + inverse_loss

            self.optimizer.zero_grad()
            if self.agent_config.simulator:
                self.sim_optimizer.zero_grad()

            actor_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.agent_config.ppo_config.clip_grad_norm,
            )

            if self.agent_config.simulator:
                simulator_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*self.actor_critic.parameters(), *self.simulator.parameters()],
                    self.agent_config.ppo_config.clip_grad_norm,
                )

            self.optimizer.step()
            if self.agent_config.simulator:
                self.sim_optimizer.step()

            # Logging metrics
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())
            total_losses.append(actor_critic_loss.item())

            term_losses.append(term_loss.item())
            block_losses.append(block_loss.item())

            term_entropies.append(term_entropy.item())
            block_entropies.append(block_entropy.item())

            if self.agent_config.simulator:
                gs_norm_sim_losses.append(gs_norm_sim_loss.item())
                time_sim_losses.append(time_sim_loss.item())
                inverse_losses.append(inverse_loss.item())

            clipped = ((ratios < 1 - self.agent_config.ppo_config.clip_epsilon)
                       | (ratios > 1 + self.agent_config.ppo_config.clip_epsilon))
            clip_fractions.append(clipped.float().mean().item())

            approx_kl = (old_log_probs - new_log_probs).mean().item()
            approx_kls.append(approx_kl)

        self.replay_buffer.empty()

        metrics = {
            "update/avg_actor_loss": np.mean(actor_losses),
            "update/avg_critic_loss": np.mean(critic_losses),
            "update/avg_term_loss": np.mean(term_losses),
            "update/avg_block_loss": np.mean(block_losses),
            "update/avg_term_entropy": np.mean(term_entropies),
            "update/avg_block_entropy": np.mean(block_entropies),
            "update/avg_entropy": np.mean(entropy_losses),
            "update/total_loss": np.mean(total_losses),
            "update/approx_kl": np.mean(approx_kl),
            "update/advantages_mean": advantages.mean().item(),
            "update/advantages_std": advantages.std().item(),
            "update/clip_fraction": np.mean(clip_fractions),
        }
        if self.agent_config.simulator:
            metrics.update(
                {
                    "update/avg_gs_norm_sim_loss": np.mean(gs_norm_sim_losses),
                    "update/avg_time_sim_loss": np.mean(time_sim_losses),
                    "update/avg_inverse_loss": np.mean(inverse_losses),
                }
            )
        return metrics

    def _update_discrete(self) -> Dict[str, float]:
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
            _, values, _ = self.actor_critic(states)
            _, next_values, _ = self.actor_critic(next_states)

        advantages, returns = generalized_advantage_estimate(
            gamma=self.agent_config.ppo_config.gamma,
            lmbda=self.agent_config.ppo_config.gae_lambda,
            state_value=values.unsqueeze(1),
            next_state_value=next_values.unsqueeze(1),
            reward=rewards.unsqueeze(1),
            done=dones.unsqueeze(1),
        )

        actor_losses, critic_losses, entropy_losses, total_losses = [], [], [], []
        clip_fractions = []
        approx_kls = []

        if self.agent_config.simulator:
            gs_norm_sim_losses = []
            time_sim_losses = []
            inverse_losses = []

        for _ in range(self.agent_config.ppo_config.epochs):
            logits, values, cached_states = self.actor_critic(states)
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
            surr2 = (
                torch.clamp(
                    ratios,
                    1 - self.agent_config.ppo_config.clip_epsilon,
                    1 + self.agent_config.ppo_config.clip_epsilon,
                )
                * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(values, returns.squeeze(1))
            entropy_loss = -dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

            actor_critic_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
            if self.agent_config.simulator:
                simulator_losses = self.get_sim_loss(
                    actions,
                    states,
                    next_states,
                    batch["time_taken"],
                )
                gs_norm_sim_loss = simulator_losses["gs_norm_loss"].mean()
                time_sim_loss = simulator_losses["time_loss"].mean()
                inverse_loss = simulator_losses["inverse_loss"].mean()
                simulator_loss = gs_norm_sim_loss + time_sim_loss + inverse_loss

            self.optimizer.zero_grad()
            if self.agent_config.simulator:
                self.sim_optimizer.zero_grad()

            actor_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.agent_config.ppo_config.clip_grad_norm,
            )

            if self.agent_config.simulator:
                simulator_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*self.actor_critic.parameters(), *self.simulator.parameters()],
                    self.agent_config.ppo_config.clip_grad_norm,
                )

            self.optimizer.step()
            if self.agent_config.simulator:
                self.sim_optimizer.step()

            # Logging metrics
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())
            total_losses.append(loss.item())

            if self.agent_config.simulator:
                gs_norm_sim_losses.append(gs_norm_sim_loss.item())
                time_sim_losses.append(time_sim_loss.item())
                inverse_losses.append(inverse_loss.item())

            clipped = ((ratios < 1 - self.agent_config.ppo_config.clip_epsilon)
                       | (ratios > 1 + self.agent_config.ppo_config.clip_epsilon))
            clip_fractions.append(clipped.float().mean().item())

            approx_kl = (old_log_probs - new_log_probs).mean().item()
            approx_kls.append(approx_kl)

        self.replay_buffer.empty()

        metrics = {
            "update/avg_actor_loss": np.mean(actor_losses),
            "update/avg_critic_loss": np.mean(critic_losses),
            "update/avg_entropy": np.mean(entropy_losses),
            "update/total_loss": np.mean(total_losses),
            "update/approx_kl": np.mean(approx_kl),
            "update/advantages_mean": advantages.mean().item(),
            "update/advantages_std": advantages.std().item(),
            "update/clip_fraction": np.mean(clip_fractions),
        }
        if self.agent_config.simulator:
            metrics.update(
                {
                    "update/avg_gs_norm_sim_loss": np.mean(gs_norm_sim_losses),
                    "update/avg_time_sim_loss": np.mean(time_sim_losses),
                    "update/avg_inverse_loss": np.mean(inverse_losses),
                }
            )
        return metrics

    def get_sim_loss(
        self,
        actions: torch.Tensor,
        states: torch.Tensor,
        next_states: torch.Tensor,
        time_taken: torch.Tensor,
    ) -> Dict[str, float]:
        current_features = self.actor_critic.preprocess_inputs(states)
        next_features = self.actor_critic.preprocess_inputs(next_states)

        gs_norm_sim_loss = torch.full_like(actions, float("nan"))
        time_sim_loss = torch.full_like(actions, float("nan"))
        inverse_loss = torch.full_like(actions, float("nan"))

        continue_mask = actions != 0
        if continue_mask.any():
            predicted_gs_norms, predicted_time, _ = self.simulate(
                current_gs_norms=current_features["gs_norms"][continue_mask],
                previous_action=current_features["last_action"][continue_mask],
                basis_dim=current_features["basis_dim"][continue_mask],
                current_action=actions[continue_mask].float(),
                target_gs_norms=next_features["gs_norms"][continue_mask],
            )

            gs_norm_sim_loss[continue_mask] = (
                (predicted_gs_norms - next_features["gs_norms"][continue_mask]) ** 2
            ).mean(dim=1)
            time_sim_loss[continue_mask] = (
                predicted_time - time_taken[continue_mask]
            ) ** 2

            attn_mask = self.actor_critic.gs_norms_encoder._generate_attn_mask(
                states["basis_dim"][continue_mask]
            )
            current_embedding = self.actor_critic.gs_norms_encoder(
                current_features["gs_norms"][continue_mask].unsqueeze(-1),
                attn_mask
            )
            next_embedding = self.actor_critic.gs_norms_encoder(
                next_features["gs_norms"][continue_mask].unsqueeze(-1),
                attn_mask
            )

            inverse_action = (
                self.inverse_model(
                    current_embedding,
                    next_embedding,
                ).squeeze(1)
                * states["basis_dim"][continue_mask]
            )
            inverse_loss[continue_mask] = (
                inverse_action - actions[continue_mask].float()
            ) ** 2

        losses = {
            "gs_norm_loss": gs_norm_sim_loss,
            "inverse_loss": inverse_loss,
            "time_loss": time_sim_loss,
        }
        return losses

    def update(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.agent_config.ppo_config.minibatch_size:
            return dict()

        if self.agent_config.pred_type == "continuous":
            metrics = self._update_continuous()
        elif self.agent_config.pred_type == "discrete":
            metrics = self._update_discrete()
        return metrics

    def collect_experiences(self) -> Dict[str, float]:
        with torch.no_grad():
            action, log_prob, value, logits = self.get_action(self.state)
            next_state, rewards, terminated, truncated, next_info = self.env.step(action)
            reward = torch.stack(list(rewards.values()), dim=0).sum(dim=0)
            if self.agent_config.simulator:
                simulator_losses = self.get_sim_loss(
                    action,
                    self.state,
                    next_state,
                    next_info["time"],
                )
                simulator_reward = self.agent_config.simulator_reward_weight * torch.stack(
                    [
                        self.agent_config.simulator_config.gs_norm_weight
                        * simulator_losses["gs_norm_loss"],
                        self.agent_config.simulator_config.time_weight
                        * simulator_losses["time_loss"],
                        self.agent_config.simulator_config.inverse_weight
                        * simulator_losses[
                            "inverse_loss"
                        ],                    ],
                    dim=0,
                ).sum(
                    dim=0
                )
                simulator_reward_ = torch.where(action == 0, 0, simulator_reward)
                reward = reward + torch.clamp(simulator_reward_, max=10)
            done = terminated | truncated

            next_state = TensorDict(
                {k: v.to(self.device) for k, v in next_state.items()}, batch_size=[]
            )
            self.store_transition(
                self.state,
                action,
                log_prob,
                reward,
                done,
                next_state,
                next_info["time"] - self.info["time"],
            )

            metrics = [
                {
                    "episode/action": float(action[i]),
                    "episode/block_size": float("nan" if action[i] == 0 else action[i] + 1),
                    "episode/block_size_rel": float(
                        "nan"
                        if action[i] == 0
                        else (action[i] + 1) / self.state["basis_dim"][i]
                    ),
                    "episode/basis_dim": float(self.state["basis_dim"][i]),
                    "episode/time_taken": float(
                        next_info["time"][i] - self.info["time"][i]
                    ),
                    "episode/time_penalty": float(rewards["time_penalty"][i]),
                    "episode/length_reward": float(rewards["length_reward"][i]),
                    "episode/defect_reward": float(rewards["defect_reward"][i]),
                    "episode/total_reward": float(reward[i]),
                    "episode/action_log_prob": float(log_prob[i]),
                    "episode/value_estimate": float(value[i]),
                    "episode/actor_terminate_logit": float(logits[i][0]),
                    "episode/actor_block_mean_logit": float(logits[i][1]),
                    "episode/actor_block_std_logit": float(logits[i][2]),
                }
                for i in range(reward.size(0))
            ]
            if self.agent_config.simulator:
                for i in range(reward.size(0)):
                    metrics[i]["episode/simulator_reward"] = float(torch.clamp(simulator_reward[i], max=10))
                    metrics[i]["episode/simulator_reward_raw"] = float(simulator_reward[i])
                    metrics[i]["episode/simulator_gs_norm_loss"] = float(simulator_losses["gs_norm_loss"][i])
                    metrics[i]["episode/simulator_time_loss"] = float(simulator_losses["time_loss"][i])
                    metrics[i]["episode/simulator_inverse_loss"] = float(simulator_losses["inverse_loss"][i])

            self.state = next_state
            self.info = next_info

            return metrics

    def evaluate(
        self,
        batch: TensorDict[str, Union[torch.Tensor, TensorDict[str, torch.Tensor]]]
    ) -> Dict:
        self.eval()
        state, info = self.env.reset(options=batch)
        log_defect_history = [info["log_defect"].item()]
        shortest_length_history = [info["shortest_length"].item()]
        time_history = [info["time"].item()]

        done = False
        episode_reward = 0
        episode_rewards = defaultdict(int)
        steps = 0

        while not done:
            state = state.to(self.device)
            action, _, _, _ = self.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            log_defect_history.append(info["log_defect"].item())
            shortest_length_history.append(info["shortest_length"].item())
            time_history.append(info["time"].item())
            done = terminated or truncated
            for key, value in reward.items():
                episode_rewards[key] += float(value)
            steps += 1
            state = next_state
        episode_reward = sum(episode_rewards.values())

        metrics = {
            "reward": episode_reward,
            "steps": steps,
            "smallest_defect": min(log_defect_history),
            "shortest_length": min(shortest_length_history),
            "success": float(min(shortest_length_history) < 1.05),
            "time": sum(time_history),
            "length_improvement": shortest_length_history[0]
                                  - min(shortest_length_history),
        }
        metrics.update(episode_rewards)

        return metrics

    def save(self, path: Path):
        checkpoint = {
            "state_dict": self.state_dict(),
            "agent_config": self.agent_config,
        }
        torch.save(checkpoint, path)
        return
