from typing import Dict, Literal, Tuple, Union

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import GSNormEncoder, ActionEncoder


class ContinuousPolicyHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        dropout_p: float = 0.1,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.dropout_p = dropout_p
        self.hidden_dim = hidden_dim

        self.termination_actor = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
            nn.Flatten(-2),
        )
        self.block_size_actor = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.hidden_dim, 2),
        )

    def forward(
        self,
        features: torch.Tensor,
        previous_action: torch.Tensor,
        basis_dim: torch.Tensor,
    ) -> torch.Tensor:
        termination_prob = self.termination_actor(features)
        block_size_output = self.block_size_actor(features)
        absolute_size = (previous_action + 1) + (
            1 - F.sigmoid(block_size_output[:, 0])
        ) * (basis_dim - previous_action)
        block_size_logits = torch.stack(
            [
                absolute_size,
                F.softplus(block_size_output[:, 1]),
            ],
            dim=1,
        )

        return termination_prob, block_size_logits


class DiscretePolicyHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        dropout_p: float = 0.1,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.dropout_p = dropout_p
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.termination_actor = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
            nn.Flatten(-2),
        )
        self.actor = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.hidden_dim, self.action_dim - 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        previous_action: torch.Tensor,
        basis_dim: torch.Tensor,
    ) -> torch.Tensor:
        termination_logit = self.termination_actor(features)

        logits = self.actor(features)

        # logits [batch_size, action_dim]
        # basis_dim [batch_size]
        # last_action [batch_size]
        indices = (
            torch.arange(start=2, end=self.action_dim + 1, device=logits.device)
            .unsqueeze(0)
            .expand(features.size(0), -1)
        )
        previous_action_ = previous_action.unsqueeze(1).expand(logits.size(0), 1)
        basis_dim_ = basis_dim.unsqueeze(1).expand(logits.size(0), 1)
        # mask entries which are False will be masked out
        # indices >= thresholds are entries not smaller than previous block size
        # indices <= basis_dim are entries with block size smaller than dim
        valid_mask = (indices >= previous_action_ + 1) & (indices <= basis_dim_)
        masked_logits = logits.masked_fill(~valid_mask, float("-inf"))

        return termination_logit, masked_logits


class JointEnergyBasedPolicyHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        dropout_p: float = 0.1,
        hidden_dim: int = 128,
        index_embedding_dim: int = 32,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.dropout_p = dropout_p
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.index_embedding_dim = index_embedding_dim

        self.termination_actor = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid(),
            nn.Flatten(-2),
        )
        self.actor = nn.Sequential(
            nn.Linear(self.feature_dim + self.index_embedding_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.hidden_dim, 1),
            nn.Flatten(-2),
        )
        self.index_embedding = nn.Sequential(
            nn.Unflatten(-1, (-1, 1)),
            nn.Linear(1, self.index_embedding_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(self.index_embedding_dim, self.index_embedding_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        previous_action: torch.Tensor,
        basis_dim: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        termination_logit = self.termination_actor(features)

        indices = (
            torch.arange(start=2, end=self.action_dim + 1, device=features.device)
            .float()
            .unsqueeze(0)
            .expand(features.size(0), -1)
        )
        index_embeddings = self.index_embedding(indices)
        features_reshaped = features.unsqueeze(1).expand(
            -1, index_embeddings.size(1), -1
        )
        logits = self.actor(torch.cat([features_reshaped, index_embeddings], dim=2))

        previous_action_ = previous_action.unsqueeze(1).expand(logits.size(0), 1)
        basis_dim_ = basis_dim.unsqueeze(1).expand(logits.size(0), 1)
        # mask entries which are False will be masked out
        valid_mask = (indices >= previous_action_) & (indices <= basis_dim_)
        masked_logits = logits.masked_fill(~valid_mask, float("-inf"))

        return termination_logit, masked_logits


class ActorCritic(nn.Module):
    def __init__(
        self,
        policy_type: Union[
            Literal["continuous"], Literal["discrete"], Literal["joint-energy"]
        ],
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

        self.combined_feature_dim = (
            self.gs_norms_embedding_hidden_dim + self.action_embedding_dim
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
        elif self.policy_type == "joint-energy":
            self.actor = JointEnergyBasedPolicyHead(
                feature_dim=self.combined_feature_dim,
                action_dim=self.max_basis_dim,
                dropout_p=self.dropout_p,
                hidden_dim=self.actor_hidden_dim,
                index_embedding_dim=32,
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
        self, tensordict: TensorDict, cached_states: Dict[str, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if cached_states is None:
            cached_states = dict()
        tensordict = self.preprocess_inputs(tensordict)

        gs_norms = tensordict["gs_norms"]  # [batch_size, basis_dim]
        basis_dim = tensordict["basis_dim"]  # [batch_size]
        previous_action = tensordict["last_action"]

        pad_mask = self.gs_norms_encoder._generate_pad_mask(basis_dim)
        gs_norms_embedding = self.gs_norms_encoder(gs_norms, pad_mask)

        prev_action_embedding = self.action_encoder(previous_action, basis_dim)
        prev_action_embedding = prev_action_embedding.unsqueeze(1).expand(
            -1, gs_norms_embedding.size(1), -1
        )

        combined = torch.cat(
            [gs_norms_embedding.mean(dim=1), prev_action_embedding.mean(dim=1)], dim=1
        )

        termination_prob, block_output = self.actor(
            combined, previous_action, basis_dim
        )
        return (
            termination_prob,
            block_output,
            self.critic(combined).squeeze(-1),
        )

    def preprocess_inputs(self, tensordict: TensorDict) -> TensorDict:
        basis = tensordict["basis"]  # [batch_size, max_basis_dim, max_basis_dim]
        basis_dim = tensordict["basis_dim"]  # [batch_size]
        _, max_basis_dim, _ = basis.shape

        mask = torch.arange(max_basis_dim, device=basis.device) < basis_dim.view(
            -1, 1, 1
        )
        masked_basis = basis * mask

        _, R = torch.linalg.qr(masked_basis)
        diag = torch.nan_to_num(torch.diagonal(R, dim1=-2, dim2=-1).abs().log(), nan=0)

        gs_norms = diag * mask.squeeze(1)

        return TensorDict(
            {
                "gs_norms": gs_norms,
                "last_action": tensordict["last_action"],
                "basis_dim": basis_dim,
            },
            batch_size=[],
        )
