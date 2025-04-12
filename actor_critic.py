from typing import Dict, Tuple

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import BasisEncoder, ActionEncoder


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
    ) -> None:
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


class ActorCritic(nn.Module):
    def __init__(
        self,
        policy_type: str,
        max_basis_dim: int,
        dropout_p: float = 0.1,
        basis_embedding_hidden_dim: int = 128,
        action_embedding_dim: int = 8,
        actor_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.policy_type = policy_type
        self.simulator = None
        self.max_basis_dim = max_basis_dim

        self.basis_embedding_hidden_dim = basis_embedding_hidden_dim
        self.action_embedding_dim = action_embedding_dim
        self.dropout_p = dropout_p

        self.basis_encoder = BasisEncoder(
            dropout_p=self.dropout_p,
            max_basis_dim=self.max_basis_dim,
            hidden_dim=self.basis_embedding_hidden_dim,
        )
        self.action_encoder = ActionEncoder(
            max_basis_dim=self.max_basis_dim,
            embedding_dim=self.action_embedding_dim,
        )

        self.log_std = nn.Parameter(torch.ones(1))

        self.combined_feature_dim = (
            self.basis_embedding_hidden_dim
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

        basis = tensordict["basis"]  # [batch_size, basis_dim]
        basis_dim = tensordict["basis_dim"]  # [batch_size]
        previous_action = tensordict["last_action"]

        if "basis_embedding" in cached_states:
            basis_embedding = cached_states["basis_embedding"]
        else:
            pad_mask = self.basis_encoder._generate_pad_mask(basis_dim)
            basis_embedding = self.basis_encoder(basis, pad_mask)

        if "prev_action_embedding" in cached_states:
            prev_action_embedding = cached_states["prev_action_embedding"]
        else:
            prev_action_embedding = self.action_encoder(previous_action, basis_dim)

        combined = torch.cat([
            basis_embedding.mean(dim=1),
            prev_action_embedding.mean(dim=1)
        ], dim=1)

        cached_states["basis_embedding"] = basis_embedding
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
                "basis": basis,
                "gs_norms": gs_norms,
                "last_action": tensordict["last_action"],
                "basis_dim": basis_dim,
            },
            batch_size=[],
        )
