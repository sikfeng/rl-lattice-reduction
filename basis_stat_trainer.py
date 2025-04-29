from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim

from modules import GSNormDecoder, GSNormEncoder, ActionEncoder
from reduction_env import ReductionEnvConfig, VectorizedReductionEnvironment


class BasisStatPredictorConfig:
    def __init__(
        self,
        lr: float = 1e-5,
        hidden_dim: int = 128,  # TODO: must equal actor critic gs norm embedding hidden dim!
    ) -> None:
        self.lr = lr
        self.hidden_dim = hidden_dim

    def __str__(self):
        self_dict = vars(self)
        return f"SimulatorConfig({', '.join(f'{k}={v}' for k, v in self_dict.items())})"


class BasisStatPredictor(nn.Module):
    def __init__(
        self,
        gs_norms_encoder: GSNormEncoder,
        action_encoder: ActionEncoder,
        dropout_p: float = 0.1,
        hidden_dim: int = 128,
        device: Union[torch.device, str] = "cpu",
        teacher_forcing: bool = False,
        normalize_gs_norms: bool = False,
    ) -> None:
        super().__init__()

        self.gs_norms_encoder = gs_norms_encoder
        self.action_encoder = action_encoder
        self.dropout_p = dropout_p
        self.hidden_dim = hidden_dim
        self.device = device
        self.teacher_forcing = teacher_forcing
        self.normalize_gs_norms = normalize_gs_norms

        self.gs_norms_decoder = GSNormDecoder(
            gs_norms_encoder=self.gs_norms_encoder,
            input_dim=self.gs_norms_encoder.hidden_dim
            + 2 * self.action_encoder.embedding_dim,
            normalize_inputs=self.normalize_gs_norms,
        )
        self.previous_action_predictor = nn.Sequential(
            nn.Linear(
                self.hidden_dim + 2 * self.action_encoder.embedding_dim,
                self.hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, 1),
        )
        self.current_action_predictor = nn.Sequential(
            nn.Linear(
                self.hidden_dim + 2 * self.action_encoder.embedding_dim,
                self.hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, 1),
        )
        self.log_defect_predictor = nn.Sequential(
            nn.Linear(
                self.hidden_dim + 2 * self.action_encoder.embedding_dim,
                self.hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, 1),
        )
        self.basis_dim_predictor = nn.Sequential(
            nn.Linear(
                self.hidden_dim + 2 * self.action_encoder.embedding_dim,
                self.hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, 1),
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

        if "gs_norms_embedding" in cached_states:
            gs_norms_embedding = cached_states["gs_norms_embedding"]
        else:
            pad_mask = self.gs_norms_encoder._generate_pad_mask(basis_dim)
            gs_norms_embedding = self.gs_norms_encoder(current_gs_norms, pad_mask)

        if "prev_action_embedding" in cached_states:
            prev_action_embedding = cached_states["prev_action_embedding"]
        else:
            prev_action_embedding = self.action_encoder(previous_action, basis_dim)

        current_action_embedding = self.action_encoder(current_action, basis_dim)

        prev_action_embedding = torch.cat(
            [
                torch.zeros(
                    [prev_action_embedding.size(0), 1, prev_action_embedding.size(2)],
                    device=prev_action_embedding.device,
                ),
                prev_action_embedding,
            ],
            dim=1,
        )
        current_action_embedding = torch.cat(
            [
                torch.zeros(
                    [
                        current_action_embedding.size(0),
                        1,
                        current_action_embedding.size(2),
                    ],
                    device=current_action_embedding.device,
                ),
                current_action_embedding,
            ],
            dim=1,
        )
        pred_context = torch.cat(
            [
                gs_norms_embedding[:, 0, :],
                prev_action_embedding.mean(dim=1),
                current_action_embedding.mean(dim=1),
            ],
            dim=1,
        )
        previous_action_pred = self.previous_action_predictor(pred_context)
        current_action_pred = self.current_action_predictor(pred_context)
        log_defect_pred = self.log_defect_predictor(pred_context)
        basis_dim_pred = self.basis_dim_predictor(pred_context)

        pred_gs_norms = self.gs_norms_decoder(
            gs_norms_embedding=gs_norms_embedding,
            prev_action_embedding=prev_action_embedding,
            current_action_embedding=current_action_embedding,
            target_gs_norms=target_gs_norms if self.teacher_forcing else None,
            basis_dim=basis_dim,
        )

        cached_states["gs_norms_embedding"] = gs_norms_embedding
        cached_states["prev_action_embedding"] = prev_action_embedding

        preds = {
            "gs_norms": pred_gs_norms,
            "previous_action": previous_action_pred.squeeze(-1),
            "current_action": current_action_pred.squeeze(-1),
            "log_defect": log_defect_pred.squeeze(-1),
            "basis_dim": basis_dim_pred.squeeze(-1),
        }

        return preds, cached_states

    def _compute_loss(
        self,
        state: TensorDict,
        action: torch.Tensor,
        continue_mask: torch.Tensor,
        current_info: Dict[str, Any],
    ) -> dict:
        """Computes loss and updates model weights."""
        if not continue_mask.any():
            return {}, torch.tensor([0.0], device=self.device, requires_grad=True)

        # Extract relevant tensors for loss calculation
        current_gs = state["gs_norms"][continue_mask]
        prev_act = state["last_action"][continue_mask]
        basis_dim = state["basis_dim"][continue_mask]
        current_act = action[continue_mask].float()
        log_defects = current_info["log_defect"][continue_mask]

        # Model predictions
        preds, _ = self(
            current_gs_norms=current_gs,
            previous_action=prev_act,
            current_action=current_act,
            basis_dim=basis_dim,
            target_gs_norms=current_gs if self.teacher_forcing else None,
        )

        losses = {
            "gs_losses": ((preds["gs_norms"] - current_gs) ** 2).mean(dim=1),
            "prev_act_losses": (preds["previous_action"] - prev_act) ** 2,
            "current_act_losses": (preds["current_action"] - current_act) ** 2,
            "log_defect_losses": (preds["log_defect"] - log_defects) ** 2,
            "basis_dim_losses": (preds["basis_dim"] - basis_dim) ** 2,
        }
        total_losses = sum(losses.values())
        total_loss = total_losses.mean()

        metrics = [
            {f"train/{loss_type}": losses[loss_type][i] for loss_type in losses}
            for i in range(total_losses.size(0))
        ]

        return metrics, total_loss


class BasisStatTrainer(nn.Module):
    def __init__(
        self,
        env_config: ReductionEnvConfig,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        device: Union[torch.device, str] = "cpu",
        teacher_forcing: bool = True,
    ) -> None:
        super().__init__()
        self.env_config = env_config
        self.env = VectorizedReductionEnvironment(env_config)
        self.device = device

        self.gs_norm_encoder = GSNormEncoder(
            max_basis_dim=self.env_config.net_dim, hidden_dim=hidden_dim, dropout_p=0.1
        )
        self.action_encoder = ActionEncoder(
            max_basis_dim=self.env_config.net_dim, embedding_dim=hidden_dim
        )
        self.basis_stat_predictor = BasisStatPredictor(
            gs_norms_encoder=self.gs_norm_encoder,
            action_encoder=self.action_encoder,
            hidden_dim=hidden_dim,
            device=device,
            teacher_forcing=teacher_forcing,
        )

        self.optimizer = optim.AdamW(
            self.basis_stat_predictor.parameters(),
            lr=lr,
        )

        state, info = self.env.reset()
        self.state = state.to(self.device)
        self.info = info.to(self.device)

        self.teacher_forcing = teacher_forcing

    def train(self) -> dict:
        """Performs one training step by interacting with the environment and updating the model."""
        with torch.no_grad():
            state = self._get_processed_state()
            action, continue_mask = self._generate_actions(state)
            next_state, next_info = self._step_environment(action)

        metrics, loss = self.basis_stat_predictor._compute_loss(
            state,
            action,
            continue_mask,
            self.info,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        self._update_state(next_state, next_info)
        return metrics

    def _get_processed_state(self) -> TensorDict:
        """Preprocesses the current state and moves it to the specified device."""
        state = self.preprocess_inputs(self.state)
        return state

    def _generate_actions(self, state: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates valid actions based on the current state."""
        batch_size = state["basis_dim"].size(0)
        terminate = torch.rand(batch_size, device=self.device) < 0.2
        action = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        continue_mask = ~terminate

        if continue_mask.any():
            last_actions = state["last_action"]
            basis_dims = state["basis_dim"]
            valid_continue_mask = (last_actions < basis_dims - 1) & continue_mask
            valid_indices = torch.where(valid_continue_mask)[0]

            for idx in valid_indices:
                min_action = int(last_actions[idx].item() + 1)
                max_action = basis_dims[idx].item() - 1
                action[idx] = torch.randint(
                    min_action, max_action + 1, (1,), device=self.device
                ).squeeze()

        return action, continue_mask

    def _step_environment(
        self, action: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, dict]:
        """Steps the environment and processes the next state."""
        next_state, _, _, _, next_info = self.env.step(action)
        return (
            self.preprocess_inputs(next_state),
            next_info,
        )

    def _update_state(self, next_state: TensorDict, next_info: dict) -> None:
        """Updates the current state and environment information."""
        self.state = next_state
        self.info = TensorDict(next_info)

    @staticmethod
    def preprocess_inputs(tensordict: TensorDict) -> TensorDict:
        """Converts basis matrices into Gram-Schmidt norms for model input."""
        basis = tensordict["basis"]
        basis_dim = tensordict["basis_dim"]
        max_basis_dim = basis.size(1)

        # Create mask for valid basis dimensions
        mask = torch.arange(max_basis_dim, device=basis.device) < basis_dim.view(
            -1, 1, 1
        )
        masked_basis = basis * mask

        # Compute Gram-Schmidt norms via QR decomposition
        _, R = torch.linalg.qr(masked_basis)
        diag = torch.nan_to_num(torch.diagonal(R, dim1=-2, dim2=-1).abs().log(), nan=0)
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

    def evaluate(self):
        """Evaluates the model on a given environment without training."""
        self.eval()
        with torch.no_grad():
            state = self._get_processed_state()
            action, continue_mask = self._generate_actions(state)
            next_state, time_taken, next_info = self._step_environment(action)

        metrics, loss = self.basis_stat_predictor._compute_loss(
            state, action, continue_mask, next_state, time_taken
        )
        self._update_state(next_state, next_info)
        return metrics

    def save(self, path: Path):
        checkpoint = {
            "state_dict": self.state_dict(),
            "env_config": self.env_config,
        }
        torch.save(checkpoint, path)
        return

    @staticmethod
    def load(path: Path, device: Union[str, torch.device]):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"]
        env_config = checkpoint["env_config"]
        trainer = BasisStatTrainer(env_config=env_config)
        trainer.load_state_dict(state_dict)

        return trainer
