from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from tensordict import TensorDict
import torch
import torch.nn as nn
import torch.optim as optim

from modules import GSNormEncoder, ActionEncoder
from reduction_env import ReductionEnvConfig, VectorizedReductionEnvironment


class SimulatorConfig:
    def __init__(
        self,
        lr: float = 1e-5,
        hidden_dim: int = 128,  # TODO: must equal actor critic gs norm embedding hidden dim!
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


class Simulator(nn.Module):
    def __init__(
        self,
        gs_norms_encoder: GSNormEncoder,
        action_encoder: ActionEncoder,
        dropout_p: float = 0.1,
        hidden_dim: int = 128,
    ) -> None:
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
        )
        self.time_simulator = nn.Sequential(
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

        device = current_gs_norms.device

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

        time_sim_context = torch.cat(
            [
                gs_norms_embedding.mean(dim=1),
                prev_action_embedding.mean(dim=1),
                current_action_embedding.mean(dim=1),
            ],
            dim=1,
        )
        simulated_time = self.time_simulator(time_sim_context).exp()

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
                    torch.ones(i + 1, i + 1, device=device) * float("-inf"), diagonal=1
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
        tgt = self.gs_norms_encoder.input_projection(target_gs_norms)
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
            torch.ones(seq_len, seq_len, device=device) * float("-inf"), diagonal=1
        )

        gs_norm_sim_context = torch.cat(
            [gs_norms_embedding, prev_action_embedding, current_action_embedding], dim=2
        )

        decoder_output = self.gs_norm_simulator(
            tgt=tgt, memory=gs_norm_sim_context, tgt_mask=tgt_mask
        )

        simulated_gs_norms = self.gs_norm_projection(decoder_output).squeeze(-1)
        return simulated_gs_norms


class SimulatorTrainer(nn.Module):
    def __init__(
        self,
        env_config: ReductionEnvConfig,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        device: Union[torch.device, str] = "cpu",
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
        self.simulator = Simulator(
            gs_norms_encoder=self.gs_norm_encoder,
            action_encoder=self.action_encoder,
            hidden_dim=hidden_dim,
        )

        self.sim_optimizer = optim.AdamW(
            [
                *self.simulator.parameters(),
                *self.gs_norm_encoder.parameters(),
                *self.action_encoder.parameters(),
            ],
            lr=lr,
        )

        state, info = self.env.reset()
        self.state = state.to(self.device)
        self.info = info.to(self.device)

    def train(self) -> dict:
        """Performs one training step by interacting with the environment and updating the model."""
        with torch.no_grad():
            state = self._get_processed_state()
            action, continue_mask = self._generate_actions(state)
            next_state, time_taken, next_info = self._step_environment(action)

        metrics = self._compute_loss_and_update(
            state, action, continue_mask, next_state, time_taken
        )
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
            next_info["time"] - self.info["time"],
            next_info,
        )

    def _compute_loss_and_update(
        self,
        state: TensorDict,
        action: torch.Tensor,
        continue_mask: torch.Tensor,
        next_state: TensorDict,
        time_taken: torch.Tensor,
    ) -> dict:
        """Computes loss and updates model weights."""
        if not continue_mask.any():
            return {}

        # Extract relevant tensors for loss calculation
        current_gs = state["gs_norms"][continue_mask]
        prev_act = state["last_action"][continue_mask]
        basis_dim = state["basis_dim"][continue_mask]
        current_act = action[continue_mask].float()
        target_gs = next_state["gs_norms"][continue_mask]
        target_time = time_taken[continue_mask]

        # Model predictions
        predicted_gs, predicted_time, _ = self.simulator(
            current_gs_norms=current_gs,
            previous_action=prev_act,
            current_action=current_act,
            basis_dim=basis_dim,
            target_gs_norms=target_gs,
        )

        gs_loss = nn.functional.mse_loss(predicted_gs, target_gs)
        time_loss = nn.functional.mse_loss(predicted_time, target_time)
        total_loss = gs_loss + time_loss

        self.sim_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.sim_optimizer.step()

        return {
            "train/gs_loss": gs_loss.item(),
            "train/time_loss": time_loss.item(),
            "train/total_loss": total_loss.item(),
        }

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
        diag = torch.diagonal(R, dim1=-2, dim2=-1).abs().log()
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

        metrics = self._compute_loss_and_update(
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
        trainer = SimulatorTrainer(env_config=env_config)
        trainer.load_state_dict(state_dict)

        return trainer