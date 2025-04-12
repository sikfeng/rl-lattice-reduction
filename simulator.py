from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from modules import BasisEncoder, ActionEncoder


class SimulatorConfig:
    def __init__(
        self,
        lr: float = 1e-5,
        hidden_dim: int = 128, # TODO: must equal actor critic gs norm embedding hidden dim!
        basis_weight: float = 1.0,
        time_weight: float = 1.0,
        inverse_weight: float = 0.05,
    ) -> None:
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.basis_weight = basis_weight
        self.time_weight = time_weight
        self.inverse_weight = inverse_weight

    def __str__(self):
        self_dict = vars(self)
        return f"SimulatorConfig({', '.join(f'{k}={v}' for k, v in self_dict.items())})"


class Simulator(nn.Module):
    def __init__(
        self,
        basis_encoder: BasisEncoder,
        action_encoder: ActionEncoder,
        dropout_p: float = 0.1,
        hidden_dim: int = 128,
        max_basis_dim: int = 64,
    ) -> None:
        super().__init__()

        self.basis_encoder = basis_encoder
        self.action_encoder = action_encoder
        self.dropout_p = dropout_p
        self.hidden_dim = hidden_dim
        self.max_basis_dim = max_basis_dim

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.basis_encoder.hidden_dim
                    + 2 * self.action_encoder.embedding_dim,
            nhead=4,
            dim_feedforward=4 * self.hidden_dim,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.basis_simulator = nn.TransformerDecoder(
            decoder_layer,
            num_layers=3,
        )
        self.basis_projection = nn.Sequential(
            nn.Linear(
                self.basis_encoder.hidden_dim
                + 2 * self.action_encoder.embedding_dim,
                self.hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, self.max_basis_dim),
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
        current_basis: torch.Tensor,
        previous_action: torch.Tensor,
        current_action: torch.Tensor,
        basis_dim: torch.Tensor,
        cached_states: Dict[str, torch.Tensor] = None,
        target_basis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if cached_states is None:
            cached_states = {}

        device = current_basis.device

        if "basis_embedding" in cached_states:
            basis_embedding = cached_states["basis_embedding"]
        else:
            pad_mask = self.basis_encoder._generate_pad_mask(basis_dim)
            basis_embedding = self.basis_encoder(current_basis, pad_mask)

        if "prev_action_embedding" in cached_states:
            prev_action_embedding = cached_states["prev_action_embedding"]
        else:
            prev_action_embedding = self.action_encoder(previous_action, basis_dim)

        current_action_embedding = self.action_encoder(current_action, basis_dim)

        time_sim_context = torch.cat(
            [
                basis_embedding.mean(dim=1),
                prev_action_embedding.mean(dim=1),
                current_action_embedding.mean(dim=1),
            ],
            dim=1,
        )
        simulated_time = self.time_simulator(time_sim_context)

        if target_basis is None:
            simulated_basis = self._autoregressive_generation(
                basis_embedding=basis_embedding,
                prev_action_embedding=prev_action_embedding,
                current_action_embedding=current_action_embedding,
                basis_dim=basis_dim,
                device=device,
            )
        else:
            simulated_basis = self._teacher_forced_generation(
                basis_embedding=basis_embedding,
                prev_action_embedding=prev_action_embedding,
                current_action_embedding=current_action_embedding,
                target_basis=target_basis,
                device=device,
            )

        cached_states["basis_embedding"] = basis_embedding
        cached_states["prev_action_embedding"] = prev_action_embedding

        return simulated_basis, simulated_time.squeeze(-1), cached_states

    def _autoregressive_generation(
        self,
        basis_embedding: torch.Tensor,
        prev_action_embedding: torch.Tensor,
        current_action_embedding: torch.Tensor,
        basis_dim: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        # buffers for storing generated output
        simulated_basis = torch.zeros(
            (basis_embedding.size(0), basis_embedding.size(1)),
            device=device,
        )
        generated_sequence = torch.zeros(
            (basis_embedding.size(0), 1, 1),
            device=device,
        )

        # encoder features
        basis_sim_context = torch.cat(
            [basis_embedding, prev_action_embedding, current_action_embedding], dim=2
        )

        for i in range(basis_dim.max()):
            # embedding features for sequence generated so far
            tgt = self.basis_encoder.input_projection(generated_sequence)
            tgt = self.basis_encoder.pos_encoding(tgt)
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

            decoder_output = self.basis_simulator(
                tgt=tgt,
                memory=basis_sim_context,
                tgt_mask=tgt_mask,
            )
            # [batch_size, 1, hidden_dim]
            current_hidden = decoder_output[:, -1:, :]
            predicted_basis = self.basis_projection(current_hidden)
            simulated_basis[:, i] = predicted_basis.squeeze(1).squeeze(1)

            if i < basis_dim.max() - 1:
                generated_sequence = torch.cat(
                    [
                        generated_sequence,
                        predicted_basis.detach(),
                    ],
                    dim=1,
                )

        return predicted_basis

    def _teacher_forced_generation(
        self,
        basis_embedding: torch.Tensor,
        prev_action_embedding: torch.Tensor,
        current_action_embedding: torch.Tensor,
        target_basis: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        tgt = self.basis_encoder.input_projection(target_basis)
        tgt = self.basis_encoder.pos_encoding(tgt)
        seq_len = target_basis.size(1)
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

        basis_sim_context = torch.cat([
            basis_embedding,
            prev_action_embedding,
            current_action_embedding
        ], dim=2)

        decoder_output = self.basis_simulator(
            tgt=tgt,
            memory=basis_sim_context,
            tgt_mask=tgt_mask
        )

        simulated_basis = self.basis_projection(decoder_output).squeeze(-1)
        return simulated_basis


class InverseModel(nn.Module):
    def __init__(
        self,
        input_embedding_dim: int,
        hidden_dim: int,
        dropout_p: float = 0.1,
    ) -> None:
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