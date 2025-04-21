from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from modules import GSNormEncoder, ActionEncoder


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
