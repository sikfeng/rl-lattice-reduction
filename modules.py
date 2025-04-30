import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 64,
    ) -> None:
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1), :]
        return x


class GSNormEncoder(nn.Module):
    def __init__(
        self,
        dropout_p: float,
        max_basis_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.max_basis_dim = max_basis_dim
        self.dropout_p = dropout_p
        self.hidden_dim = hidden_dim

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.input_projection = nn.Sequential(
            nn.Unflatten(1, (-1, 1)),
            nn.Linear(1, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.pos_encoding = PositionalEncoding(
            self.hidden_dim,
            max_len=self.max_basis_dim + 1,
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=4,
            dim_feedforward=4 * self.hidden_dim,
            dropout=dropout_p,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=3,
        )
        self.encoder_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(
        self,
        gs_norms: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_projection(gs_norms)

        cls_tokens = self.cls_token.expand(gs_norms.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        x = self.encoder_projection(x)
        return x

    def _generate_pad_mask(
        self,
        seq_lengths: torch.Tensor,
    ):
        batch_size = seq_lengths.size(0)
        padding_mask = torch.zeros(
            batch_size,
            self.max_basis_dim + 1,
            dtype=torch.bool,
            device=seq_lengths.device,
        )

        for i, length in enumerate(seq_lengths.int()):
            padding_mask[i, length + 1 :] = True

        return padding_mask


class GSNormDecoder(nn.Module):
    def __init__(
        self,
        gs_norms_encoder: GSNormEncoder,
        input_dim: int,
        hidden_dim: int = 128,
        dropout_p: float = 0.1,
    ):
        super().__init__()

        self.gs_norms_encoder = gs_norms_encoder
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout_p

        self.bos_token = nn.Parameter(torch.randn(1, 1))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.input_dim,
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
                self.input_dim,
                self.hidden_dim,
            ),
            nn.LeakyReLU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        gs_norms_embedding: torch.Tensor,
        prev_action_embedding: torch.Tensor,
        current_action_embedding: torch.Tensor,
        basis_dim: torch.Tensor,
        target_gs_norms: torch.Tensor,
    ):
        device = gs_norms_embedding.device
        if target_gs_norms is None:
            predicted_gs_norms = self._autoregressive_generation(
                gs_norms_embedding=gs_norms_embedding,
                prev_action_embedding=prev_action_embedding,
                current_action_embedding=current_action_embedding,
                basis_dim=basis_dim,
                device=device,
            )
        else:
            predicted_gs_norms = self._teacher_forced_generation(
                gs_norms_embedding=gs_norms_embedding,
                prev_action_embedding=prev_action_embedding,
                current_action_embedding=current_action_embedding,
                target_gs_norms=target_gs_norms,
                device=device,
            )

        pad_mask = self._generate_pad_mask(basis_dim)
        predicted_gs_norms[pad_mask] = 0
        return predicted_gs_norms

    def _generate_pad_mask(
        self,
        seq_lengths: torch.Tensor,
    ):
        batch_size = seq_lengths.size(0)
        padding_mask = torch.zeros(
            batch_size,
            self.gs_norms_encoder.max_basis_dim,
            dtype=torch.bool,
            device=seq_lengths.device,
        )

        for i, length in enumerate(seq_lengths.int()):
            padding_mask[i, length:] = True

        return padding_mask

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
        generated_sequence = self.bos_token.expand(gs_norms_embedding.size(0), 1)
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
            predicted_gs_norm = self.gs_norm_projection(current_hidden).squeeze(1)
            simulated_gs_norms[:, i] = predicted_gs_norm.squeeze(1)

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
        bos = self.bos_token.expand(target_gs_norms.size(0), 1)
        tgt = torch.cat([bos, target_gs_norms], dim=1)
        tgt = self.gs_norms_encoder.input_projection(tgt)
        tgt = self.gs_norms_encoder.pos_encoding(tgt)
        seq_len = tgt.size(1)
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

        simulated_gs_norms = self.gs_norm_projection(decoder_output[:, 1:, :]).squeeze(
            -1
        )
        return simulated_gs_norms


class ActionEncoder(nn.Module):
    def __init__(
        self,
        max_basis_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.max_basis_dim = max_basis_dim
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(2, self.embedding_dim),
            nn.LeakyReLU(),
        )

    def forward(
        self,
        action: torch.Tensor,
        basis_dim: torch.Tensor,
    ) -> torch.Tensor:
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
