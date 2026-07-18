"""Core modules of the manuscript-aligned Emo-UID architecture."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .functional import (
    categorical_entropy,
    jensen_shannon_divergence,
    masked_mean,
    soft_ordinal_target,
)


MODALITIES: Tuple[str, ...] = ("language", "vision", "acoustic")


class BertLanguageEncoder(nn.Module):
    """BERT sequence encoder used for CMU-MOSI and CMU-MOSEI.

    The expected input follows the benchmark preprocessing convention
    `[batch, 3, time]`: token ids, attention mask, and token-type ids. The
    pooling head is disabled because Emo-UID consumes token-level states.
    """

    def __init__(
        self,
        model_name: str,
        fine_tune: bool,
        model: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if model is None:
            try:
                from transformers import BertModel
            except ImportError as error:
                raise ImportError(
                    "BERT inputs require the optional dependency; install "
                    "Emo-UID with `pip install -e '.[bert]'`."
                ) from error
            model = BertModel.from_pretrained(model_name, add_pooling_layer=False)

        hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("The BERT model must expose config.hidden_size.")
        if hasattr(model, "pooler"):
            model.pooler = None
        self.model = model
        self.output_dim = int(hidden_size)
        self.fine_tune = fine_tune
        if not fine_tune:
            self.model.requires_grad_(False)

    def forward(self, token_bundle: Tensor) -> Tensor:
        if token_bundle.ndim != 3 or token_bundle.shape[1] != 3:
            raise ValueError(
                "BERT language input must have shape [batch, 3, time]: "
                "token ids, attention mask, and token-type ids."
            )
        input_ids = token_bundle[:, 0, :].long()
        attention_mask = token_bundle[:, 1, :]
        token_type_ids = token_bundle[:, 2, :].long()
        context = nullcontext() if self.fine_tune else torch.no_grad()
        with context:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output[0]


class TemporalProjector(nn.Module):
    """Modality-specific temporal front-end from Eq. (2)."""

    def __init__(self, input_dim: int, model_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.projection = nn.Conv1d(
            input_dim,
            model_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.normalization = nn.LayerNorm(model_dim)

    def forward(self, sequence: Tensor, valid_mask: Tensor) -> Tensor:
        if sequence.ndim != 3:
            raise ValueError("Each modality input must have shape [batch, time, feature].")
        sequence = sequence * valid_mask.unsqueeze(-1).to(sequence.dtype)
        projected = self.projection(sequence.transpose(1, 2)).transpose(1, 2)
        projected = self.normalization(F.gelu(projected))
        return projected * valid_mask.unsqueeze(-1).to(projected.dtype)


class ResidualFeatureEncoder(nn.Module):
    def __init__(self, model_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )
        self.normalization = nn.LayerNorm(model_dim)

    def forward(self, sequence: Tensor) -> Tensor:
        return self.normalization(sequence + self.layers(sequence))


class SharedPrivateFactorization(nn.Module):
    """Shared-private decomposition and Eq. (4) factorization loss."""

    def __init__(self, model_dim: int, dropout: float, epsilon: float) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.shared_encoder = ResidualFeatureEncoder(model_dim, dropout)
        self.private_encoders = nn.ModuleDict(
            {name: ResidualFeatureEncoder(model_dim, dropout) for name in MODALITIES}
        )
        self.decoders = nn.ModuleDict(
            {name: nn.Linear(2 * model_dim, model_dim) for name in MODALITIES}
        )

    def _orthogonality_loss(
        self,
        shared: Tensor,
        private: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        weights = valid_mask.unsqueeze(-1).to(shared.dtype)
        shared = shared * weights
        private = private * weights
        inner = (shared * private).sum(dim=(1, 2))
        shared_norm = shared.square().sum(dim=(1, 2)).sqrt()
        private_norm = private.square().sum(dim=(1, 2)).sqrt()
        normalized_overlap = inner / (shared_norm * private_norm + self.epsilon)
        return normalized_overlap.square().mean()

    def _reconstruction_loss(
        self,
        reconstruction: Tensor,
        target: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        weights = valid_mask.unsqueeze(-1).to(target.dtype)
        squared_error = (reconstruction - target).square() * weights
        # Eq. (4): squared Frobenius norm per sample, averaged over the batch.
        return squared_error.sum(dim=(1, 2)).mean()

    def _cycle_consistency_loss(
        self,
        reencoded_private: Tensor,
        private: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        weights = valid_mask.unsqueeze(-1).to(private.dtype)
        squared_error = (reencoded_private - private).square() * weights
        # Eq. (4): re-encode the reconstruction and preserve its private factor.
        return squared_error.sum(dim=(1, 2)).mean()

    def forward(
        self,
        features: Mapping[str, Tensor],
        masks: Mapping[str, Tensor],
    ) -> Dict[str, object]:
        shared: Dict[str, Tensor] = {}
        private: Dict[str, Tensor] = {}
        reconstructed: Dict[str, Tensor] = {}
        reencoded_private: Dict[str, Tensor] = {}
        orthogonality = features[MODALITIES[0]].new_zeros(())
        reconstruction = features[MODALITIES[0]].new_zeros(())
        private_cycle = features[MODALITIES[0]].new_zeros(())

        for modality in MODALITIES:
            mask_values = masks[modality].unsqueeze(-1).to(features[modality].dtype)
            shared[modality] = self.shared_encoder(features[modality]) * mask_values
            private[modality] = self.private_encoders[modality](features[modality]) * mask_values
            reconstructed[modality] = self.decoders[modality](
                torch.cat([shared[modality], private[modality]], dim=-1)
            ) * mask_values
            reencoded_private[modality] = self.private_encoders[modality](
                reconstructed[modality]
            ) * mask_values
            orthogonality = orthogonality + self._orthogonality_loss(
                shared[modality], private[modality], masks[modality]
            )
            reconstruction = reconstruction + self._reconstruction_loss(
                reconstructed[modality], features[modality], masks[modality]
            )
            private_cycle = private_cycle + self._cycle_consistency_loss(
                reencoded_private[modality], private[modality], masks[modality]
            )

        return {
            "shared": shared,
            "private": private,
            "reconstructed": reconstructed,
            "reencoded_private": reencoded_private,
            "losses": {
                "orthogonality": orthogonality,
                "reconstruction": reconstruction,
                "private_cycle": private_cycle,
                "factorization": orthogonality + reconstruction + private_cycle,
            },
        }


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, model_dim: int, maximum_length: int) -> None:
        super().__init__()
        position = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        even_indices = torch.arange(0, model_dim, 2, dtype=torch.float32)
        frequencies = torch.exp(-math.log(10000.0) * even_indices / model_dim)
        encoding = torch.zeros(maximum_length, model_dim)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        odd_width = encoding[:, 1::2].shape[1]
        encoding[:, 1::2] = torch.cos(position * frequencies[:odd_width])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"Sequence length {sequence.shape[1]} exceeds configured maximum "
                f"{self.encoding.shape[1]}."
            )
        return sequence + self.encoding[:, : sequence.shape[1]].to(
            device=sequence.device,
            dtype=sequence.dtype,
        )


def _future_attention_mask(
    query_length: int,
    context_length: int,
    device: torch.device,
) -> Tensor:
    diagonal = 1 + abs(context_length - query_length)
    return torch.triu(
        torch.ones(query_length, context_length, dtype=torch.bool, device=device),
        diagonal=diagonal,
    )


def _initialize_linear(linear: nn.Linear) -> None:
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class DMDStyleTransformerLayer(nn.Module):
    """Pre-norm self- or cross-attention layer used by the DMD backbone."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(model_dim)
        self.feedforward_norm = nn.LayerNorm(model_dim)
        self.feedforward_in = nn.Linear(model_dim, feedforward_dim)
        self.feedforward_out = nn.Linear(feedforward_dim, model_dim)
        self.residual_dropout = nn.Dropout(dropout)
        self.activation_dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        _initialize_linear(self.feedforward_in)
        _initialize_linear(self.feedforward_out)

    def forward(
        self,
        query: Tensor,
        context: Optional[Tensor],
        context_mask: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        residual = query
        normalized_query = self.attention_norm(query)
        if context is None:
            key = value = normalized_query
        else:
            key = value = self.attention_norm(context)
        attended, _ = self.attention(
            normalized_query,
            key,
            value,
            attn_mask=attention_mask,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        query = residual + self.residual_dropout(attended)

        residual = query
        hidden = self.feedforward_norm(query)
        hidden = F.relu(self.feedforward_in(hidden))
        hidden = self.activation_dropout(hidden)
        hidden = self.feedforward_out(hidden)
        return residual + self.residual_dropout(hidden)


class DMDStyleTransformer(nn.Module):
    """DMD Transformer stack with optional fixed cross-modal context."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        layers: int,
        feedforward_dim: int,
        dropout: float,
        maximum_length: int,
        causal_attention: bool,
    ) -> None:
        super().__init__()
        self.scale = math.sqrt(model_dim)
        self.causal_attention = causal_attention
        self.position = SinusoidalPositionEncoding(model_dim, maximum_length)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                DMDStyleTransformerLayer(
                    model_dim,
                    num_heads,
                    feedforward_dim,
                    dropout,
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)

    def _embed(self, sequence: Tensor) -> Tensor:
        return self.embedding_dropout(self.position(sequence * self.scale))

    def forward(
        self,
        query: Tensor,
        query_mask: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
    ) -> Tensor:
        query = self._embed(query)
        if context is None:
            embedded_context = None
            effective_context_mask = query_mask
            context_length = query.shape[1]
        else:
            if context_mask is None:
                raise ValueError("context_mask is required for cross-modal attention.")
            embedded_context = self._embed(context)
            effective_context_mask = context_mask
            context_length = context.shape[1]

        attention_mask = None
        if self.causal_attention:
            attention_mask = _future_attention_mask(
                query.shape[1], context_length, query.device
            )
        for layer in self.layers:
            query = layer(
                query,
                embedded_context,
                effective_context_mask,
                attention_mask,
            )
        query = self.final_norm(query)
        return query * query_mask.unsqueeze(-1).to(query.dtype)


class SharedStreamTransformer(nn.Module):
    """One modality-indexed DMD Transformer for a common stream."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        layers: int,
        feedforward_dim: int,
        dropout: float,
        maximum_length: int,
        causal_attention: bool,
    ) -> None:
        super().__init__()
        self.encoder = DMDStyleTransformer(
            model_dim=model_dim,
            num_heads=num_heads,
            layers=layers,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            maximum_length=maximum_length,
            causal_attention=causal_attention,
        )

    def forward(self, sequence: Tensor, valid_mask: Tensor) -> Tensor:
        return self.encoder(sequence, valid_mask)


class DMDStylePrivateTransformer(nn.Module):
    """Six directed cross-modal stacks followed by three memory stacks."""

    _SOURCE_ORDER = {
        "language": ("acoustic", "vision"),
        "acoustic": ("language", "vision"),
        "vision": ("language", "acoustic"),
    }

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        layers: int,
        feedforward_dim: int,
        dropout: float,
        maximum_length: int,
        causal_attention: bool,
    ) -> None:
        super().__init__()
        self.cross_transformers = nn.ModuleDict(
            {
                f"{target}_from_{source}": DMDStyleTransformer(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    layers=layers,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout,
                    maximum_length=maximum_length,
                    causal_attention=causal_attention,
                )
                for target, sources in self._SOURCE_ORDER.items()
                for source in sources
            }
        )
        memory_dim = 2 * model_dim
        self.memory_transformers = nn.ModuleDict(
            {
                modality: DMDStyleTransformer(
                    model_dim=memory_dim,
                    num_heads=num_heads,
                    layers=max(layers, 3),
                    feedforward_dim=2 * feedforward_dim,
                    dropout=dropout,
                    maximum_length=maximum_length,
                    causal_attention=causal_attention,
                )
                for modality in MODALITIES
            }
        )

    def forward(
        self,
        streams: Mapping[str, Tensor],
        masks: Mapping[str, Tensor],
    ) -> Dict[str, Tensor]:
        enhanced: Dict[str, Tensor] = {}
        for target, sources in self._SOURCE_ORDER.items():
            cross_outputs = [
                self.cross_transformers[f"{target}_from_{source}"](
                    streams[target],
                    masks[target],
                    context=streams[source],
                    context_mask=masks[source],
                )
                for source in sources
            ]
            target_context = torch.cat(cross_outputs, dim=-1)
            enhanced[target] = self.memory_transformers[target](
                target_context,
                masks[target],
            )
        return enhanced


class ConsensusPool(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(len(MODALITIES) * model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )

    def forward(
        self,
        streams: Mapping[str, Tensor],
        masks: Mapping[str, Tensor],
        epsilon: float,
    ) -> Tensor:
        pooled = [masked_mean(streams[name], masks[name], epsilon) for name in MODALITIES]
        return self.projection(torch.cat(pooled, dim=-1))


class PrototypeGramUnity(nn.Module):
    """Gram-volume contrastive learning and soft ordinal prototypes."""

    def __init__(
        self,
        model_dim: int,
        anchors: Tuple[float, ...],
        prototypes_per_anchor: int,
        gram_temperature: float,
        ordinal_temperature: float,
        prototype_momentum: float,
        neighborhood_weight: float,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.gram_temperature = gram_temperature
        self.ordinal_temperature = ordinal_temperature
        self.prototype_momentum = prototype_momentum
        self.neighborhood_weight = neighborhood_weight
        self.epsilon = epsilon
        self.register_buffer("anchors", torch.tensor(anchors, dtype=torch.float32))
        prototypes = F.normalize(
            torch.randn(len(anchors), prototypes_per_anchor, model_dim), dim=-1
        )
        self.register_buffer("prototypes", prototypes)

    def _gram_volume_logits(self, normalized: Mapping[str, Tensor]) -> Tensor:
        language = normalized["language"]
        vision = normalized["vision"]
        acoustic = normalized["acoustic"]
        language_vision = language @ vision.transpose(0, 1)
        language_acoustic = language @ acoustic.transpose(0, 1)
        vision_acoustic = (vision * acoustic).sum(dim=-1).unsqueeze(0)
        vision_acoustic = vision_acoustic.expand_as(language_vision)
        diagonal = torch.ones_like(language_vision)

        first_row = torch.stack(
            [diagonal, language_vision, language_acoustic], dim=-1
        )
        second_row = torch.stack(
            [language_vision, diagonal, vision_acoustic], dim=-1
        )
        third_row = torch.stack(
            [language_acoustic, vision_acoustic, diagonal], dim=-1
        )
        gram = torch.stack([first_row, second_row, third_row], dim=-2)
        volume = torch.linalg.det(gram).clamp_min(0.0)
        return -volume / self.gram_temperature

    def _soft_ordinal_prototype_loss(
        self,
        normalized: Mapping[str, Tensor],
        ordinal_target: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        stacked = torch.stack([normalized[name] for name in MODALITIES], dim=1)
        shared_mean = stacked.mean(dim=1)
        prototype_snapshot = self.prototypes.detach().clone()

        mean_distance = (
            shared_mean[:, None, None, :] - prototype_snapshot[None, :, :, :]
        ).square().sum(dim=-1)
        nearest_indices = mean_distance.argmin(dim=-1)
        expanded_bank = prototype_snapshot.unsqueeze(0).expand(
            shared_mean.shape[0], -1, -1, -1
        )
        gather_index = nearest_indices[:, :, None, None].expand(
            -1, -1, 1, prototype_snapshot.shape[-1]
        )
        selected = expanded_bank.gather(dim=2, index=gather_index).squeeze(2)
        sample_anchor = (ordinal_target.unsqueeze(-1) * selected).sum(dim=1)

        anchor_term = (stacked - sample_anchor.unsqueeze(1)).square().sum(dim=-1).mean()
        neighborhood_terms = []
        for modality in MODALITIES:
            distance = (
                normalized[modality][:, None, None, :]
                - prototype_snapshot[None, :, :, :]
            ).square().sum(dim=-1)
            nearest_distance = distance.min(dim=-1).values
            neighborhood_terms.append((ordinal_target * nearest_distance).sum(dim=-1).mean())
        neighborhood_term = torch.stack(neighborhood_terms).mean()
        loss = anchor_term + self.neighborhood_weight * neighborhood_term
        return loss, nearest_indices

    @torch.no_grad()
    def _update_prototypes(
        self,
        normalized: Mapping[str, Tensor],
        ordinal_target: Tensor,
        nearest_indices: Tensor,
    ) -> None:
        shared_mean = torch.stack([normalized[name] for name in MODALITIES], dim=1).mean(dim=1)
        for anchor_index in range(self.prototypes.shape[0]):
            for prototype_index in range(self.prototypes.shape[1]):
                assigned = nearest_indices[:, anchor_index] == prototype_index
                weights = ordinal_target[:, anchor_index] * assigned.to(ordinal_target.dtype)
                denominator = weights.sum()
                if denominator.item() <= self.epsilon:
                    continue
                centroid = (weights.unsqueeze(-1) * shared_mean).sum(dim=0) / denominator
                self.prototypes[anchor_index, prototype_index].mul_(self.prototype_momentum)
                self.prototypes[anchor_index, prototype_index].add_(
                    centroid, alpha=1.0 - self.prototype_momentum
                )

    def forward(
        self,
        shared_vectors: Mapping[str, Tensor],
        labels: Tensor,
        update_prototypes: bool,
    ) -> Dict[str, Tensor]:
        normalized = {
            name: F.normalize(shared_vectors[name], dim=-1, eps=self.epsilon)
            for name in MODALITIES
        }
        logits = self._gram_volume_logits(normalized)
        targets = torch.arange(logits.shape[0], device=logits.device)
        gram_loss = 0.5 * (
            F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets)
        )
        ordinal_target = soft_ordinal_target(
            labels, self.anchors, self.ordinal_temperature
        )
        prototype_loss, nearest_indices = self._soft_ordinal_prototype_loss(
            normalized, ordinal_target
        )
        if update_prototypes:
            self._update_prototypes(normalized, ordinal_target, nearest_indices)
        return {
            "gram": gram_loss,
            "sop": prototype_loss,
            "pgu": gram_loss + prototype_loss,
            "ordinal_target": ordinal_target,
            "prototype_indices": nearest_indices,
        }


class DiversityClassification(nn.Module):
    """Ordinal evidence heads and Confidence Product Suppression."""

    def __init__(
        self,
        consensus_dim: int,
        private_dim: int,
        num_anchors: int,
        cps_weight: float,
    ) -> None:
        super().__init__()
        self.cps_weight = cps_weight
        self.consensus_head = nn.Linear(consensus_dim, num_anchors)
        self.private_heads = nn.ModuleDict(
            {name: nn.Linear(private_dim, num_anchors) for name in MODALITIES}
        )

    def forward(
        self,
        consensus: Tensor,
        private: Mapping[str, Tensor],
        ordinal_target: Optional[Tensor],
    ) -> Dict[str, object]:
        logits = {"consensus": self.consensus_head(consensus)}
        logits.update({name: self.private_heads[name](private[name]) for name in MODALITIES})
        probabilities = {name: torch.softmax(value, dim=-1) for name, value in logits.items()}
        losses: Dict[str, Tensor] = {}

        if ordinal_target is not None:
            head_losses = [
                -(ordinal_target * torch.log_softmax(value, dim=-1)).sum(dim=-1).mean()
                for value in logits.values()
            ]
            ordinal_loss = torch.stack(head_losses).mean()
            language = probabilities["language"]
            vision = probabilities["vision"]
            acoustic = probabilities["acoustic"]
            pairwise_surface = (
                language * vision + language * acoustic + vision * acoustic
            ) / 3.0
            trimodal_volume = language * vision * acoustic
            cps_loss = (
                (1.0 - ordinal_target) * (pairwise_surface + trimodal_volume)
            ).mean()
            losses = {
                "ordinal": ordinal_loss,
                "cps": cps_loss,
                "dc": ordinal_loss + self.cps_weight * cps_loss,
            }

        return {"logits": logits, "probabilities": probabilities, "losses": losses}


class ReliabilityGatedResidualFusion(nn.Module):
    """Asymmetric consensus-plus-private-residual fusion from Eqs. (20)-(23)."""

    def __init__(
        self,
        consensus_dim: int,
        private_dim: int,
        gate_hidden_dim: int,
        epsilon: float,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.private_projections = nn.ModuleDict(
            {name: nn.Linear(private_dim, consensus_dim) for name in MODALITIES}
        )
        self.gates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(2 * consensus_dim + 2, gate_hidden_dim),
                    nn.GELU(),
                    nn.Linear(gate_hidden_dim, 1),
                )
                for name in MODALITIES
            }
        )

    def forward(
        self,
        consensus: Tensor,
        private: Mapping[str, Tensor],
        probabilities: Mapping[str, Tensor],
    ) -> Dict[str, object]:
        residuals: Dict[str, Tensor] = {}
        gate_values: Dict[str, Tensor] = {}
        uncertainties: Dict[str, Tensor] = {}
        discrepancies: Dict[str, Tensor] = {}
        fused = consensus

        for modality in MODALITIES:
            residual = self.private_projections[modality](private[modality])
            uncertainty = categorical_entropy(probabilities[modality], self.epsilon)
            discrepancy = jensen_shannon_divergence(
                probabilities[modality], probabilities["consensus"], self.epsilon
            )
            gate_input = torch.cat(
                [residual, consensus, uncertainty, discrepancy], dim=-1
            )
            gate = torch.sigmoid(self.gates[modality](gate_input))
            fused = fused + gate * residual
            residuals[modality] = residual
            gate_values[modality] = gate
            uncertainties[modality] = uncertainty
            discrepancies[modality] = discrepancy

        return {
            "fused": fused,
            "residuals": residuals,
            "gates": gate_values,
            "uncertainty": uncertainties,
            "discrepancy": discrepancies,
        }
