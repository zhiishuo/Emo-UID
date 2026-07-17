"""End-to-end Emo-UID architecture and grouped training objective."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import EmoUIDConfig
from .functional import masked_mean
from .modules import (
    MODALITIES,
    ConsensusPool,
    DiversityClassification,
    MultimodalPrivateTransformer,
    PrototypeGramUnity,
    ReliabilityGatedResidualFusion,
    SharedPrivateFactorization,
    SharedStreamTransformer,
    TemporalProjector,
)


class EmoUID(nn.Module):
    """Reliability-controlled unity-in-diversity sentiment model.

    The forward path follows Sec. III of the revised manuscript. Inputs are
    modality feature sequences; language-model feature extraction and dataset
    preprocessing remain outside this architecture package.
    """

    def __init__(self, config: EmoUIDConfig) -> None:
        super().__init__()
        self.config = config
        self.frontends = nn.ModuleDict(
            {
                modality: TemporalProjector(
                    config.input_dims[modality],
                    config.model_dim,
                    config.kernel_sizes[modality],
                )
                for modality in MODALITIES
            }
        )
        self.factorization = SharedPrivateFactorization(
            config.model_dim, config.dropout, config.epsilon
        )
        self.pgu = PrototypeGramUnity(
            model_dim=config.model_dim,
            anchors=config.sentiment_anchors,
            prototypes_per_anchor=config.prototypes_per_anchor,
            gram_temperature=config.gram_temperature,
            ordinal_temperature=config.ordinal_temperature,
            prototype_momentum=config.prototype_momentum,
            neighborhood_weight=config.prototype_neighborhood_weight,
            epsilon=config.epsilon,
        )
        self.shared_enhancer = SharedStreamTransformer(
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            layers=config.shared_transformer_layers,
            feedforward_dim=config.feedforward_dim,
            dropout=config.dropout,
            maximum_length=config.max_sequence_length,
        )
        self.private_enhancer = MultimodalPrivateTransformer(
            model_dim=config.model_dim,
            num_heads=config.num_heads,
            layers=config.private_transformer_layers,
            feedforward_dim=config.feedforward_dim,
            dropout=config.dropout,
            maximum_length=config.max_sequence_length,
        )
        self.consensus_pool = ConsensusPool(config.model_dim)
        self.diversity_classification = DiversityClassification(
            config.model_dim, len(config.sentiment_anchors), config.cps_weight
        )
        self.reliability_fusion = ReliabilityGatedResidualFusion(
            config.model_dim, config.gate_hidden_dim, config.epsilon
        )
        self.regression_head = nn.Sequential(
            nn.Linear(config.model_dim, config.regression_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.regression_hidden_dim, 1),
        )

    def _prepare_inputs(
        self,
        inputs: Mapping[str, Tensor],
        masks: Optional[Mapping[str, Tensor]],
    ) -> tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        prepared_inputs: Dict[str, Tensor] = {}
        prepared_masks: Dict[str, Tensor] = {}
        batch_size = inputs[MODALITIES[0]].shape[0]

        for modality in MODALITIES:
            sequence = inputs[modality]
            if sequence.ndim != 3:
                raise ValueError(
                    f"{modality} must have shape [batch, time, feature], "
                    f"received {tuple(sequence.shape)}."
                )
            if sequence.shape[0] != batch_size:
                raise ValueError("All modalities must have the same batch size.")
            expected_dim = self.config.input_dims[modality]
            if sequence.shape[-1] != expected_dim:
                raise ValueError(
                    f"{modality} feature dimension must be {expected_dim}, "
                    f"received {sequence.shape[-1]}."
                )
            if sequence.shape[1] > self.config.max_sequence_length:
                raise ValueError(
                    f"{modality} sequence length exceeds max_sequence_length="
                    f"{self.config.max_sequence_length}."
                )

            if masks is None or modality not in masks:
                valid_mask = torch.ones(
                    sequence.shape[:2], dtype=torch.bool, device=sequence.device
                )
            else:
                valid_mask = masks[modality].to(device=sequence.device, dtype=torch.bool)
                if valid_mask.shape != sequence.shape[:2]:
                    raise ValueError(
                        f"{modality} mask must have shape {tuple(sequence.shape[:2])}."
                    )
            if not valid_mask.any(dim=1).all():
                raise ValueError(f"Every {modality} sample must contain a valid time step.")

            prepared_inputs[modality] = sequence
            prepared_masks[modality] = valid_mask

        return prepared_inputs, prepared_masks

    def forward(
        self,
        language: Tensor,
        vision: Tensor,
        acoustic: Tensor,
        labels: Optional[Tensor] = None,
        masks: Optional[Mapping[str, Tensor]] = None,
        update_prototypes: bool = True,
    ) -> Dict[str, object]:
        """Run sentiment inference and, when labels are given, compute Eq. (25).

        Args:
            language: Language feature sequence `[B, T_L, d_L]`.
            vision: Visual feature sequence `[B, T_V, d_V]`.
            acoustic: Acoustic feature sequence `[B, T_A, d_A]`.
            labels: Optional continuous sentiment labels `[B]` or `[B, 1]`.
            masks: Optional valid-step masks keyed by modality.
            update_prototypes: Allow EMA prototype updates during training.
        """

        inputs, valid_masks = self._prepare_inputs(
            {"language": language, "vision": vision, "acoustic": acoustic}, masks
        )
        projected = {
            modality: self.frontends[modality](inputs[modality], valid_masks[modality])
            for modality in MODALITIES
        }
        factorized = self.factorization(projected, valid_masks)
        shared_pre = factorized["shared"]
        private_pre = factorized["private"]

        # PGU is intentionally applied before Transformer enhancement.
        pooled_shared_pre = {
            modality: masked_mean(
                shared_pre[modality], valid_masks[modality], self.config.epsilon
            )
            for modality in MODALITIES
        }
        pgu_output: Optional[Dict[str, Tensor]] = None
        ordinal_target: Optional[Tensor] = None
        normalized_labels: Optional[Tensor] = None
        if labels is not None:
            normalized_labels = labels.reshape(-1).to(
                device=language.device, dtype=language.dtype
            )
            if normalized_labels.shape[0] != language.shape[0]:
                raise ValueError("labels must contain one value per sample.")
            pgu_output = self.pgu(
                pooled_shared_pre,
                normalized_labels,
                update_prototypes=self.training and update_prototypes,
            )
            ordinal_target = pgu_output["ordinal_target"]

        shared_enhanced = {
            modality: self.shared_enhancer(
                shared_pre[modality], valid_masks[modality]
            )
            for modality in MODALITIES
        }
        private_enhanced = self.private_enhancer(private_pre, valid_masks)
        consensus = self.consensus_pool(
            shared_enhanced, valid_masks, self.config.epsilon
        )
        private_vectors = {
            modality: masked_mean(
                private_enhanced[modality],
                valid_masks[modality],
                self.config.epsilon,
            )
            for modality in MODALITIES
        }

        dc_output = self.diversity_classification(
            consensus, private_vectors, ordinal_target
        )
        fusion_output = self.reliability_fusion(
            consensus, private_vectors, dc_output["probabilities"]
        )
        prediction = self.regression_head(fusion_output["fused"])

        losses: Dict[str, Tensor] = {}
        if normalized_labels is not None and pgu_output is not None:
            task_loss = F.l1_loss(prediction.reshape(-1), normalized_labels)
            factorization_losses = factorized["losses"]
            dc_losses = dc_output["losses"]
            weights = self.config.loss_weights
            total = (
                task_loss
                + weights.factorization * factorization_losses["factorization"]
                + weights.pgu * pgu_output["pgu"]
                + weights.dc * dc_losses["dc"]
            )
            losses = {
                "task": task_loss,
                "orthogonality": factorization_losses["orthogonality"],
                "reconstruction": factorization_losses["reconstruction"],
                "factorization": factorization_losses["factorization"],
                "gram": pgu_output["gram"],
                "sop": pgu_output["sop"],
                "pgu": pgu_output["pgu"],
                "ordinal": dc_losses["ordinal"],
                "cps": dc_losses["cps"],
                "dc": dc_losses["dc"],
                "total": total,
            }

        return {
            "prediction": prediction,
            "gates": fusion_output["gates"],
            "auxiliary_probabilities": dc_output["probabilities"],
            "uncertainty": fusion_output["uncertainty"],
            "discrepancy": fusion_output["discrepancy"],
            "ordinal_target": ordinal_target,
            "losses": losses,
            "representations": {
                "consensus": consensus,
                "private": private_vectors,
                "residuals": fusion_output["residuals"],
                "fused": fusion_output["fused"],
            },
            "streams": {
                "shared_pre_enhancement": shared_pre,
                "private_pre_enhancement": private_pre,
                "shared_enhanced": shared_enhanced,
                "private_enhanced": private_enhanced,
            },
        }
