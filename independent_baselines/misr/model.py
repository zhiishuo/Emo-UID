"""Independent, equation-guided MISR; see METHOD_MAP.md for fidelity limits."""
from dataclasses import dataclass
from itertools import combinations
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MISRConfig:
    language_dim: int = 768
    audio_dim: int = 5
    vision_dim: int = 20
    hidden_dim: int = 128
    heads: int = 4
    dropout: float = 0.1
    lambda_sp: float = 1.0
    lambda_dc: float = 0.4
    lambda_ic: float = 1.0
    min_std: float = 1e-4


def gaussian_kl(mu_p, std_p, mu_q, std_q):
    """KL(p||q), sum over latent coordinates, isotropic std may broadcast."""
    # Divide before squaring: equal tiny/large scales must not become 0/0 or inf/inf.
    return (torch.log(std_q) - torch.log(std_p)
            + 0.5 * ((std_p / std_q).square()
                     + ((mu_p - mu_q) / std_q).square() - 1)).sum(-1)


def masked_mean(x, mask):
    return x.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)


def masked_risk(x, mask):
    # Equal utterance weight, independent of padded length.
    return (x.masked_fill(~mask, 0).sum(1) / mask.sum(1)).mean()


def position_encoding(x):
    t, d = x.shape[1:]
    positions = torch.arange(t, device=x.device, dtype=x.dtype)[:, None]
    frequency = torch.exp(torch.arange(0, d, 2, device=x.device, dtype=x.dtype)
                          * (-math.log(10000.0) / d))
    pe = x.new_zeros(t, d)
    pe[:, 0::2] = torch.sin(positions * frequency)
    pe[:, 1::2] = torch.cos(positions * frequency)
    return x + pe


def transformer(config):
    layer = nn.TransformerEncoderLayer(
        config.hidden_dim, config.heads, config.hidden_dim * 4,
        config.dropout, batch_first=True, activation="relu")
    return nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)


class ModalityBranch(nn.Module):
    def __init__(self, dim, config):
        super().__init__()
        d = config.hidden_dim
        self.min_std = config.min_std
        self.embedding = nn.Linear(dim, d)
        self.me = transformer(config)
        self.ve_mapping = nn.Linear(d, d)
        self.ve = transformer(config)
        self.mean = nn.Linear(d, d)
        self.scale = nn.Linear(d, 1)
        self.vd_mapping = nn.Linear(d, d)
        self.vd = transformer(config)
        self.reconstruction = nn.Linear(d, d)

    def forward(self, x, mask, sample):
        x = x.masked_fill(~mask[..., None], 0)
        h = self.me(position_encoding(self.embedding(x)), src_key_padding_mask=~mask)
        h = h.masked_fill(~mask[..., None], 0)
        q = self.ve(F.relu(self.ve_mapping(h)), src_key_padding_mask=~mask)
        mu = self.mean(q)
        std = F.softplus(self.scale(q)) + self.min_std
        z = mu + std * torch.randn_like(mu) if sample else mu
        decoded = self.vd(F.relu(self.vd_mapping(z)), src_key_padding_mask=~mask)
        reconstructed = self.reconstruction(decoded)
        na = masked_risk(gaussian_kl(mu, std, torch.zeros_like(mu), torch.ones_like(std)), mask)
        dr = masked_risk((h - reconstructed).square().sum(-1), mask)
        return h, mu, std, na, dr


class MISR(nn.Module):
    """MOSI regression interface. Boolean masks use True for valid positions.

    Inputs must share aligned token positions. No unimodal labels are required.
    `losses` contains total only when multimodal regression labels are provided.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config or MISRConfig()
        c = self.config
        if c.hidden_dim % 2 or c.hidden_dim % c.heads or c.min_std <= 0:
            raise ValueError("hidden_dim must be even and divisible by heads; min_std > 0")
        self.names = ("language", "audio", "vision")
        self.branches = nn.ModuleDict({name: ModalityBranch(dim, c) for name, dim in
                                      zip(self.names, (c.language_dim, c.audio_dim, c.vision_dim))})
        self.environment_norm = nn.LayerNorm(c.hidden_dim)
        self.phi = nn.Sequential(nn.Linear(c.hidden_dim, c.hidden_dim), nn.ReLU(),
                                 nn.Linear(c.hidden_dim, c.hidden_dim))
        self.predictor = nn.Sequential(nn.Linear(c.hidden_dim, c.hidden_dim), nn.ReLU(),
                                       nn.Linear(c.hidden_dim, 1))
        self.subsets = tuple(subset for k in range(4) for subset in combinations(range(3), k))
        self.environment_names = tuple("u" + "".join("+" + self.names[i] for i in s)
                                       for s in self.subsets)

    def forward(self, language, audio, vision, masks=None, labels=None, *, sample=None):
        c = self.config
        inputs = (language, audio, vision)
        dims = (c.language_dim, c.audio_dim, c.vision_dim)
        if language.ndim != 3 or min(language.shape[:2]) == 0:
            raise ValueError("Inputs must be nonempty [B,T,D]")
        b, t = language.shape[:2]
        masks = {} if masks is None else masks
        if set(masks) - set(self.names):
            raise ValueError("Mask keys must be language, audio, vision")
        valid = []
        for name, x, dim in zip(self.names, inputs, dims):
            if x.shape != (b, t, dim) or not x.is_floating_point():
                raise ValueError(f"{name} must be floating [B,T,{dim}] on aligned positions")
            if x.device != language.device or x.dtype != language.dtype:
                raise ValueError("Inputs must share device and dtype")
            mask = masks.get(name, torch.ones((b, t), dtype=torch.bool, device=x.device))
            if mask.shape != (b, t) or mask.dtype != torch.bool or mask.device != x.device:
                raise ValueError("Masks must be boolean [B,T] on input device")
            if not mask.any(1).all():
                raise ValueError("Fully missing modality is unsupported")
            if not torch.isfinite(x[mask]).all():
                raise ValueError("Non-finite valid input")
            valid.append(mask)
        for i, j in combinations(range(3), 2):
            if not (valid[i] & valid[j]).any(1).all():
                raise ValueError("Cross-modal KL needs overlapping aligned valid positions")
        if labels is not None:
            if labels.shape not in ((b,), (b, 1)) or not labels.is_floating_point():
                raise ValueError("labels must be multimodal floating regression targets [B] or [B,1]")
            if labels.device != language.device or not torch.isfinite(labels).all():
                raise ValueError("Labels must be finite on input device")
        sample = self.training if sample is None else sample
        branches = [self.branches[name](x, m, sample) for name, x, m in zip(self.names, inputs, valid)]
        h, mu, std, na, dr = zip(*branches)
        mask_stack = torch.stack(valid, dim=2)
        union = mask_stack.any(2)
        logits = torch.stack([s.reciprocal().squeeze(-1) for s in std], dim=2)
        logits = logits.masked_fill(~mask_stack, -torch.inf)
        # Avoid all-negative-infinity softmax at completely padded positions.
        logits = logits.masked_fill(~union[..., None], 0)
        weights = logits.softmax(2).masked_fill(~mask_stack, 0)
        joint_tokens = (torch.stack(mu, dim=2) * weights[..., None]).sum(2)
        joint = masked_mean(joint_tokens, union)
        unimodal = [masked_mean(x, m) for x, m in zip(h, valid)]
        environments = torch.stack([joint + sum((unimodal[i] for i in s), torch.zeros_like(joint))
                                    for s in self.subsets], dim=1)
        invariant = self.phi(self.environment_norm(environments))
        environment_predictions = self.predictor(invariant).squeeze(-1)
        cma = sum(masked_risk(gaussian_kl(mu[i], std[i], mu[j], std[j]), valid[i] & valid[j])
                  for i in range(3) for j in range(3) if i != j)
        losses = {"normalized_alignment": sum(na), "cross_modal_alignment": cma,
                  "data_reconstruction": sum(dr)}
        losses["distribution_constraint"] = sum(losses.values())
        risks = None
        if labels is not None:
            risks = (environment_predictions - labels.reshape(b, 1)).abs().mean(0)
            losses["sentiment_prediction"] = risks[0]
            losses["invariant_constraint"] = risks.var(unbiased=False)
            losses["total"] = (c.lambda_sp * losses["sentiment_prediction"]
                               + c.lambda_dc * losses["distribution_constraint"]
                               + c.lambda_ic * losses["invariant_constraint"])
        return {"prediction": environment_predictions[:, 0], "losses": losses,
                "environment_predictions": environment_predictions, "environment_risks": risks,
                "environment_names": self.environment_names, "fusion_weights": weights,
                "joint_representation": joint, "invariant_representation": invariant[:, 0]}
