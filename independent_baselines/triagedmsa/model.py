"""Independent, interpretation-bounded TriagedMSA; see METHOD_MAP.md.

No dataset access, optimizer, BERT wrapper, or automatic APD state mutation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def masked_mean(x, mask):
    return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)


def mlp(in_dim, hidden, out_dim, dropout=0.0):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(hidden, out_dim))


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden, heads, dropout, recurrent):
        super().__init__()
        self.recurrent = recurrent
        self.project = (nn.LSTM(input_dim, hidden // 2, batch_first=True,
                                bidirectional=True) if recurrent else nn.Linear(input_dim, hidden))
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.ff = mlp(hidden, hidden * 2, hidden, dropout)

    def forward(self, x, mask):
        if self.recurrent:
            packed = pack_padded_sequence(x, mask.sum(1).cpu(), batch_first=True,
                                          enforce_sorted=False)
            x, _ = self.project(packed)
            x, _ = pad_packed_sequence(x, batch_first=True, total_length=mask.shape[1])
            pos = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
            freq = torch.exp(torch.arange(0, x.shape[2], 2, device=x.device,
                                          dtype=x.dtype) * (-math.log(10000) / x.shape[2]))
            pe = torch.zeros_like(x[0])
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
            x = x + pe
        else:
            x = self.project(x)
        a, _ = self.attention(x, x, x, key_padding_mask=~mask, need_weights=False)
        # Eq. 4 has residual + LN; text Eq. 3 does not specify that residual.
        x = self.ff(self.norm(x + a) if self.recurrent else a)
        return x.masked_fill(~mask.unsqueeze(-1), 0)


class SDT(nn.Module):
    def __init__(self, hidden, max_length):
        super().__init__()
        self.max_length = max_length
        self.pairs = nn.ModuleList(nn.Linear(max_length ** 2, hidden) for _ in range(3))
        self.out = nn.Linear(3 * hidden, 1)

    def forward(self, xs, masks):
        sims = []
        for layer, (i, j) in zip(self.pairs, ((0, 1), (0, 2), (1, 2))):
            s = torch.bmm(xs[i], xs[j].transpose(1, 2))
            s = s.masked_fill(~(masks[i][:, :, None] & masks[j][:, None, :]), 0)
            s = F.pad(s, (0, self.max_length - s.shape[2], 0, self.max_length - s.shape[1]))
            sims.append(torch.tanh(layer(s.flatten(1))))
        return torch.tanh(self.out(torch.cat(sims, -1))).squeeze(-1)


class SSA(nn.Module):
    """Every semantic head queries every head, including itself (Eq. 9)."""
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.heads, self.width = heads, hidden // heads
        self.project = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(3))
        self.q = nn.Linear(self.width, self.width, bias=False)
        self.k = nn.Linear(self.width, self.width, bias=False)
        self.v = nn.Linear(self.width, self.width, bias=False)
        self.n = 3 * heads
        total = self.n * self.n * self.width
        self.gate = mlp(total, hidden, self.n)
        self.out = mlp(total, hidden, hidden, dropout)

    def forward(self, xs, masks):
        b, t, _ = xs[0].shape
        z = torch.cat([p(x).reshape(b, t, self.heads, self.width).transpose(1, 2)
                       for p, x in zip(self.project, xs)], dim=1)
        mask = torch.cat([m[:, None].expand(-1, self.heads, -1) for m in masks], 1)
        # [B, source head, target head, query time, key time].
        scores = torch.einsum('bitd,bjsd->bijts', self.q(z), self.k(z)) / math.sqrt(self.width)
        weights = scores.masked_fill(~mask[:, None, :, None, :], -torch.inf).softmax(-1)
        pairs = torch.einsum('bijts,bjsd->bijtd', weights, self.v(z))
        pairs = pairs.masked_fill(~mask[:, :, None, :, None], 0)
        heads = pairs.permute(0, 3, 1, 2, 4).reshape(b, t, self.n, self.n * self.width)
        valid_heads = mask.transpose(1, 2)
        logits = self.gate(heads.flatten(2)).masked_fill(~valid_heads, -torch.inf)
        # Fully padded time positions must not softmax an all-infinite row.
        valid_time = valid_heads.any(-1)
        logits = logits.masked_fill(~valid_time.unsqueeze(-1), 0)
        gates = logits.softmax(-1) * valid_heads
        fused = self.out((heads * gates.unsqueeze(-1)).flatten(2))
        return fused.masked_fill(~valid_time.unsqueeze(-1), 0), gates


class SCA(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.text = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(2))
        self.attention = nn.ModuleList(nn.MultiheadAttention(hidden, heads, dropout=dropout,
                                                          batch_first=True) for _ in range(2))
        self.out = mlp(2 * hidden, hidden, hidden, dropout)

    def forward(self, xs, masks):
        outputs = []
        for i, (p, attn) in enumerate(zip(self.text, self.attention), 1):
            a, _ = attn(p(xs[0]), xs[i], xs[i], key_padding_mask=~masks[i], need_weights=False)
            # Eq. 12/13 literal text query + nontext residual needs aligned time.
            outputs.append((a + xs[i]).masked_fill(~masks[0].unsqueeze(-1), 0))
        return self.out(torch.cat(outputs, -1)).masked_fill(~masks[0].unsqueeze(-1), 0)


class APD(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.attention = nn.ModuleList(nn.MultiheadAttention(hidden, heads, dropout=dropout,
                                                          batch_first=True) for _ in range(3))
        self.linear = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(3))
        self.bn = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(3))
        self.register_buffer('positive', torch.zeros(hidden))
        self.register_buffer('negative', torch.zeros(hidden))
        self.register_buffer('counts', torch.zeros(2, dtype=torch.long))
        self.register_buffer('reference_version', torch.zeros((), dtype=torch.long))

    @property
    def ready(self):
        return bool((self.counts > 0).all())

    @torch.no_grad()
    def set_references(self, fused, labels, *, source_split):
        """Replace using one complete train-only snapshot, never a mixed-split batch.

        Caller must enforce provenance and full training coverage. Nothing here can
        infer whether a caller has falsely tagged validation/test samples as train.
        """
        if not self.training or source_split != 'train':
            raise RuntimeError('APD reference mutation requires training mode and source_split=train')
        if fused.ndim != 2 or fused.shape[1:] != self.positive.shape or labels.shape != fused.shape[:1]:
            raise ValueError('Expected fused [N,H], labels [N]')
        if not torch.isfinite(fused).all() or not torch.isfinite(labels).all():
            raise ValueError('Nonfinite APD snapshot')
        pos = labels >= 0
        if not pos.any() or pos.all():
            raise ValueError('Both polarity classes required; previous references preserved')
        self.positive.copy_(fused[pos].mean(0))
        self.negative.copy_(fused[~pos].mean(0))
        self.counts.copy_(torch.stack((pos.sum(), (~pos).sum())))
        self.reference_version.add_(1)

    @staticmethod
    def smooth_targets(previous, raw, epoch):
        """Literal Eq. 21, not Algorithm 1; parent owns sample-ID history."""
        if not isinstance(epoch, int) or epoch < 1:
            raise ValueError('epoch must be a positive integer')
        if not torch.isfinite(raw).all() or not ((raw == -1) | (raw == 1)).all():
            raise ValueError('raw APD labels must be -1/+1')
        if epoch == 1:
            return torch.zeros_like(raw)
        if previous is None or previous.shape != raw.shape or not torch.isfinite(previous).all() or (previous.abs() > 1).any():
            raise ValueError('Previous per-sample targets in [-1,1] required')
        return (((epoch - 1) * previous.detach() + raw.detach()) / (epoch + 1)).detach()

    @torch.no_grad()
    def infer_targets(self, shared):
        if not self.ready:
            return None, None
        dp = (shared - self.positive).square().sum(-1)
        dn = (shared - self.negative).square().sum(-1)
        d = (dn - dp) / (dp + 1e-8)
        disagreement = (d[:, 0] * d[:, 1] < 0) | (d[:, 0] * d[:, 2] < 0)
        return torch.where(disagreement, -torch.ones_like(d[:, 0]), torch.ones_like(d[:, 0])), d

    def forward(self, xs, masks, fused, fused_mask):
        shared, consistency = [], []
        for x, mask, attn, linear, bn in zip(xs, masks, self.attention, self.linear, self.bn):
            z, _ = attn(x, fused, fused, key_padding_mask=~fused_mask, need_weights=False)
            z = linear(z)
            valid = z[mask]
            if self.training and valid.shape[0] < 2:
                raise ValueError('APD BatchNorm training requires at least two valid tokens')
            mapped = torch.zeros_like(z)
            mapped[mask] = bn(valid)
            consistency.append(masked_mean((mapped - x).abs(), mask).mean())
            shared.append(masked_mean(mapped, mask))
        shared = torch.stack(shared, 1)
        raw, polarity = self.infer_targets(shared.detach())
        return shared, sum(consistency), raw, polarity


class TriagedMSA(nn.Module):
    def __init__(self, language_dim=768, audio_dim=5, vision_dim=20,
                 hidden=256, heads=2, dropout=0.5, max_length=50):
        super().__init__()
        if hidden < 2 or heads < 1 or hidden % 2 or hidden % heads or max_length < 1:
            raise ValueError('hidden must be even and divisible by positive heads; max_length > 0')
        self.input_dims = (language_dim, audio_dim, vision_dim)
        self.max_length = max_length
        self.encoders = nn.ModuleList(Encoder(d, hidden, heads, dropout, i != 0)
                                      for i, d in enumerate(self.input_dims))
        self.sdt = SDT(hidden, max_length)
        self.ssa = SSA(hidden, heads, dropout)
        self.sca = SCA(hidden, heads, dropout)
        self.apd = APD(hidden, heads, dropout)
        self.regression = mlp(hidden, hidden, 1, dropout)

    def forward(self, language, audio, vision, masks=None, labels=None, relationship_targets=None):
        xs = (language, audio, vision)
        shape = language.shape[:2]
        if len(shape) != 2 or min(shape) < 1 or shape[1] > self.max_length:
            raise ValueError('Nonempty aligned input with T <= max_length required')
        for x, dim in zip(xs, self.input_dims):
            if x.ndim != 3 or x.shape[:2] != shape or x.shape[2] != dim:
                raise ValueError('Expected aligned [B,T,D] modality features')
        if masks is None:
            masks = tuple(torch.ones(shape, device=x.device, dtype=torch.bool) for x in xs)
        elif isinstance(masks, torch.Tensor):
            masks = (masks, masks, masks)
        if len(masks) != 3:
            raise ValueError('masks must be a boolean tensor or three boolean tensors')
        for x, m in zip(xs, masks):
            if m.shape != shape or m.dtype != torch.bool or not m.any(1).all():
                raise ValueError('Each sample/modality needs at least one valid token')
            if ((~m[:, :-1]) & m[:, 1:]).any():
                raise ValueError('Only right-padded masks supported by packed BiLSTM')
            if not torch.isfinite(x[m]).all():
                raise ValueError('Nonfinite valid features')
        if labels is not None:
            if labels.shape != shape[:1] or not torch.isfinite(labels).all():
                raise ValueError('labels must be finite [B]')
            if self.training and relationship_targets is None:
                raise ValueError('Training requires cached train-only APD targets (zeros at epoch 1)')
        if relationship_targets is not None:
            if relationship_targets.shape != shape[:1] or not torch.isfinite(relationship_targets).all() or (relationship_targets.abs() > 1).any():
                raise ValueError('relationship_targets must be finite [B] in [-1,1]')
        xs = [e(x.masked_fill(~m.unsqueeze(-1), 0), m) for e, x, m in zip(self.encoders, xs, masks)]
        triage = self.sdt(xs, masks)
        ssa, gates = self.ssa(xs, masks)
        sca = self.sca(xs, masks)
        disagreement = triage < 0
        fused = torch.where(disagreement[:, None, None], ssa, sca)
        union = torch.stack(tuple(masks)).any(0)
        fused_mask = torch.where(disagreement[:, None], union, masks[0])
        pooled = masked_mean(fused, fused_mask)
        prediction = self.regression(pooled).squeeze(-1)
        shared, consistency, raw, polarity = self.apd(xs, masks, fused, fused_mask)
        losses = {'consistency': consistency}
        if labels is not None:
            losses['sentiment'] = F.l1_loss(prediction, labels)
        if relationship_targets is not None:
            losses['triage'] = F.l1_loss(triage, relationship_targets.detach())
        if labels is not None and relationship_targets is not None:
            losses['total'] = losses['sentiment'] + losses['triage'] + consistency
        return {'prediction': prediction, 'losses': losses, 'triage_score': triage,
                'disagreement': disagreement, 'ssa_gates': gates, 'fused': pooled,
                'shared': shared, 'apd_raw_targets': raw, 'apd_polarity': polarity}
