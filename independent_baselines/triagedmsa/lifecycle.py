"""Train-only, epoch-boundary APD refresh; independent timing interpretation.

The runner owns preprocessing, optimization and checkpoint selection. Never run
an optimizer concurrently with refresh. ``batches`` is a re-iterable or a factory
returning (stable_ids, payload) batches; ``forward(core, payload)`` must only do a
forward pass and return the normal model output. It must not change weights,
buffers, modes or data. Use deterministic, unaugmented train inputs for both
passes. Labels are supplied separately in canonical train_ids order.
"""
from dataclasses import dataclass

import torch


def _ids(values):
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.dtype not in (torch.int32, torch.int64):
            raise ValueError('Tensor IDs must be one-dimensional integers')
        values = values.detach().cpu().tolist()
    ids = tuple(values)
    if not ids or any(type(i) not in (int, str) for i in ids):
        raise ValueError('Nonempty integer/string stable IDs required')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate IDs')
    return ids


@dataclass(frozen=True)
class APDCache:
    """Checkpoint this cache alongside model/optimizer/RNG and split hashes."""
    train_ids: tuple
    targets: torch.Tensor
    epoch: int

    @classmethod
    def initialize(cls, train_ids):
        ids = _ids(train_ids)
        return cls(ids, torch.zeros(len(ids)), 1)

    def for_ids(self, ids, *, device=None):
        positions = {key: i for i, key in enumerate(self.train_ids)}
        try:
            rows = [positions[key] for key in _ids(ids)]
        except KeyError as error:
            raise ValueError('ID outside train cache') from error
        return self.targets[rows].detach().clone().to(device=device)

    def state_dict(self):
        return {'train_ids': self.train_ids, 'targets': self.targets.detach().cpu().clone(),
                'epoch': self.epoch}

    @classmethod
    def from_state_dict(cls, state):
        result = cls(_ids(state['train_ids']), state['targets'].detach().cpu().clone(),
                     state['epoch'])
        result.validate()
        return result

    def validate(self):
        ids = _ids(self.train_ids)
        if type(self.epoch) is not int or self.epoch < 1:
            raise ValueError('Cache epoch must be a positive integer')
        if (self.targets.shape != (len(ids),) or not self.targets.is_floating_point()
                or not torch.isfinite(self.targets).all() or (self.targets.abs() > 1).any()):
            raise ValueError('Invalid cached targets')
        if self.epoch == 1 and self.targets.count_nonzero():
            raise ValueError('Epoch one targets must be zero')


@torch.no_grad()
def refresh_apd_cache(core, batches, forward, *, cache, train_labels,
                completed_epoch, source_split):
    """Return next epoch's cache, committing references only on full success.

First pass collects every train ID's fused feature. Then Eq.15/16 references
are installed once. Second pass maps every train sample using the same frozen
model and the new references; Eq.21 generates targets for completed_epoch + 1.
Batch order may differ; duplicate/missing/foreign IDs fail. A one-shot iterator
fails second-pass coverage and rolls back. No validation/test refresh is allowed.
The explicit split tag cannot detect dishonestly tagged data: runner owns IDs.
"""
    if source_split != 'train':
        raise ValueError('APD refresh is train-only')
    cache.validate()
    if type(completed_epoch) is not int or completed_epoch != cache.epoch:
        raise ValueError('Refresh must follow the cache epoch exactly once')
    labels = torch.as_tensor(train_labels).detach().cpu().clone()
    if labels.shape != cache.targets.shape or not torch.isfinite(labels).all():
        raise ValueError('Finite labels in canonical train_ids order required')
    if not (labels >= 0).any() or not (labels < 0).any():
        raise ValueError('Both training polarity classes required')
    positions = {key: i for i, key in enumerate(cache.train_ids)}
    modes = [(module, module.training) for module in core.modules()]
    apd = core.apd
    names = ('positive', 'negative', 'counts', 'reference_version')
    old = {name: getattr(apd, name).clone() for name in names}

    def collect(key):
        collected = [None] * len(positions)
        stream = batches() if callable(batches) else batches
        for ids, payload in stream:
            ids = _ids(ids)
            if any(i not in positions for i in ids):
                raise ValueError('Snapshot contains an ID outside train')
            rows = [positions[i] for i in ids]
            if any(collected[i] is not None for i in rows):
                raise ValueError('Snapshot repeats a train ID')
            output = forward(core, payload)
            value = output[key]
            if key == 'shared':
                if value.ndim != 3 or value.shape[1:] != (3, apd.positive.numel()):
                    raise ValueError('Expected shared [B,3,H]')
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite shared snapshot')
                value, _ = apd.infer_targets(value)
            elif value.ndim != 2 or value.shape[1:] != apd.positive.shape:
                raise ValueError('Expected fused [B,H]')
            if value.shape[0] != len(ids) or not torch.isfinite(value).all():
                raise ValueError('Invalid snapshot batch')
            value = value.detach().cpu().clone()
            for row, item in zip(rows, value):
                collected[row] = item
        if any(value is None for value in collected):
            raise ValueError('Snapshot does not cover all train IDs')
        return torch.stack(collected)

    try:
        core.eval()
        fused = collect('fused')
        # Setter requires training mode; no forward occurs until eval is restored.
        apd.train()
        apd.set_references(fused.to(apd.positive), labels.to(apd.positive.device),
                           source_split='train')
        core.eval()
        raw = collect('shared')
        targets = apd.smooth_targets(cache.targets.detach().cpu(), raw,
                                     epoch=completed_epoch + 1)
        result = APDCache(cache.train_ids, targets, completed_epoch + 1)
        result.validate()
        return result
    except BaseException:
        for name, value in old.items():
            getattr(apd, name).copy_(value)
        raise
    finally:
        # Preserve mixed-mode trees, e.g. a frozen encoder inside a training model.
        for module, mode in modes:
            module.training = mode


@torch.no_grad()
def refresh_apd(core, forward_batches, train_ids, previous_targets, next_epoch,
                *, source_split='train'):
    """Runner adapter: callback yields (CPU long IDs, output, CPU float labels).

Returns a CPU target tensor in train_ids order. Parent must freeze external BERT
and preprocessing across both calls, restore their modes afterwards, and persist
targets plus epoch and IDs. next_epoch=2 follows training epoch one with zeros.
The callback is invoked exactly twice on success, in eval/no-grad mode, with new
references installed between invocations. Outputs need fused and shared keys.
"""
    if source_split != 'train' or type(next_epoch) is not int or next_epoch < 2:
        raise ValueError('Train-only refresh requires next_epoch >= 2')
    ids = _ids(train_ids)
    cache = APDCache(ids, previous_targets.detach().cpu().clone(), next_epoch - 1)
    cache.validate()
    positions = {key: i for i, key in enumerate(ids)}
    labels = torch.empty(len(ids))
    seen = set()
    first = []
    modes = [(module, module.training) for module in core.modules()]

    def validate_batch(indices, batch_labels):
        if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.long
                or indices.device.type != 'cpu'):
            raise ValueError('Callback indices must be CPU LongTensor')
        keys = _ids(indices)
        if (not isinstance(batch_labels, torch.Tensor) or batch_labels.device.type != 'cpu'
                or not batch_labels.is_floating_point() or batch_labels.shape != (len(keys),)
                or not torch.isfinite(batch_labels).all()):
            raise ValueError('Callback labels must be finite CPU float [B]')
        if any(key not in positions for key in keys):
            raise ValueError('Callback ID outside train')
        return keys, [positions[key] for key in keys]

    try:
        core.eval()
        for indices, output, batch_labels in forward_batches():
            keys, rows = validate_batch(indices, batch_labels)
            if seen.intersection(keys):
                raise ValueError('Duplicate train IDs')
            seen.update(keys)
            labels[rows] = batch_labels
            first.append((keys, {'fused': output['fused'].detach().cpu().clone()}))
        if len(seen) != len(ids):
            raise ValueError('Snapshot does not cover all train IDs')
        pass_number = 0

        def batches():
            nonlocal pass_number
            pass_number += 1
            if pass_number == 1:
                yield from first
            else:
                for indices, output, batch_labels in forward_batches():
                    keys, rows = validate_batch(indices, batch_labels)
                    if not torch.equal(labels[rows], batch_labels.to(labels.dtype)):
                        raise ValueError('Train labels changed between passes')
                    yield keys, output

        result = refresh_apd_cache(core, batches, lambda model, output: output,
                                   cache=cache, train_labels=labels,
                                   completed_epoch=next_epoch - 1, source_split=source_split)
        return result.targets
    finally:
        for module, mode in modes:
            module.training = mode
