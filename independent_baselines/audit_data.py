"""Read-only feature/split admission; never changes benchmark arrays."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from run_real_data_smoke import CompatUnpickler, digest


def load(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rb') as f:
        return CompatUnpickler(f).load()


def sample_ids(split, name):
    n = len(split['regression_labels'])
    key = 'id' if 'id' in split else 'ids' if 'ids' in split else None
    if key is None:
        return [f'{name}:row:{i}' for i in range(n)], 'row_only'
    ids = [str(x) for x in split[key]]
    if len(ids) != n or len(set(ids)) != n:
        raise ValueError(f'Nonunique/invalid {name} IDs')
    return ids, key


def audit(path):
    payload = load(path)
    result = {'path': str(path), 'sha256': digest(path), 'splits': {}}
    all_ids = []
    for name in ['train', 'valid', 'test']:
        s = payload[name]
        t, a, v, y = [np.asarray(s[k]) for k in ['text_bert', 'audio', 'vision', 'regression_labels']]
        mask = t[:, 1].astype(bool)
        assert a.shape[:2] == v.shape[:2] == mask.shape
        assert mask.any(1).all() and not ((~mask[:, :-1]) & mask[:, 1:]).any()
        assert np.isfinite(a[mask]).all() and np.isfinite(v[mask]).all() and np.isfinite(y).all()
        ids, key = sample_ids(s, name)
        all_ids.append(set(ids))
        result['splits'][name] = {'keys': list(s), 'n': len(y), 'text_shape': list(t.shape),
                                'audio_shape': list(a.shape), 'vision_shape': list(v.shape),
                                'label_min': float(y.min()), 'label_max': float(y.max()),
                                'id_key': key, 'id_preview': ids[:3]}
    assert not any(all_ids[i] & all_ids[j] for i in range(3) for j in range(i))
    result['status'] = 'passed_structural_checks_alignment_assumes_benchmark_aligned_artifact'
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('paths', nargs='+', type=Path)
    args = p.parse_args()
    for path in args.paths:
        print(json.dumps(audit(path)), flush=True)
