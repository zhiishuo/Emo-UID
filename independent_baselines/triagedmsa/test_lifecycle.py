"""CPU-only lifecycle regression tests, not scientific results."""
import unittest

import torch

from lifecycle import APDCache, refresh_apd_cache as refresh_apd, refresh_apd as runner_refresh
from model import TriagedMSA


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.core = TriagedMSA(3, 2, 4, hidden=8, heads=2, dropout=.5, max_length=4)
        self.x = tuple(torch.randn(4, 3, d) for d in (3, 2, 4))
        self.labels = torch.tensor([-1., 0., 2., -2.])
        self.cache = APDCache.initialize(('a', 'b', 'c', 'd'))
        self.calls = 0

    def batches(self):
        self.calls += 1
        order = [2, 0, 3, 1] if self.calls % 2 else [1, 3, 0, 2]
        for start in (0, 2):
            rows = order[start:start + 2]
            yield tuple(self.cache.train_ids[i] for i in rows), tuple(x[rows] for x in self.x)

    def forward(self, core, x):
        self.assertFalse(torch.is_grad_enabled())
        self.assertTrue(all(not m.training for m in core.modules()))
        return core(*x)

    def refresh(self, **overrides):
        args = dict(cache=self.cache, train_labels=self.labels,
                    completed_epoch=self.cache.epoch, source_split='train')
        args.update(overrides)
        return refresh_apd(self.core, self.batches, self.forward, **args)

    def test_exact_full_snapshot_modes_and_cache(self):
        self.core.eval()
        with torch.no_grad():
            fused = self.core(*self.x)['fused']
        self.core.train()
        self.core.encoders[0].eval()
        modes = [m.training for m in self.core.modules()]
        old = {k: v.clone() for k, v in self.core.state_dict().items()}
        new = self.refresh()
        self.assertEqual(modes, [m.training for m in self.core.modules()])
        self.assertEqual(new.epoch, 2)
        self.assertEqual(self.calls, 2)
        torch.testing.assert_close(self.core.apd.positive, fused[[1, 2]].mean(0))
        torch.testing.assert_close(self.core.apd.negative, fused[[0, 3]].mean(0))
        for key, value in self.core.state_dict().items():
            if key not in ('apd.positive', 'apd.negative', 'apd.counts', 'apd.reference_version'):
                torch.testing.assert_close(value, old[key], rtol=0, atol=0)
        self.core.eval()
        with torch.no_grad():
            raw = self.core(*self.x)['apd_raw_targets']
        torch.testing.assert_close(new.targets, raw / 3)
        torch.testing.assert_close(self.cache.targets, torch.zeros(4))
        torch.testing.assert_close(new.for_ids(('d', 'a')), new.targets[[3, 0]])
        restored = APDCache.from_state_dict(new.state_dict())
        torch.testing.assert_close(restored.targets, new.targets)
        third = self.refresh(cache=new, completed_epoch=2)
        self.assertTrue(all(not m.training for m in self.core.modules()))
        torch.testing.assert_close(third.targets, (2 * new.targets + raw) / 4)

    def test_failed_second_pass_rolls_back(self):
        self.refresh()
        old = {k: v.clone() for k, v in self.core.state_dict().items()}
        modes = [m.training for m in self.core.modules()]
        once = iter(list(self.batches()))
        with self.assertRaisesRegex(ValueError, 'cover all'):
            refresh_apd(self.core, once, self.forward, cache=self.cache,
                        train_labels=self.labels, completed_epoch=1, source_split='train')
        for key, value in self.core.state_dict().items():
            torch.testing.assert_close(value, old[key], rtol=0, atol=0)
        self.assertEqual(modes, [m.training for m in self.core.modules()])

    def test_split_epoch_and_class_guards(self):
        for kwargs in ({'source_split': 'test'}, {'source_split': 'validation'},
                       {'completed_epoch': 2}, {'train_labels': torch.ones(4)},
                       {'train_labels': torch.tensor([1., -1., float('nan'), 2.])}):
            with self.assertRaises(ValueError):
                self.refresh(**kwargs)
        self.assertEqual(self.calls, 0)
        self.assertEqual(self.core.apd.reference_version.item(), 0)

    def test_duplicate_missing_foreign_ids(self):
        for ids in (('a', 'a'), ('a', 'other'), ('a', 'b')):
            batches = [(ids, tuple(x[:2] for x in self.x))]
            with self.assertRaises(ValueError):
                refresh_apd(self.core, batches, self.forward, cache=self.cache,
                            train_labels=self.labels, completed_epoch=1, source_split='train')
            self.assertTrue(self.core.training)
            self.assertFalse(self.core.apd.ready)

    def test_callback_failure_restores_mixed_modes(self):
        self.core.encoders[1].eval()
        modes = [m.training for m in self.core.modules()]
        def fail(core, batch):
            raise RuntimeError('forward failed')
        with self.assertRaisesRegex(RuntimeError, 'forward failed'):
            refresh_apd(self.core, self.batches, fail, cache=self.cache,
                        train_labels=self.labels, completed_epoch=1, source_split='train')
        self.assertEqual(modes, [m.training for m in self.core.modules()])

    def test_cache_validation(self):
        for ids in ((), ('x', 'x'), torch.tensor([1.5])):
            with self.assertRaises(ValueError):
                APDCache.initialize(ids)
        with self.assertRaises(ValueError):
            self.cache.for_ids(('validation-id',))
        bad = self.cache.state_dict()
        bad['targets'].fill_(1)
        with self.assertRaises(ValueError):
            APDCache.from_state_dict(bad)

    def test_runner_adapter(self):
        calls = []
        def callback():
            calls.append(int(self.core.apd.reference_version))
            for rows in ([2, 0], [3, 1]):
                indices = torch.tensor(rows)
                yield indices, self.forward(self.core, tuple(x[rows] for x in self.x)), self.labels[rows]
        targets = runner_refresh(self.core, callback, torch.arange(4), torch.zeros(4), 2)
        self.assertEqual(calls, [0, 1])
        self.assertTrue(self.core.training)
        self.core.eval()
        with torch.no_grad():
            raw = self.core(*self.x)['apd_raw_targets']
        torch.testing.assert_close(targets, raw / 3)

    def test_runner_changed_labels_roll_back(self):
        calls = 0
        def callback():
            nonlocal calls
            calls += 1
            yield torch.arange(4), self.forward(self.core, self.x), self.labels + (calls - 1)
        with self.assertRaisesRegex(ValueError, 'labels changed'):
            runner_refresh(self.core, callback, torch.arange(4), torch.zeros(4), 2)
        self.assertTrue(self.core.training)
        self.assertEqual(int(self.core.apd.reference_version), 0)


if __name__ == '__main__':
    unittest.main()
