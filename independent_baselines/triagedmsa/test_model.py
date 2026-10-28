"""Synthetic CPU engineering tests, not benchmark evidence."""
import io
import unittest

import torch

from model import APD, TriagedMSA


class TriagedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261026)
        torch.set_num_threads(1)
        self.model = TriagedMSA(hidden=16, heads=2, dropout=0, max_length=8).cpu()
        self.x = tuple(torch.randn(4, 6, d) for d in (768, 5, 20))
        self.mask = torch.arange(6)[None] < torch.tensor([6, 5, 3, 2])[:, None]
        self.labels = torch.tensor([-1., 1., 0., -2.])
        self.targets = torch.zeros(4)

    def test_backward_both_routes(self):
        for sign, branch in [(-1, 'ssa'), (1, 'sca')]:
            self.model.zero_grad(set_to_none=True)
            with torch.no_grad():
                self.model.sdt.out.bias.fill_(sign * 5)
            out = self.model(*self.x, masks=self.mask, labels=self.labels,
                             relationship_targets=self.targets)
            self.assertEqual(out['prediction'].shape, (4,))
            self.assertEqual(set(out['losses']), {'total', 'sentiment', 'triage', 'consistency'})
            self.assertEqual(out['disagreement'].all().item(), sign < 0)
            out['losses']['total'].backward()
            for prefix in ('encoders.0', 'encoders.1', 'encoders.2', branch, 'sdt', 'apd', 'regression'):
                grads = [p.grad for n, p in self.model.named_parameters()
                         if n.startswith(prefix) and p.grad is not None]
                self.assertTrue(grads, prefix)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads), prefix)
                self.assertGreater(sum(g.abs().sum().item() for g in grads), 0, prefix)

    def test_default_mosi_backward(self):
        model = TriagedMSA().cpu()
        out = model(*(x[:2, :3] for x in self.x), labels=self.labels[:2],
                    relationship_targets=self.targets[:2])
        out['losses']['total'].backward()
        self.assertTrue(torch.isfinite(out['losses']['total']))
        self.assertEqual(model.input_dims, (768, 5, 20))

    def test_eval_no_updates_or_label_dependency(self):
        self.model.apd.set_references(torch.randn(4, 16), self.labels, source_split='train')
        self.model.eval()
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        out1 = self.model(*self.x, masks=self.mask)
        out2 = self.model(*self.x, masks=self.mask, labels=-self.labels)
        torch.testing.assert_close(out1['prediction'], out2['prediction'], rtol=0, atol=0)
        torch.testing.assert_close(out1['apd_raw_targets'], out2['apd_raw_targets'])
        for k, v in self.model.state_dict().items():
            torch.testing.assert_close(v, state[k], rtol=0, atol=0)
        with self.assertRaises(RuntimeError):
            self.model.apd.set_references(torch.randn(4, 16), self.labels, source_split='train')

    def test_train_forward_does_not_update_references(self):
        before = self.model.apd.positive.clone()
        self.model(*self.x, labels=self.labels, relationship_targets=self.targets)
        torch.testing.assert_close(self.model.apd.positive, before)
        self.assertEqual(self.model.apd.reference_version.item(), 0)
        self.assertFalse(self.model.apd.ready)

    def test_reference_rejection_and_means(self):
        fused = torch.arange(64.).reshape(4, 16)
        for split in ('validation', 'test', 'dev', ''):
            with self.assertRaises(RuntimeError):
                self.model.apd.set_references(fused, self.labels, source_split=split)
        self.model.apd.set_references(fused, self.labels, source_split='train')
        torch.testing.assert_close(self.model.apd.positive, fused[[1, 2]].mean(0))
        torch.testing.assert_close(self.model.apd.negative, fused[[0, 3]].mean(0))
        before = self.model.apd.positive.clone()
        with self.assertRaises(ValueError):
            self.model.apd.set_references(fused, torch.ones(4), source_split='train')
        torch.testing.assert_close(before, self.model.apd.positive)
        self.assertEqual(self.model.apd.reference_version.item(), 1)

    def test_threshold_and_smoothing(self):
        apd = self.model.apd
        apd.set_references(torch.stack((torch.ones(16), -torch.ones(16))),
                           torch.tensor([1., -1.]), source_split='train')
        # Weak same-positive polarities have products <1 but >0: Algorithm 1 agreement.
        shared = torch.full((3, 3, 16), 0.01)
        shared[1, 1] = -0.01
        shared[2] = 0  # exactly equidistant is agreement under strict <0.
        raw, d = apd.infer_targets(shared)
        torch.testing.assert_close(raw, torch.tensor([1., -1., 1.]))
        self.assertTrue(0 < (d[0, 0] * d[0, 1]).item() < 1)
        initial = APD.smooth_targets(None, raw, 1)
        torch.testing.assert_close(initial, torch.zeros(3))
        second = APD.smooth_targets(initial, raw, 2)
        torch.testing.assert_close(second, raw / 3)
        torch.testing.assert_close(APD.smooth_targets(second, raw, 3), raw * 5 / 12)
        with self.assertRaises(ValueError):
            APD.smooth_targets(None, raw, 2)

    def test_state_roundtrip(self):
        self.model.apd.set_references(torch.randn(4, 16), self.labels, source_split='train')
        self.model(*self.x, masks=self.mask, labels=self.labels, relationship_targets=self.targets)
        self.model.eval()
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        other = TriagedMSA(hidden=16, heads=2, dropout=0, max_length=8).eval()
        other.load_state_dict(torch.load(buffer, weights_only=True))
        for key in ('prediction', 'apd_raw_targets', 'apd_polarity'):
            torch.testing.assert_close(self.model(*self.x, masks=self.mask)[key],
                                       other(*self.x, masks=self.mask)[key])

    def test_mask_padding_invariance(self):
        self.model.eval()
        baseline = self.model(*self.x, masks=self.mask)
        changed = tuple(x.masked_fill(~self.mask.unsqueeze(-1), float('nan')) for x in self.x)
        actual = self.model(*changed, masks=self.mask)
        torch.testing.assert_close(actual['prediction'], baseline['prediction'])
        padded = tuple(torch.cat((x, torch.full((4, 2, x.shape[-1]), float('nan'))), 1) for x in self.x)
        pm = torch.cat((self.mask, torch.zeros(4, 2, dtype=torch.bool)), 1)
        out = self.model(*padded, masks=pm)
        torch.testing.assert_close(out['prediction'], baseline['prediction'], atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.isfinite(out['ssa_gates']).all())

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.model(*self.x, labels=self.labels)
        bad = self.mask.clone()
        bad[0] = False
        with self.assertRaises(ValueError):
            self.model(*self.x, masks=bad)
        bad = self.mask.clone()
        bad[0, 1] = False
        with self.assertRaises(ValueError):
            self.model(*self.x, masks=bad)
        with self.assertRaises(ValueError):
            self.model(self.x[0], self.x[1][:, :-1], self.x[2])
        with self.assertRaises(ValueError):
            self.model(*(x[:1, :1] for x in self.x))
        self.model.eval()
        out = self.model(*(x[:1, :1] for x in self.x))
        self.assertTrue(torch.isfinite(out['prediction']).all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
