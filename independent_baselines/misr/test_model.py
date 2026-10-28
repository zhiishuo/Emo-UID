"""Synthetic CPU engineering checks only; not scientific evaluation."""
import unittest
import torch
from torch.distributions import Normal, kl_divergence
from model import MISR, MISRConfig, gaussian_kl


class MISRTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(1111)
        self.model = MISR(MISRConfig(hidden_dim=16, dropout=0.0)).cpu()
        self.inputs = [torch.randn(3, 5, d) for d in (768, 5, 20)]
        self.labels = torch.tensor([-1.0, 0.0, 2.0])

    def test_full_backward_all_parameters(self):
        out = self.model(*self.inputs, labels=self.labels)
        self.assertEqual(out['prediction'].shape, (3,))
        self.assertEqual(out['environment_predictions'].shape, (3, 8))
        self.assertEqual(len(set(out['environment_names'])), 8)
        for value in out['losses'].values():
            self.assertTrue(torch.isfinite(value).item())
        out['losses']['total'].backward()
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all().item(), name)
        for name, branch in self.model.branches.items():
            self.assertGreater(branch.scale.weight.grad.abs().sum().item(), 0, name)
            self.assertGreater(branch.reconstruction.weight.grad.abs().sum().item(), 0, name)

    def test_kl_matches_torch(self):
        a, b = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        s, t = torch.rand(2, 4, 1) + .1, torch.rand(2, 4, 1) + .1
        expected = kl_divergence(Normal(a, s), Normal(b, t)).sum(-1)
        torch.testing.assert_close(gaussian_kl(a, s, b, t), expected)
        torch.testing.assert_close(gaussian_kl(a, s, a, s), torch.zeros(2, 4))

    def test_equations_11_12_13(self):
        out = self.model(*self.inputs, labels=self.labels, sample=False)
        risk = (out['environment_predictions'] - self.labels[:, None]).abs().mean(0)
        loss = out['losses']
        torch.testing.assert_close(loss['invariant_constraint'], risk.var(unbiased=False))
        torch.testing.assert_close(loss['sentiment_prediction'], risk[0])
        torch.testing.assert_close(loss['distribution_constraint'], loss['normalized_alignment']
                                   + loss['cross_modal_alignment'] + loss['data_reconstruction'])
        torch.testing.assert_close(loss['total'], risk[0] + .4 * loss['distribution_constraint']
                                   + risk.var(unbiased=False))

    def test_kl_extreme_equal_scales_remain_finite(self):
        mu = torch.zeros(2, 3, 128)
        for scale in (1e-30, 1e30):
            std = torch.full((2, 3, 1), scale, requires_grad=True)
            kl = gaussian_kl(mu, std, mu, std)
            torch.testing.assert_close(kl, torch.zeros(2, 3), rtol=0, atol=0)
            kl.sum().backward()
            self.assertTrue(torch.isfinite(std.grad).all())

    def test_kl_sums_latent_coordinates_without_hidden_rescaling(self):
        # 128 unit-variance coordinates shifted by one have KL = 128 / 2.
        mu = torch.ones(2, 3, 128)
        std = torch.ones(2, 3, 1)
        torch.testing.assert_close(gaussian_kl(mu, std, torch.zeros_like(mu), std),
                                   torch.full((2, 3), 64.0))

    def test_reconstruction_is_squared_norm_and_target_has_gradient(self):
        branch = self.model.branches['language']
        captured = []
        handle = branch.reconstruction.register_forward_hook(
            lambda module, args, result: captured.append(result))
        try:
            mask = torch.ones(3, 5, dtype=torch.bool)
            mask[0, -2:] = False
            h, _, _, _, dr = branch(self.inputs[0], mask, sample=False)
        finally:
            handle.remove()
        residual = h - captured[0]
        expected = ((residual.square().sum(-1) * mask).sum(1) / mask.sum(1)).mean()
        torch.testing.assert_close(dr, expected)
        # Capture both paths; this distinguishes a detached reconstruction target.
        grad_h, grad_reconstructed = torch.autograd.grad(dr, (h, captured[0]), retain_graph=True)
        direct = 2 * residual * mask[..., None] / mask.sum(1)[:, None, None] / 3
        torch.testing.assert_close(grad_reconstructed, -direct)
        # grad_h also includes the decoder path, so test direct-target contribution
        # by subtracting the path propagated from the reconstructed output.
        decoder_path = torch.autograd.grad(captured[0], h, grad_outputs=grad_reconstructed)[0]
        torch.testing.assert_close(grad_h - decoder_path, direct)

    def test_cross_modal_kl_keeps_all_six_directed_pairs(self):
        captured = {}
        handles = [branch.register_forward_hook(
            lambda module, args, result, name=name: captured.__setitem__(name, result))
            for name, branch in self.model.branches.items()]
        try:
            out = self.model(*self.inputs, sample=False)
        finally:
            for handle in handles:
                handle.remove()
        expected = torch.zeros(())
        for a in self.model.names:
            for b in self.model.names:
                if a != b:
                    _, mu_a, std_a, _, _ = captured[a]
                    _, mu_b, std_b, _, _ = captured[b]
                    expected += kl_divergence(Normal(mu_a, std_a),
                                              Normal(mu_b, std_b)).sum(-1).mean()
        torch.testing.assert_close(out['losses']['cross_modal_alignment'], expected)

    def test_padding_invariance_and_nan_padding(self):
        self.model.eval()
        ref = self.model(*self.inputs, labels=self.labels)
        padded = [torch.cat((x, torch.full((3, 2, x.shape[2]), float('nan'))), 1)
                  for x in self.inputs]
        masks = {name: torch.tensor([[True]*5 + [False]*2]*3) for name in self.model.names}
        result = self.model(*padded, labels=self.labels, masks=masks)
        torch.testing.assert_close(result['prediction'], ref['prediction'], atol=1e-6, rtol=1e-5)
        for name in ref['losses']:
            torch.testing.assert_close(result['losses'][name], ref['losses'][name], atol=1e-4, rtol=1e-5)
        self.assertEqual(result['fusion_weights'][:, 5:].abs().sum().item(), 0)

    def test_masks_and_normalized_weights(self):
        masks = {name: torch.ones(3, 5, dtype=torch.bool) for name in self.model.names}
        masks['audio'][:, -1] = False
        out = self.model(*self.inputs, masks=masks, labels=self.labels)
        torch.testing.assert_close(out['fusion_weights'].sum(-1), torch.ones(3, 5))
        self.assertEqual(out['fusion_weights'][:, -1, 1].abs().sum().item(), 0)
        out['losses']['total'].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_reject_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.model(self.inputs[0], self.inputs[1][:, :-1], self.inputs[2])
        with self.assertRaises(ValueError):
            self.model(*self.inputs, masks={'audio': torch.zeros(3, 5, dtype=torch.bool)})
        with self.assertRaises(ValueError):
            self.model(*self.inputs, labels=torch.zeros(3, 2))
        with self.assertRaises(ValueError):
            self.model(*self.inputs, masks={'acoustic': torch.ones(3, 5, dtype=torch.bool)})
        bad = self.inputs[0].clone()
        bad[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.model(bad, *self.inputs[1:])
        masks = {'language': torch.tensor([[True, False, False, False, False]]*3),
                 'audio': torch.tensor([[False, True, False, False, False]]*3)}
        with self.assertRaises(ValueError):
            self.model(*self.inputs, masks=masks)

    def test_unlabeled_deterministic_inference(self):
        self.model.eval()
        with torch.no_grad():
            a = self.model(*self.inputs)
            b = self.model(*self.inputs)
        torch.testing.assert_close(a['prediction'], b['prediction'], rtol=0, atol=0)
        self.assertNotIn('total', a['losses'])
        self.assertIsNone(a['environment_risks'])

    def test_sampling_and_input_gradients(self):
        inputs = [x.clone().requires_grad_() for x in self.inputs]
        a = self.model(*inputs, labels=self.labels, sample=True)
        b = self.model(*inputs, labels=self.labels, sample=True)
        self.assertNotEqual(a['losses']['data_reconstruction'].item(), b['losses']['data_reconstruction'].item())
        torch.testing.assert_close(a['prediction'], b['prediction'])
        a['losses']['total'].backward()
        for x in inputs:
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_invariant_penalty_has_gradient(self):
        out = self.model(*self.inputs, labels=self.labels, sample=False)
        out['losses']['invariant_constraint'].backward()
        self.assertGreater(self.model.phi[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.branches['language'].mean.weight.grad.abs().sum().item(), 0)

    def test_default_mosi_T50_backward(self):
        model = MISR().cpu()
        xs = [torch.randn(2, 50, d) for d in (768, 5, 20)]
        result = model(*xs, labels=torch.tensor([-.5, .5]))
        result['losses']['total'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_mosei_base_and_large_contextual_interfaces(self):
        for language_dim in (768, 1024):
            model = MISR(MISRConfig(language_dim=language_dim, audio_dim=74,
                                   vision_dim=35, hidden_dim=16, dropout=0.0)).cpu()
            xs = [torch.randn(2, 5, d, requires_grad=True)
                  for d in (language_dim, 74, 35)]
            result = model(*xs, labels=torch.tensor([-.5, .5]), sample=False)
            result['losses']['total'].backward()
            self.assertEqual(result['prediction'].shape, (2,))
            for x in xs:
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(x.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
