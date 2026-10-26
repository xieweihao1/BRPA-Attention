"""Self-contained BRPA tests; run via unittest discovery (no pytest required)."""

import importlib
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import BRPA as public_brpa  # noqa: E402


class TestBRPA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        importlib.reload(public_brpa)
        torch.manual_seed(5)

    def test_original_default_configuration(self):
        self.assertEqual(public_brpa.COMPONENT_VARIANT, "seed_prop_boundary_refine")
        self.assertEqual(public_brpa.PROP_STEPS, 4)
        module = public_brpa.BRPA(channels=8)
        self.assertFalse(module.use_channel)
        self.assertEqual(module.steps, 4)

    def test_spatial_dimensions_preserved(self):
        for steps in (0, 1, 2, 5):
            with self.subTest(steps=steps):
                module = public_brpa.BRPA(channels=8, steps=steps).eval()
                x = torch.randn(2, 8, 11, 15)
                with torch.no_grad():
                    y = module(x)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all().item())

    def test_full_variant_has_all_components(self):
        public_brpa.COMPONENT_VARIANT = "full"
        module = public_brpa.BRPA(channels=8, steps=2)
        self.assertTrue(module.use_seed)
        self.assertTrue(module.use_boundary)
        self.assertTrue(module.use_prop)
        self.assertTrue(module.use_refine)
        self.assertTrue(module.use_channel)
        self.assertEqual(module(torch.randn(2, 8, 11, 11)).shape, (2, 8, 11, 11))

    def test_identity_is_exact(self):
        public_brpa.COMPONENT_VARIANT = "identity"
        x = torch.randn(2, 8, 10, 10)
        y = public_brpa.BRPA(8)(x)
        self.assertIs(y, x)

    def test_backward(self):
        public_brpa.COMPONENT_VARIANT = "full"
        module = public_brpa.BRPA(8, steps=2)
        x = torch.randn(2, 8, 10, 10, requires_grad=True)
        module(x).square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all().item())
        self.assertIsNotNone(module.seed[0].weight.grad)

    def test_learnable_rate_matches_initialization(self):
        module = public_brpa.BRPA(8)
        rate = module.get_effective_p()
        self.assertAlmostEqual(rate, torch.sigmoid(torch.tensor(0.5)).item(), places=7)

    def test_invalid_steps_raise(self):
        with self.assertRaises(ValueError):
            public_brpa.BRPA(channels=8, steps=6)


if __name__ == "__main__":
    unittest.main()
