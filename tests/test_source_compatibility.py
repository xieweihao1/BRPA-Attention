"""Check that the cleaned BRPA preserves the archived source forward results."""

import importlib
import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import BRPA as cleaned  # noqa: E402

original_path = ROOT / "reference" / "BRPA_ablation_original.py"
spec = importlib.util.spec_from_file_location("original_brpa_for_tests", original_path)


class TestSourceCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        importlib.reload(cleaned)
        self.original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.original)
        torch.manual_seed(19)

    def check_case(self, **overrides):
        for name, value in overrides.items():
            setattr(cleaned, name, value)
            setattr(self.original, name, value)

        torch.manual_seed(23)
        public_module = cleaned.BRPA(channels=8, steps=2).eval()
        torch.manual_seed(23)
        original_module = self.original.BRPA(channels=8, steps=2).eval()

        self.assertEqual(public_module.state_dict().keys(), original_module.state_dict().keys())
        for key in public_module.state_dict():
            torch.testing.assert_close(public_module.state_dict()[key], original_module.state_dict()[key], rtol=0, atol=0)

        inputs = torch.randn(2, 8, 9, 12)
        with torch.no_grad():
            actual = public_module(inputs)
            expected = original_module(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_default_nonfull_variant(self):
        self.check_case()

    def test_complete_variant(self):
        self.check_case(COMPONENT_VARIANT="full")

    def test_fixed_rate_and_hard_barrier(self):
        self.check_case(COMPONENT_VARIANT="full", PROP_RATE_MODE="fixed", BARRIER_MODE="hard")

    def test_average_pool_variant(self):
        self.check_case(COMPONENT_VARIANT="seed_prop", PROP_OPERATOR="avgpool3")

    def test_learnable_conv_variant(self):
        self.check_case(COMPONENT_VARIANT="full", PROP_OPERATOR="conv3", BARRIER_MODE="power")

    def test_identity_variant(self):
        self.check_case(COMPONENT_VARIANT="identity")


if __name__ == "__main__":
    unittest.main()
