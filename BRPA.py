import torch
from torch import nn
from torch.nn import functional as F

COMPONENT_VARIANT = "seed_prop_boundary_refine"

PROP_STEPS = 4

PROP_OPERATOR = "maxpool3"

PROP_RATE_MODE = "learnable"
ETA_INIT = 0.5
FIXED_P = 0.75

BARRIER_MODE = "soft"
BARRIER_THRESHOLD = 0.70
BARRIER_GAMMA = 1.0
BARRIER_SHIFT_RATIO = 0.50

RECALIBRATION_MODE = "full"

ENABLE_CHANNEL = False


def _check_config():
    valid_components = {
        "identity", "seed", "seed_prop", "seed_prop_boundary",
        "seed_prop_boundary_refine", "full",
    }
    valid_ops = {"maxpool3", "avgpool3", "maxpool5", "conv3"}
    valid_rates = {"learnable", "fixed"}
    valid_barriers = {"none", "soft", "hard", "shifted", "power"}
    valid_recal = {"spatial", "spatial_channel", "spatial_residual", "full"}

    assert COMPONENT_VARIANT in valid_components, f"Invalid COMPONENT_VARIANT: {COMPONENT_VARIANT}"
    assert PROP_OPERATOR in valid_ops, f"Invalid PROP_OPERATOR: {PROP_OPERATOR}"
    assert PROP_RATE_MODE in valid_rates, f"Invalid PROP_RATE_MODE: {PROP_RATE_MODE}"
    assert BARRIER_MODE in valid_barriers, f"Invalid BARRIER_MODE: {BARRIER_MODE}"
    assert RECALIBRATION_MODE in valid_recal, f"Invalid RECALIBRATION_MODE: {RECALIBRATION_MODE}"
    assert PROP_STEPS in {0, 1, 2, 3, 4, 5}, "PROP_STEPS must be between 0 and 5"
    assert 0.0 <= FIXED_P <= 1.0, "FIXED_P must be in [0, 1]"
    assert 0.0 < BARRIER_THRESHOLD < 1.0
    assert BARRIER_GAMMA > 0.0
    assert 0.0 <= BARRIER_SHIFT_RATIO <= 1.0
    assert isinstance(ENABLE_CHANNEL, bool)


_check_config()


class BRPA(nn.Module):
    def __init__(self, channels, steps=None):
        super().__init__()
        self.channels = channels
        self.steps = PROP_STEPS if steps is None else int(steps)
        if self.steps not in {0, 1, 2, 3, 4, 5}:
            raise ValueError("BRPA propagation steps should be in {0,1,2,3,4,5}.")

        self.variant = COMPONENT_VARIANT
        self.prop_operator = PROP_OPERATOR
        self.rate_mode = PROP_RATE_MODE
        self.barrier_mode = BARRIER_MODE
        self.recalibration_mode = RECALIBRATION_MODE

        self.use_seed = self.variant != "identity"
        self.use_prop = self.variant in {
            "seed_prop", "seed_prop_boundary", "seed_prop_boundary_refine", "full",
        }
        self.use_boundary = self.variant in {
            "seed_prop_boundary", "seed_prop_boundary_refine", "full",
        }
        self.use_refine = self.variant in {"seed_prop_boundary_refine", "full"}
        self.use_channel = (self.variant == "full") or ENABLE_CHANNEL

        if not self.use_channel and self.recalibration_mode == "spatial_channel":
            print("Warning: 'spatial_channel' requires channel branch, falling back to 'spatial'.")
            self.recalibration_mode = "spatial"

        if self.use_seed:
            self.seed = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.BatchNorm2d(channels),
                nn.SiLU(),
                nn.Conv2d(channels, 1, 1),
                nn.Sigmoid(),
            )

        if self.use_boundary:
            self.boundary = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.BatchNorm2d(channels),
                nn.SiLU(),
                nn.Conv2d(channels, 1, 1),
                nn.Sigmoid(),
            )

        if self.use_prop:
            if self.rate_mode == "learnable":
                self.eta = nn.Parameter(torch.tensor(float(ETA_INIT)))
            else:
                self.register_buffer(
                    "fixed_p", torch.tensor(float(FIXED_P)), persistent=False
                )

        if self.use_prop and self.prop_operator == "conv3":
            self.prop_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
            nn.init.constant_(self.prop_conv.weight, 1.0 / 9.0)

        if self.use_refine:
            self.refine = nn.Sequential(
                nn.Conv2d(3, 1, 3, padding=1),
                nn.Sigmoid(),
            )

        if self.use_channel:
            hidden = max(channels // 8, 16)
            self.channel = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels, 1),
                nn.Sigmoid(),
            )

    def propagation_operator(self, x):
        if self.prop_operator == "maxpool3":
            return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        if self.prop_operator == "avgpool3":
            return F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        if self.prop_operator == "maxpool5":
            return F.max_pool2d(x, kernel_size=5, stride=1, padding=2)
        if self.prop_operator == "conv3":
            learned = self.prop_conv(x)
            return torch.maximum(x, learned)
        raise RuntimeError(f"Unsupported propagation operator: {self.prop_operator}")

    def propagation_rate(self):
        if self.rate_mode == "learnable":
            return torch.sigmoid(self.eta)
        return self.fixed_p

    def barrier_gate(self, b):
        if b is None or self.barrier_mode == "none":
            return 1.0
        if self.barrier_mode == "soft":
            return 1.0 - b
        if self.barrier_mode == "hard":
            return (b < BARRIER_THRESHOLD).to(dtype=b.dtype)
        if self.barrier_mode == "shifted":
            h, w = b.shape[-2:]
            sh = max(1, int(round(h * BARRIER_SHIFT_RATIO))) if h > 1 else 0
            sw = max(1, int(round(w * BARRIER_SHIFT_RATIO))) if w > 1 else 0
            b_shift = torch.roll(b, shifts=(sh, sw), dims=(-2, -1))
            return 1.0 - b_shift
        if self.barrier_mode == "power":
            return torch.pow((1.0 - b).clamp(0.0, 1.0), BARRIER_GAMMA)
        raise RuntimeError(f"Unsupported barrier mode: {self.barrier_mode}")

    def recalibrate(self, x, a, c=None):
        if self.recalibration_mode == "spatial":
            return a * x
        if self.recalibration_mode == "spatial_channel":
            if c is None:
                raise RuntimeError(
                    "spatial_channel requires the channel branch to be enabled. "
                    "Please set ENABLE_CHANNEL=True or use 'full' variant."
                )
            return a * x * c
        if self.recalibration_mode == "spatial_residual":
            return x + a * x
        if self.recalibration_mode == "full":
            if c is None:
                return x + a * x
            return x + a * x * c
        raise RuntimeError(f"Unsupported recalibration mode: {self.recalibration_mode}")

    def forward(self, x):
        if self.variant == "identity":
            return x

        s0 = self.seed(x)
        s = s0

        b = self.boundary(x) if self.use_boundary else None

        if self.use_prop and self.steps > 0:
            p = self.propagation_rate()
            gate = self.barrier_gate(b)
            for _ in range(self.steps):
                expanded = self.propagation_operator(s)
                delta = F.relu(expanded - s)
                s = s + p * gate * delta
                s = s.clamp(0.0, 1.0)

        if self.use_refine:
            a = self.refine(torch.cat([s, s0, b], dim=1))
        else:
            a = s

        c = self.channel(x) if self.use_channel else None

        return self.recalibrate(x, a, c)

    @torch.no_grad()
    def get_effective_p(self):
        if not self.use_prop:
            return None
        return float(self.propagation_rate().detach().cpu())

    def extra_repr(self):
        p_desc = (
            f"learnable(sigmoid({ETA_INIT}))"
            if self.rate_mode == "learnable"
            else f"fixed({FIXED_P})"
        )
        return (
            f"variant={self.variant}, steps={self.steps}, "
            f"op={self.prop_operator}, p={p_desc}, "
            f"barrier={self.barrier_mode}, recal={self.recalibration_mode}, "
            f"channel_enabled={self.use_channel}"
        )


def get_brpa_ablation_tag():
    p_tag = (
        f"pLearn_eta{ETA_INIT}"
        if PROP_RATE_MODE == "learnable"
        else f"p{FIXED_P}"
    )
    channel_tag = "Chan" if ENABLE_CHANNEL else "NoChan"
    return (
        f"{COMPONENT_VARIANT}"
        f"_K{PROP_STEPS}"
        f"_{PROP_OPERATOR}"
        f"_{p_tag}"
        f"_{BARRIER_MODE}"
        f"_{RECALIBRATION_MODE}"
        f"_{channel_tag}"
    )


if __name__ == "__main__":
    x = torch.randn(2, 64, 40, 40)
    model = BRPA(64)
    y = model(x)
    print(model)
    print("Input :", x.shape)
    print("Output:", y.shape)
    print("Tag   :", get_brpa_ablation_tag())
    print("p     :", model.get_effective_p())