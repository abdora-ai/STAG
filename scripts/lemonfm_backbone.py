"""
Loader for the frozen LemonFM teacher backbone (ConvNeXt-L).

Exposes:

    FrozenLemonFM(ckpt_path) -> nn.Module

    forward(x: [B, 3, 224, 224]) -> (pooled, spatial)
        pooled  : [B, 1536]        LayerNorm2d + flattened global average pool
        spatial : [B, 49, 1536]    LayerNorm2d + spatial-flatten of the pre-pool
                                    ConvNeXt-L stage-4 feature map (7 x 7 tokens)

Both outputs are drawn from the same LemonFM ConvNeXt-L teacher, matching the
convention used in the paper's own `src/model_loader.py::build_LemonFM` for the
pooled feature. The 49 spatial tokens are additionally normalized with the same
LayerNorm2d, so slot-attention pooling in the STAG head operates on features
that live in the same geometry as the pooled representation.

Frozen: every parameter has requires_grad=False and the module is forced into
eval() mode.
"""
import torch
import torch.nn as nn
import torchvision


class FrozenLemonFM(nn.Module):
    def __init__(self, ckpt_path: str):
        super().__init__()

        # Instantiate a fresh, unpretrained ConvNeXt-L
        m = torchvision.models.convnext_large(weights=None)

        # Load the LemonFM teacher checkpoint
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "teacher" not in state:
            raise KeyError(
                f"Expected 'teacher' key in checkpoint {ckpt_path}, "
                f"got keys: {list(state.keys())}"
            )
        sd = state["teacher"]

        # Strip `module.` (DDP) and `backbone.` (MultiCropWrapper) prefixes
        stripped = {}
        for k, v in sd.items():
            nk = k
            for prefix in ("module.", "backbone."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            stripped[nk] = v

        # Drop the DINO projection head - not part of torchvision ConvNeXt
        stripped = {k: v for k, v in stripped.items() if not k.startswith("head.")}

        # Replace ImageNet-1000 classifier linear with Identity so `m(x)` returns 1536-D
        m.classifier[2] = nn.Identity()

        msg = m.load_state_dict(stripped, strict=False)
        if len(msg.missing_keys) != 0:
            raise RuntimeError(f"missing keys when loading LemonFM teacher: {msg.missing_keys}")

        # Extract the modules we need
        self.features = m.features           # ConvNeXt-L trunk -> [B, 1536, 7, 7]
        self.avgpool = m.avgpool             # AdaptiveAvgPool2d(1) -> [B, 1536, 1, 1]
        self.norm2d = m.classifier[0]        # LayerNorm2d(1536)
        self.flatten = m.classifier[1]       # Flatten (from dim 1)

        # Freeze
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        # Force eval() to keep BatchNorm/LayerNorm statistics fixed.
        return super().train(False)

    def forward(self, x: torch.Tensor):
        """
        x : [B, 3, 224, 224], normalized with ImageNet mean/std.
        Returns:
            pooled  : [B, 1536]
            spatial : [B, 49, 1536]
        """
        f = self.features(x)                              # [B, 1536, 7, 7]

        # Pooled: matches paper's build_LemonFM
        p = self.avgpool(f)                               # [B, 1536, 1, 1]
        p = self.norm2d(p)                                # [B, 1536, 1, 1]
        p = self.flatten(p)                               # [B, 1536]

        # Spatial: same LayerNorm2d, then flatten spatial dims to 49 tokens
        s = self.norm2d(f)                                # [B, 1536, 7, 7]
        s = s.flatten(2).transpose(1, 2).contiguous()     # [B, 49, 1536]

        return p, s
