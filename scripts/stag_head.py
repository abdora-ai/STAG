"""

STAG: Spatio-Temporal Affinity Graph head, and its ablation variants.


Implements the three modules of the paper (Sec. 3):

    1. Class-conditional slot pooling over the frozen backbone's 7x7 token grid.

    2. Cosine prototype layer, placing task classes in the backbone's metric space.

    3. Temporal affinity graph: non-parametric label propagation over a video

       graph whose edges are the backbone's own cosine similarities.


Four modes are supported via `mode="..."`, corresponding to the rows of the

ablation table (paper Table 2):


    mode          use_slot   use_prototype   use_temporal   what it is
    -----------   --------   -------------   ------------   -----------------------------------------

    "linear"        False        False          False       Linear(pooled -> N_c). Head-dagger anchor.

    "prototype"     False        True           False       Linear(pooled -> d), cosine to prototypes.

    "slot_proto"    True         True           False       Slot-attention pool + cosine to prototypes.

    "full"          True         True           True        Full STAG: slot + prototype + temporal graph.


Task specialization is controlled by:

    - num_slots      : 1 for phase, N_c for tool/verb (only used when use_slot=True)

    - window, depth  : temporal graph window W and iterations K (only when use_temporal=True)


Pipeline (mode="full"):


    pooled  [T, 1536]           spatial [T, 49, 1536]

        │                            │

        │              ┌─────────────┘

        │              ▼

        │        Slot Pooling  ──> slots [T, K_s, d]

        │              │

        │              ▼

        │         Cosine Prototypes  ──> Y_0 [T, N_c]

        ▼              │

   Temporal Affinity ──> │

   Graph (K steps)     │

                       ▼

                    Y_K [T, N_c]

"""

import math

import torch

import torch.nn as nn

import torch.nn.functional as F



class SlotAttentionPool(nn.Module):

    """One-layer cross-attention: K_s learnable class queries over 49 spatial tokens."""


    def __init__(self, in_dim=1536, slot_dim=256, num_slots=1,

                 num_heads=4, dropout=0.0):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        self.queries = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)

        self.k_proj = nn.Linear(in_dim, slot_dim, bias=False)

        self.v_proj = nn.Linear(in_dim, slot_dim, bias=False)

        self.attn = nn.MultiheadAttention(

            embed_dim=slot_dim, num_heads=num_heads,

            dropout=dropout, batch_first=True,
        )

        self.norm = nn.LayerNorm(slot_dim)


    def forward(self, spatial):

        T = spatial.size(0)

        k = self.k_proj(spatial)

        v = self.v_proj(spatial)

        q = self.queries.unsqueeze(0).expand(T, -1, -1)

        out, _ = self.attn(q, k, v)
        return self.norm(out)



VALID_MODES = ("linear", "prototype", "slot_proto", "full")



def _flags_from_mode(mode):

    return {

        "linear":     (False, False, False),

        "prototype":  (False, True,  False),

        "slot_proto": (True,  True,  False),

        "full":       (True,  True,  True),

    }[mode]



class STAG(nn.Module):
    def __init__(
        self,

        in_dim=1536,

        num_classes=7,

        task_type="multi-class",     # "multi-class" | "multi-label"

        num_slots=1,                 # 1 for phase, N_c for tool/verb

        slot_dim=256,

        mode="full",                 # linear | prototype | slot_proto | full

        window=256,

        depth=8,

        temperature=20.0,

        sigma_init=1.5,

        alpha_init=0.5,

    ):
        super().__init__()

        if mode not in VALID_MODES:

            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

        assert task_type in ("multi-class", "multi-label")
        self.mode = mode

        self.task_type = task_type

        self.num_classes = num_classes
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        self.window = window

        self.depth = depth
        self.temperature = temperature


        use_slot, use_prototype, use_diffusion = _flags_from_mode(mode)
        self.use_slot = use_slot

        self.use_prototype = use_prototype
        self.use_diffusion = use_diffusion


        # ---- Frame feature ----

        if use_slot:

            self.slot_pool = SlotAttentionPool(
                in_dim=in_dim, slot_dim=slot_dim, num_slots=num_slots,
            )


        # ---- Unary ----

        if use_prototype:

            # Prototypes always live in slot_dim. If no slot pooling, project pooled -> slot_dim.

            if not use_slot:

                self.pool_to_proto = nn.Linear(in_dim, slot_dim)

            self.prototypes = nn.Parameter(torch.randn(num_classes, slot_dim) * 0.02)

        else:

            # Plain linear head (mode="linear"): 1536 -> N_c

            self.linear = nn.Linear(in_dim, num_classes)


        # ---- Diffusion params ----

        if use_diffusion:

            self.alpha_raw = nn.Parameter(

                torch.tensor(math.log(alpha_init / (1.0 - alpha_init)), dtype=torch.float32)
            )

            self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma_init), dtype=torch.float32))

            self.pool_proj = nn.Linear(in_dim, slot_dim)


    # ------------------------------------------------ Unary logits ---


    def _unary(self, pooled, spatial):

        """Returns [T, N_c] logits before diffusion."""

        if self.use_prototype:

            p_norm = F.normalize(self.prototypes, dim=-1)          # [N_c, d]


            if self.use_slot:

                slots = self.slot_pool(spatial)                    # [T, K_s, d]

                s_norm = F.normalize(slots, dim=-1)                # [T, K_s, d]

                if self.num_slots == 1:

                    logits = s_norm.squeeze(1) @ p_norm.t()        # [T, N_c]

                elif self.num_slots == self.num_classes:

                    logits = (s_norm * p_norm.unsqueeze(0)).sum(dim=-1)   # [T, N_c]

                else:

                    raise ValueError(

                        f"num_slots={self.num_slots} must be 1 or num_classes={self.num_classes}"
                    )

            else:

                f = self.pool_to_proto(pooled)                     # [T, d]

                f_norm = F.normalize(f, dim=-1)

                logits = f_norm @ p_norm.t()                       # [T, N_c]


            return self.temperature * logits

        else:

            # mode="linear"

            return self.linear(pooled)                             # [T, N_c]


    # ------------------------------------------------ Diffusion ---


    def _diffuse(self, Y0, pooled):

        phi = F.normalize(self.pool_proj(pooled), dim=-1)          # [T, d]

        T, _ = phi.shape

        cos = phi @ phi.t()

        d2 = torch.clamp(2.0 - 2.0 * cos, min=0.0)

        sigma = torch.clamp(self.log_sigma.exp(), min=1e-4)

        A = torch.exp(-d2 / (sigma ** 2))


        idx = torch.arange(T, device=phi.device)

        band = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.window

        A = A * band.to(A.dtype)


        deg = A.sum(dim=1) + 1e-8

        d_inv_sqrt = deg.rsqrt()

        S = A * d_inv_sqrt.unsqueeze(0) * d_inv_sqrt.unsqueeze(1)


        alpha = torch.sigmoid(self.alpha_raw)

        Y = Y0

        for _ in range(self.depth):

            Y = alpha * (S @ Y) + (1.0 - alpha) * Y0

        return Y


    # ------------------------------------------------ Forward ---


    def forward(self, pooled, spatial):

        Y0 = self._unary(pooled, spatial)

        if not self.use_diffusion:

            return Y0

        return self._diffuse(Y0, pooled)


    # ------------------------------------------------ Diagnostics ---


    def diffusion_params(self):

        if not self.use_diffusion:

            return {"alpha": 0.0, "sigma": 0.0}

        with torch.no_grad():

            return {

                "alpha": float(torch.sigmoid(self.alpha_raw).item()),

                "sigma": float(self.log_sigma.exp().item()),

            }

