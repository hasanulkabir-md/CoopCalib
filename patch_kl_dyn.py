"""
Patch SocialVAE to add L_KL_dyn (dynamic KL prior scaled by crowd density).

Changes:
1. main.py  — add --lambda3 argument; pass it to model.loss()
2. social_vae.py — modify learn() to compute density-scaled prior std
                 — modify loss() to accept lambda3 and apply KL scaling
"""

import re

# ── 1. Patch main.py ──────────────────────────────────────────────────────────
main_path = "models/socialvae/main.py"
with open(main_path) as f:
    src = f.read()

# Add --lambda3 argument after --fpc-finetune line
old = 'parser.add_argument("--fpc-finetune", action="store_true", default=False)'
new = old + '\nparser.add_argument("--lambda3", type=float, default=0.0)'
assert old in src, "Could not find --fpc-finetune in main.py"
src = src.replace(old, new, 1)

# Pass lambda3 to model.loss() — find `loss = model.loss(*res)` and replace
old = "loss = model.loss(*res)"
new = "loss = model.loss(*res, lambda3=settings.lambda3)"
assert old in src, "Could not find model.loss(*res) in main.py"
src = src.replace(old, new, 1)

with open(main_path, "w") as f:
    f.write(src)
print("main.py patched OK")


# ── 2. Patch social_vae.py ────────────────────────────────────────────────────
vae_path = "models/socialvae/social_vae.py"
with open(vae_path) as f:
    src = f.read()

# ── 2a. Modify learn() to compute neighbor count and scale prior std ──────────
# Target the KL loop:
#     kl = []
#     for p, q, z in zip(P, Q, Z):
#         kl.append(q.log_prob(z) - p.log_prob(z))
old_kl = (
    "        kl = []\n"
    "        for p, q, z in zip(P, Q, Z):\n"
    "            kl.append(q.log_prob(z) - p.log_prob(z))\n"
    "        kl = torch.stack(kl)"
)
new_kl = (
    "        # L_KL_dyn: compute density factor from neighbour count\n"
    "        # neighbor shape: (ob_len, N, Nn, feat) or (ob_len, N, Nn)\n"
    "        if neighbor is not None and neighbor.dim() >= 3:\n"
    "            # count valid neighbours (not sentinel 1e9) at last obs step\n"
    "            nb = neighbor[-1]  # (N, Nn, feat) or (N, Nn)\n"
    "            if nb.dim() == 3:\n"
    "                valid = (nb.abs().sum(-1) < 1e8).float()  # (N, Nn)\n"
    "            else:\n"
    "                valid = (nb.abs() < 1e8).float()\n"
    "            n_neighbors = valid.sum(-1).clamp(min=0)  # (N,)\n"
    "            # density factor: 1 + log(1 + n_neighbors), shape (N,1)\n"
    "            density_scale = (1.0 + torch.log1p(n_neighbors)).unsqueeze(-1)\n"
    "        else:\n"
    "            density_scale = None\n"
    "        kl = []\n"
    "        for p, q, z in zip(P, Q, Z):\n"
    "            if density_scale is not None:\n"
    "                # widen prior by density factor -> less KL penalty in crowds\n"
    "                p_dyn = torch.distributions.Normal(\n"
    "                    p.loc, p.scale * density_scale)\n"
    "                kl.append(q.log_prob(z) - p_dyn.log_prob(z))\n"
    "            else:\n"
    "                kl.append(q.log_prob(z) - p.log_prob(z))\n"
    "        kl = torch.stack(kl)"
)
assert old_kl in src, "Could not find KL loop in social_vae.py"
src = src.replace(old_kl, new_kl, 1)

# ── 2b. Modify loss() to accept lambda3 and scale KL term ────────────────────
old_loss = (
    "    def loss(self, err, kl):\n"
    "        rec = err.mean()\n"
    "        kl = kl.mean()\n"
    "\n"
    "        return {\n"
    '            "loss": kl+rec,\n'
    '            "rec": rec,\n'
    '            "kl": kl\n'
    "        }"
)
new_loss = (
    "    def loss(self, err, kl, lambda3=0.0):\n"
    "        rec = err.mean()\n"
    "        kl = kl.mean()\n"
    "        # L_KL_dyn: lambda3 scales the KL term (0 = standard ELBO)\n"
    "        kl_weight = 1.0 + lambda3\n"
    "\n"
    "        return {\n"
    '            "loss": kl_weight*kl + rec,\n'
    '            "rec": rec,\n'
    '            "kl": kl\n'
    "        }"
)
assert old_loss in src, "Could not find loss() in social_vae.py"
src = src.replace(old_loss, new_loss, 1)

with open(vae_path, "w") as f:
    f.write(src)
print("social_vae.py patched OK")

print("\nAll patches applied. Test with:")
print("  python main.py --config config/eth.py --train data/eth/train --test data/eth/test --device cuda --lambda3 0.1")
