"""
Fix the density_scale broadcasting issue in social_vae.py learn().
Replaces the p_dyn construction with a safer version.
"""

vae_path = "models/socialvae/social_vae.py"
with open(vae_path) as f:
    src = f.read()

old = (
    "            if density_scale is not None:\n"
    "                # widen prior by density factor -> less KL penalty in crowds\n"
    "                p_dyn = torch.distributions.Normal(\n"
    "                    p.loc, p.scale * density_scale)\n"
    "                kl.append(q.log_prob(z) - p_dyn.log_prob(z))\n"
    "            else:\n"
    "                kl.append(q.log_prob(z) - p.log_prob(z))\n"
)

new = (
    "            if density_scale is not None:\n"
    "                # widen prior by density factor -> less KL penalty in crowds\n"
    "                # density_scale: (N,1), p.scale: (N, z_dim) -> broadcasts OK\n"
    "                new_scale = (p.scale * density_scale.to(p.scale)).clamp(min=1e-6)\n"
    "                p_dyn = torch.distributions.Normal(\n"
    "                    p.loc, new_scale, validate_args=False)\n"
    "                kl.append(q.log_prob(z) - p_dyn.log_prob(z))\n"
    "            else:\n"
    "                kl.append(q.log_prob(z) - p.log_prob(z))\n"
)

assert old in src, "Could not find p_dyn block — already patched differently?"
src = src.replace(old, new, 1)

with open(vae_path, "w") as f:
    f.write(src)
print("social_vae.py fix applied OK")
print("Now run: python main.py --config config/eth.py --train data/eth/train --test data/eth/test --device cuda --lambda3 0.1")
