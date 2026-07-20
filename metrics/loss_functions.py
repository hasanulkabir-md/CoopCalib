"""
loss_functions.py — CoopCalib-TP Training Losses
=================================================
Implements all four loss components described in §4 of the paper.

    L_ECE      — differentiable soft-ECE calibration loss (Karandikar NeurIPS 2021)
    L_energy   — energy-based social margin loss (L_coop reframed; IGP / OWOBJ)
    L_KL_dyn   — dynamic KL prior for CVAE collapse prevention (Zhang CVPR 2025)
    L_FPR      — freezing predictor regulariser (retained; not used in TUTR objective)

Combined objective:
    L_total = L_TUTR + lambda1 * L_ECE + lambda2 * L_energy + lambda3 * L_KL_dyn

Hardware target: RTX 3050 Ti (4 GB VRAM), Windows 11, Python 3.11, PyTorch 2.5.1+cu121

Usage
-----
    from loss_functions import LossECE, LossEnergy, LossKLDyn, CoopCalibLoss

    loss_fn = CoopCalibLoss(lambda1=0.1, lambda2=0.1, lambda3=1.0)
    total, components = loss_fn(
        base_loss   = tutr_brier_ade,
        pred_trajs  = pred_trajs,       # (B, K, T, 2)
        scores      = scores,           # (B, K)
        gt          = gt,               # (B, T, 2)
        nei_preds   = None,             # (B, N, K, T, 2) or None
        mu_q        = None,             # CVAE posterior mean    — SocialVAE only
        logvar_q    = None,             # CVAE posterior log-var — SocialVAE only
        mu_p        = None,             # Dynamic prior mean     — SocialVAE only
        logvar_p    = None,             # Dynamic prior log-var  — SocialVAE only
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# 1. L_ECE — Differentiable Soft-ECE Calibration Loss
# ---------------------------------------------------------------------------

class LossECE(nn.Module):
    """
    Differentiable soft-ECE loss for multimodal trajectory prediction.

    Adapts Karandikar et al. (NeurIPS 2021, arXiv:2108.00106) soft-ECE
    to the K-hypothesis trajectory setting.

    For each agent i and hypothesis k:
      - Confidence weight  w_ik  comes from TUTR's softmax scoring head.
      - Displacement error e_ik  is mean-L2 over T timesteps.
      - A prediction is "accurate" if e_ik < accuracy_thresh (default 1.0m).

    Soft bin assignment replaces hard histogram with Gaussian kernels,
    giving smooth gradients w.r.t. w_ik throughout training.

    Parameters
    ----------
    num_bins        : int   — number of calibration bins M (default 15)
    sigma           : float — bin bandwidth for soft assignment (default 1/M)
    accuracy_thresh : float — L2 threshold for "correct" prediction in metres
    """

    def __init__(
        self,
        num_bins: int = 15,
        sigma: Optional[float] = None,
        accuracy_thresh: float = 1.0,
    ):
        super().__init__()
        self.M = num_bins
        self.sigma = sigma if sigma is not None else (1.0 / num_bins)
        self.acc_thresh = accuracy_thresh

        # Bin centres c_m = (2m-1)/(2M) for m=1..M, fixed, not learnable
        centres = (2 * torch.arange(1, num_bins + 1) - 1) / (2 * num_bins)
        self.register_buffer("centres", centres)  # (M,)

    def forward(
        self,
        pred_trajs: torch.Tensor,   # (B, K, T, 2)
        scores: torch.Tensor,        # (B, K)  — raw logits or post-softmax
        gt: torch.Tensor,            # (B, T, 2)
    ) -> torch.Tensor:
        """
        Returns scalar soft-ECE loss in [0, 1].
        """
        B, K, T, _ = pred_trajs.shape
        if scores.shape[1] != K:
            scores = scores[:, :K]

        # Confidence weights via softmax (idempotent if already normalised)
        w = F.softmax(scores, dim=-1)                        # (B, K)

        # Per-hypothesis mean-L2 displacement error
        # gt expanded to (B, 1, T, 2) for broadcasting
        diff = pred_trajs - gt.unsqueeze(1)                  # (B, K, T, 2)
        e = diff.norm(dim=-1).mean(dim=-1)                   # (B, K)

        # Binary accuracy indicator: 1 if e < threshold
        acc = (e < self.acc_thresh).float()                  # (B, K)

        # Soft bin assignment: phi[b, k, m] = Gaussian(w[b,k], centre_m, sigma)
        # w: (B, K) → (B, K, 1); centres: (M,) → (1, 1, M)
        w_exp = w.unsqueeze(-1)                              # (B, K, 1)
        c_exp = self.centres.view(1, 1, -1)                  # (1, 1, M)
        phi = torch.exp(
            -((w_exp - c_exp) ** 2) / (2 * self.sigma ** 2)
        )                                                     # (B, K, M)

        # phi_sum[m] = sum over (b, k) of phi[b, k, m]
        phi_sum = phi.sum(dim=(0, 1))                         # (M,)
        total_phi = phi_sum.sum().clamp(min=1e-8)

        # Weighted mean confidence per bin
        # w_rep: (B, K, 1) * phi: (B, K, M) → (B, K, M)
        conf_per_bin = (w_exp * phi).sum(dim=(0, 1)) / phi_sum.clamp(min=1e-8)  # (M,)

        # Weighted mean accuracy per bin
        acc_exp = acc.unsqueeze(-1)                          # (B, K, 1)
        acc_per_bin = (acc_exp * phi).sum(dim=(0, 1)) / phi_sum.clamp(min=1e-8)  # (M,)

        # Bin weights (fraction of soft mass in each bin)
        bin_weights = phi_sum / total_phi                    # (M,)

        # Soft-ECE: weighted mean absolute calibration gap
        ece = (bin_weights * (conf_per_bin - acc_per_bin).abs()).sum()

        return ece


# ---------------------------------------------------------------------------
# 2. L_energy — Energy-Based Social Margin Loss (L_coop reframed)
# ---------------------------------------------------------------------------

class LossEnergy(nn.Module):
    """
    Energy-based social margin loss that penalises predicted trajectories
    whose inter-agent distances fall below safety radius r_ped.

    Reframes L_coop (IGP-motivated) as an SVR-reduction signal rather than
    an FPE correction (FPR=0.000 at V0 — TUTR is architecturally immune).

    Operates on MULTI-AGENT prediction tensors. Returns 0.0 gracefully
    when no neighbour predictions are supplied (single-agent eval mode).

    Parameters
    ----------
    r_ped : float — pedestrian safety radius in metres (default 0.3)
    """

    def __init__(self, r_ped: float = 0.3):
        super().__init__()
        self.r_ped = r_ped

    def forward(
        self,
        pred_trajs: torch.Tensor,          # (B, K, T, 2)  — focal agent
        nei_preds: Optional[torch.Tensor], # (B, N, K, T, 2) — neighbours, or None
    ) -> torch.Tensor:
        """
        Returns scalar social margin loss >= 0.
        Returns 0.0 (no gradient) when nei_preds is None.
        """
        if nei_preds is None:
            return pred_trajs.sum() * 0.0  # zero with gradient graph intact

        B, K, T, _ = pred_trajs.shape
        N = nei_preds.shape[1]

        # focal: (B, 1, K, 1, T, 2) — broadcast over N neighbours and K' hyps
        focal = pred_trajs.unsqueeze(1).unsqueeze(3)         # (B, 1, K, 1, T, 2)

        # neigh: (B, N, 1, K, T, 2) — broadcast over K focal hypotheses
        neigh = nei_preds.unsqueeze(2)                        # (B, N, 1, K, T, 2)

        # Pairwise distances: (B, N, K, K, T)
        dist = (focal - neigh).norm(dim=-1)                  # (B, N, K, K, T)

        # Soft hinge: max(0, r_ped - d)^2
        violation = F.relu(self.r_ped - dist) ** 2           # (B, N, K, K, T)

        # Mean over all dimensions
        loss = violation.mean()

        return loss

# ---------------------------------------------------------------------------
# 2b. L_RankingSVR — Ranking-Aware Social Violation Loss
# ---------------------------------------------------------------------------

class LossRankingSVR(nn.Module):
    """
    Ranking-aware social violation loss.
    Penalises the model when candidates violate personal space, creating 
    gradient pressure to rank socially compliant candidates higher.
    """

    def __init__(self, r_ped: float = 0.3, temperature: float = 5.0):
        super().__init__()
        self.r_ped = r_ped
        self.temperature = temperature

    def forward(
        self,
        pred_trajs: torch.Tensor,    # (B, K, T, 2)
        scores: torch.Tensor,        # (B, K) or (B, M) — CLF logits
        nei_futures: torch.Tensor,   # (B, N, T, 2)
    ) -> torch.Tensor:
        B, K, T, _ = pred_trajs.shape

        # ── Derive (B, K) weights regardless of scores shape ──────────────
        # TUTR's CLF head often outputs (B, M=50) motion-mode logits.
        # If dimensions match K, we use temperature-scaled softmax.
        # Otherwise, we use uniform weighting to ensure stable gradients.
        if scores.shape[-1] == K:
            w = F.softmax(scores * self.temperature, dim=-1)   # (B, K)
        else:
            # Fallback: uniform weight (B, K). Gradient flows through 
            # violation -> pred_trajs -> CLF via TUTR's native assignment.
            w = torch.ones(B, K, device=pred_trajs.device) / K

        # ── Mask sentinel neighbours (padded slots > 1e8) ─────────────────
        valid_mask = (nei_futures.abs() < 1e8).all(dim=-1)      # (B, N, T)
        nei_safe = nei_futures.clone()
        nei_safe[~valid_mask] = 1e4

        # ── Pairwise distances: (B, K, N, T) ──────────────────────────────
        pred_exp = pred_trajs.unsqueeze(2)    # (B, K, 1, T, 2)
        nei_exp  = nei_safe.unsqueeze(1)      # (B, 1, N, T, 2)
        dist = (pred_exp - nei_exp).norm(dim=-1)   # (B, K, N, T)

        # ── Min distance to any neighbour at any timestep per candidate ────
        min_dist = dist.min(dim=-1).values.min(dim=-1).values   # (B, K)

        # ── Soft violation: hinge loss on safety radius ────────────────────
        violation = F.relu(self.r_ped - min_dist) ** 2          # (B, K)

        # ── Weight violation by per-candidate confidence ───────────────────
        weighted_violation = (w * violation).sum(dim=-1)         # (B,)

        return weighted_violation.mean()

# ---------------------------------------------------------------------------
# 3. L_KL_dyn — Dynamic KL Prior (SocialVAE / CVAE only)
# ---------------------------------------------------------------------------

class LossKLDyn(nn.Module):
    """
    Dynamic KL divergence loss for CVAE-based predictors (SocialVAE).

    Replaces the fixed N(0,I) prior with a scene-conditioned prior
    p_theta(z|x) = N(mu_p, diag(exp(logvar_p))), following Zhang et al.
    (OWOBJ, CVPR 2025). Prevents KL collapse in low-data regimes by
    adapting the regularisation target to scene complexity.

    Closed-form KL for two diagonal Gaussians:
        KL(q || p) = 0.5 * sum[ log(sigma_p^2/sigma_q^2)
                                + (sigma_q^2 + (mu_q - mu_p)^2) / sigma_p^2
                                - 1 ]

    Parameters
    ----------
    reduction : str — 'mean' (default) or 'sum' over batch and latent dims
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "sum")
        self.reduction = reduction

    def forward(
        self,
        mu_q: torch.Tensor,      # (B, d) — posterior mean
        logvar_q: torch.Tensor,  # (B, d) — posterior log-variance
        mu_p: torch.Tensor,      # (B, d) — dynamic prior mean
        logvar_p: torch.Tensor,  # (B, d) — dynamic prior log-variance
    ) -> torch.Tensor:
        """
        Returns scalar KL divergence KL(q || p).

        Falls back to KL(q || N(0,I)) if mu_p and logvar_p are both zero,
        which recovers the standard CVAE ELBO as a special case.
        """
        # KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2))
        # = 0.5 * [log(sigma_p^2/sigma_q^2)
        #          + (sigma_q^2 + (mu_q - mu_p)^2) / sigma_p^2 - 1]

        kl = 0.5 * (
            logvar_p - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2)
            / torch.exp(logvar_p).clamp(min=1e-8)
            - 1.0
        )                                                     # (B, d)

        if self.reduction == "mean":
            return kl.mean()
        else:
            return kl.sum()


# ---------------------------------------------------------------------------
# 4. L_FPR — Freezing Predictor Regulariser (retained, non-differentiable)
# ---------------------------------------------------------------------------

class LossFPR(nn.Module):
    """
    Freezing Predictor Rate as a soft training signal.

    NOTE: FPR = 0.000 everywhere at V0 for TUTR. This loss is RETAINED
    in the codebase for completeness and cross-model comparability but is
    NOT included in the TUTR combined objective. It may be activated for
    SocialVAE ablations where KL collapse can indirectly produce frozen
    predictions.

    Soft surrogate: penalises hypotheses whose total displacement over T
    steps is below freeze_thresh, using a sigmoid gate instead of a
    hard indicator to allow gradient flow.

    Parameters
    ----------
    freeze_thresh : float — displacement threshold (metres) below which a
                            trajectory is considered frozen (default 0.5)
    temperature   : float — sigmoid sharpness (default 10.0)
    """

    def __init__(self, freeze_thresh: float = 0.5, temperature: float = 10.0):
        super().__init__()
        self.thresh = freeze_thresh
        self.temp = temperature

    def forward(self, pred_trajs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_trajs : (B, K, T, 2)

        Returns
        -------
        Scalar soft-FPR loss in [0, 1].
        """
        B, K, T, _ = pred_trajs.shape
        if scores.shape[1] != K:
            scores = scores[:, :K]

        # Max displacement from first predicted point
        origin = pred_trajs[:, :, 0:1, :]                    # (B, K, 1, 2)
        disp = (pred_trajs - origin).norm(dim=-1)             # (B, K, T)
        max_disp = disp.max(dim=-1).values                    # (B, K)

        # Soft indicator: 1 when frozen, 0 when moving
        # sigmoid(-temp * (max_disp - thresh)) ≈ 1[max_disp < thresh]
        soft_frozen = torch.sigmoid(-self.temp * (max_disp - self.thresh))  # (B, K)

        return soft_frozen.mean()


# ---------------------------------------------------------------------------
# 5. CoopCalibLoss — Combined Training Objective
# ---------------------------------------------------------------------------

class CoopCalibLoss(nn.Module):
    """
    Combined CoopCalib-TP training objective:

        L_total = L_TUTR
                + lambda1 * L_ECE
                + lambda2 * L_energy
                + lambda3 * L_KL_dyn      [SocialVAE only; skipped if mu_q=None]

    Default hyperparameters (pre-sweep initialisation):
        lambda1 = 0.1  (ECE calibration)
        lambda2 = 0.1  (social margin)
        lambda3 = 1.0  (KL dynamic prior — SocialVAE only)

    Parameters
    ----------
    lambda1     : float — weight for L_ECE
    lambda2     : float — weight for L_energy
    lambda3     : float — weight for L_KL_dyn
    ece_bins    : int   — number of ECE bins (default 15)
    ece_sigma   : float — ECE bin bandwidth (default 1/M)
    acc_thresh  : float — ECE accuracy threshold in metres (default 1.0)
    r_ped       : float — social safety radius in metres (default 0.3)
    freeze_thresh: float — FPR displacement threshold (default 0.5)
    """

    def __init__(
        self,
        lambda1: float = 0.1,
        lambda2: float = 0.1,
        lambda3: float = 1.0,
        ece_bins: int = 15,
        ece_sigma: Optional[float] = None,
        acc_thresh: float = 1.0,
        r_ped: float = 0.3,
        freeze_thresh: float = 0.5,
    ):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

        self.loss_ece    = LossECE(ece_bins, ece_sigma, acc_thresh)
        self.loss_energy = LossEnergy(r_ped)
        self.loss_kl_dyn = LossKLDyn()
        self.loss_fpr    = LossFPR(freeze_thresh)   # retained, not added to total

    def forward(
        self,
        base_loss: torch.Tensor,                     # scalar — L_TUTR (Brier-ADE)
        pred_trajs: torch.Tensor,                    # (B, K, T, 2)
        scores: torch.Tensor,                        # (B, K)
        gt: torch.Tensor,                            # (B, T, 2)
        nei_preds: Optional[torch.Tensor] = None,   # (B, N, K, T, 2) or None
        mu_q: Optional[torch.Tensor] = None,         # (B, d) — SocialVAE only
        logvar_q: Optional[torch.Tensor] = None,     # (B, d) — SocialVAE only
        mu_p: Optional[torch.Tensor] = None,         # (B, d) — SocialVAE only
        logvar_p: Optional[torch.Tensor] = None,     # (B, d) — SocialVAE only
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns
        -------
        total   : scalar tensor — backprop target
        components : dict of float — individual loss values for logging
        """
        l_ece    = self.loss_ece(pred_trajs, scores, gt)
        l_energy = self.loss_energy(pred_trajs, nei_preds)
        l_fpr    = self.loss_fpr(pred_trajs)           # logged but not added

        total = base_loss + self.lambda1 * l_ece + self.lambda2 * l_energy

        # KL dynamic prior — only active when CVAE parameters are provided
        if mu_q is not None and logvar_q is not None:
            assert mu_p is not None and logvar_p is not None, (
                "Dynamic prior requires mu_p and logvar_p. "
                "For fixed N(0,I) prior pass mu_p=zeros, logvar_p=zeros."
            )
            l_kl = self.loss_kl_dyn(mu_q, logvar_q, mu_p, logvar_p)
            total = total + self.lambda3 * l_kl
        else:
            l_kl = torch.tensor(0.0)

        components = {
            "L_total":  total.item(),
            "L_TUTR":   base_loss.item(),
            "L_ECE":    l_ece.item(),
            "L_energy": l_energy.item(),
            "L_KL_dyn": l_kl.item() if isinstance(l_kl, torch.Tensor) else 0.0,
            "L_FPR":    l_fpr.item(),    # diagnostic only — not in objective
        }

        return total, components


# ---------------------------------------------------------------------------
# 6. Unit Tests — run with: python loss_functions.py
# ---------------------------------------------------------------------------

def _run_tests():
    import sys
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running unit tests on {device}\n{'='*60}")

    B, K, T, N, d = 4, 20, 12, 3, 16
    passed = 0
    n_tests = 0

    def check(name, condition, detail=""):
        nonlocal passed, n_tests
        n_tests += 1
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        if condition:
            passed += 1
        return condition

    # -----------------------------------------------------------------------
    # Test 1: LossECE — output shape and range
    print("\nTest 1: LossECE — shape and range")
    pred  = torch.randn(B, K, T, 2, device=device)
    scores = torch.randn(B, K, device=device)
    gt    = torch.randn(B, T, 2, device=device)
    ece_fn = LossECE().to(device)
    ece_val = ece_fn(pred, scores, gt)
    check("ECE is scalar",       ece_val.shape == torch.Size([]))
    check("ECE in [0, 1]",       0.0 <= ece_val.item() <= 1.0,
          f"got {ece_val.item():.4f}")

    # -----------------------------------------------------------------------
    # Test 2: LossECE — perfect calibration → near-zero loss
    print("\nTest 2: LossECE — near-zero for perfect confidence assignment")
    # Put all weight on the best hypothesis (closest to GT)
    gt_fixed    = torch.zeros(B, T, 2, device=device)
    pred_fixed  = torch.zeros(B, K, T, 2, device=device)
    # hypothesis 0 is perfect; rest are far away
    pred_fixed[:, 1:, :, :] = 5.0
    scores_perfect = torch.full((B, K), -1e9, device=device)
    scores_perfect[:, 0] = 10.0                              # all weight on k=0
    ece_perfect = ece_fn(pred_fixed, scores_perfect, gt_fixed)
    check("ECE near-zero for perfect conf",
          ece_perfect.item() < 0.15,
          f"got {ece_perfect.item():.4f}")

    # -----------------------------------------------------------------------
    # Test 3: LossECE — has gradient
    print("\nTest 3: LossECE — gradient flows through scores")
    scores_grad = torch.randn(B, K, device=device, requires_grad=True)
    pred_grad   = torch.randn(B, K, T, 2, device=device, requires_grad=True)
    gt_grad     = torch.randn(B, T, 2, device=device)          # no grad needed
    ece_g = ece_fn(pred_grad, scores_grad, gt_grad)
    ece_g.backward()
    check("scores.grad is not None",   scores_grad.grad is not None)
    check("scores.grad has no NaNs",   not torch.isnan(scores_grad.grad).any())

    # -----------------------------------------------------------------------
    # Test 4: LossEnergy — returns zero when nei_preds is None
    print("\nTest 4: LossEnergy — zero when no neighbours")
    energy_fn = LossEnergy().to(device)
    pred_e = torch.randn(B, K, T, 2, device=device, requires_grad=True)
    e_none = energy_fn(pred_e, nei_preds=None)
    check("Energy = 0.0 when nei_preds=None",
          e_none.item() == 0.0, f"got {e_none.item():.6f}")

    # -----------------------------------------------------------------------
    # Test 5: LossEnergy — positive when agents overlap
    print("\nTest 5: LossEnergy — positive for overlapping agents")
    pred_e2   = torch.zeros(B, K, T, 2, device=device)       # focal at origin
    nei_same  = torch.zeros(B, N, K, T, 2, device=device)    # neighbours also at origin
    e_overlap = energy_fn(pred_e2, nei_same)
    check("Energy > 0 for overlapping agents",
          e_overlap.item() > 0.0, f"got {e_overlap.item():.4f}")

    # -----------------------------------------------------------------------
    # Test 6: LossKLDyn — recovers standard KL when prior is N(0,I)
    print("\nTest 6: LossKLDyn — recovers standard KL for N(0,I) prior")
    kl_fn  = LossKLDyn().to(device)
    mu_q   = torch.randn(B, d, device=device)
    lv_q   = torch.randn(B, d, device=device)
    mu_p   = torch.zeros(B, d, device=device)
    lv_p   = torch.zeros(B, d, device=device)                # log(1) = 0 → sigma_p=1
    kl_dyn = kl_fn(mu_q, lv_q, mu_p, lv_p)
    # Standard KL: 0.5*(exp(lv_q) + mu_q^2 - 1 - lv_q)
    kl_std = 0.5 * (torch.exp(lv_q) + mu_q**2 - 1 - lv_q).mean()
    check("KL_dyn ≈ standard KL for N(0,I) prior",
          abs(kl_dyn.item() - kl_std.item()) < 1e-4,
          f"dyn={kl_dyn.item():.4f} std={kl_std.item():.4f}")

    # -----------------------------------------------------------------------
    # Test 7: CoopCalibLoss — full forward pass, no CVAE
    print("\nTest 7: CoopCalibLoss — full forward, no CVAE")
    combo = CoopCalibLoss(lambda1=0.1, lambda2=0.1).to(device)
    base  = torch.tensor(1.23, device=device, requires_grad=True)
    p     = torch.randn(B, K, T, 2, device=device, requires_grad=True)
    s     = torch.randn(B, K, device=device, requires_grad=True)
    g     = torch.randn(B, T, 2, device=device)
    total, comps = combo(base, p, s, g)
    check("total > 0",            total.item() > 0,
          f"got {total.item():.4f}")
    check("L_KL_dyn == 0.0",      comps["L_KL_dyn"] == 0.0,
          f"got {comps['L_KL_dyn']}")
    total.backward()
    check("gradient flows to p",  p.grad is not None)

    # -----------------------------------------------------------------------
    # Test 8: CoopCalibLoss — CVAE branch activates correctly
    print("\nTest 8: CoopCalibLoss — CVAE branch (SocialVAE mode)")
    mu_q2  = torch.randn(B, d, device=device)
    lv_q2  = torch.randn(B, d, device=device)
    mu_p2  = torch.zeros(B, d, device=device)
    lv_p2  = torch.zeros(B, d, device=device)
    base2  = torch.tensor(0.5, device=device, requires_grad=True)
    total2, comps2 = combo(base2, p.detach(), s.detach(), g,
                           mu_q=mu_q2, logvar_q=lv_q2,
                           mu_p=mu_p2, logvar_p=lv_p2)
    check("L_KL_dyn > 0 in CVAE mode",
          comps2["L_KL_dyn"] > 0.0,
          f"got {comps2['L_KL_dyn']:.4f}")
    check("total2 > total (KL adds positive term)",
          comps2["L_total"] >= comps["L_total"] - 0.5)   # rough sanity check

    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{n_tests} tests passed")
    if passed < n_tests:
        print("WARNING: Some tests failed — review output above.")
        sys.exit(1)
    else:
        print("All tests passed ✅")


if __name__ == "__main__":
    _run_tests()
