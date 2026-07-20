"""
spsr_planner.py — CoopCalib-TP SPSR Virtual Planner
=====================================================
Implements the Safe Planning Success Rate (SPSR) deployment proxy metric.

SPSR answers the question:
    "For what fraction of agents does at least one predicted hypothesis
     allow a downstream planner to reach the goal safely?"

Three-gate algorithm (applied per agent):
    Gate 1 — Social Clearance : no waypoint within r_ped of any neighbour traj
    Gate 2 — Goal Alignment   : endpoint within goal_radius of GT final position
    Gate 3 — Path Safety      : no waypoint within planner_radius of any obstacle
                                 (obstacles = neighbour GT trajectories at eval time)

SPSR = fraction of agents with at least one hypothesis passing all three gates.

Relationship to eval_suite.py
------------------------------
    eval_suite.compute_spsr() — fast scalar, single-agent mode (no neighbours)
    spsr_planner.SPSRPlanner  — full multi-agent deployment simulation

Both are valid. The planner version is used from V1 onward when neighbour
prediction tensors are available. The eval_suite version produced the V0
baseline numbers (baseline.json).

Hardware target: RTX 3050 Ti (4 GB VRAM), Windows 11, Python 3.11
Dependencies   : torch, numpy (no additional installs needed)

Usage
-----
    from metrics.spsr_planner import SPSRPlanner

    planner = SPSRPlanner(
        planner_radius = 0.5,   # metres — collision radius for path safety
        goal_radius    = 1.0,   # metres — goal acceptance radius
        r_ped          = 0.3,   # metres — social clearance radius
    )

    # Single-agent mode (V0 compatible — matches eval_suite.compute_spsr)
    spsr = planner.compute(
        pred_trajs = pred_trajs,   # (B, K, T, 2)
        gt         = gt,           # (B, T, 2)
    )

    # Multi-agent mode (V1+)
    spsr = planner.compute(
        pred_trajs = pred_trajs,   # (B, K, T, 2)
        gt         = gt,           # (B, T, 2)
        nei_gt     = nei_gt,       # (B, N, T, 2) — neighbour GT as obstacles
        nei_preds  = nei_preds,    # (B, N, K, T, 2) — neighbour predictions
    )
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# SPSRPlanner
# ---------------------------------------------------------------------------

class SPSRPlanner:
    """
    Virtual planner for Safe Planning Success Rate (SPSR).

    Parameters
    ----------
    planner_radius : float
        Collision radius for path safety gate (Gate 3). A waypoint is
        unsafe if it falls within this distance of any obstacle position.
        Default: 0.5m (half a body-width clearance for a robot base).

    goal_radius : float
        Acceptance radius for goal alignment gate (Gate 2). A hypothesis
        "reaches the goal" if its final predicted position is within this
        distance of the GT final position.
        Default: 1.0m (generous — accounts for prediction horizon drift).

    r_ped : float
        Social clearance radius for Gate 1. A hypothesis fails social
        clearance if any of its waypoints is within r_ped of any neighbour
        trajectory waypoint (at the same timestep).
        Default: 0.3m (standard pedestrian body radius).

    use_gpu : bool
        Move tensors to CUDA if available. Default: True.
    """

    def __init__(
        self,
        planner_radius: float = 0.5,
        goal_radius:    float = 1.0,
        r_ped:          float = 0.3,
        use_gpu:        bool  = True,
    ):
        self.planner_radius = planner_radius
        self.goal_radius    = goal_radius
        self.r_ped          = r_ped
        self.device = (
            torch.device("cuda")
            if use_gpu and torch.cuda.is_available()
            else torch.device("cpu")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        pred_trajs: torch.Tensor,                    # (B, K, T, 2)
        gt:         torch.Tensor,                    # (B, T, 2)
        nei_gt:     Optional[torch.Tensor] = None,   # (B, N, T, 2)
        nei_preds:  Optional[torch.Tensor] = None,   # (B, N, K, T, 2)
    ) -> float:
        """
        Compute SPSR scalar in [0, 1].

        Parameters
        ----------
        pred_trajs : (B, K, T, 2) — K predicted hypotheses per agent
        gt         : (B, T, 2)   — ground-truth future trajectory
        nei_gt     : (B, N, T, 2) optional — neighbour GT as static obstacles
        nei_preds  : (B, N, K, T, 2) optional — neighbour predictions (Gate 1)

        Returns
        -------
        float — fraction of agents with at least one viable hypothesis
        """
        pred_trajs = pred_trajs.to(self.device)
        gt         = gt.to(self.device)
        if nei_gt    is not None: nei_gt    = nei_gt.to(self.device)
        if nei_preds is not None: nei_preds = nei_preds.to(self.device)

        # viable[b, k] = True if hypothesis k passes all active gates for agent b
        viable = self._gate1_social_clearance(pred_trajs, nei_preds)  # (B, K) bool
        viable = viable & self._gate2_goal_alignment(pred_trajs, gt)   # (B, K) bool
        viable = viable & self._gate3_path_safety(pred_trajs, nei_gt)  # (B, K) bool

        # SPSR = fraction of agents with at least one viable hypothesis
        has_viable = viable.any(dim=-1).float()   # (B,)
        return has_viable.mean().item()

    def compute_detailed(
        self,
        pred_trajs: torch.Tensor,
        gt:         torch.Tensor,
        nei_gt:     Optional[torch.Tensor] = None,
        nei_preds:  Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Returns SPSR broken down by gate for diagnostics and ablation.

        Returns dict with keys:
            spsr            — full three-gate SPSR
            gate1_pass_rate — fraction of (agent, hyp) pairs passing Gate 1
            gate2_pass_rate — fraction of (agent, hyp) pairs passing Gate 2
            gate3_pass_rate — fraction of (agent, hyp) pairs passing Gate 3
            spsr_g1g2       — SPSR with only Gates 1+2 (no path safety)
            spsr_g2only     — SPSR with only Gate 2 (matches eval_suite V0)
            viable_per_agent_mean — mean number of viable hypotheses per agent
        """
        pred_trajs = pred_trajs.to(self.device)
        gt         = gt.to(self.device)
        if nei_gt    is not None: nei_gt    = nei_gt.to(self.device)
        if nei_preds is not None: nei_preds = nei_preds.to(self.device)

        g1 = self._gate1_social_clearance(pred_trajs, nei_preds)
        g2 = self._gate2_goal_alignment(pred_trajs, gt)
        g3 = self._gate3_path_safety(pred_trajs, nei_gt)

        B, K = g1.shape
        total = float(B * K)

        viable_full = g1 & g2 & g3
        viable_g1g2 = g1 & g2

        return {
            "spsr":                    viable_full.any(dim=-1).float().mean().item(),
            "gate1_pass_rate":         g1.float().mean().item(),
            "gate2_pass_rate":         g2.float().mean().item(),
            "gate3_pass_rate":         g3.float().mean().item(),
            "spsr_g1g2":               viable_g1g2.any(dim=-1).float().mean().item(),
            "spsr_g2only":             g2.any(dim=-1).float().mean().item(),
            "viable_per_agent_mean":   viable_full.float().sum(dim=-1).mean().item(),
        }

    # ------------------------------------------------------------------
    # Gate implementations
    # ------------------------------------------------------------------

    def _gate1_social_clearance(
        self,
        pred_trajs: torch.Tensor,                    # (B, K, T, 2)
        nei_preds:  Optional[torch.Tensor],          # (B, N, K, T, 2) or None
    ) -> torch.Tensor:                               # (B, K) bool
        """
        Gate 1: Social Clearance.

        A hypothesis (b, k) passes if NO waypoint of pred_trajs[b, k, t]
        falls within r_ped of ANY neighbour hypothesis waypoint
        nei_preds[b, n, k', t] at the same timestep t.

        When nei_preds is None (single-agent mode), all hypotheses pass.
        """
        B, K, T, _ = pred_trajs.shape

        if nei_preds is None:
            return torch.ones(B, K, dtype=torch.bool, device=self.device)

        N = nei_preds.shape[1]

        # focal: (B, K, 1, 1, T, 2) — broadcast over N neighbours and K' hyps
        focal = pred_trajs.unsqueeze(2).unsqueeze(3)           # (B, K, 1, 1, T, 2)

        # neigh: (B, 1, N, K, T, 2) — broadcast over K focal hypotheses
        neigh = nei_preds.unsqueeze(1)                          # (B, 1, N, K, T, 2)

        # Pairwise distances: (B, K, N, K', T)
        dist = (focal - neigh).norm(dim=-1)                    # (B, K, N, K', T)

        # Min distance over neighbours, their hypotheses, and timesteps
        min_dist = dist.amin(dim=(2, 3, 4))                    # (B, K)

        return min_dist >= self.r_ped                          # (B, K) bool

    def _gate2_goal_alignment(
        self,
        pred_trajs: torch.Tensor,   # (B, K, T, 2)
        gt:         torch.Tensor,   # (B, T, 2)
    ) -> torch.Tensor:              # (B, K) bool
        """
        Gate 2: Goal Alignment.

        A hypothesis (b, k) passes if its final predicted position
        pred_trajs[b, k, -1] is within goal_radius of the GT final
        position gt[b, -1].
        """
        pred_end = pred_trajs[:, :, -1, :]                    # (B, K, 2)
        gt_end   = gt[:, -1, :].unsqueeze(1)                  # (B, 1, 2)

        dist_to_goal = (pred_end - gt_end).norm(dim=-1)       # (B, K)

        return dist_to_goal <= self.goal_radius                # (B, K) bool

    def _gate3_path_safety(
        self,
        pred_trajs: torch.Tensor,              # (B, K, T, 2)
        nei_gt:     Optional[torch.Tensor],    # (B, N, T, 2) or None
    ) -> torch.Tensor:                         # (B, K) bool
        """
        Gate 3: Path Safety.

        Uses neighbour GT trajectories as ground-truth obstacle positions.
        A hypothesis (b, k) passes if NO waypoint pred_trajs[b, k, t]
        falls within planner_radius of ANY neighbour GT position
        nei_gt[b, n, t] at the same timestep t.

        When nei_gt is None (single-agent mode), all hypotheses pass.
        """
        B, K, T, _ = pred_trajs.shape

        if nei_gt is None:
            return torch.ones(B, K, dtype=torch.bool, device=self.device)

        N = nei_gt.shape[1]

        # focal: (B, K, 1, T, 2) — broadcast over N neighbours
        focal = pred_trajs.unsqueeze(2)                        # (B, K, 1, T, 2)

        # obstacles: (B, 1, N, T, 2) — broadcast over K hypotheses
        obs   = nei_gt.unsqueeze(1)                            # (B, 1, N, T, 2)

        # Pairwise distances: (B, K, N, T)
        dist = (focal - obs).norm(dim=-1)                      # (B, K, N, T)

        # Min distance over neighbours and timesteps
        min_dist = dist.amin(dim=(2, 3))                       # (B, K)

        return min_dist >= self.planner_radius                 # (B, K) bool


# ---------------------------------------------------------------------------
# Convenience wrapper — matches eval_suite.compute_spsr signature
# ---------------------------------------------------------------------------

def compute_spsr_planner(
    samples:        np.ndarray,                      # (N, K, T, 2)
    gt:             np.ndarray,                      # (N, T, 2)
    planner_radius: float = 0.5,
    goal_radius:    float = 1.0,
) -> float:
    """
    Drop-in replacement for eval_suite.compute_spsr using the planner class.
    Accepts numpy arrays, returns scalar float.
    Single-agent mode — Gates 1 and 3 pass trivially (no neighbours).
    """
    planner = SPSRPlanner(planner_radius=planner_radius, goal_radius=goal_radius)
    pred_t  = torch.from_numpy(samples.astype(np.float32))
    gt_t    = torch.from_numpy(gt.astype(np.float32))
    return planner.compute(pred_t, gt_t)


# ---------------------------------------------------------------------------
# Integration Tests — run with: python spsr_planner.py
# ---------------------------------------------------------------------------

def _run_tests():
    import sys
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running integration tests on {device}\n{'='*60}")

    B, K, T, N = 8, 20, 12, 4
    passed  = 0
    n_tests = 0

    def check(name, condition, detail=""):
        nonlocal passed, n_tests
        n_tests += 1
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        if condition:
            passed += 1

    planner = SPSRPlanner(planner_radius=0.5, goal_radius=1.0, r_ped=0.3)

    # -----------------------------------------------------------------------
    # Test 1: Single-agent mode — SPSR == 1.0 when pred == GT
    print("\nTest 1: Perfect predictions → SPSR = 1.0")
    gt_perfect   = torch.randn(B, T, 2)
    pred_perfect = gt_perfect.unsqueeze(1).expand(B, K, T, 2).clone()
    spsr = planner.compute(pred_perfect, gt_perfect)
    check("SPSR = 1.0 for perfect preds", abs(spsr - 1.0) < 1e-6,
          f"got {spsr:.6f}")

    # -----------------------------------------------------------------------
    # Test 2: Single-agent mode — SPSR == 0.0 when all preds far from GT
    print("\nTest 2: All predictions far from GT → SPSR = 0.0")
    gt_zero    = torch.zeros(B, T, 2)
    pred_far   = torch.full((B, K, T, 2), fill_value=100.0)
    spsr_zero  = planner.compute(pred_far, gt_zero)
    check("SPSR = 0.0 for distant preds", abs(spsr_zero - 0.0) < 1e-6,
          f"got {spsr_zero:.6f}")

    # -----------------------------------------------------------------------
    # Test 3: Single-agent mode — SPSR in [0, 1] for random inputs
    print("\nTest 3: Random inputs → SPSR ∈ [0, 1]")
    pred_rand = torch.randn(B, K, T, 2)
    gt_rand   = torch.randn(B, T, 2)
    spsr_rand = planner.compute(pred_rand, gt_rand)
    check("SPSR in [0, 1]", 0.0 <= spsr_rand <= 1.0,
          f"got {spsr_rand:.4f}")

    # -----------------------------------------------------------------------
    # Test 4: Gate 2 only — compute_detailed spsr_g2only matches compute_spsr
    print("\nTest 4: spsr_g2only matches single-agent compute()")
    detail = planner.compute_detailed(pred_rand, gt_rand)
    spsr_g2 = detail["spsr_g2only"]
    check("spsr_g2only == compute() in single-agent mode",
          abs(spsr_g2 - spsr_rand) < 1e-6,
          f"g2only={spsr_g2:.4f} compute={spsr_rand:.4f}")

    # -----------------------------------------------------------------------
    # Test 5: Gate 1 blocks overlapping predictions
    print("\nTest 5: Gate 1 blocks all preds when agents fully overlap")
    # Focal agent and all neighbours at the exact same position → Gate 1 fails
    pred_overlap = torch.zeros(B, K, T, 2)
    nei_overlap  = torch.zeros(B, N, K, T, 2)   # neighbours at same location
    gt_overlap   = torch.zeros(B, T, 2)          # GT also at origin → Gate 2 passes
    spsr_overlap = planner.compute(pred_overlap, gt_overlap, nei_preds=nei_overlap)
    check("SPSR = 0.0 when all agents overlap (Gate 1 blocks)",
          abs(spsr_overlap - 0.0) < 1e-6,
          f"got {spsr_overlap:.4f}")

    # -----------------------------------------------------------------------
    # Test 6: Gate 3 blocks predictions near neighbour GT obstacles
    print("\nTest 6: Gate 3 blocks preds too close to neighbour GT")
    # All predicted waypoints at (0,0); neighbour GT also at (0,0) → Gate 3 fails
    pred_g3   = torch.zeros(B, K, T, 2)
    gt_g3     = torch.zeros(B, T, 2)             # GT at origin → Gate 2 passes
    nei_gt_g3 = torch.zeros(B, N, T, 2)          # neighbours at same location
    spsr_g3   = planner.compute(pred_g3, gt_g3, nei_gt=nei_gt_g3)
    check("SPSR = 0.0 when preds collide with neighbour GT (Gate 3 blocks)",
          abs(spsr_g3 - 0.0) < 1e-6,
          f"got {spsr_g3:.4f}")

    # -----------------------------------------------------------------------
    # Test 7: Multi-agent mode — well-separated agents → SPSR = 1.0
    print("\nTest 7: Well-separated agents → SPSR = 1.0 in multi-agent mode")
    pred_sep   = torch.zeros(B, K, T, 2)         # focal at (0, 0)
    gt_sep     = torch.zeros(B, T, 2)             # GT at (0, 0) → Gate 2 passes
    # Neighbours and their predictions 10m away → Gates 1 and 3 pass
    nei_gt_sep    = torch.full((B, N, T, 2),    fill_value=10.0)
    nei_preds_sep = torch.full((B, N, K, T, 2), fill_value=10.0)
    spsr_sep = planner.compute(
        pred_sep, gt_sep,
        nei_gt=nei_gt_sep, nei_preds=nei_preds_sep
    )
    check("SPSR = 1.0 for well-separated multi-agent scene",
          abs(spsr_sep - 1.0) < 1e-6,
          f"got {spsr_sep:.6f}")

    # -----------------------------------------------------------------------
    # Test 8: compute_detailed keys present and gate rates sum correctly
    print("\nTest 8: compute_detailed — all keys present, rates in [0, 1]")
    detail_full = planner.compute_detailed(
        pred_rand, gt_rand,
        nei_gt=nei_gt_sep[:B], nei_preds=nei_preds_sep[:B]
    )
    required_keys = {
        "spsr", "gate1_pass_rate", "gate2_pass_rate", "gate3_pass_rate",
        "spsr_g1g2", "spsr_g2only", "viable_per_agent_mean"
    }
    check("All required keys present",
          required_keys.issubset(detail_full.keys()),
          f"keys={list(detail_full.keys())}")
    all_in_range = all(
        0.0 <= v <= K + 1e-6
        for v in detail_full.values()
    )
    check("All rate values in valid range",
          all_in_range,
          f"{detail_full}")

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
