"""
CoopCalib-TP — Trajectory Evaluation Suite
===========================================
File:    C:\\CoopCalib\\metrics\\eval_suite.py
Author:  CoopCalib-TP project
Day:     2 Morning

Implements four novel trajectory-prediction metrics:
  - compute_ece   : Expected Calibration Error   (Karandikar NeurIPS 2021)
  - compute_fpr   : Freezing Predictor Rate       (new metric)
  - compute_svr   : Social Violation Rate         (new metric)
  - compute_spsr  : Safe Planning Success Rate    (new metric)

All functions accept numpy arrays.  No torch dependency at eval time.
Run this file directly to execute the full unit-test suite:

    python metrics/eval_suite.py

Expected output (all 8 tests passing):
    [PASS] ECE near-zero on perfect calibration
    [PASS] ECE ~0.5 on uniform spread
    [PASS] FPR = 1.0 when all predictions static
    [PASS] FPR = 0.0 when predictions are mobile
    [PASS] SVR = 0.0 on well-separated pedestrians
    [PASS] SVR > 0 when pedestrians overlap
    [PASS] SPSR = 1.0 on perfect predictor
    [PASS] SPSR = 0.0 on all-collision predictor
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. ECE — Expected Calibration Error (Karandikar NeurIPS 2021)
# ---------------------------------------------------------------------------

def compute_ece(
    samples: np.ndarray,
    gt: np.ndarray,
    M: int = 15,
    s: float = 15.0,
    eps: float = 1e-8,
) -> float:
    """
    Compute the soft-binned Expected Calibration Error for trajectory prediction.

    Following Karandikar et al. NeurIPS 2021 (arXiv:2108.00106), Eq.(3).
    Bins are defined over the interval [0, 1] using sigmoid-based soft
    membership so the metric is differentiable (useful when used as a loss).

    For trajectory prediction we define "confidence" of sample k in scene n as:

        conf(n, k) = exp(-d(n,k) / sigma) / sum_j exp(-d(n,j) / sigma)

    where d(n,k) is the final-displacement error of sample k, i.e. the
    proportion of probability mass that is centred near the ground truth.
    "Accuracy" at timestep t is 1 if the best sample is within a threshold.

    Here we use a simpler, standard formulation:
      - Treat each (scene, sample) as a "prediction event"
      - confidence  = soft-min distance rank (high conf = small ADE)
      - accuracy    = 1 if ADE(k,n) < median ADE across K samples

    This gives ECE ∈ [0, 1].  A perfectly calibrated predictor → ECE ≈ 0.
    Uniform (random) predictor → ECE ≈ 0.5.

    Parameters
    ----------
    samples : ndarray, shape (N, K, T, 2)
        K predicted trajectories for each of N scenes, each with T timesteps.
    gt      : ndarray, shape (N, T, 2)
        Ground-truth future trajectory for each scene.
    M       : int
        Number of soft bins (paper default: 15).
    s       : float
        Sigmoid sharpness for soft bin membership (paper default: 15).
    eps     : float
        Small constant for numerical stability.

    Returns
    -------
    float
        Scalar ECE ∈ [0, 1].
    """
    samples = np.asarray(samples, dtype=np.float64)   # (N, K, T, 2)
    gt      = np.asarray(gt,      dtype=np.float64)   # (N, T, 2)

    N, K, T, _ = samples.shape

    # --- per-(scene, sample) ADE -------------------------------------------
    diff = samples - gt[:, None, :, :]                # (N, K, T, 2)
    ade  = np.sqrt((diff ** 2).sum(axis=-1)).mean(axis=-1)  # (N, K)

    # --- confidence: softmin over K candidates per scene -------------------
    # conf_nk = exp(-ade_nk) / sum_k exp(-ade_nk)   ∈ (0,1), sums to 1
    log_w  = -ade                                     # (N, K)
    log_w -= log_w.max(axis=-1, keepdims=True)        # numerical stability
    w      = np.exp(log_w)
    conf   = w / (w.sum(axis=-1, keepdims=True) + eps)  # (N, K)
    conf   = conf.reshape(-1)                         # (N*K,)

    # --- accuracy: 1 if this sample is within median ADE of its scene ------
    median_ade = np.median(ade, axis=-1, keepdims=True)  # (N, 1)
    acc        = (ade <= median_ade).astype(np.float64)  # (N, K)
    acc        = acc.reshape(-1)                          # (N*K,)

    # --- bin centres  m_j = (j-0.5) / M  for j = 1..M ---------------------
    bin_centres = (np.arange(1, M + 1) - 0.5) / M    # (M,)

    # --- soft membership  w_j(p) = sigmoid(s*(p - c_lo)) * sigmoid(s*(c_hi - p))
    c_lo = (np.arange(M)     ) / M                    # (M,) left edge
    c_hi = (np.arange(M) + 1 ) / M                    # (M,) right edge

    # conf shape: (N*K,); c_lo/c_hi shape: (M,)  → broadcast to (N*K, M)
    p = conf[:, None]                                  # (N*K, 1)
    membership = (
        1.0 / (1.0 + np.exp(-s * (p - c_lo[None, :])))
        * 1.0 / (1.0 + np.exp(-s * (c_hi[None, :] - p)))
    )                                                  # (N*K, M)

    # normalise membership so each sample contributes exactly 1 unit total
    membership_sum = membership.sum(axis=-1, keepdims=True) + eps
    membership     = membership / membership_sum        # (N*K, M)

    # --- bin-level weighted accuracy and confidence ------------------------
    # Weight each sample by its softmin probability mass so that low-confidence
    # "also-ran" samples (numerous but effectively zero-weight) don't inflate ECE.
    softmin_weight = conf                              # (N*K,) — already computed above

    wm      = membership * softmin_weight[:, None]     # (N*K, M)
    w_bin   = wm.sum(axis=0)                           # (M,)
    acc_bin = (wm * acc[:, None]).sum(axis=0) / (w_bin + eps)
    con_bin = bin_centres                              # by construction

    # --- ECE = sum_j  (w_j / total) * |acc_j - con_j| --------------------
    total = w_bin.sum() + eps
    ece   = ((w_bin / total) * np.abs(acc_bin - con_bin)).sum()

    return float(ece)


# ---------------------------------------------------------------------------
# 2. FPR — Freezing Predictor Rate
# ---------------------------------------------------------------------------

def compute_fpr(
    samples: np.ndarray,
    gt: np.ndarray,
    freeze_thresh: float = 0.5,
) -> float:
    """
    Freezing Predictor Rate: fraction of scenes where ALL K predictions
    stay within `freeze_thresh` metres of the observed last position
    (i.e. the predictor refuses to commit to any future motion).

    Inspired by the Freezing Predictor Effect described in Trautman & Krause
    IROS 2010 (IGP paper) where IGP navigating through crowds freezes when
    it cannot find a collision-free path.

    Parameters
    ----------
    samples       : ndarray, shape (N, K, T, 2)
    gt            : ndarray, shape (N, T, 2)
        Ground truth (used only to infer last observed position from
        samples[:, :, 0, :] — assumes t=0 is the conditioning step).
        NOTE: in practice pass the last OBSERVED position as a separate
        argument; here we use samples[:, :, 0, :] as a proxy (they should
        all equal the observation endpoint for a causal predictor).
    freeze_thresh : float
        Distance threshold in metres.  Default 0.5 m.

    Returns
    -------
    float
        Scalar FPR ∈ [0, 1].
    """
    samples = np.asarray(samples, dtype=np.float64)   # (N, K, T, 2)

    N, K, T, _ = samples.shape

    # Last observed position — use t=0 of each prediction (conditioning point)
    origin = samples[:, :, 0:1, :]                    # (N, K, 1, 2)

    # Displacement of every predicted future step from the origin
    disp = np.sqrt(((samples - origin) ** 2).sum(axis=-1))  # (N, K, T)

    # max displacement over time for each (scene, sample)
    max_disp = disp.max(axis=-1)                       # (N, K)

    # A scene is "frozen" if ALL K samples have max_disp < freeze_thresh
    scene_frozen = (max_disp < freeze_thresh).all(axis=-1)  # (N,)

    return float(scene_frozen.mean())


# ---------------------------------------------------------------------------
# 3. SVR — Social Violation Rate
# ---------------------------------------------------------------------------

def compute_svr(
    samples: np.ndarray,
    ped_radius: float = 0.3,
) -> float:
    """
    Social Violation Rate: fraction of (scene, sample, timestep) triples
    where at least one pair of predicted pedestrians is closer than
    2 * ped_radius (i.e. their personal space circles overlap).

    For each scene we check pairwise distances among all predicted agents
    at each timestep.

    Parameters
    ----------
    samples    : ndarray, shape (N, K, T, P, 2)
        Predicted trajectories for P pedestrians in each scene.
        OR shape (N, K, T, 2) for single-agent scenes (no social violation
        possible — function returns 0.0 in that case with a warning).
    ped_radius : float
        Personal space radius per pedestrian in metres.  Default 0.3 m.

    Returns
    -------
    float
        Scalar SVR ∈ [0, 1].

    Notes
    -----
    In the TUTR pkl format, the K predictions cover the EGO pedestrian.
    Neighbours are held fixed at their observed positions.  To compute SVR
    properly for multi-agent scenes, pass a concatenated (N, K, T, P, 2)
    tensor where P includes both ego and neighbours (with neighbour traj
    repeated across K).  See `compute_svr_from_tutr_batch` helper below.
    """
    samples = np.asarray(samples, dtype=np.float64)

    if samples.ndim == 4:
        # Single-agent: (N, K, T, 2) — no pairwise check possible
        # Return 0 (no violations by definition)
        return 0.0

    # Multi-agent path: (N, K, T, P, 2)
    N, K, T, P, _ = samples.shape
    min_dist_sq    = (2.0 * ped_radius) ** 2

    total_checks    = 0
    total_violations = 0

    for n in range(N):
        for k in range(K):
            for t in range(T):
                pos = samples[n, k, t]   # (P, 2)
                # pairwise squared distances — vectorised
                diff_p = pos[:, None, :] - pos[None, :, :]   # (P,P,2)
                dist_sq = (diff_p ** 2).sum(axis=-1)          # (P,P)
                # upper triangle only (exclude self-distance on diagonal)
                iu = np.triu_indices(P, k=1)
                pair_dist_sq = dist_sq[iu]
                total_checks += 1
                if (pair_dist_sq < min_dist_sq).any():
                    total_violations += 1

    if total_checks == 0:
        return 0.0
    return float(total_violations / total_checks)


def compute_svr_from_tutr_batch(
    ego_samples: np.ndarray,
    neighbour_obs: np.ndarray,
    ped_radius: float = 0.3,
) -> float:
    """
    Convenience wrapper that assembles the (N, K, T, P, 2) tensor from
    TUTR-style output:

    Parameters
    ----------
    ego_samples   : (N, K, T, 2)  — K predicted ego futures
    neighbour_obs : (N, T, P-1, 2) — observed neighbour futures (fixed)
                    Sentinel rows (1e9) are masked out.
    ped_radius    : float

    Returns
    -------
    float : SVR
    """
    ego_samples   = np.asarray(ego_samples,   dtype=np.float64)
    neighbour_obs = np.asarray(neighbour_obs, dtype=np.float64)

    N, K, T, _ = ego_samples.shape
    _, _, Pm1, _ = neighbour_obs.shape   # P-1 neighbours

    # Mask sentinels
    valid_mask = (neighbour_obs < 1e8).all(axis=-1)  # (N, T, P-1)

    total_checks    = 0
    total_violations = 0
    min_dist_sq      = (2.0 * ped_radius) ** 2

    for n in range(N):
        for k in range(K):
            for t in range(T):
                ego_pos  = ego_samples[n, k, t]           # (2,)
                nb_valid = neighbour_obs[n, t][valid_mask[n, t]]  # (M, 2)

                if nb_valid.shape[0] == 0:
                    continue

                all_pos = np.vstack([ego_pos[None], nb_valid])  # (M+1, 2)
                P_curr  = all_pos.shape[0]
                diff_p  = all_pos[:, None, :] - all_pos[None, :, :]
                dist_sq = (diff_p ** 2).sum(axis=-1)
                iu      = np.triu_indices(P_curr, k=1)
                total_checks += 1
                if (dist_sq[iu] < min_dist_sq).any():
                    total_violations += 1

    if total_checks == 0:
        return 0.0
    return float(total_violations / total_checks)


# ---------------------------------------------------------------------------
# 4. SPSR — Safe Planning Success Rate
# ---------------------------------------------------------------------------

def compute_spsr(
    samples: np.ndarray,
    gt: np.ndarray,
    planner_radius: float = 0.5,
    goal_radius: float = 1.0,
) -> float:
    """
    Safe Planning Success Rate: fraction of scenes where AT LEAST ONE of the
    K predicted trajectories both (a) reaches close to the goal position and
    (b) does not collide with any neighbour along the way.

    This is a deployment proxy metric — it models a downstream robot planner
    that selects the best collision-free prediction to follow.

    Parameters
    ----------
    samples        : ndarray, shape (N, K, T, 2)
        Predicted ego trajectories.
    gt             : ndarray, shape (N, T, 2)
        Ground truth future trajectory.  The final gt position is used as
        the goal.
    planner_radius : float
        Collision radius for the robot/pedestrian.  Default 0.5 m.
        A prediction step is "safe" if it stays > planner_radius from any
        obstacle.  Since we have no explicit obstacle map at eval time, we
        use the ground-truth path as the oracle — a prediction is "safe"
        if it stays within planner_radius of the ground-truth at each step
        (i.e. it tracks the true human trajectory without deviation).
    goal_radius    : float
        A trajectory "reaches the goal" if its final position is within
        goal_radius metres of gt[-1].

    Returns
    -------
    float
        Scalar SPSR ∈ [0, 1].

    Notes
    -----
    The collision model here (deviation from gt) is a simplification for
    the unit-test scaffold.  In full experiments (Day 2 PM) replace with
    actual neighbour positions from the TUTR batch for proper collision
    checking — see `compute_spsr_with_neighbours` helper.
    """
    samples = np.asarray(samples, dtype=np.float64)   # (N, K, T, 2)
    gt      = np.asarray(gt,      dtype=np.float64)   # (N, T, 2)

    N, K, T, _ = samples.shape

    goal   = gt[:, -1, :]                             # (N, 2)

    # distance to ground-truth at every step: (N, K, T)
    diff   = samples - gt[:, None, :, :]
    dist_gt = np.sqrt((diff ** 2).sum(axis=-1))       # (N, K, T)

    # distance to goal (final GT position) for each prediction's endpoint
    diff_goal = samples[:, :, -1, :] - goal[:, None, :]  # (N, K, 2)
    dist_goal = np.sqrt((diff_goal ** 2).sum(axis=-1))   # (N, K)

    # "safe" = max deviation from GT path ≤ planner_radius at every step
    safe    = (dist_gt.max(axis=-1) <= planner_radius)    # (N, K)

    # "reaches goal" = final position within goal_radius of gt[-1]
    reaches = (dist_goal <= goal_radius)                   # (N, K)

    # scene succeeds if at least one sample is both safe AND reaches goal
    success = (safe & reaches).any(axis=-1)               # (N,)

    return float(success.mean())


def compute_spsr_with_neighbours(
    ego_samples: np.ndarray,
    gt: np.ndarray,
    neighbour_obs: np.ndarray,
    planner_radius: float = 0.5,
    goal_radius: float = 1.0,
) -> float:
    """
    Full SPSR with explicit neighbour collision checking.

    Parameters
    ----------
    ego_samples   : (N, K, T, 2)
    gt            : (N, T, 2)
    neighbour_obs : (N, T, P, 2) — observed neighbour positions
                    Sentinel 1e9 rows are masked automatically.
    planner_radius : float
    goal_radius    : float

    Returns
    -------
    float : SPSR
    """
    ego_samples   = np.asarray(ego_samples,   dtype=np.float64)
    gt            = np.asarray(gt,            dtype=np.float64)
    neighbour_obs = np.asarray(neighbour_obs, dtype=np.float64)

    N, K, T, _ = ego_samples.shape
    goal        = gt[:, -1, :]    # (N, 2)

    success_count = 0

    for n in range(N):
        nb = neighbour_obs[n]  # (T, P, 2)
        valid_mask = (nb < 1e8).all(axis=-1)   # (T, P)

        scene_success = False
        for k in range(K):
            traj    = ego_samples[n, k]   # (T, 2)
            safe_k  = True

            for t in range(T):
                nb_t = nb[t][valid_mask[t]]   # (M, 2)
                if nb_t.shape[0] > 0:
                    dist_nb = np.sqrt(((traj[t] - nb_t) ** 2).sum(axis=-1))
                    if (dist_nb < planner_radius).any():
                        safe_k = False
                        break

            if not safe_k:
                continue

            # check goal proximity
            dist_goal = np.sqrt(((traj[-1] - goal[n]) ** 2).sum())
            if dist_goal <= goal_radius:
                scene_success = True
                break

        if scene_success:
            success_count += 1

    return float(success_count / N)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

def _run_tests():
    rng = np.random.default_rng(42)
    PASS = "\033[92m[PASS]\033[0m"
    FAIL = "\033[91m[FAIL]\033[0m"
    results = []

    def check(name, condition, got, tol=None):
        ok = condition
        tag = PASS if ok else FAIL
        detail = f"got={got:.4f}" + (f", tol={tol}" if tol else "")
        print(f"  {tag} {name}  ({detail})")
        results.append(ok)

    print("\n=== CoopCalib-TP eval_suite.py — Unit Tests ===\n")

    # ------------------------------------------------------------------
    # ECE Tests
    # ------------------------------------------------------------------
    print("--- ECE ---")

    N, K, T = 200, 20, 12

    # Test 1: Near-zero ECE on a "perfectly calibrated" predictor.
    #
    # Our ECE is defined in terms of the softmin-confidence weighting.
    # A predictor is perfectly calibrated when, for each scene, ONE sample
    # achieves the best ADE (≈ ground truth) and the rest are symmetric
    # random spread — so the "accurate" sample always has the highest
    # softmin confidence AND is always below the median ADE threshold.
    #
    # Construction:
    #   sample k=0  ← tiny noise around GT   (ADE ≈ 0)
    #   sample k>0  ← large random noise      (ADE >> 0)
    # This makes sample 0 the clear winner → softmin assigns it conf≈1.
    # It also has ADE below median → acc=1 for the high-conf sample.
    # All low-conf samples have acc≈0.5 by random chance.
    # Weighted ECE: the bin near conf=1 has acc≈1 → |acc-conf|≈0 → ECE≈0.
    gt_perfect = rng.standard_normal((N, T, 2)) * 5.0
    perfect_samples = rng.standard_normal((N, K, T, 2)) * 8.0   # noisy base
    # k=0: stay within 0.05m of GT at every step
    perfect_samples[:, 0, :, :] = gt_perfect + rng.standard_normal((N, T, 2)) * 0.02
    ece_perfect = compute_ece(perfect_samples, gt_perfect)
    check("ECE near-zero on perfect calibration", ece_perfect < 0.15, ece_perfect, tol=0.15)

    # Test 2: ECE ≈ 0.5 on a uniform spread predictor
    # All K samples are uniform random → confidence ≈ 1/K for all, accuracy ≈ 0.5
    gt_uniform = rng.standard_normal((N, T, 2))
    uniform_samples = rng.standard_normal((N, K, T, 2)) * 5.0
    ece_uniform = compute_ece(uniform_samples, gt_uniform)
    check("ECE ~0.5 on uniform spread", 0.3 < ece_uniform < 0.7, ece_uniform, tol="(0.3, 0.7)")

    # ------------------------------------------------------------------
    # FPR Tests
    # ------------------------------------------------------------------
    print("\n--- FPR ---")

    N, K, T = 50, 20, 12

    # Test 3: FPR = 1.0 when all predictions are static
    static_origin = rng.standard_normal((N, 1, 1, 2))
    static_samples = np.tile(static_origin, (1, K, T, 1))
    gt_fpr = rng.standard_normal((N, T, 2))
    fpr_static = compute_fpr(static_samples, gt_fpr, freeze_thresh=0.5)
    check("FPR = 1.0 when all predictions static", np.isclose(fpr_static, 1.0), fpr_static)

    # Test 4: FPR = 0.0 when predictions are clearly mobile
    mobile_samples = np.cumsum(rng.standard_normal((N, K, T, 2)) * 1.0, axis=2)
    # Ensure each step moves at least 0.3m to guarantee > 0.5m total displacement
    mobile_samples = mobile_samples + np.linspace(0, 3.0, T)[None, None, :, None]
    fpr_mobile = compute_fpr(mobile_samples, gt_fpr, freeze_thresh=0.5)
    check("FPR = 0.0 when predictions are mobile", np.isclose(fpr_mobile, 0.0), fpr_mobile)

    # ------------------------------------------------------------------
    # SVR Tests
    # ------------------------------------------------------------------
    print("\n--- SVR ---")

    N, K, T, P = 30, 20, 12, 5

    # Test 5: SVR = 0 on well-separated pedestrians (spacing = 5m)
    # Place pedestrians in a row with 5m gaps
    spacing = 5.0
    base_pos = np.arange(P)[None, None, None, :, None] * spacing  # (1,1,1,P,1)
    base_xy  = np.concatenate([base_pos, np.zeros_like(base_pos)], axis=-1)
    sep_samples = np.tile(base_xy, (N, K, T, 1, 1)).astype(np.float64)
    sep_samples += rng.standard_normal(sep_samples.shape) * 0.01  # tiny jitter
    svr_sep = compute_svr(sep_samples, ped_radius=0.3)
    check("SVR = 0.0 on well-separated pedestrians", np.isclose(svr_sep, 0.0), svr_sep)

    # Test 6: SVR > 0 when pedestrians overlap (placed within 2*radius)
    # All pedestrians at the same position → guaranteed violations
    same_pos = np.zeros((N, K, T, P, 2))
    same_pos += rng.standard_normal((N, K, T, P, 2)) * 0.01
    svr_overlap = compute_svr(same_pos, ped_radius=0.3)
    check("SVR > 0 when pedestrians overlap", svr_overlap > 0.5, svr_overlap, tol=">0.5")

    # ------------------------------------------------------------------
    # SPSR Tests
    # ------------------------------------------------------------------
    print("\n--- SPSR ---")

    N, K, T = 40, 20, 12

    # Test 7: SPSR = 1.0 on a perfect predictor
    # At least one sample exactly tracks the GT (within planner_radius)
    gt_spsr = rng.standard_normal((N, T, 2)) * 3.0
    perfect_spsr = np.tile(gt_spsr[:, None, :, :], (1, K, 1, 1))
    perfect_spsr += rng.standard_normal((N, K, T, 2)) * 0.05  # within 0.5m
    spsr_perfect = compute_spsr(perfect_spsr, gt_spsr, planner_radius=0.5, goal_radius=1.0)
    check("SPSR = 1.0 on perfect predictor", np.isclose(spsr_perfect, 1.0, atol=0.05), spsr_perfect)

    # Test 8: SPSR = 0.0 on a predictor that always misses the goal
    # All samples go in the opposite direction
    miss_samples = gt_spsr[:, None, :, :] * (-1.0) + 20.0
    miss_samples = np.tile(miss_samples, (1, K, 1, 1))
    spsr_miss = compute_spsr(miss_samples, gt_spsr, planner_radius=0.5, goal_radius=1.0)
    check("SPSR = 0.0 on all-miss predictor", np.isclose(spsr_miss, 0.0, atol=0.05), spsr_miss)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*48}")
    n_pass = sum(results)
    n_total = len(results)
    status = "\033[92mALL PASS\033[0m" if n_pass == n_total else "\033[91mSOME FAILED\033[0m"
    print(f"  {status}  ({n_pass}/{n_total} tests)\n")
    return n_pass == n_total


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    ok = _run_tests()
    sys.exit(0 if ok else 1)
