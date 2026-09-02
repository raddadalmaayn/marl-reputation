"""
Basin classifier — reusable, training-time signal (Tier 1A, WS3).

Factored out of the Tier 0 logic in evaluation/recompute_metrics.py so that
train.py / train_ablation.py can self-report each seed's basin at end-of-training
instead of us reconstructing it later. Thresholds are EXACTLY the Tier 0 values:

  abstention            : participation < 0.01
  adversarial-dominant  : honest_given_rating <= 0.05 OR final_mean_stake <= 0.01*reset_stake
  honest-dominant       : honest_given_rating >= 0.95 AND participation >= 0.01
  mixed                 : otherwise

Action-count convention matches Tier 0: counts record the POLICY-SELECTED action
(pre any in-env forcing of actions >=7 to noop for non-adversarial agents). The
forcing only touches non-rating buckets, so participation / honest_given_rating
are invariant to it.
"""

import math
from collections import Counter

# Explicit, Tier 0-identical thresholds.
THR_PARTICIPATION_MIN = 0.01   # below this => abstention
THR_ADV_HGR_MAX       = 0.05   # honest_given_rating <= this => adversarial-dominant
THR_ADV_STAKE_FRAC    = 0.01   # final mean stake <= this*reset => "bankrupt"
THR_HON_HGR_MIN       = 0.95   # honest_given_rating >= this => honest-dominant


def honest_metrics_from_counts(action_counts: dict) -> dict:
    """Compute participation / honest_given_rating / honest_unconditional from A0..A11."""
    A = {i: int(action_counts.get(i, action_counts.get(str(i), 0))) for i in range(12)}
    rating = A[1] + A[2] + A[3] + A[4]
    honest = A[1] + A[2]
    total = sum(A.values())
    participation = rating / total if total else float("nan")
    hgr = (honest / rating) if rating > 0 else float("nan")
    huncond = hgr if rating > 0 else 0.0
    return {
        "rating_actions": rating,
        "honest_actions": honest,
        "total_actions": total,
        "participation_rate": participation,
        "honest_given_rating": hgr,
        "honest_unconditional": huncond,
    }


def classify_basin(participation_rate: float, honest_given_rating: float,
                   final_mean_stake: float, reset_stake: float) -> str:
    """Return one of: abstention | adversarial-dominant | honest-dominant | mixed."""
    part = participation_rate
    hgr = honest_given_rating
    stake_frac = (final_mean_stake / reset_stake) if reset_stake else 1.0
    if (not math.isnan(part)) and part < THR_PARTICIPATION_MIN:
        return "abstention"
    bankrupt = stake_frac <= THR_ADV_STAKE_FRAC
    if (not math.isnan(hgr) and hgr <= THR_ADV_HGR_MAX) or bankrupt:
        return "adversarial-dominant"
    if (not math.isnan(hgr) and hgr >= THR_HON_HGR_MIN) and \
       (not math.isnan(part) and part >= THR_PARTICIPATION_MIN):
        return "honest-dominant"
    return "mixed"


def collect_basin_report(env, select_action, eval_episodes: int = 20,
                         eval_base: int = 10000, with_rho: bool = False) -> dict:
    """Run deterministic eval and produce the full basin report for one seed.

    `select_action(agent, obs) -> int` must return the deterministic (argmax)
    action for the given agent/observation (e.g. a closure over a loaded trainer).
    Episodes use fixed seeds (eval_base + i) for reproducibility/idempotency.

    Returns a dict ready to merge into a per-seed summary: action_counts (A0..A11),
    participation, honest_given_rating, final_mean_stake, reset_stake, basin.

    with_rho (Tier 1B, default False so Tier 1A callers are unaffected): additionally
    pool per-agent (score, true_quality) across eval episodes and add Spearman
    `system_spearman_rho` + `system_mae` + `n_score_points`. Tier 0 established rho
    (NOT MAE) is the valid discrimination metric (true quality ~ Beta(8,2)).
    """
    import numpy as np
    counts = Counter()
    final_stakes = []
    scores, quals = [], []
    reset_stake = env.config.min_stake_required * 2.0
    for e in range(eval_episodes):
        env.reset(seed=eval_base + e)
        while env.agents:
            ag = env.agent_selection
            if env.terminations.get(ag, False) or env.truncations.get(ag, False):
                env.step(None)
                continue
            obs = env.observe(ag)
            a = int(select_action(ag, obs))
            env.step(a)
            counts[a] += 1
        m = env.get_metrics()
        final_stakes.append(float(np.mean(list(m["stakes"].values()))))
        if with_rho:
            scores.extend(m["agent_scores"])
            quals.extend(m["agent_true_qualities"])

    action_counts = {str(a): int(counts.get(a, 0)) for a in range(12)}
    hm = honest_metrics_from_counts(action_counts)
    final_mean_stake = float(np.mean(final_stakes)) if final_stakes else float("nan")
    basin = classify_basin(hm["participation_rate"], hm["honest_given_rating"],
                           final_mean_stake, reset_stake)
    out = {
        "action_counts": action_counts,
        "participation_rate": hm["participation_rate"],
        "honest_given_rating": hm["honest_given_rating"],
        "honest_unconditional": hm["honest_unconditional"],
        "final_mean_stake": final_mean_stake,
        "reset_stake": reset_stake,
        "basin": basin,
        "basin_eval_episodes": eval_episodes,
        "basin_eval_base": eval_base,
    }
    if with_rho:
        out.update(_score_quality_metrics(scores, quals))
    return out


def _score_quality_metrics(scores, quals) -> dict:
    """Spearman rho + MAE of system score vs true quality (Tier 1B endpoint)."""
    import numpy as np
    s = np.asarray(scores, dtype=float)
    q = np.asarray(quals, dtype=float)
    if s.size == 0:
        return {"system_spearman_rho": float("nan"), "system_mae": float("nan"),
                "n_score_points": 0}
    mae = float(np.mean(np.abs(s - q)))
    if np.std(s) < 1e-12 or np.std(q) < 1e-12:
        rho = float("nan")
    else:
        from scipy.stats import spearmanr
        rho = float(spearmanr(s, q).statistic)
    return {"system_spearman_rho": rho, "system_mae": mae,
            "n_score_points": int(s.size)}
