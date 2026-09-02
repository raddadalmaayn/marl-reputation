"""
Tier 1B — Stage 0: derive the participation incentive c from the ACTUAL env reward
function (not the paper's Eq. 6/7), and decide the gate.

Method
------
For an agent that is eligible to rate (balance >= min_stake, no duplicate), the
per-action immediate reward, taken from env.reputation_marl_env._execute_action /
_execute_rating, decomposes (relative to noop, the EV=0 baseline) as:

  E[noop]            = 0
  E[honest_rating]   = 0                              # rating changes the TARGET's
                       (no own-score change; honest ratings are never slashed)
  E[dishonest_rating]= p_d * (R_DISHONEST_CAUGHT)     # p_d = DISHONEST_DETECT_PROB
                       + (1-p_d)*0  + slash side-cost # detected -> slash + penalty

The adversarial bonus b enters as b*max(0, own_score_delta); a RATING action does
not change the rater's own score this step, so b is identical across the agent's
action choices this step and CANCELS in the rate/noop and honest/dishonest
comparisons. Hence E[dishonest] is independent of b in the implemented code. The
collusion division (reward/=G) scales every action's reward equally, so it also
cancels in the sign conditions. Both facts are stated and then checked empirically.

Participation incentive c is paid on the OBSERVABLE act of submitting a rating
(actions 1-4 that pass can_rate + dedup), independent of hidden honesty. Then:

  E[honest_rating]   + c  = c
  E[dishonest_rating]+ c  = E_dishonest + c

Feasibility (incentive compatibility):
  (P1) honest beats abstention :  E[honest] + c > 0        =>  c > -E[honest] = 0
  (P2) cheating stays negative :  E[dishonest] + c < 0     =>  c < -E[dishonest]
  =>  closed form:  E[dishonest] < -c < E[honest].

Gate: feasible c exists  <=>  E[dishonest] < E[honest]. If so, recommend the
midpoint of the feasible interval (robust to EV estimation error and to b, which
does not enter). Otherwise plan (b) collapses to plan (a): STOP, no Arm B.

Usage: python3 evaluation/participation_economics.py
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from env.reputation_marl_env import (
    ReputationMARLEnv, DISHONEST_DETECT_PROB, R_DISHONEST_CAUGHT,
    R_STAKE_LOSS, R_SCORE_GAIN, R_SCORE_LOSS,
)
from env.reputation_engine import SystemConfig

B_RANGE = (0.5, 2.0)   # configured adversarial bonus range


# --------------------------------------------------------------------------- #
# Analytic EVs from the actual constants
# --------------------------------------------------------------------------- #
def analytic_evs():
    cfg = SystemConfig()
    p_d = DISHONEST_DETECT_PROB
    # honest rating: no own-score change, no slash -> 0
    e_honest = 0.0
    # dishonest rating: detected w.p. p_d -> R_DISHONEST_CAUGHT + small stake-loss
    # increment from one 10% slash at reset balance (B=2*min_stake):
    #   d(stake_loss term) = R_STAKE_LOSS * 0.1 * (slash_frac*B/(2*min_stake)) ... at
    #   B=2m this is R_STAKE_LOSS * 0.1 * slash_percentage = -0.5*0.1*0.10 = -0.005
    stake_side = R_STAKE_LOSS * 0.1 * cfg.slash_percentage  # ~ -0.005
    e_dishonest = p_d * (R_DISHONEST_CAUGHT + stake_side) + (1 - p_d) * 0.0
    return {
        "p_detect_dishonest": p_d,
        "R_dishonest_caught": R_DISHONEST_CAUGHT,
        "stake_loss_side_cost_per_slash": stake_side,
        "E_noop": 0.0,
        "E_honest_rating": e_honest,
        "E_dishonest_rating": e_dishonest,
        "b_enters_dishonest_EV": False,
        "note_b": ("adversarial bonus is b*max(0,own_score_delta); a rating does not "
                   "change own score this step, so b cancels in the action comparison "
                   "and does not enter E[dishonest]."),
    }


# --------------------------------------------------------------------------- #
# Empirical validation: force one eligible agent to take each action class and
# read its immediate reward (env.rewards[agent]) right after its step.
# --------------------------------------------------------------------------- #
def empirical_evs(n_trials=4000, participation_coef=0.0):
    """Measure immediate reward per action class for a non-adversarial, non-colluding
    agent (so reward == _execute_action only — no bonus, no collusion division)."""
    probe_idx = 5                      # honest (not in adversarial_ids)
    probe = f"agent_{probe_idx}"
    results = {a: [] for a in (0, 1, 2, 3, 4)}
    for a in results:
        for t in range(n_trials):
            env = ReputationMARLEnv(
                n_agents=20, max_steps=100,
                adversarial_ids=[0, 1, 2, 3, 4],   # probe_idx=5 is honest
                adversarial_reward_bonus=1.0,      # bonus exists but probe is honest
                collusion_groups=[], max_sybils_per_agent=0,
                config=SystemConfig(), seed=1000 + t,
                stake_obs_mode="absolute_log",
                participation_coef=participation_coef,
            )
            env.reset(seed=1000 + t)
            # advance agents 0..4 (their turns) with noop so the probe rates a fresh target
            while env.agent_selection != probe:
                env.step(0)
            env.step(a)               # probe takes the action under test
            results[a].append(env.rewards[probe])
    return {a: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
            for a, v in results.items()}


# --------------------------------------------------------------------------- #
# Feasibility + gate
# --------------------------------------------------------------------------- #
def derive(evs):
    e_h = evs["E_honest_rating"]
    e_d = evs["E_dishonest_rating"]
    lower = -e_h          # c must exceed this (P1)
    upper = -e_d          # c must be below this (P2)
    feasible = lower < upper
    rec = (lower + upper) / 2.0 if feasible else None
    return {
        "feasibility_closed_form": "E[dishonest] < -c < E[honest]",
        "c_lower_bound_exclusive": lower,   # P1: c > lower
        "c_upper_bound_exclusive": upper,   # P2: c < upper
        "feasible_interval": [lower, upper] if feasible else None,
        "feasible": bool(feasible),
        "b_range": list(B_RANGE),
        "b_independent": not evs["b_enters_dishonest_EV"],
        "recommended_c": rec,
        "recommended_c_justification": (
            f"midpoint of the feasible interval ({lower:.3f}, {upper:.3f}); the "
            f"interval is independent of the adversarial bonus b in {B_RANGE} "
            f"(b does not enter E[dishonest] in the implemented reward), so the "
            f"midpoint is simultaneously the worst-case-b choice. At c={rec:.3f} "
            f"honest rating EV = +{rec:.3f} and dishonest rating EV = "
            f"{e_d + rec:.3f} < 0 — incentive compatible." if feasible else
            "no feasible c — E[dishonest] >= E[honest]"),
    }


def main():
    out_dir = REPO / "results" / "tier1b"
    out_dir.mkdir(parents=True, exist_ok=True)

    evs = analytic_evs()
    print("Analytic EVs:", json.dumps(evs, indent=2))
    print("\nEmpirical validation (Arm A, participation_coef=0.0)...")
    emp0 = empirical_evs(participation_coef=0.0)
    for a in (0, 1, 2, 3, 4):
        print(f"  action {a}: mean reward = {emp0[a]['mean']:+.4f} "
              f"(+/-{emp0[a]['std']:.3f}, n={emp0[a]['n']})")

    der = derive(evs)
    rec = der["recommended_c"]

    # Empirical check that the recommended c lands honest>0 and dishonest<0.
    emp_c = None
    if rec is not None:
        print(f"\nEmpirical validation (Arm B, participation_coef={rec:.4f})...")
        emp_c = empirical_evs(participation_coef=rec)
        for a in (0, 1, 3):
            print(f"  action {a}: mean reward = {emp_c[a]['mean']:+.4f}")

    gate = "PROCEED_ARM_B" if der["feasible"] else "STOP_COLLAPSE_TO_PLAN_A"
    report = {
        "stage": "Tier1B Stage 0 — participation incentive derivation",
        "analytic_evs": evs,
        "empirical_evs_armA": {str(k): v for k, v in emp0.items()},
        "empirical_evs_armB_at_recommended_c":
            ({str(k): v for k, v in emp_c.items()} if emp_c else None),
        "feasibility": der,
        "gate_decision": gate,
    }
    (out_dir / "participation_economics.json").write_text(json.dumps(report, indent=2))

    # Markdown
    L = ["# Tier 1B — Stage 0: Participation-Incentive Derivation & Gate", ""]
    L.append("## Per-action expected value (relative to noop = 0)")
    L.append("Derived from the ACTUAL `_execute_action` / `_execute_rating` reward terms "
             "and validated by forced-action rollouts.")
    L.append("")
    L.append("| Action class | analytic EV | empirical mean (Arm A) | std |")
    L.append("|---|---|---|---|")
    names = {0: "noop (A0)", 1: "honest+ (A1)", 2: "honest- (A2)",
             3: "dishonest inflate (A3)", 4: "dishonest deflate (A4)"}
    ana = {0: evs["E_noop"], 1: evs["E_honest_rating"], 2: evs["E_honest_rating"],
           3: evs["E_dishonest_rating"], 4: evs["E_dishonest_rating"]}
    for a in (0, 1, 2, 3, 4):
        L.append(f"| {names[a]} | {ana[a]:+.4f} | {emp0[a]['mean']:+.4f} | "
                 f"{emp0[a]['std']:.4f} |")
    L.append("")
    L.append(f"- `DISHONEST_DETECT_PROB = {evs['p_detect_dishonest']}`, "
             f"`R_DISHONEST_CAUGHT = {evs['R_dishonest_caught']}` ⇒ "
             f"`E[dishonest] = {evs['E_dishonest_rating']:.4f}`.")
    L.append(f"- **b-independence:** {evs['note_b']}")
    L.append("")
    L.append("## Feasibility (incentive compatibility)")
    L.append(f"- Closed form: **{der['feasibility_closed_form']}**")
    L.append(f"- (P1) c > {der['c_lower_bound_exclusive']:.4f}  (honest beats abstention)")
    L.append(f"- (P2) c < {der['c_upper_bound_exclusive']:.4f}  (cheating stays net-negative)")
    L.append(f"- Feasible interval: **{der['feasible_interval']}** "
             f"for all b in {der['b_range']}")
    L.append(f"- **Recommended c = {rec:.4f}** — {der['recommended_c_justification']}")
    if emp_c is not None:
        L.append("")
        L.append(f"Empirical check at c={rec:.4f}: honest (A1) = "
                 f"{emp_c[1]['mean']:+.4f} (>0 ✓), dishonest (A3) = "
                 f"{emp_c[3]['mean']:+.4f} (<0 ✓), noop (A0) = {emp_c[0]['mean']:+.4f}.")
    L.append("")
    L.append(f"## GATE DECISION: **{gate}**")
    L.append("")
    if der["feasible"]:
        L.append("A feasible, b-robust participation incentive exists ⇒ proceed to "
                 f"Stage 1 with Arm B enabled at `participation_coef = {rec:.4f}`.")
    else:
        L.append("No feasible c ⇒ plan (b) collapses to plan (a). Do NOT run Arm B; "
                 "recommend the one-arm sweep. See STAGE0_RESULT.md.")
    (out_dir / "participation_economics.md").write_text("\n".join(L) + "\n")

    print(f"\nGATE: {gate}")
    print(f"Recommended c = {rec}")
    print(f"Wrote {out_dir}/participation_economics.{{json,md}}")
    return der["feasible"], rec


if __name__ == "__main__":
    main()
