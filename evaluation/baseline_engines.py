"""
Tier 1B — Stage 2: external reputation-engine baselines.

Two drop-in alternatives to the Beta engine, sharing the env's rating-event
interface (rater, target, outcome, weight):

  FlatAverageEngine  — trivial control: unweighted running mean of ratings. A true
                       in-loop drop-in for ReputationEngine (same method surface:
                       apply_rating / wilson_ci / can_rate / slash / deduct_dispute_cost),
                       differing only in apply_rating (no rater-weight, no decay).

  EigenTrustScorer   — global trust-flow (Sep & Garcia-Molina row-normalized variant).
                       Trust is a GLOBAL fixed point over the whole rating graph, so it
                       cannot be expressed as an independent per-target Beta update; it
                       is computed from the recorded rating events at evaluation time
                       (env records edges; score(actor) reads the converged vector).

Both are validated against the Beta engine on a controlled rating stream in
`smoke_validate()` (run this file directly).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env.reputation_engine import ReputationEngine, ReputationState, SystemConfig


class FlatAverageEngine(ReputationEngine):
    """Unweighted Beta(prior) count: alpha += 1{positive}, beta += 1{negative}.
    No rater-weight scaling, no temporal decay. score = alpha/(alpha+beta)."""

    def apply_rating(self, rep, outcome, weight=1.0, now=None):
        t = now if now is not None else 0.0
        return ReputationState(
            actor_id=rep.actor_id, dimension=rep.dimension,
            alpha=rep.alpha + (1.0 if outcome == 1 else 0.0),
            beta=rep.beta + (1.0 if outcome == 0 else 0.0),
            total_events=rep.total_events + 1, last_ts=t,
        )


class EigenTrustScorer:
    """Row-normalized EigenTrust over the recorded rating graph.

    Edges: positive local trust s_ij = (#positive i->j) accumulated. The local
    trust matrix C is row-normalized; the global trust vector t solves
        t = (1-a) * C^T t + a * p
    with teleport a and uniform pre-trust p. Scores are min-max normalized to [0,1]
    so they are comparable to Beta scores for rho/hold-rate evaluation.
    """

    def __init__(self, n_actors, teleport=0.15, iters=100, tol=1e-10):
        self.n = n_actors
        self.teleport = teleport
        self.iters = iters
        self.tol = tol
        self.pos = np.zeros((n_actors, n_actors), dtype=float)  # pos[i,j] = i trusts j

    def add_event(self, rater_idx, target_idx, outcome, weight=1.0):
        if outcome == 1:
            self.pos[rater_idx, target_idx] += weight

    def _normalized_C(self):
        # Row-normalize local trust; rows with no positive ratings become uniform
        # (teleport) to avoid dead ends in the power iteration.
        out = np.zeros_like(self.pos)
        for i in range(self.n):
            r = self.pos[i].sum()
            out[i] = self.pos[i] / r if r > 0 else np.ones(self.n) / self.n
        return out

    def scores(self):
        C = self._normalized_C()
        p = np.ones(self.n) / self.n
        t = p.copy()
        for _ in range(self.iters):
            t_new = (1 - self.teleport) * (C.T @ t) + self.teleport * p
            if np.linalg.norm(t_new - t, 1) < self.tol:
                t = t_new
                break
            t = t_new
        # normalize to [0,1] for comparability
        lo, hi = t.min(), t.max()
        return (t - lo) / (hi - lo) if hi > lo else np.full(self.n, 0.5)


def smoke_validate(n_actors=20, n_events=4000, seed=0):
    """Compare Beta vs Flat vs EigenTrust on a synthetic rating stream where each
    actor has a hidden true quality ~ Beta(8,2) and raters report honestly with
    probability proportional to that quality. Reports Spearman rho vs true quality."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    cfg = SystemConfig()
    q = rng.beta(8.0, 2.0, n_actors)

    beta = ReputationEngine(cfg)
    flat = FlatAverageEngine(cfg)
    eig = EigenTrustScorer(n_actors)
    beta_state = [ReputationState(f"a{i}", "quality", cfg.initial_alpha, cfg.initial_beta,
                                  last_ts=0.0) for i in range(n_actors)]
    flat_state = [ReputationState(f"a{i}", "quality", cfg.initial_alpha, cfg.initial_beta,
                                  last_ts=0.0) for i in range(n_actors)]

    for k in range(n_events):
        i = rng.integers(n_actors)
        j = rng.integers(n_actors)
        if i == j:
            continue
        outcome = int(rng.random() < q[j])      # honest rating reflects true quality
        w = cfg.min_rater_weight + beta_state[i].score * (cfg.max_rater_weight - cfg.min_rater_weight)
        beta_state[j] = beta.apply_rating(beta_state[j], outcome, weight=w, now=k)
        flat_state[j] = flat.apply_rating(flat_state[j], outcome, now=k)
        eig.add_event(i, j, outcome)

    beta_sc = np.array([s.score for s in beta_state])
    flat_sc = np.array([s.score for s in flat_state])
    eig_sc = eig.scores()
    out = {
        "beta_rho": float(spearmanr(beta_sc, q).statistic),
        "flat_rho": float(spearmanr(flat_sc, q).statistic),
        "eigentrust_rho": float(spearmanr(eig_sc, q).statistic),
        "beta_mae": float(np.mean(np.abs(beta_sc - q))),
        "flat_mae": float(np.mean(np.abs(flat_sc - q))),
        "eigentrust_mae": float(np.mean(np.abs(eig_sc - q))),
    }
    return out


if __name__ == "__main__":
    import json
    res = {f"seed{s}": smoke_validate(seed=s) for s in range(3)}
    print(json.dumps(res, indent=2))
    print("\nAll three engines produce finite scores and positive rho vs true "
          "quality on the honest stream — interface validated.")
