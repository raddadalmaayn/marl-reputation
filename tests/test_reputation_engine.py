"""
T0.4 — Validation of env/reputation_engine.py against its SPECIFIED math and
against the canonical Go chaincode.

Go reference located at:
  ~/am-unified/chaincode/unified/helpers.go            (applyDynamicDecay, calculateWilsonCI)
  ~/am-unified/chaincode/unified/reputation_contract.go (slash, dispute, stake gate, Bayesian update)

Golden cross-checks below are derived directly from those Go functions. Where the
Python engine diverges from the Go DEFAULTS, the divergence is asserted explicitly
(see test_rater_weight_clamp_divergence) rather than papered over.

Run:  python3 -m pytest tests/test_reputation_engine.py -v
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env.reputation_engine import (
    ReputationEngine, ReputationState, StakeState, SystemConfig,
)

TOL = 1e-9


@pytest.fixture
def engine():
    return ReputationEngine(SystemConfig())


# --------------------------------------------------------------------------- #
# Decay: alpha_eff = alpha0 + (alpha - alpha0) * lambda^(elapsed/T), floor alpha0
# --------------------------------------------------------------------------- #
def test_decay_pulls_toward_prior(engine):
    cfg = engine.config
    # one full decay period elapsed => decay_factor = lambda^1 = 0.98
    rep = ReputationState("a", "quality", alpha=10.0, beta=4.0, last_ts=0.0)
    rep.last_ts = 1.0  # nonzero so decay runs
    now = 1.0 + cfg.decay_period  # elapsed = 1.0
    out = engine.apply_decay(rep, now=now)
    # alpha_eff = 2 + (10-2)*0.98 = 9.84 ; beta_eff = 2 + (4-2)*0.98 = 3.96
    assert out.alpha == pytest.approx(2.0 + 8.0 * 0.98, abs=TOL)
    assert out.beta == pytest.approx(2.0 + 2.0 * 0.98, abs=TOL)


def test_decay_floors_at_prior(engine):
    cfg = engine.config
    # alpha below the prior must be floored back up to alpha0
    rep = ReputationState("a", "quality", alpha=1.0, beta=2.0, last_ts=1.0)
    now = 1.0 + cfg.decay_period
    out = engine.apply_decay(rep, now=now)
    assert out.alpha == pytest.approx(cfg.initial_alpha, abs=TOL)  # 2.0
    assert out.beta == pytest.approx(cfg.initial_beta, abs=TOL)


def test_decay_skipped_when_no_history(engine):
    rep = ReputationState("a", "quality", alpha=10.0, beta=4.0, last_ts=0.0)
    out = engine.apply_decay(rep, now=999999.0)
    assert out.alpha == 10.0 and out.beta == 4.0  # last_ts==0 => unchanged


def test_decay_skipped_when_nonpositive_elapsed(engine):
    rep = ReputationState("a", "quality", alpha=10.0, beta=4.0, last_ts=100.0)
    out = engine.apply_decay(rep, now=50.0)  # elapsed < 0
    assert out.alpha == 10.0 and out.beta == 4.0


# --------------------------------------------------------------------------- #
# Bayesian update: alpha += weight (positive), beta += weight (otherwise)
# Go: rep.Alpha += weight*value ; rep.Beta += weight*(1-value), value in {0,1}
# --------------------------------------------------------------------------- #
def test_update_positive_outcome(engine):
    rep = ReputationState("a", "quality", alpha=2.0, beta=2.0, last_ts=0.0)
    out = engine.apply_rating(rep, outcome=1, weight=1.0, now=10.0)
    assert out.alpha == pytest.approx(3.0, abs=TOL)
    assert out.beta == pytest.approx(2.0, abs=TOL)
    assert out.total_events == 1


def test_update_negative_outcome(engine):
    rep = ReputationState("a", "quality", alpha=2.0, beta=2.0, last_ts=0.0)
    out = engine.apply_rating(rep, outcome=0, weight=1.0, now=10.0)
    assert out.alpha == pytest.approx(2.0, abs=TOL)
    assert out.beta == pytest.approx(3.0, abs=TOL)


def test_update_weight_within_bounds(engine):
    rep = ReputationState("a", "quality", alpha=2.0, beta=2.0, last_ts=0.0)
    out = engine.apply_rating(rep, outcome=1, weight=1.5, now=10.0)
    assert out.alpha == pytest.approx(3.5, abs=TOL)  # 2 + 1.5


def test_update_weight_clamped_high(engine):
    rep = ReputationState("a", "quality", alpha=2.0, beta=2.0, last_ts=0.0)
    # Tier 1A: max_rater_weight default is now Go-matching 5.0 -> 2.0 + 5.0 = 7.0
    assert engine.config.max_rater_weight == 5.0
    out = engine.apply_rating(rep, outcome=1, weight=9.0, now=10.0)
    assert out.alpha == pytest.approx(7.0, abs=TOL)


def test_update_weight_clamped_low(engine):
    rep = ReputationState("a", "quality", alpha=2.0, beta=2.0, last_ts=0.0)
    # Tier 1A: min_rater_weight default is now Go-matching 0.1 -> 2.0 + 0.1 = 2.1
    assert engine.config.min_rater_weight == 0.1
    out = engine.apply_rating(rep, outcome=1, weight=0.001, now=10.0)
    assert out.alpha == pytest.approx(2.1, abs=TOL)


# --------------------------------------------------------------------------- #
# Wilson CI — Go uses n = alpha + beta (helpers.go:275). Paper Eq.(2) also uses
# n = alpha + beta. The report alleged the code uses n = alpha+beta-2*alpha0;
# this test PINS the implemented denominator to n = alpha + beta and FAILS if
# the engine ever switches to the n-2*alpha0 form.
# --------------------------------------------------------------------------- #
def _wilson_ref(alpha, beta, n, z=1.96):
    p = alpha / n
    denom = 1.0 + (z * z) / n
    centre = (p + (z * z) / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def test_wilson_matches_n_equals_alpha_plus_beta():
    alpha, beta = 8.0, 4.0
    lo, hi, width = ReputationEngine.wilson_ci(alpha, beta)
    n_correct = alpha + beta            # paper Eq.(2) and Go helpers.go:275
    elo, ehi = _wilson_ref(alpha, beta, n_correct)
    assert lo == pytest.approx(elo, abs=TOL)
    assert hi == pytest.approx(ehi, abs=TOL)
    assert width == pytest.approx(ehi - elo, abs=TOL)


def test_wilson_is_not_n_minus_2alpha0():
    """Guard: the engine must NOT implement the n = alpha+beta-2*alpha0 form."""
    alpha, beta, alpha0 = 8.0, 4.0, 2.0
    lo, hi, _ = ReputationEngine.wilson_ci(alpha, beta)
    n_wrong = alpha + beta - 2 * alpha0  # = 8 ; the alleged (wrong) denominator
    wlo, whi = _wilson_ref(alpha, beta, n_wrong)
    # The two forms give materially different widths; assert engine != wrong form.
    assert not (lo == pytest.approx(wlo, abs=1e-6) and hi == pytest.approx(whi, abs=1e-6))


def test_wilson_hand_value_prior():
    # alpha=beta=2 -> symmetric, centre exactly 0.5
    lo, hi, width = ReputationEngine.wilson_ci(2.0, 2.0)
    assert (lo + hi) / 2 == pytest.approx(0.5, abs=TOL)
    # hand: n=4, p=0.5, z=1.96 -> denom=1.9604, margin=0.34996429,
    # width = 2*margin = 0.69992858
    assert width == pytest.approx(0.6999285824, abs=1e-8)


# --------------------------------------------------------------------------- #
# Stake mechanics — Go reputation_contract.go: gate, slash, dispute
# --------------------------------------------------------------------------- #
def test_can_rate_threshold(engine):
    # Go: reject if Balance < MinStakeRequired  => can_rate iff balance >= min
    assert engine.can_rate(StakeState("a", balance=10_000.0)) is True
    assert engine.can_rate(StakeState("a", balance=9_999.99)) is False


def test_slash_by_percentage(engine):
    # Go: slashAmount = balance * 0.10 ; balance -= slashAmount  => balance*0.9
    out = engine.slash(StakeState("a", balance=10_000.0))
    assert out.balance == pytest.approx(9_000.0, abs=TOL)


def test_dispute_cost_deduction(engine):
    # Go: stake.Balance -= config.DisputeCost (100)
    out = engine.deduct_dispute_cost(StakeState("a", balance=10_000.0))
    assert out.balance == pytest.approx(9_900.0, abs=TOL)


def test_slash_floors_at_zero(engine):
    out = engine.slash(StakeState("a", balance=0.0))
    assert out.balance == 0.0


# --------------------------------------------------------------------------- #
# Golden cross-check vs Go chaincode + documented divergence
# --------------------------------------------------------------------------- #
def test_golden_decay_matches_go_formula():
    """Reproduce Go applyDynamicDecay for a concrete vector (helpers.go:239)."""
    cfg = SystemConfig()  # InitialAlpha=2, InitialBeta=2, DecayRate=0.98, T=86400
    eng = ReputationEngine(cfg)
    rep = ReputationState("a", "quality", alpha=20.0, beta=6.0, last_ts=1000.0)
    elapsed_periods = 3.0
    now = 1000.0 + elapsed_periods * cfg.decay_period
    df = math.pow(cfg.decay_rate, elapsed_periods)
    exp_alpha = cfg.initial_alpha + (20.0 - cfg.initial_alpha) * df
    exp_beta = cfg.initial_beta + (6.0 - cfg.initial_beta) * df
    out = eng.apply_decay(rep, now=now)
    assert out.alpha == pytest.approx(exp_alpha, abs=TOL)
    assert out.beta == pytest.approx(exp_beta, abs=TOL)


def test_rater_weight_clamp_matches_go_default():
    """Tier 1A (WS2): the SystemConfig DEFAULT rater-weight clamp now MATCHES the
    canonical Go default config [0.1, 5.0] (reputation_contract.go:52-53). This is
    the inverted form of the Tier 0 divergence test — it must FAIL if the default
    ever drifts back to the legacy [0.5, 2.0]."""
    GO_MIN_RATER_WEIGHT, GO_MAX_RATER_WEIGHT = 0.1, 5.0
    py = SystemConfig()
    assert (py.min_rater_weight, py.max_rater_weight) == \
        (GO_MIN_RATER_WEIGHT, GO_MAX_RATER_WEIGHT)


def test_rater_weight_legacy_override_still_selectable():
    """Tier 1A (WS2): the legacy v1 clamp [0.5, 2.0] remains selectable via an
    explicit SystemConfig override, so v1 behavior is exactly reproducible."""
    legacy = SystemConfig(min_rater_weight=0.5, max_rater_weight=2.0)
    assert (legacy.min_rater_weight, legacy.max_rater_weight) == (0.5, 2.0)
    eng = ReputationEngine(legacy)
    rep = ReputationState("a", "quality", alpha=2.0, beta=2.0, last_ts=0.0)
    # weight 9.0 clamps to the legacy max 2.0 -> alpha = 4.0 (not the new 7.0)
    out = eng.apply_rating(rep, outcome=1, weight=9.0, now=10.0)
    assert out.alpha == pytest.approx(4.0, abs=TOL)


def test_config_defaults_match_go():
    """Core SystemConfig defaults that DO match the Go chaincode defaults."""
    c = SystemConfig()
    assert c.initial_alpha == 2.0 and c.initial_beta == 2.0
    assert c.decay_rate == 0.98 and c.decay_period == 86400.0
    assert c.min_stake_required == 10_000.0
    assert c.dispute_cost == 100.0
    assert c.slash_percentage == 0.10
