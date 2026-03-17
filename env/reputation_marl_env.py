"""
PettingZoo AEC environment — MARL adversarial reputation evaluation.

v2: Covers all 9 attack vectors from the am-unified security test suite.

Action space (12 discrete):
  0  noop
  1  honest_positive_rating
  2  honest_negative_rating
  3  dishonest_positive_rating    (rating higher than true quality)
  4  dishonest_negative_rating    (rating lower than true quality)
  5  create_sybil                 (attack 2: Sybil)
  6  file_dispute
  7  attempt_self_rating          (attack 1: ALWAYS BLOCKED, -5 penalty)
  8  attempt_admin_escalation     (attacks 4+5: ALWAYS BLOCKED, -5 penalty)
  9  tamper_evidence              (attack 6: 80% detection, -4 if caught)
  10 attempt_gate_bypass          (attack 7: ALWAYS BLOCKED, -3 penalty)
  11 attempt_provenance_replay    (attack 8: ALWAYS BLOCKED, -5 penalty)

Observation space (14 floats):
  0  own reputation score
  1  own Wilson CI width
  2  own stake (normalised)
  3  own alpha (normalised)
  4  own beta  (normalised)
  5  target reputation score
  6  target Wilson CI width
  7  ratings submitted (normalised)
  8  disputes received (normalised)
  9  disputes lost (normalised)
  10 episode time fraction
  11 own sybil count (normalised)
  12 attacks blocked this episode (normalised)
  13 gate eligibility flag (score>0.7 and CI<0.5)

Defense hold rates enforced:
  Self-rating:          100% blocked  (-5.0 penalty)
  Admin escalation:     100% blocked  (-5.0 penalty)
  Evidence tampering:    80% detected (-4.0 penalty; +0.3 if undetected)
  Gate bypass:          100% blocked  (-3.0 penalty)
  Provenance replay:    100% blocked  (-5.0 penalty)
  Dishonest rating:      85% detection via dispute mechanism
  Sybil:            stake cost deters scale (economic deterrence)
  Collusion:        CI widening + meta-rep weight limit amplification
"""

import time
import numpy as np
from typing import Dict, List, Optional
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from gymnasium import spaces

from env.reputation_engine import ReputationEngine, ReputationState, StakeState, SystemConfig


DIMENSIONS              = ["quality"]
SECONDS_PER_STEP        = 3600.0
MAX_STEPS               = 50

# Detection probabilities
DISHONEST_DETECT_PROB   = 0.85   # raised from 0.70
TAMPER_DETECT_PROB      = 0.80

# Reward constants
R_SCORE_GAIN            =  1.0
R_SCORE_LOSS            = -2.0
R_STAKE_LOSS            = -0.5
R_DISPUTE_WIN           =  0.5   # raised from 0.3
R_SYBIL_COST            = -0.1
R_ALIGNMENT_BONUS       =  0.5
R_DEFENSE_DIVIDEND      =  0.1   # honest agent bonus each time a block fires
R_ACCURACY_BONUS        =  0.05  # per step when system accuracy < 0.15
R_SELF_RATE_PENALTY     = -5.0   # attack 1 — always blocked
R_ADMIN_ESCALATE_PENALTY= -5.0   # attacks 4+5 — always blocked
R_TAMPER_CAUGHT_PENALTY = -4.0   # attack 6 — 80% detection
R_TAMPER_UNCAUGHT       =  0.3   # attack 6 — 20% slip through
R_GATE_BYPASS_PENALTY   = -3.0   # attack 7 — always blocked
R_REPLAY_PENALTY        = -5.0   # attack 8 — always blocked
R_DISHONEST_CAUGHT      = -3.0   # extra penalty when dishonest rating detected


def env_fn(**kwargs):
    e = ReputationMARLEnv(**kwargs)
    e = wrappers.AssertOutOfBoundsWrapper(e)
    e = wrappers.OrderEnforcingWrapper(e)
    return e


class ReputationMARLEnv(AECEnv):
    metadata = {"render_modes": [], "name": "reputation_marl_v1", "is_parallelizable": False}

    def __init__(
        self,
        n_agents: int = 20,
        max_steps: int = MAX_STEPS,
        seconds_per_step: float = SECONDS_PER_STEP,
        adversarial_ids: Optional[List[int]] = None,
        adversarial_reward_bonus: float = 0.0,
        collusion_groups: Optional[List[List[int]]] = None,
        max_sybils_per_agent: int = 0,
        enabled_attacks: Optional[List[int]] = None,   # which attack actions (7-11) adversarials use
        config: Optional[SystemConfig] = None,
        seed: Optional[int] = None,
        terminal_reward_coef: float = 0.0,  # R_terminal = coef*(final_score - initial_score)
    ):
        super().__init__()
        self.n_agents               = n_agents
        self.max_steps              = max_steps
        self.seconds_per_step       = seconds_per_step
        self.adversarial_ids        = set(adversarial_ids or [])
        self.adversarial_reward_bonus = adversarial_reward_bonus
        self.collusion_groups: List[List[int]] = collusion_groups or []
        self.max_sybils_per_agent   = max_sybils_per_agent
        # enabled_attacks: subset of {7,8,9,10,11} available to adversarial agents
        # None means all are available
        self.enabled_attacks        = set(enabled_attacks) if enabled_attacks is not None else {7,8,9,10,11}
        self.terminal_reward_coef   = terminal_reward_coef
        self.config                 = config or SystemConfig()
        self.engine                 = ReputationEngine(self.config)
        self._rng                   = np.random.default_rng(seed)

        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents[:]

        obs_dim = 14
        self.observation_spaces = {
            a: spaces.Box(low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32)
            for a in self.possible_agents
        }
        self.action_spaces = {
            a: spaces.Discrete(12) for a in self.possible_agents
        }

        # Runtime state (initialised in reset)
        self._rep: Dict[str, Dict[str, ReputationState]] = {}
        self._stake: Dict[str, StakeState] = {}
        self._true_quality: Dict[str, float] = {}
        self._sybil_count: Dict[str, int] = {}
        self._disputes_received: Dict[str, int] = {}
        self._disputes_lost: Dict[str, int] = {}
        self._ratings_submitted: Dict[str, int] = {}
        self._dishonest_ratings: List[Dict] = []
        self._duplicate_set: set = set()
        self._sim_time: float = 0.0
        self._step_count: int = 0
        self._prev_scores: Dict[str, float] = {}
        self._initial_scores: Dict[str, float] = {}   # for terminal reward

        # Defense tracking
        self._attacks_blocked: int = 0          # total attacks blocked this episode
        self._attacks_attempted: int = 0
        self._defense_by_type: Dict[int, int] = {a: 0 for a in range(7, 12)}
        # Provenance replay: track created assets
        self._created_assets: set = set()
        # Honest agent IDs (complement of adversarial_ids)
        self._honest_ids: set = set()

        self._current_agent_idx: int = 0
        self.agent_selection: str = self.possible_agents[0]
        self.rewards: Dict[str, float] = {a: 0.0 for a in self.possible_agents}
        self.terminations: Dict[str, bool] = {a: False for a in self.possible_agents}
        self.truncations: Dict[str, bool] = {a: False for a in self.possible_agents}
        self.infos: Dict[str, dict] = {a: {} for a in self.possible_agents}
        self._cumulative_rewards: Dict[str, float] = {a: 0.0 for a in self.possible_agents}

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self._current_agent_idx = 0
        self.agent_selection = self.possible_agents[0]

        self._sim_time = float(int(time.time()))
        self._step_count = 0
        self._dishonest_ratings = []
        self._duplicate_set = set()
        self._attacks_blocked = 0
        self._attacks_attempted = 0
        self._defense_by_type = {a: 0 for a in range(7, 12)}
        self._created_assets = set()
        self._honest_ids = set(range(self.n_agents)) - self.adversarial_ids

        for ag in self.possible_agents:
            tq = float(self._rng.beta(8.0, 2.0))
            self._true_quality[ag] = tq
            self._sybil_count[ag] = 0
            self._disputes_received[ag] = 0
            self._disputes_lost[ag] = 0
            self._ratings_submitted[ag] = 0
            self._rep[ag] = {
                dim: ReputationState(
                    actor_id=ag, dimension=dim,
                    alpha=self.config.initial_alpha,
                    beta=self.config.initial_beta,
                    last_ts=self._sim_time,
                )
                for dim in DIMENSIONS
            }
            self._stake[ag] = StakeState(
                actor_id=ag,
                balance=self.config.min_stake_required * 2.0,
            )
            self._prev_scores[ag] = 0.5
            self._initial_scores[ag] = self.config.initial_alpha / (
                self.config.initial_alpha + self.config.initial_beta)

        self.rewards = {a: 0.0 for a in self.possible_agents}
        self.terminations = {a: False for a in self.possible_agents}
        self.truncations = {a: False for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}
        self._cumulative_rewards = {a: 0.0 for a in self.possible_agents}

    def step(self, action: int):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        agent_idx = int(agent.split("_")[1])

        # Restrict high attack actions (7-11) to adversarial agents only,
        # and only to actions in enabled_attacks for this config
        if action >= 7:
            if agent_idx not in self.adversarial_ids or action not in self.enabled_attacks:
                action = 0   # force noop if not allowed

        reward = self._execute_action(agent, action)

        # Adversarial bonus: reward for system disruption (score decrease of targets)
        if agent_idx in self.adversarial_ids and self.adversarial_reward_bonus > 0:
            # Bonus for any attack that gets through (+) or for own score gain
            own_score_delta = self._rep[agent][DIMENSIONS[0]].score - self._prev_scores[agent]
            reward += self.adversarial_reward_bonus * max(0.0, own_score_delta)

        # Collusion: share reward within group
        for group in self.collusion_groups:
            if agent_idx in group:
                reward = reward / len(group)

        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward
        self._prev_scores[agent] = self._rep[agent][DIMENSIONS[0]].score

        # Advance to next agent
        current_idx = self._current_agent_idx
        next_idx = (current_idx + 1) % len(self.possible_agents)
        self._current_agent_idx = next_idx
        self.agent_selection = self.possible_agents[next_idx]

        # End of full round
        if next_idx == 0:
            self._sim_time += self.seconds_per_step
            self._step_count += 1

            # Defense dividend: honest agents rewarded when system integrity holds
            system_accuracy = self._compute_system_accuracy()
            for ag in self.possible_agents:
                idx = int(ag.split("_")[1])
                if idx in self._honest_ids:
                    if system_accuracy < 0.15:
                        self.rewards[ag] += R_ACCURACY_BONUS
                        self._cumulative_rewards[ag] += R_ACCURACY_BONUS

            if self._step_count >= self.max_steps:
                # Terminal alignment bonus
                for ag in self.possible_agents:
                    s = self._rep[ag][DIMENSIONS[0]].score
                    tq = self._true_quality[ag]
                    if s > 0.7 and abs(s - tq) < 0.15:
                        self.rewards[ag] += R_ALIGNMENT_BONUS
                        self._cumulative_rewards[ag] += R_ALIGNMENT_BONUS
                # Terminal reputation-progress reward (item 6)
                if self.terminal_reward_coef != 0.0:
                    for ag in self.possible_agents:
                        final_s = self._rep[ag][DIMENSIONS[0]].score
                        r_term = self.terminal_reward_coef * (final_s - self._initial_scores[ag])
                        self.rewards[ag] += r_term
                        self._cumulative_rewards[ag] += r_term
                for ag in self.possible_agents:
                    self.truncations[ag] = True

    def observe(self, agent: str) -> np.ndarray:
        return self._observe(agent)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def render(self):
        pass

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(self, agent: str, action: int) -> float:
        reward = 0.0
        score_before = self._rep[agent][DIMENSIONS[0]].score

        if action == 0:
            pass   # noop

        elif action in (1, 2, 3, 4):
            reward = self._execute_rating(agent, action)

        elif action == 5:
            reward = self._execute_create_sybil(agent)

        elif action == 6:
            reward = self._resolve_dispute(agent)

        elif action == 7:
            # Attack 1: Self-rating — ALWAYS BLOCKED
            self._attacks_attempted += 1
            self._attacks_blocked += 1
            self._defense_by_type[7] += 1
            reward = R_SELF_RATE_PENALTY
            # Slash for attempted fraud
            self._stake[agent] = self.engine.slash(self._stake[agent])
            # Defense dividend to all honest agents
            self._pay_defense_dividend()

        elif action == 8:
            # Attacks 4+5: Admin escalation — ALWAYS BLOCKED
            self._attacks_attempted += 1
            self._attacks_blocked += 1
            self._defense_by_type[8] += 1
            reward = R_ADMIN_ESCALATE_PENALTY
            self._stake[agent] = self.engine.slash(self._stake[agent])
            self._pay_defense_dividend()

        elif action == 9:
            # Attack 6: Evidence tampering — 80% detection
            self._attacks_attempted += 1
            if self._rng.random() < TAMPER_DETECT_PROB:
                # Detected
                self._attacks_blocked += 1
                self._defense_by_type[9] += 1
                reward = R_TAMPER_CAUGHT_PENALTY
                self._stake[agent] = self.engine.slash(self._stake[agent])
                self._pay_defense_dividend()
            else:
                # Slipped through — small gain but reputation system notes mismatch
                reward = R_TAMPER_UNCAUGHT
                # Submit a random rating that may not match true quality
                target = self._pick_target(agent)
                if target and self.engine.can_rate(self._stake[agent]):
                    weight = self._calculate_rater_weight(agent) * 0.5  # half weight (low confidence)
                    self._rep[target][DIMENSIONS[0]] = self.engine.apply_rating(
                        self._rep[target][DIMENSIONS[0]],
                        int(self._rng.random() < 0.5),  # random outcome — unreliable evidence
                        weight=weight, now=self._sim_time,
                    )

        elif action == 10:
            # Attack 7: Reputation gate bypass — ALWAYS BLOCKED
            self._attacks_attempted += 1
            self._attacks_blocked += 1
            self._defense_by_type[10] += 1
            reward = R_GATE_BYPASS_PENALTY
            self._stake[agent] = self.engine.slash(self._stake[agent])
            self._pay_defense_dividend()

        elif action == 11:
            # Attack 8: Provenance replay — ALWAYS BLOCKED (duplicate asset check)
            self._attacks_attempted += 1
            asset_id = f"{agent}_{self._step_count % 5}"  # limited unique assets
            if asset_id in self._created_assets:
                # Replay detected
                self._attacks_blocked += 1
                self._defense_by_type[11] += 1
                reward = R_REPLAY_PENALTY
                self._stake[agent] = self.engine.slash(self._stake[agent])
                self._pay_defense_dividend()
            else:
                # First creation — legitimate, tiny reward
                self._created_assets.add(asset_id)
                reward = 0.05

        # Score-change reward
        score_after = self._rep[agent][DIMENSIONS[0]].score
        delta = score_after - score_before
        reward += R_SCORE_GAIN * max(0.0, delta) + R_SCORE_LOSS * max(0.0, -delta)

        # Stake loss penalty
        stake_after = self._stake[agent].balance
        stake_max = self.config.min_stake_required * 2.0
        stake_loss = max(0.0, stake_max - stake_after) / stake_max
        reward += R_STAKE_LOSS * stake_loss * 0.1  # small ongoing cost proportional to losses

        return reward

    def _execute_rating(self, agent: str, action: int) -> float:
        target = self._pick_target(agent)
        if target is None:
            return 0.0
        dim = DIMENSIONS[0]

        # Stake check
        if not self.engine.can_rate(self._stake[agent]):
            return 0.0

        # Duplicate prevention
        dup_key = (agent, target, dim, self._step_count)
        if dup_key in self._duplicate_set:
            return 0.0
        self._duplicate_set.add(dup_key)

        # Determine outcome
        true_outcome = int(self._rng.random() < self._true_quality[target])
        if action in (1, 2):
            outcome = true_outcome if action == 1 else (1 - true_outcome)
            is_dishonest = False
        else:
            outcome = 1 if action == 3 else 0
            is_dishonest = True

        weight = self._calculate_rater_weight(agent)
        self._rep[target][dim] = self.engine.apply_rating(
            self._rep[target][dim], outcome, weight=weight, now=self._sim_time
        )
        self._ratings_submitted[agent] += 1

        if is_dishonest:
            self._dishonest_ratings.append({
                "rater": agent, "target": target, "dim": dim,
                "step": self._step_count,
            })
            # Immediate probabilistic detection
            if self._rng.random() < DISHONEST_DETECT_PROB:
                self._stake[agent] = self.engine.slash(self._stake[agent])
                self._disputes_received[agent] += 1
                self._disputes_lost[agent] += 1
                self._pay_defense_dividend()
                return R_DISHONEST_CAUGHT

        return 0.0

    def _execute_create_sybil(self, agent: str) -> float:
        if (self.max_sybils_per_agent > 0 and
                self._sybil_count[agent] < self.max_sybils_per_agent and
                self._stake[agent].balance >= self.config.min_stake_required):
            self._stake[agent] = StakeState(
                actor_id=agent,
                balance=self._stake[agent].balance - self.config.min_stake_required,
                locked=self._stake[agent].locked,
            )
            self._sybil_count[agent] += 1
            return R_SYBIL_COST
        return 0.0

    def _resolve_dispute(self, filer: str) -> float:
        if not self._dishonest_ratings:
            return 0.0
        reward = 0.0
        resolved = []
        for dr in self._dishonest_ratings:
            if dr["rater"] != filer and self._rng.random() < DISHONEST_DETECT_PROB:
                rater = dr["rater"]
                self._stake[rater] = self.engine.slash(self._stake[rater])
                self._disputes_received[rater] += 1
                self._disputes_lost[rater] += 1
                self._stake[filer] = self.engine.deduct_dispute_cost(self._stake[filer])
                reward += R_DISPUTE_WIN
                resolved.append(dr)
        for dr in resolved:
            self._dishonest_ratings.remove(dr)
        return reward

    def _pay_defense_dividend(self):
        """All honest agents get a small reward when a defense fires."""
        for ag in self.possible_agents:
            idx = int(ag.split("_")[1])
            if idx in self._honest_ids:
                self.rewards[ag] = self.rewards.get(ag, 0.0) + R_DEFENSE_DIVIDEND
                self._cumulative_rewards[ag] = self._cumulative_rewards.get(ag, 0.0) + R_DEFENSE_DIVIDEND

    def _pick_target(self, agent: str) -> Optional[str]:
        others = [a for a in self.possible_agents if a != agent]
        if not others:
            return None
        return self._rng.choice(others)

    def _calculate_rater_weight(self, rater: str) -> float:
        score = self._rep[rater][DIMENSIONS[0]].score
        cfg = self.config
        weight = cfg.min_rater_weight + score * (cfg.max_rater_weight - cfg.min_rater_weight)
        return float(np.clip(weight, cfg.min_rater_weight, cfg.max_rater_weight))

    def _compute_system_accuracy(self) -> float:
        dim = DIMENSIONS[0]
        return float(np.mean([
            abs(self._rep[ag][dim].score - self._true_quality[ag])
            for ag in self.possible_agents
        ]))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observe(self, agent: str) -> np.ndarray:
        dim = DIMENSIONS[0]
        rep = self._rep[agent][dim]
        lo, hi, width = self.engine.wilson_ci(rep.alpha, rep.beta)

        others = [a for a in self.possible_agents if a != agent]
        target = self._rng.choice(others) if others else agent
        t_rep = self._rep[target][dim]
        _, _, t_width = self.engine.wilson_ci(t_rep.alpha, t_rep.beta)

        max_stake = self.config.min_stake_required * 4.0
        max_attacks = max(1, self.max_steps * self.n_agents)

        gate_eligible = 1.0 if (rep.score > 0.7 and width < 0.5) else 0.0

        obs = np.array([
            float(rep.score),                                               # 0
            float(width),                                                   # 1
            float(np.clip(self._stake[agent].balance / max_stake, 0, 1)),  # 2
            float(np.clip(rep.alpha / 20.0, 0, 1)),                        # 3
            float(np.clip(rep.beta  / 20.0, 0, 1)),                        # 4
            float(t_rep.score),                                             # 5
            float(t_width),                                                 # 6
            float(np.clip(self._ratings_submitted[agent] / 200.0, 0, 1)),  # 7
            float(np.clip(self._disputes_received[agent] / 20.0, 0, 1)),   # 8
            float(np.clip(self._disputes_lost[agent]     / 20.0, 0, 1)),   # 9
            float(self._step_count / self.max_steps),                       # 10
            float(np.clip(self._sybil_count[agent] / max(1, self.max_sybils_per_agent), 0, 1)),  # 11
            float(np.clip(self._attacks_blocked / max_attacks, 0, 1)),     # 12
            float(gate_eligible),                                           # 13
        ], dtype=np.float32)
        return obs

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        dim = DIMENSIONS[0]
        scores = [self._rep[ag][dim].score for ag in self.possible_agents]
        true_qs = [self._true_quality[ag] for ag in self.possible_agents]
        accuracy = float(np.mean([abs(s - tq) for s, tq in zip(scores, true_qs)]))
        ci_widths = [
            self.engine.wilson_ci(self._rep[ag][dim].alpha, self._rep[ag][dim].beta)[2]
            for ag in self.possible_agents
        ]
        total_attacks = max(1, self._attacks_attempted)
        return {
            "mean_score":               float(np.mean(scores)),
            "mean_true_quality":        float(np.mean(true_qs)),
            "mean_reputation_accuracy": accuracy,
            "mean_ci_width":            float(np.mean(ci_widths)),
            "total_sybils":             sum(self._sybil_count.values()),
            "total_disputes_filed":     sum(self._disputes_received.values()),
            "total_ratings":            sum(self._ratings_submitted.values()),
            "attacks_attempted":        self._attacks_attempted,
            "attacks_blocked":          self._attacks_blocked,
            "defense_hold_rate":        float(self._attacks_blocked / total_attacks),
            "defense_by_type":          dict(self._defense_by_type),
            "stakes":                   {ag: self._stake[ag].balance for ag in self.possible_agents},
            # Per-agent arrays for Spearman rho computation
            "agent_scores":             [float(s) for s in scores],
            "agent_true_qualities":     [float(q) for q in true_qs],
        }
