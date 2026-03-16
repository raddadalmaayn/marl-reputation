"""
MAPPO — Multi-Agent PPO with shared policy network.
Architecture: MLP 256→128→ReLU, shared actor+critic heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional


class SharedActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, int] = (256, 128)):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden[1], act_dim)
        self.critic = nn.Linear(hidden[1], 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


class RolloutBuffer:
    def __init__(self):
        self.obs:      List[np.ndarray] = []
        self.actions:  List[int]        = []
        self.logprobs: List[float]      = []
        self.rewards:  List[float]      = []
        self.values:   List[float]      = []
        self.dones:    List[bool]       = []

    def store(self, obs, action, logprob, reward, value, done):
        self.obs.append(obs)
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


class MAPPOTrainer:
    """
    Shared-policy MAPPO. All agents share one network; each agent's
    observation is fed independently.
    """

    def __init__(
        self,
        obs_dim: int = 14,
        act_dim: int = 12,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.05,   # higher entropy → more exploration of all 12 actions
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        minibatch_size: int = 64,
        device: str = "cpu",
    ):
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_ratio    = clip_ratio
        self.entropy_coef  = entropy_coef
        self.vf_coef       = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs      = n_epochs
        self.minibatch_size = minibatch_size
        self.device        = torch.device(device)

        self.policy = SharedActorCritic(obs_dim, act_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.buffers: Dict[str, RolloutBuffer] = {}
        self.total_updates = 0

    def init_agents(self, agent_ids: List[str]):
        self.buffers = {ag: RolloutBuffer() for ag in agent_ids}

    def select_action(self, agent: str, obs: np.ndarray, deterministic: bool = False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, logprob, _, value = self.policy.get_action(obs_t, deterministic)
        return int(action.item()), float(logprob.item()), float(value.item())

    def store_transition(self, agent: str, obs, action, logprob, reward, value, done):
        self.buffers[agent].store(obs, action, logprob, reward, value, done)

    def update(self, lr_multiplier: float = 1.0) -> Dict[str, float]:
        """Combine all agent buffers and run PPO update."""
        all_obs, all_acts, all_lps, all_rets, all_advs = [], [], [], [], []

        for buf in self.buffers.values():
            if len(buf) == 0:
                continue
            returns, advantages = self._compute_gae(buf)
            all_obs.extend(buf.obs)
            all_acts.extend(buf.actions)
            all_lps.extend(buf.logprobs)
            all_rets.extend(returns)
            all_advs.extend(advantages)
            buf.clear()

        if not all_obs:
            return {}

        obs_t   = torch.FloatTensor(np.array(all_obs)).to(self.device)
        acts_t  = torch.LongTensor(all_acts).to(self.device)
        lps_t   = torch.FloatTensor(all_lps).to(self.device)
        rets_t  = torch.FloatTensor(all_rets).to(self.device)
        advs_t  = torch.FloatTensor(all_advs).to(self.device)
        advs_t  = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        # Adjust LR
        for g in self.optimizer.param_groups:
            g["lr"] = 3e-4 * lr_multiplier

        total_loss = pg_loss = vf_loss = ent_loss = 0.0
        n = len(all_obs)

        for _ in range(self.n_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                logits, values = self.policy(obs_t[mb])
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(acts_t[mb])
                entropy = dist.entropy().mean()

                ratio = (new_lp - lps_t[mb]).exp()
                adv   = advs_t[mb]
                pg1   = ratio * adv
                pg2   = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
                pg    = -torch.min(pg1, pg2).mean()
                vf    = F.mse_loss(values, rets_t[mb])
                loss  = pg + self.vf_coef * vf - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                pg_loss += pg.item()
                vf_loss += vf.item()
                ent_loss += entropy.item()

        self.total_updates += 1
        n_updates = max(1, self.n_epochs * max(1, n // self.minibatch_size))
        return {
            "total_loss": total_loss / n_updates,
            "pg_loss":    pg_loss    / n_updates,
            "vf_loss":    vf_loss    / n_updates,
            "entropy":    ent_loss   / n_updates,
        }

    def _compute_gae(self, buf: RolloutBuffer):
        rewards = buf.rewards
        values  = buf.values
        dones   = buf.dones
        T = len(rewards)

        returns = [0.0] * T
        advantages = [0.0] * T
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae   = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t]    = gae + values[t]
            next_value    = values[t]

        return returns, advantages

    def save(self, path: str):
        torch.save({
            "policy_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "total_updates": self.total_updates,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.total_updates = ckpt.get("total_updates", 0)
