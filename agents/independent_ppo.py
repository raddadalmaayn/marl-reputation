"""
Independent PPO — each agent has its own policy network (no parameter sharing).
Identical architecture to MAPPO; used as comparison baseline.
"""

import torch
import numpy as np
from typing import Dict, List
from agents.mappo import SharedActorCritic, RolloutBuffer


class IndependentPPOTrainer:
    def __init__(
        self,
        agent_ids: List[str],
        obs_dim: int = 14,
        act_dim: int = 12,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
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

        self.policies: Dict[str, SharedActorCritic] = {
            ag: SharedActorCritic(obs_dim, act_dim).to(self.device)
            for ag in agent_ids
        }
        self.optimizers = {
            ag: torch.optim.Adam(self.policies[ag].parameters(), lr=lr)
            for ag in agent_ids
        }
        self.buffers: Dict[str, RolloutBuffer] = {
            ag: RolloutBuffer() for ag in agent_ids
        }

    def select_action(self, agent: str, obs: np.ndarray, deterministic: bool = False):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, logprob, _, value = self.policies[agent].get_action(obs_t, deterministic)
        return int(action.item()), float(logprob.item()), float(value.item())

    def store_transition(self, agent, obs, action, logprob, reward, value, done):
        self.buffers[agent].store(obs, action, logprob, reward, value, done)

    def update(self, lr_multiplier: float = 1.0) -> Dict[str, float]:
        losses = {}
        for ag, buf in self.buffers.items():
            if len(buf) == 0:
                continue
            losses[ag] = self._update_agent(ag, buf, lr_multiplier)
            buf.clear()
        if not losses:
            return {}
        return {k: float(np.mean([l[k] for l in losses.values() if k in l]))
                for k in next(iter(losses.values()))}

    def _update_agent(self, agent: str, buf: RolloutBuffer, lr_multiplier: float):
        import torch.nn.functional as F
        from torch.distributions import Categorical

        T = len(buf)
        returns, advantages = [], []
        gae, nv = 0.0, 0.0
        for t in reversed(range(T)):
            mask = 1.0 - float(buf.dones[t])
            d = buf.rewards[t] + self.gamma * nv * mask - buf.values[t]
            gae = d + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + buf.values[t])
            nv = buf.values[t]

        obs_t  = torch.FloatTensor(np.array(buf.obs)).to(self.device)
        acts_t = torch.LongTensor(buf.actions).to(self.device)
        lps_t  = torch.FloatTensor(buf.logprobs).to(self.device)
        rets_t = torch.FloatTensor(returns).to(self.device)
        advs_t = torch.FloatTensor(advantages).to(self.device)
        advs_t = (advs_t - advs_t.mean()) / (advs_t.std() + 1e-8)

        for g in self.optimizers[agent].param_groups:
            g["lr"] = 3e-4 * lr_multiplier

        total_loss = 0.0
        n = T
        for _ in range(self.n_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size]
                logits, values = self.policies[agent](obs_t[mb])
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(acts_t[mb])
                ratio = (new_lp - lps_t[mb]).exp()
                adv = advs_t[mb]
                pg = -torch.min(ratio*adv, torch.clamp(ratio,1-self.clip_ratio,1+self.clip_ratio)*adv).mean()
                vf = F.mse_loss(values, rets_t[mb])
                loss = pg + self.vf_coef*vf - self.entropy_coef*dist.entropy().mean()
                self.optimizers[agent].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policies[agent].parameters(), self.max_grad_norm)
                self.optimizers[agent].step()
                total_loss += loss.item()
        return {"total_loss": total_loss}

    def save(self, path: str):
        torch.save({ag: self.policies[ag].state_dict() for ag in self.policies}, path)

    def load(self, path: str):
        states = torch.load(path, map_location=self.device)
        for ag, sd in states.items():
            if ag in self.policies:
                self.policies[ag].load_state_dict(sd)
