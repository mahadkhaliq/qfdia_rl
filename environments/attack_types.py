"""
environments/attack_types.py
============================
Baseline (non-RL) FDIA generators used for comparison against the Q-NPG-FDIA
policy and for seeding the QGrid-Synth dataset.

The GridSTAGE-style rule-based attacks (ramp, step, random, ...) are Python
re-implementations of the PNNL GridSTAGE attack families; liu_stealthy is the
exact a = H c construction (perfectly BDD-stealthy by Liu-Ning-Reiter).

These are intentionally simple. Mahad's *contribution* is the rl_learned
attacker (Q-NPG-FDIA); these baselines exist so the paper's Table I has
non-quantum rows to beat.
"""
from __future__ import annotations
import numpy as np


class FDIAAttackFactory:
    def __init__(self, env, seed: int = 0):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.m, self.n = env.m, env.n
        self.eps = env.cfg.epsilon

    def _target_meters(self, k: int) -> np.ndarray:
        return self.rng.choice(self.m, size=min(k, self.m), replace=False)

    def ramp(self, amp=None, frac=0.1):
        a = np.zeros(self.m); idx = self._target_meters(int(frac * self.m))
        a[idx] = (amp if amp is not None else self.eps)
        return a

    def step_attack(self, amp=None, frac=0.1):
        a = np.zeros(self.m); idx = self._target_meters(int(frac * self.m))
        a[idx] = (amp if amp is not None else self.eps)
        return a

    def random(self, frac=0.1):
        a = np.zeros(self.m); idx = self._target_meters(int(frac * self.m))
        a[idx] = self.rng.uniform(-self.eps, self.eps, size=idx.size)
        return a

    def trapezoidal(self, amp=None, frac=0.1):
        return self.ramp(amp, frac)  # final-state plateau value

    def multiplicative(self, scale=1.05):
        return (scale - 1.0) * self.env._z0

    def replay(self):
        # inject a previously seen (jittered) clean snapshot as the "attack" delta
        z_old = self.env._h(self.env.x_star + self.rng.normal(0, 0.02, self.env.x_star.shape))
        return z_old - self.env._z0

    def freeze(self):
        return np.zeros(self.m)  # stale reading == no change relative to a held z0

    def coordinated_sparse(self, k=4, amp=None):
        a = np.zeros(self.m); idx = self._target_meters(k)
        a[idx] = (amp if amp is not None else self.eps) * self.rng.choice([-1, 1], size=idx.size)
        return a

    def liu_stealthy(self, scale=0.02):
        c = self.rng.normal(0, scale, 2 * self.n)
        return self.env.liu_stealthy_attack(c)

    def sample(self, kind: str):
        return {
            "ramp": self.ramp, "step": self.step_attack, "random": self.random,
            "trapezoidal": self.trapezoidal, "multiplicative": self.multiplicative,
            "replay": self.replay, "freeze": self.freeze,
            "coordinated_sparse": self.coordinated_sparse, "liu_stealthy": self.liu_stealthy,
        }[kind]()

    ALL_TYPES = ["ramp", "step", "random", "trapezoidal", "multiplicative",
                 "replay", "freeze", "coordinated_sparse", "liu_stealthy"]
