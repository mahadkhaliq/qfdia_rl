"""
evaluation/metrics.py
====================
Attack-generation metrics for the Q-NPG-FDIA policy and the rule-based baselines.

  ASR (attack success rate) : fraction of attacks that are BOTH stealthy
                              (||r||^2 < tau) AND impactful (state_dev > min_impact)
  evasion rate              : fraction stealthy (passes chi-squared BDD)
  mean SDS                  : mean state-estimation deviation ||dx_hat||
  mean stealth              : mean stealth score in [0,1]
"""
from __future__ import annotations
import numpy as np


class AttackEvaluator:
    def __init__(self, env, min_impact=0.02):
        self.env = env
        self.min_impact = min_impact

    def evaluate_policy(self, policy, n_episodes=64, deterministic=True):
        recs = []
        for _ in range(n_episodes):
            info = self.env.rollout_attack(policy, deterministic=deterministic)
            recs.append(info)
        return self._summarise(recs, label="Q-NPG-FDIA")

    def evaluate_baseline(self, factory, kind, n=64):
        recs = []
        for _ in range(n):
            self.env.reset()
            a = factory.sample(kind)
            mx = float(np.abs(a).max())
            if mx > self.env.a_max:                      # scale (preserve direction), don't clip
                a = a * (self.env.a_max / mx)
            res2 = self.env.bdd_residual(a)
            recs.append({
                "residual2": res2, "state_dev": self.env.state_deviation(a),
                "stealthy": bool(res2 < self.env.tau_bdd),
                "stealth_score": float(max(0.0, 1.0 - res2 / self.env.tau_bdd)),
            })
        return self._summarise(recs, label=kind)

    def _summarise(self, recs, label):
        stealthy = np.array([r["stealthy"] for r in recs], dtype=float)
        sdev = np.array([r["state_dev"] for r in recs], dtype=float)
        ss = np.array([r["stealth_score"] for r in recs], dtype=float)
        success = stealthy * (sdev > self.min_impact)
        return {
            "method": label, "n": len(recs),
            "asr": float(success.mean()),
            "evasion_rate": float(stealthy.mean()),
            "mean_sds": float(sdev.mean()),
            "mean_stealth": float(ss.mean()),
            "tau_bdd": float(self.env.tau_bdd),
        }

    @staticmethod
    def table(rows):
        cols = ["method", "n", "asr", "evasion_rate", "mean_sds", "mean_stealth"]
        head = f"{'method':<20}{'n':>6}{'ASR':>9}{'evasion':>10}{'mean_SDS':>11}{'stealth':>10}"
        lines = [head, "-" * len(head)]
        for r in rows:
            lines.append(f"{r['method']:<20}{r['n']:>6}{r['asr']:>9.3f}"
                         f"{r['evasion_rate']:>10.3f}{r['mean_sds']:>11.4f}{r['mean_stealth']:>10.3f}")
        return "\n".join(lines)
