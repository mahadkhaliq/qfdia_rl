"""
config.py
=========
Centralised configuration for Q-NPG-FDIA (Mahad's task within QT-FDIA-RL).

Two dataclasses:
  * BusConfig      -- per-bus-size topology / VQC / reward parameters
  * TrainingConfig -- global natural-policy-gradient hyperparameters

Reward weights follow Section 12 of the Theoretical Framework. VQC sizes and
epsilon/lambda/mu follow Section 3.2 of the Code Documentation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class BusConfig:
    bus_size: int                 # 30, 57, or 118
    n_qubits: int                 # VQC width for the policy
    vqc_layers: int               # StronglyEntanglingLayers depth
    epsilon: float                # per-step attack increment bound (p.u.)
    a_max_mult: float = 6.0       # cumulative attack bound = a_max_mult * epsilon
    # --- compound reward weights (Eq. 12.1 / Section 12 table) ---
    w_impact: float = 1.0
    w_stealth: float = 3.0
    w_physics: float = 1.0
    w_adv: float = 0.0            # phi: 0 in Phase 1 (detector inactive), set in Q-ASP phase
    w_time: float = 0.05          # beta: survival bonus per non-terminal step
    stealth_bonus: float = 2.0    # reward cliff for being under the chi-squared threshold
    # --- reward magnitude scaling so the five terms are comparable ---
    impact_scale: float = 10.0    # state-deviation is small (~0.01-0.1 p.u.); scale up
    phys_scale: float = 1.0
    horizon: int = 4              # steps per episode (non-accumulating attacks on a drifting grid)
    kappa_detect: float = 6.0     # (unused in non-accumulating mode; kept for reference)


# Per-bus presets. n_qubits = ceil(log2 m) rounded to a practical width.
BUS_CONFIGS: Dict[int, BusConfig] = {
    30:  BusConfig(bus_size=30,  n_qubits=4, vqc_layers=3, epsilon=0.05,
                   w_stealth=3.0, w_physics=1.0),
    57:  BusConfig(bus_size=57,  n_qubits=6, vqc_layers=4, epsilon=0.03,
                   w_stealth=3.5, w_physics=1.5),
    118: BusConfig(bus_size=118, n_qubits=8, vqc_layers=4, epsilon=0.02,
                   w_stealth=4.0, w_physics=2.0),
}


@dataclass
class TrainingConfig:
    # ---- rollout ----
    total_updates: int = 200
    steps_per_update: int = 1024     # transitions collected before each NPG step
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # ---- natural-gradient step (Algorithm 4.1) ----
    nat_lr: float = 0.30             # base scale on the natural-gradient direction
    kl_trust: float = 0.02           # KL trust-region bound (epsilon_KL); caps the step
    damping: float = 1e-2            # Tikhonov damping delta added to F_FDIA
    mu_phys: float = 1.0             # physics-augmentation weight in F_FDIA (Def. 4.1)
    qfim_batch: int = 16             # # states the QFIM is averaged over per update
    grad_batch: int = 256            # # transitions used for the policy-gradient estimate

    # ---- classical-parameter optimiser (encoder / heads / critic via Adam) ----
    adam_lr: float = 3e-3
    value_coef: float = 0.5
    entropy_coef: float = 1e-3
    max_grad_norm: float = 1.0

    # ---- policy noise (sized to the stealthy slab; std is a fraction of the bound) ----
    init_log_std: float = -3.4       # std ~= 0.033 * bound  (exploration stays ~stealthy)
    min_log_std: float = -6.0
    max_log_std: float = -2.8

    # ---- bookkeeping ----
    seed: int = 42
    device_name: str = "default.qubit"   # falls back automatically if lightning absent
    eval_episodes: int = 64
    min_impact: float = 0.02         # state-dev threshold for an attack to "succeed" (ASR)
    log_every: int = 5
    out_dir: str = "outputs"


# Convenience: "quick" overrides for a ~10-minute smoke test on a laptop / login node.
def quick_overrides(tc: TrainingConfig) -> TrainingConfig:
    tc.total_updates = 30
    tc.steps_per_update = 256
    tc.qfim_batch = 6
    tc.grad_batch = 96
    tc.eval_episodes = 24
    tc.log_every = 2
    return tc
