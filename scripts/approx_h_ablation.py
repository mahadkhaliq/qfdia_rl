#!/usr/bin/env python3
"""
Approximate-H diagnostic for the physics-augmented QNPG term.

The reported fixed-H experiments keep the learned attacks on the stealthy
manifold, so J^T S J is near-inactive. This script tests the reviewer-facing
question: does detector/model mismatch make those attacks drift off manifold?

The runnable mode is diagnostic/evaluation-only: roll out a trained policy,
perturb the detector Jacobian H by a relative Frobenius amount, and compare
fixed-H evasion with approximate-H evasion. If the evasion drop is substantial,
then retraining with mu_phys on/off is a worthwhile stronger C1 ablation.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_env(bus: int, seed: int):
    from config import BUS_CONFIGS
    from environments.grid_env import FDIAGridEnv

    return FDIAGridEnv(bus, cfg=BUS_CONFIGS[bus], seed=seed)


def load_policy(bus: int, policy_path: Path, env, seed: int):
    from config import BUS_CONFIGS, TrainingConfig
    from models.vqc_policy import VQCPolicy

    cfg = BUS_CONFIGS[bus]
    tc = TrainingConfig(seed=seed)
    policy = VQCPolicy(
        env.obs_dim,
        env.action_dim,
        cfg.n_qubits,
        cfg.vqc_layers,
        env.a_max,
        init_log_std=tc.init_log_std,
        device_name=tc.device_name,
        seed=seed,
    )
    state = np.load(policy_path)
    policy.load_state_dict({key: state[key] for key in state.files})
    return policy


def perturb_H(H: np.ndarray, rel: float, rng: np.random.Generator) -> np.ndarray:
    delta = rng.standard_normal(H.shape)
    delta *= rel * np.linalg.norm(H) / (np.linalg.norm(delta) + 1e-12)
    return H + delta


def residual_chi2(attack: np.ndarray, H: np.ndarray, sigma: float) -> float:
    projector = np.eye(H.shape[0]) - H @ np.linalg.pinv(H)
    residual = projector @ attack
    return float(residual @ residual) / (sigma ** 2)


def diagnose_rel(env, policy, rel: float, episodes: int, k_perturb: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    fixed_residuals = []
    approx_residuals = []
    sds = []

    for _ in range(episodes):
        info = env.rollout_attack(policy, deterministic=True)
        attack = np.asarray(info["attack"], dtype=float)
        fixed_residuals.append(float(info.get("residual2", env.bdd_residual(attack))))
        sds.append(float(info.get("state_dev", env.state_deviation(attack))))
        for _ in range(k_perturb):
            Ht = perturb_H(env.H, rel, rng)
            approx_residuals.append(residual_chi2(attack, Ht, env.sigma))

    fixed_residuals = np.asarray(fixed_residuals)
    approx_residuals = np.asarray(approx_residuals)
    fixed_evasion = float(np.mean(fixed_residuals < env.tau_bdd))
    approx_evasion = float(np.mean(approx_residuals < env.tau_bdd))
    return {
        "bus": env.bus_size,
        "rel_h": rel,
        "episodes": episodes,
        "k_perturb": k_perturb,
        "mean_sds": float(np.mean(sds)),
        "fixed_evasion_rate": fixed_evasion,
        "approx_h_evasion_rate": approx_evasion,
        "evasion_drop_points": float(100.0 * (fixed_evasion - approx_evasion)),
        "fixed_median_chi2_over_tau": float(np.median(fixed_residuals) / env.tau_bdd),
        "approx_h_median_chi2_over_tau": float(np.median(approx_residuals) / env.tau_bdd),
        "tau_bdd": float(env.tau_bdd),
        "sigma": float(env.sigma),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_diagnose(args) -> list[dict]:
    env = build_env(args.bus, args.seed)
    policy_path = args.policy or Path(f"outputs/qnpg_{args.bus}_policy.npz")
    policy = load_policy(args.bus, policy_path, env, args.seed)

    rows = [diagnose_rel(env, policy, rel, args.episodes, args.k_perturb, args.seed) for rel in args.rel]
    print(f"IEEE {args.bus}-bus approximate-H diagnostic | policy={policy_path}")
    print(f"{'relH':>7} {'mean SDS':>9} {'fixed ev':>9} {'approx ev':>10} "
          f"{'drop pts':>9} {'approx med/tau':>15}")
    for row in rows:
        print(f"{row['rel_h']:>7.3f} {row['mean_sds']:>9.4f} "
              f"{row['fixed_evasion_rate']:>9.3f} {row['approx_h_evasion_rate']:>10.3f} "
              f"{row['evasion_drop_points']:>9.1f} {row['approx_h_median_chi2_over_tau']:>15.3f}")
    write_csv(args.csv_out, rows)
    print(f"\nwrote {args.csv_out}")
    return rows


def run_ablate(args) -> None:
    raise NotImplementedError(
        "The full mu=0 versus mu>0 retraining ablation needs a trainer variant "
        "that updates the residual projector used by the physics-Fisher term "
        "under approximate-H sampling. Run --diagnose first; if it shows a large "
        "evasion drop, wire this path to a Slurm training job rather than running "
        "on the Hellbender login node."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Approximate-H diagnostic for Q-NPG-FDIA policies.")
    parser.add_argument("--bus", type=int, choices=[30, 57, 118], required=True)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--rel", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--k-perturb", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--diagnose", action="store_true", help="run evaluation-only approximate-H diagnostic")
    parser.add_argument("--ablate", action="store_true", help="reserved for mu=0/mu>0 retraining ablation")
    args = parser.parse_args()

    if args.csv_out is None:
        args.csv_out = Path(f"paper_tables/approx_h_diagnostic_{args.bus}_bus.csv")
    if args.ablate:
        run_ablate(args)
    else:
        run_diagnose(args)


if __name__ == "__main__":
    main()
