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


def build_approx_h_env(bus: int, seed: int, rel: float):
    env = build_env(bus, seed)
    H0 = env.H.copy()
    S0 = env.S.copy()
    Hpinv0 = env.Hpinv.copy()
    rng = np.random.default_rng(seed)
    base_reset = env.reset

    def apply_detector_projector(Ht: np.ndarray) -> None:
        detector_projector = np.eye(Ht.shape[0]) - Ht @ np.linalg.pinv(Ht)
        env.S = 0.5 * (detector_projector + detector_projector.T)
        env.H_detector = Ht

    def reset(*args, **kwargs):
        result = base_reset(*args, **kwargs)
        apply_detector_projector(perturb_H(H0, rel, rng))
        return result

    def nominal_state_deviation(attack: np.ndarray) -> float:
        return float(np.linalg.norm(Hpinv0 @ attack))

    env.H_nominal = H0
    env.S_nominal = S0
    env.Hpinv_nominal = Hpinv0
    env.reset = reset
    env.state_deviation = nominal_state_deviation
    apply_detector_projector(perturb_H(H0, rel, rng))
    return env


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


def build_policy(bus: int, env, seed: int, device_name: str):
    from config import BUS_CONFIGS, TrainingConfig
    from models.vqc_policy import VQCPolicy

    cfg = BUS_CONFIGS[bus]
    tc = TrainingConfig(seed=seed, device_name=device_name)
    return VQCPolicy(
        env.obs_dim,
        env.action_dim,
        cfg.n_qubits,
        cfg.vqc_layers,
        env.a_max,
        init_log_std=tc.init_log_std,
        device_name=device_name,
        seed=seed,
    )


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


class ApproxHTrainer:
    """QNPGTrainer wrapper that averages the current detector projector.

    The base trainer stores one residual projector at construction. In this
    approximate-H ablation, the detector projector changes on reset, so the
    physics-Fisher term should use the rollout-average projector rather than the
    nominal fixed-H projector.
    """

    def __init__(self, env, policy, tc):
        from training.qnpg_trainer import QNPGTrainer

        self.base = QNPGTrainer(env, policy, tc)
        self.env = env
        self.policy = policy
        self.tc = tc

    def collect(self):
        original_step = self.env.step
        S_sum = np.zeros_like(self.env.S)
        S_count = 0

        def step_with_projector_record(action):
            nonlocal S_sum, S_count
            out = original_step(action)
            S_sum += self.env.S
            S_count += 1
            return out

        self.env.step = step_with_projector_record
        try:
            out = self.base.collect()
        finally:
            self.env.step = original_step
        if S_count:
            self.base.S = S_sum / S_count
        return out

    def update(self, buf, adv, ret):
        return self.base.update(buf, adv, ret)

    def train(self, callback=None):
        history = []
        for update in range(self.tc.total_updates):
            buf, adv, ret, stats = self.collect()
            uinfo = self.update(buf, adv, ret)
            rec = {"update": update + 1, **stats, **uinfo}
            history.append(rec)
            if callback:
                callback(rec)
        return history


def evaluate_scaled_direction(bus: int, policy, rel: float, episodes: int, k_perturb: int,
                              seed: int, scales: list[float]) -> list[dict]:
    env = build_env(bus, seed)
    rng = np.random.default_rng(seed)
    attacks = []
    for _ in range(episodes):
        info = env.rollout_attack(policy, deterministic=True)
        attacks.append(np.asarray(info["attack"], dtype=float))

    rows = []
    for scale in scales:
        fixed_residuals = []
        approx_residuals = []
        sds = []
        for attack in attacks:
            scaled = np.clip(scale * attack, -env.a_max, env.a_max)
            fixed_residuals.append(env.bdd_residual(scaled))
            sds.append(env.state_deviation(scaled))
            for _ in range(k_perturb):
                Ht = perturb_H(env.H, rel, rng)
                approx_residuals.append(residual_chi2(scaled, Ht, env.sigma))
        fixed_residuals = np.asarray(fixed_residuals)
        approx_residuals = np.asarray(approx_residuals)
        fixed_evasion = float(np.mean(fixed_residuals < env.tau_bdd))
        approx_evasion = float(np.mean(approx_residuals < env.tau_bdd))
        rows.append({
            "scale": float(scale),
            "mean_sds": float(np.mean(sds)),
            "fixed_evasion_rate": fixed_evasion,
            "approx_h_evasion_rate": approx_evasion,
            "evasion_drop_points": float(100.0 * (fixed_evasion - approx_evasion)),
            "fixed_median_chi2_over_tau": float(np.median(fixed_residuals) / env.tau_bdd),
            "approx_h_median_chi2_over_tau": float(np.median(approx_residuals) / env.tau_bdd),
        })
    return rows


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
    from config import TrainingConfig

    summary_rows = []
    scale_rows = []
    scales = [float(item) for item in args.scales]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        for mu in args.mu_values:
            env = build_approx_h_env(args.bus, seed, args.rel[0])
            policy = build_policy(args.bus, env, seed, args.device)
            tc = TrainingConfig(seed=seed, device_name=args.device, out_dir=str(out_dir))
            tc.total_updates = args.updates
            tc.steps_per_update = args.steps_per_update
            tc.qfim_batch = args.qfim_batch
            tc.grad_batch = args.grad_batch
            tc.eval_episodes = args.eval_episodes
            tc.log_every = max(1, args.log_every)
            tc.mu_phys = float(mu)

            trainer = ApproxHTrainer(env, policy, tc)
            history = []

            def log(rec):
                history.append(rec)
                if rec["update"] == 1 or rec["update"] % tc.log_every == 0:
                    print(f"mu={mu:.3g} seed={seed} upd {rec['update']:>4}/{tc.total_updates} "
                          f"ASR {rec['asr']:.3f} SDS {rec['state_dev']:.4f} "
                          f"stealth {rec['stealth']:.3f} phys {rec.get('phys_trace', 0.0):.3e}")

            trainer.train(callback=log)

            run_name = f"approx_h_bus{args.bus}_rel{args.rel[0]:.3f}_mu{mu:g}_seed{seed}"
            run_dir = out_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            if history:
                write_csv(run_dir / "history.csv", history)
            np.savez(run_dir / "policy.npz", **policy.state_dict())

            eval_rows = evaluate_scaled_direction(
                args.bus,
                policy,
                args.rel[0],
                args.eval_episodes,
                args.k_perturb,
                seed + 1000,
                scales,
            )
            for row in eval_rows:
                full = {
                    "bus": args.bus,
                    "rel_h": args.rel[0],
                    "mu_phys": float(mu),
                    "seed": seed,
                    "updates": args.updates,
                    **row,
                }
                scale_rows.append(full)
            write_csv(run_dir / "scaled_eval.csv", [
                {"bus": args.bus, "rel_h": args.rel[0], "mu_phys": float(mu), "seed": seed, "updates": args.updates, **row}
                for row in eval_rows
            ])

            unscaled = next(row for row in eval_rows if abs(row["scale"] - 1.0) < 1e-12)
            last = history[-1] if history else {}
            summary_rows.append({
                "bus": args.bus,
                "rel_h": args.rel[0],
                "mu_phys": float(mu),
                "seed": seed,
                "updates": args.updates,
                "mean_sds": unscaled["mean_sds"],
                "fixed_evasion_rate": unscaled["fixed_evasion_rate"],
                "approx_h_evasion_rate": unscaled["approx_h_evasion_rate"],
                "evasion_drop_points": unscaled["evasion_drop_points"],
                "approx_h_median_chi2_over_tau": unscaled["approx_h_median_chi2_over_tau"],
                "train_final_asr": last.get("asr", ""),
                "train_final_sds": last.get("state_dev", ""),
                "train_final_stealth": last.get("stealth", ""),
                "train_final_phys_trace": last.get("phys_trace", ""),
                "run_dir": str(run_dir),
            })

    write_csv(args.csv_out, summary_rows)
    scaled_out = args.scaled_csv_out or args.csv_out.with_name(args.csv_out.stem + "_scaled.csv")
    write_csv(scaled_out, scale_rows)

    print(f"\nwrote {args.csv_out}")
    print(f"wrote {scaled_out}")
    print(f"{'mu':>8} {'seed':>6} {'SDS':>9} {'approx ev':>10} {'drop':>8} {'phys tr':>10}")
    for row in summary_rows:
        print(f"{row['mu_phys']:>8.3g} {row['seed']:>6} {row['mean_sds']:>9.4f} "
              f"{row['approx_h_evasion_rate']:>10.3f} {row['evasion_drop_points']:>8.1f} "
              f"{row['train_final_phys_trace'] if row['train_final_phys_trace'] != '' else '-':>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Approximate-H diagnostic for Q-NPG-FDIA policies.")
    parser.add_argument("--bus", type=int, choices=[30, 57, 118], required=True)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--rel", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--k-perturb", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--scaled-csv-out", type=Path, default=None)
    parser.add_argument("--diagnose", action="store_true", help="run evaluation-only approximate-H diagnostic")
    parser.add_argument("--ablate", action="store_true", help="train mu=0 versus mu>0 under approximate-H residuals")
    parser.add_argument("--mu-values", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--steps-per-update", type=int, default=256)
    parser.add_argument("--qfim-batch", type=int, default=6)
    parser.add_argument("--grad-batch", type=int, default=96)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    parser.add_argument("--device", type=str, default="default.qubit")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/approx_h_ablation"))
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    if args.csv_out is None:
        suffix = "ablation" if args.ablate else "diagnostic"
        args.csv_out = Path(f"paper_tables/approx_h_{suffix}_{args.bus}_bus.csv")
    if args.ablate:
        run_ablate(args)
    else:
        run_diagnose(args)


if __name__ == "__main__":
    main()
