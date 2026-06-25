"""
main.py
=======
Q-NPG-FDIA pipeline orchestrator (Mahad's task).

Usage
-----
  python main.py --bus 30 --quick                 # ~10-min smoke test (CPU)
  python main.py --bus 30                          # full 30-bus run
  python main.py --bus 57                          # 57-bus (HPC recommended)
  python main.py --bus 118 --device lightning.qubit
  python main.py --bus 30 --classical             # classical-PPO-style baseline policy

Outputs (under --out):
  qnpg_<bus>_history.csv     per-update training curve
  qnpg_<bus>_results.csv     final attack-generation table (Q-NPG vs baselines)
  qnpg_<bus>_policy.npz       trained policy parameters
"""
from __future__ import annotations
import os, sys, csv, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np

# make sibling packages importable when run from inside qfdia_rl/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BUS_CONFIGS, TrainingConfig, quick_overrides
from environments.grid_env import FDIAGridEnv
from environments.attack_types import FDIAAttackFactory
from models.vqc_policy import VQCPolicy, ClassicalGaussianPolicy
from training.qnpg_trainer import QNPGTrainer
from evaluation.metrics import AttackEvaluator


def main():
    ap = argparse.ArgumentParser(description="Q-NPG-FDIA attacker training")
    ap.add_argument("--bus", type=int, default=30, choices=[30, 57, 118])
    ap.add_argument("--quick", action="store_true", help="short smoke-test run")
    ap.add_argument("--classical", action="store_true", help="use classical MLP policy baseline")
    ap.add_argument("--device", type=str, default="default.qubit")
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="outputs")
    ap.add_argument("--eval-only", action="store_true", help="skip training; just evaluate (use with --load)")
    ap.add_argument("--load", type=str, default=None, help="path to a saved policy .npz to load")
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    bcfg = BUS_CONFIGS[args.bus]
    tc = TrainingConfig(seed=args.seed, device_name=args.device, out_dir=args.out)
    if args.quick:
        tc = quick_overrides(tc)
    if args.updates is not None:
        tc.total_updates = args.updates

    print(f"\n{'='*64}\n  Q-NPG-FDIA  |  IEEE {args.bus}-bus  |  "
          f"{'CLASSICAL MLP' if args.classical else f'VQC ({bcfg.n_qubits}q, {bcfg.vqc_layers}L)'}"
          f"\n{'='*64}")

    env = FDIAGridEnv(args.bus, cfg=bcfg, seed=args.seed)
    print(f"  measurements m = {env.m}   states 2n = {2*env.n}   "
          f"tau_BDD = {env.tau_bdd:.2f} (df={env.chi2_df})")
    print(f"  H-matrix {env.H.shape}, rank {np.linalg.matrix_rank(env.H)}   "
          f"epsilon = {bcfg.epsilon}")

    PolicyCls = ClassicalGaussianPolicy if args.classical else VQCPolicy
    policy = PolicyCls(env.obs_dim, env.action_dim, bcfg.n_qubits, bcfg.vqc_layers,
                       env.a_max, init_log_std=tc.init_log_std,
                       device_name=args.device, seed=args.seed)
    if not args.classical:
        print(f"  VQC params (theta_q) = {policy.d_theta}   sim backend = {policy.device_name}")

    if args.load:
        sd = np.load(args.load)
        policy.load_state_dict({k: sd[k] for k in sd.files})
        print(f"  loaded policy from {args.load}")

    trainer = QNPGTrainer(env, policy, tc)

    # ---- train ----
    t0 = time.time(); history = []

    def log(rec):
        history.append(rec)
        if rec["update"] % tc.log_every == 0 or rec["update"] == 1:
            extra = (f" | natstep {rec.get('kl_step', 0):.3f}"
                     f" | QFIM tr {rec.get('qfim_trace', 0):.2f}"
                     f" | phys tr {rec.get('phys_trace', 0):.2f}") if 'kl_step' in rec else ""
            print(f"  upd {rec['update']:3d}/{tc.total_updates} | "
                  f"R {rec['ep_return']:7.3f} | ASR {rec['asr']:.3f} | "
                  f"stealth {rec['stealth']:.3f} | SDS {rec['state_dev']:.4f}{extra}")

    if args.eval_only:
        print("  --eval-only: skipping training")
    else:
        trainer.train(callback=log)
        print(f"  training done in {time.time()-t0:.1f}s")

    # ---- evaluate Q-NPG vs baselines ----
    ev = AttackEvaluator(env, min_impact=tc.min_impact)
    rows = [ev.evaluate_policy(policy, n_episodes=tc.eval_episodes)]
    fac = FDIAAttackFactory(env, seed=args.seed)
    for kind in ["liu_stealthy", "step", "random", "multiplicative", "coordinated_sparse"]:
        rows.append(ev.evaluate_baseline(fac, kind, n=tc.eval_episodes))

    print("\n  Attack-generation results (paper Table I):\n")
    print("  " + AttackEvaluator.table(rows).replace("\n", "\n  "))

    # ---- save ----
    tag = f"qnpg_{args.bus}{'_classical' if args.classical else ''}"
    with open(os.path.join(args.out, f"{tag}_history.csv"), "w", newline="") as f:
        keys = sorted({k for r in history for k in r})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(history)
    with open(os.path.join(args.out, f"{tag}_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    np.savez(os.path.join(args.out, f"{tag}_policy.npz"), **policy.state_dict())
    print(f"\n  saved: {tag}_history.csv, {tag}_results.csv, {tag}_policy.npz  (in {args.out}/)\n")


if __name__ == "__main__":
    main()
