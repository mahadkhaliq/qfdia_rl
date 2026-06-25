#!/usr/bin/env python3
"""
Analytical ceiling for the stealthy max-SDS attack.

For each grid, solve

    maximize   SDS(a) = ||H^+ a||_2
    subject to a = H c
               ||a||_inf <= a_max

The construction a = Hc is exactly stealthy in the linearised BDD model, so the
result is a direct worst-case ceiling under the same box constraint used by the
Q-NPG policy. The learned/ceiling ratio is the number to report in the paper.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FALLBACK_LEARNED_SDS = {30: 0.4700599047057844, 57: 0.21667642976011264, 118: 0.07216626118858749}


def build_env(bus: int):
    from config import BUS_CONFIGS
    from environments.grid_env import FDIAGridEnv

    cfg = BUS_CONFIGS[bus]
    return FDIAGridEnv(bus, cfg=cfg, seed=0)


def read_learned_sds(bus: int, results_dir: Path) -> float:
    path = results_dir / f"qnpg_{bus}_results.csv"
    if not path.exists():
        return FALLBACK_LEARNED_SDS[bus]
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("method") == "Q-NPG-FDIA":
                return float(row["mean_sds"])
    return FALLBACK_LEARNED_SDS[bus]


def max_sds(H: np.ndarray, a_max: float, restarts: int, iters: int, seed: int, tol: float = 1e-9):
    rng = np.random.default_rng(seed)
    m, d = H.shape
    Hpinv = np.linalg.pinv(H)
    row_projector = Hpinv @ H

    a_ub = np.vstack([H, -H])
    b_ub = np.full(2 * m, a_max, dtype=float)
    bounds = [(None, None)] * d

    def solve_lp(direction: np.ndarray):
        result = linprog(-direction, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        return result.x if result.success else None

    best_c = None
    best_val = -np.inf
    for _ in range(restarts):
        direction = row_projector @ rng.standard_normal(d)
        norm = np.linalg.norm(direction)
        if norm < tol:
            continue
        c = solve_lp(direction / norm)
        if c is None:
            continue

        prev = -np.inf
        for _ in range(iters):
            projected = row_projector @ c
            val = float(np.linalg.norm(projected))
            if val < tol or abs(val - prev) < tol:
                break
            prev = val
            nxt = solve_lp(projected / val)
            if nxt is None:
                break
            c = nxt

        val = float(np.linalg.norm(row_projector @ c))
        if val > best_val:
            best_c = c
            best_val = val

    if best_c is None:
        raise RuntimeError("No feasible LP solution found.")
    return best_c, best_val


def run_bus(bus: int, restarts: int, iters: int, seed: int, results_dir: Path) -> dict:
    env = build_env(bus)
    c, ceiling = max_sds(env.H, env.a_max, restarts=restarts, iters=iters, seed=seed)
    attack = env.H @ c
    learned = read_learned_sds(bus, results_dir)
    box_max = float(np.max(np.abs(attack)))
    residual = float(env.bdd_residual(attack))
    return {
        "bus": bus,
        "a_max": float(env.a_max),
        "SDS_ceiling": float(ceiling),
        "SDS_learned": float(learned),
        "learned_over_ceiling": float(learned / ceiling),
        "stealth_residual": residual,
        "tau_bdd": float(env.tau_bdd),
        "box_max": box_max,
        "box_ok": bool(box_max <= env.a_max * (1.0 + 1e-6)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute analytical max-SDS stealth ceilings.")
    parser.add_argument("--bus", type=int, choices=[30, 57, 118], help="single grid to run")
    parser.add_argument("--all", action="store_true", help="run 30, 57, and 118")
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--csv-out", type=Path, default=Path("paper_tables/sds_ceiling_ratios.csv"))
    args = parser.parse_args()

    buses = [30, 57, 118] if args.all else [args.bus]
    if not buses or buses == [None]:
        parser.error("use --bus {30,57,118} or --all")

    rows = [run_bus(bus, args.restarts, args.iters, args.seed, args.results_dir) for bus in buses]

    print(f"{'grid':>5} {'a_max':>8} {'SDS_ceiling':>12} {'SDS_learned':>12} "
          f"{'ratio':>9} {'stealth_resid':>14} {'tau':>10}")
    for row in rows:
        print(f"{row['bus']:>5} {row['a_max']:>8.4f} {row['SDS_ceiling']:>12.4f} "
              f"{row['SDS_learned']:>12.4f} {row['learned_over_ceiling']:>9.3f} "
              f"{row['stealth_residual']:>14.3e} {row['tau_bdd']:>10.3f}")

    write_csv(args.csv_out, rows)
    print(f"\nwrote {args.csv_out}")


if __name__ == "__main__":
    main()
