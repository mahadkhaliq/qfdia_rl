from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

METRICS = ("asr", "evasion_rate", "mean_sds", "mean_stealth", "tau_bdd")
RUN_RE = re.compile(r"bus(?P<bus>\d+)_seed(?P<seed>\d+)(?P<classical>_classical)?$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def collect(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in sorted(path for path in root.glob("bus*_seed*") if path.is_dir()):
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        bus = int(match.group("bus"))
        seed = int(match.group("seed"))
        family = "Classical-MLP" if match.group("classical") else "Q-NPG-FDIA"
        result_name = f"qnpg_{bus}{'_classical' if match.group('classical') else ''}_results.csv"
        result_path = run_dir / result_name
        if not result_path.exists():
            continue
        for row in read_csv(result_path):
            out = {
                "bus": bus,
                "seed": seed,
                "family": family,
                "method": row.get("method", ""),
                "n": int(float(row.get("n", 0) or 0)),
                "result_path": str(result_path),
            }
            for metric in METRICS:
                out[metric] = float(row.get(metric, "nan"))
            rows.append(out)
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["bus"], row["family"], row["method"])].append(row)

    summary: list[dict[str, object]] = []
    for (bus, family, method), group in sorted(grouped.items()):
        out: dict[str, object] = {
            "bus": bus,
            "family": family,
            "method": method,
            "seeds": " ".join(str(row["seed"]) for row in sorted(group, key=lambda item: int(item["seed"]))),
            "n_seeds": len(group),
            "eval_n_per_seed": int(group[0]["n"]),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in group]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = sample_std(values)
        summary.append(out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Q-NPG-FDIA seed-sweep results.")
    parser.add_argument("--root", type=Path, default=Path("runs/qnpg_seed_sweep"))
    parser.add_argument("--raw-out", type=Path, default=Path("paper_tables/qnpg_seed_sweep_raw.csv"))
    parser.add_argument("--summary-out", type=Path, default=Path("paper_tables/qnpg_seed_sweep_summary.csv"))
    args = parser.parse_args()

    rows = collect(args.root)
    summary = summarize(rows)
    write_csv(args.raw_out, rows)
    write_csv(args.summary_out, summary)
    print(f"wrote {len(rows)} raw rows -> {args.raw_out}")
    print(f"wrote {len(summary)} summary rows -> {args.summary_out}")


if __name__ == "__main__":
    main()
