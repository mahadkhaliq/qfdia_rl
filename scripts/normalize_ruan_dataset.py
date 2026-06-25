"""
Normalize the Ruan/STGDL FDIA dataset into the detector Parquet schema.

Input MATLAB files:
  - AdmittanceMatrix_<bus>.mat
  - CAISO_normal_operation_data_<bus>.mat
  - CAISO_attack_data_<bus>.mat

Output columns match QGrid-Synth enough for scripts/train_detector_cnn.py:
  sample_id, bus, attack_type, label, z, a

Ruan measurements are state-estimation outputs:
  z = [Vm_1..Vm_n, theta_1..theta_n]

Attack samples are stored as 10k MATLAB cells, each with three attacked time
steps. We flatten those to 30k attack rows and sample an equal number of normal
rows by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.io import loadmat


def _cell_vec(cell) -> np.ndarray:
    return np.asarray(cell, dtype=np.float32).reshape(-1)


def _normal_z(outputvoltage, row: int) -> np.ndarray:
    vm = _cell_vec(outputvoltage[row, 0])
    th = _cell_vec(outputvoltage[row, 1])
    return np.concatenate([vm, th]).astype(np.float32)


def normalize_ruan(root: Path, bus: int, out: Path, normal_ratio: float, seed: int, chunk: int):
    rng = np.random.default_rng(seed)
    root = Path(root)
    normal_path = root / f"{bus}-bus" / "extracted" / f"CAISO_normal_operation_data_{bus}.mat"
    attack_path = root / f"{bus}-bus" / "extracted" / f"CAISO_attack_data_{bus}.mat"
    adm_path = root / f"{bus}-bus" / "extracted" / f"AdmittanceMatrix_{bus}.mat"

    for path in [normal_path, attack_path, adm_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    normal = loadmat(normal_path, squeeze_me=False)["outputvoltage"]
    attack = loadmat(attack_path, squeeze_me=False)["total_attack"]
    adm = loadmat(adm_path, squeeze_me=False)
    n = int(adm["G"].shape[0])
    if n != bus:
        raise ValueError(f"expected {bus} buses from admittance matrix, got {n}")

    n_attack = int(attack.shape[0] * 3)
    n_normal = int(round(normal_ratio * n_attack))
    normal_idx = rng.choice(normal.shape[0], size=min(n_normal, normal.shape[0]), replace=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = {"w": None}
    buffer = []
    sid = 0

    def emit(row):
        nonlocal sid
        row["sample_id"] = sid
        sid += 1
        buffer.append(row)
        if len(buffer) >= chunk:
            flush()

    def flush():
        if not buffer:
            return
        table = pa.Table.from_pandas(pd.DataFrame(buffer), preserve_index=False)
        if writer["w"] is None:
            writer["w"] = pq.ParquetWriter(out, table.schema, compression="snappy")
        writer["w"].write_table(table)
        buffer.clear()

    zero_a = np.zeros(2 * n, dtype=np.float32)
    for idx in normal_idx:
        z = _normal_z(normal, int(idx))
        emit(
            {
                "bus": bus,
                "attack_type": "normal",
                "label": 0,
                "source_index": int(idx) + 1,  # MATLAB-style reference
                "z": z,
                "a": zero_a,
            }
        )

    for i in range(attack.shape[0]):
        time_idx = np.asarray(attack[i, 0]).reshape(-1).astype(int)
        atk_vm = np.asarray(attack[i, 1], dtype=np.float32)
        atk_th = np.asarray(attack[i, 2], dtype=np.float32)
        for t in range(atk_vm.shape[0]):
            z = np.concatenate([atk_vm[t].reshape(-1), atk_th[t].reshape(-1)]).astype(np.float32)
            src = int(time_idx[t]) if t < len(time_idx) else -1
            if 1 <= src <= normal.shape[0]:
                clean = _normal_z(normal, src - 1)
                a = (z - clean).astype(np.float32)
            else:
                a = zero_a
            emit(
                {
                    "bus": bus,
                    "attack_type": "ruan_fdia",
                    "label": 1,
                    "source_index": src,
                    "z": z,
                    "a": a,
                }
            )

    flush()
    if writer["w"] is not None:
        writer["w"].close()

    pf = pq.ParquetFile(out)
    print(f"wrote {out}")
    print(f"rows={pf.metadata.num_rows} bus={bus} z_dim={2*n} attacks={n_attack} normals={len(normal_idx)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to extracted ruan_fdia repo")
    ap.add_argument("--bus", type=int, required=True, choices=[30, 118])
    ap.add_argument("--out", required=True)
    ap.add_argument("--normal-ratio", type=float, default=1.0, help="normal rows per attack row")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args()
    normalize_ruan(Path(args.root), args.bus, Path(args.out), args.normal_ratio, args.seed, args.chunk)


if __name__ == "__main__":
    main()
