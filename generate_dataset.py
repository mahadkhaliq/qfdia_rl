"""
generate_dataset.py
===================
QGrid-Synth dataset generator for the QT-FDIA-RL framework.

Produces a labelled measurement dataset for training the DETECTOR side
(QB-DETECT / QLSTM) and for Q-ASP's normal/attack mix. This is downstream of
the Q-NPG-FDIA attacker: the trained policy supplies the high-value
"rl_learned" attack rows.

What makes this a real dataset (vs. the training env):
  * Each sample uses a FRESH operating point -- loads are randomly scaled and
    the AC power flow is RE-SOLVED, and the measurement model (z, H) is rebuilt
    at that operating point. This gives genuine operating-point diversity
    instead of a single base case with jitter.

Rows: one "normal" (clean + noise) or one "attack" (clean + a + noise) per draw,
balanced across attack types. Each row stores the measurement vector the
operator sees, the attack vector, the label, and physics stats.

Usage:
    python generate_dataset.py --bus 30 --load outputs/qnpg_30_policy.npz \
        --n-normal 10000 --n-attack 10000 --out outputs

    # all three grids, if the policies exist:
    for B in 30 57 118; do
      python generate_dataset.py --bus $B --load outputs/qnpg_${B}_policy.npz
    done

Output: outputs/qgrid_synth_<bus>.parquet
Columns:
    sample_id, bus, attack_type, label(0/1), load_scale,
    residual(chi2 stat), bdd_flagged(0/1), state_dev, stealth_score, piqe_alpha,
    z(list<float32>, the observed measurement), a(list<float32>, the attack)

Runtime note: dominated by the AC power-flow solve per operating point.
30/57-bus are fast; 118-bus is heavier -- install numba for a large speedup,
or reduce --n-normal/--n-attack. Use --light to drop the raw z/a vectors and
keep only the scalar stats (much smaller file).
"""
from __future__ import annotations
import os, sys, argparse, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pandapower as pp
import pandapower.networks as pn
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BUS_CONFIGS
from models.vqc_policy import VQCPolicy

CASE = {30: pn.case30, 57: pn.case57, 118: pn.case118}


# ----------------------------------------------------------------- physics
def measurement_model(net, use_numba=False):
    """Run AC power flow on `net` and build (z_clean, H, S, H+, tau) at that point."""
    pp.runpp(net, calculate_voltage_angles=True, init="dc", numba=use_numba)
    if not net.converged:
        return None
    ppc = net._ppc
    Ybus = np.asarray(ppc["internal"]["Ybus"].todense())
    Vm = ppc["bus"][:, 7].astype(float)
    Va = np.deg2rad(ppc["bus"][:, 8].astype(float))
    n = Ybus.shape[0]; m = 4 * n
    x = np.concatenate([Va, Vm])

    def h(xv):
        th, vm = xv[:n], xv[n:]
        V = vm * np.exp(1j * th)
        S = V * np.conj(Ybus @ V)
        return np.concatenate([S.real, S.imag, vm, th])

    z = h(x)
    H = np.zeros((m, 2 * n)); eps = 1e-6
    for k in range(2 * n):
        xp = x.copy(); xm = x.copy(); xp[k] += eps; xm[k] -= eps
        H[:, k] = (h(xp) - h(xm)) / (2 * eps)
    Hp = np.linalg.pinv(H)
    S = np.eye(m) - H @ Hp; S = 0.5 * (S + S.T)
    tau = float(chi2.ppf(0.95, df=m - 2 * n))
    return dict(n=n, m=m, z=z, H=H, Hp=Hp, S=S, tau=tau)


def stats(mm, a, sigma):
    r = mm["S"] @ a
    res = float(r @ r) / (sigma ** 2)
    sdev = float(np.linalg.norm(mm["Hp"] @ a))
    stealth = float(max(0.0, 1.0 - res / mm["tau"]))
    alpha = float(np.arctan2(float(np.sqrt(r @ r)), np.linalg.norm(mm["z"]) + 1e-12))
    return res, int(res > mm["tau"]), sdev, stealth, alpha


# ----------------------------------------------------------------- attacks
def baseline_attack(kind, mm, cfg, a_max, rng):
    m, n = mm["m"], mm["n"]
    eps = cfg.epsilon

    def targets(k):
        return rng.choice(m, size=min(k, m), replace=False)

    if kind == "ramp" or kind == "step":
        a = np.zeros(m); idx = targets(int(0.1 * m)); a[idx] = eps
    elif kind == "random":
        a = np.zeros(m); idx = targets(int(0.1 * m)); a[idx] = rng.uniform(-eps, eps, idx.size)
    elif kind == "multiplicative":
        a = 0.05 * mm["z"]
    elif kind == "coordinated_sparse":
        a = np.zeros(m); idx = targets(4); a[idx] = eps * rng.choice([-1, 1], idx.size)
    elif kind == "liu_stealthy":
        c = rng.normal(0, 0.02, 2 * n); a = mm["H"] @ c
    else:
        a = np.zeros(m)
    mx = float(np.abs(a).max())
    if mx > a_max:
        a = a * (a_max / mx)            # scale, preserve direction (don't break Liu structure)
    return a


def rl_attack(policy, mm, sigma, a_max, rng):
    obs = np.concatenate([mm["z"] + rng.normal(0, sigma, mm["m"]), np.zeros(3)])
    a = np.asarray(policy.act(obs, deterministic=True)[0])
    return np.clip(a, -a_max, a_max)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="QGrid-Synth dataset generator")
    ap.add_argument("--bus", type=int, default=30, choices=[30, 57, 118])
    ap.add_argument("--load", type=str, default=None, help="trained Q-NPG policy .npz (adds rl_learned rows)")
    ap.add_argument("--n-normal", type=int, default=10000)
    ap.add_argument("--n-attack", type=int, default=10000)
    ap.add_argument("--load-var", type=float, default=0.30, help="+/- load scaling range for operating-point diversity")
    ap.add_argument("--sigma", type=float, default=1e-2, help="measurement noise std")
    ap.add_argument("--device", type=str, default="default.qubit")
    ap.add_argument("--light", action="store_true", help="store only scalar stats, drop raw z/a vectors")
    ap.add_argument("--per-op", type=int, default=4, help="samples per class per operating-point solve (higher = faster, slightly less diverse)")
    ap.add_argument("--chunk", type=int, default=20000, help="rows buffered in memory before flushing to disk (caps memory)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    cfg = BUS_CONFIGS[args.bus]
    a_max = cfg.a_max_mult * cfg.epsilon
    sigma = args.sigma

    try:
        import numba  # noqa: F401
        _NUMBA = True
    except Exception:
        _NUMBA = False
        print("  [warning] numba not found -- power flow will be slow. "
              "Install it (pip install numba) for a large speedup, or raise --per-op.")

    # attack-type rotation (rl_learned only if a policy is provided)
    types = ["ramp", "step", "random", "multiplicative", "coordinated_sparse", "liu_stealthy"]
    policy = None
    if args.load:
        net0 = CASE[args.bus]()
        pp.runpp(net0, numba=False)
        m0 = 4 * len(net0._ppc["bus"])
        policy = VQCPolicy(m0 + 3, m0, cfg.n_qubits, cfg.vqc_layers, a_max,
                           device_name=args.device, seed=args.seed)
        sd = np.load(args.load); policy.load_state_dict({k: sd[k] for k in sd.files})
        types.append("rl_learned")
        print(f"  loaded policy {args.load}  -> rl_learned rows enabled")

    print(f"\n  QGrid-Synth | IEEE {args.bus}-bus | target {args.n_normal} normal + {args.n_attack} attack "
          f"| load var +/-{args.load_var:.0%}")
    base = CASE[args.bus]()
    p0 = base.load.p_mw.values.copy()
    q0 = base.load.q_mvar.values.copy()

    path = os.path.join(args.out, f"qgrid_synth_{args.bus}.parquet")
    buffer = []
    writer = {"w": None}          # pyarrow ParquetWriter, opened on first flush
    # running stats (so we never hold the full dataset in memory)
    from collections import defaultdict
    type_count = defaultdict(int)
    type_sum = defaultdict(lambda: np.zeros(3))   # [stealth, state_dev, bdd_flagged]
    label_count = defaultdict(int)

    def flush():
        if not buffer:
            return
        table = pa.Table.from_pandas(pd.DataFrame(buffer), preserve_index=False)
        if writer["w"] is None:
            writer["w"] = pq.ParquetWriter(path, table.schema, compression="snappy")
        writer["w"].write_table(table)
        buffer.clear()

    def emit(r):
        buffer.append(r)
        label_count[r["label"]] += 1
        if r["label"] == 1:
            type_count[r["attack_type"]] += 1
            type_sum[r["attack_type"]] += [r["stealth_score"], r["state_dev"], r["bdd_flagged"]]
        if len(buffer) >= args.chunk:
            flush()

    n_norm = n_atk = 0
    sid = 0; ti = 0; t0 = time.time(); op = 0; fails = 0
    target = args.n_normal + args.n_attack
    while n_norm < args.n_normal or n_atk < args.n_attack:
        # fresh operating point: scale loads, re-solve power flow, rebuild model
        net = CASE[args.bus]()
        sc = rng.uniform(1 - args.load_var, 1 + args.load_var, size=len(p0))
        net.load.p_mw = p0 * sc
        net.load.q_mvar = q0 * sc
        mm = measurement_model(net, use_numba=_NUMBA)
        op += 1
        if mm is None:
            fails += 1
            if fails > 300:
                print("  power flow keeps diverging; reduce --load-var"); break
            continue
        fails = 0
        load_scale = float(sc.mean())

        # amortise the solve: several samples per class from this operating point
        for _ in range(args.per_op):
            if n_norm < args.n_normal:
                z_obs = mm["z"] + rng.normal(0, sigma, mm["m"])
                res, flag, sdev, st, al = stats(mm, np.zeros(mm["m"]), sigma)
                emit(_row(sid, args.bus, "normal", 0, load_scale, res, flag, 0.0, 1.0, al,
                          z_obs, np.zeros(mm["m"]), args.light)); sid += 1; n_norm += 1
            if n_atk < args.n_attack:
                kind = types[ti % len(types)]; ti += 1
                a = rl_attack(policy, mm, sigma, a_max, rng) if kind == "rl_learned" \
                    else baseline_attack(kind, mm, cfg, a_max, rng)
                z_obs = mm["z"] + a + rng.normal(0, sigma, mm["m"])
                res, flag, sdev, st, al = stats(mm, a, sigma)
                emit(_row(sid, args.bus, kind, 1, load_scale, res, flag, sdev, st, al,
                          z_obs, a, args.light)); sid += 1; n_atk += 1
        if sid % 20000 < 2 * args.per_op:
            print(f"    {sid}/{target} rows  ({op} ops, {time.time()-t0:.0f}s)")

    flush()
    if writer["w"] is not None:
        writer["w"].close()
    dt = time.time() - t0
    print(f"\n  wrote {sid} rows -> {path}  ({dt:.0f}s = {dt/60:.1f} min, {op} operating points)")
    print("  class balance:", {("normal" if k == 0 else "attack"): v for k, v in sorted(label_count.items())})
    print("  attack types:", dict(sorted(type_count.items())))
    print("  mean stealth / state_dev / bdd_flagged by attack type:")
    for k in sorted(type_count):
        m = type_sum[k] / max(type_count[k], 1)
        print(f"    {k:<20} stealth={m[0]:.3f}  state_dev={m[1]:.3f}  flagged={m[2]:.3f}")


def _row(sid, bus, kind, label, load_scale, res, flag, sdev, st, al, z, a, light):
    r = dict(sample_id=sid, bus=bus, attack_type=kind, label=label, load_scale=round(load_scale, 4),
             residual=round(res, 4), bdd_flagged=flag, state_dev=round(sdev, 5),
             stealth_score=round(st, 4), piqe_alpha=round(al, 5))
    if not light:
        r["z"] = z.astype(np.float32)
        r["a"] = a.astype(np.float32)
    return r


if __name__ == "__main__":
    main()
