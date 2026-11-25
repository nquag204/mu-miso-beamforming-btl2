"""Monte-Carlo experiment orchestrators for BTL2 Topic 1 (MU-MISO Beamforming via GA).

Runs six reproducible experiments on the polar-coordinate circular cell and
direct-sum Rician channel model, persisting raw Monte-Carlo arrays (.npz) and
summary CSV tables (.csv) to btl2/results/data/:
  1. exp1_geometry_beampattern     : Polar circular deployment & 360-deg beampattern
  2. exp2_power_sweep              : Average sum-rate vs. BS transmit power P_max
  3. exp3_antenna_sweep            : Average sum-rate vs. BS antenna count N_t
  4. exp4_user_sweep               : Average sum-rate vs. user count K (incl. K > N_t)
  5. exp5_rician_k_sweep           : Average sum-rate vs. Rician K-factor
  6. exp6_convergence_stability    : GA convergence curves, 30-run stability & runtime
"""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATA_DIR, DEFAULT_CONFIG, ExperimentConfig
from src.baselines import (
    mrt_beamformer,
    rzf_beamformer,
    wmmse_beamformer,
    zf_beamformer,
)
from src.channel import generate_rician_channel
from src.metrics import compute_beampattern
from src.optimizers import (
    direct_ga_beamformer,
    hybrid_ga_beamformer,
    struct_ga_beamformer,
)
from src.utils import dbm_to_watt, ensure_dirs


ALL_METHODS: tuple[str, ...] = (
    "MRT",
    "ZF",
    "RZF",
    "WMMSE",
    "Direct-GA",
    "Hybrid-GA",
    "Struct-GA",
)


def _evaluate_all_methods_on_channel(
    H: np.ndarray,
    p_max_watt: float,
    noise_watt: float,
    cfg: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run all baselines & GA variants on one paired channel realization."""
    rates: dict[str, float] = {}
    times: dict[str, float] = {}

    for name, fn in (
        ("MRT", mrt_beamformer),
        ("ZF", zf_beamformer),
        ("RZF", rzf_beamformer),
    ):
        res = fn(H, p_max_watt, noise_watt)
        rates[name] = res.sum_rate
        times[name] = res.runtime_sec

    res_wmmse = wmmse_beamformer(
        H,
        p_max_watt,
        noise_watt,
        max_iter=cfg.optimizer.wmmse_max_iter,
        tol=cfg.optimizer.wmmse_tol,
    )
    rates["WMMSE"] = res_wmmse.sum_rate
    times["WMMSE"] = res_wmmse.runtime_sec

    child_seeds = rng.integers(0, 2**31 - 1, size=3)
    ga_solvers = (
        ("Direct-GA", direct_ga_beamformer),
        ("Hybrid-GA", hybrid_ga_beamformer),
        ("Struct-GA", struct_ga_beamformer),
    )
    for (name, solver_fn), c_seed in zip(ga_solvers, child_seeds):
        res_ga = solver_fn(
            H,
            p_max_watt,
            noise_watt,
            cfg.optimizer,
            np.random.default_rng(int(c_seed)),
        )
        rates[name] = res_ga.best_sum_rate
        times[name] = res_ga.runtime_sec

    return rates, times


def _run_sweep(
    x_label: str,
    x_values: list[float] | list[int],
    channel_params_fn,
    cfg: ExperimentConfig,
    seed_offset: int,
    desc: str,
    out_stem: str,
    data_dir: Path,
) -> pd.DataFrame:
    """Generic paired Monte-Carlo sweep runner saving .npz and summary .csv."""
    n_pts = len(x_values)
    n_trials = cfg.num_trials

    raw_rates = {m: np.zeros((n_pts, n_trials), dtype=np.float64) for m in ALL_METHODS}
    raw_times = {m: np.zeros((n_pts, n_trials), dtype=np.float64) for m in ALL_METHODS}

    rng = np.random.default_rng(cfg.seed + seed_offset)

    for idx, x_val in enumerate(tqdm(x_values, desc=desc, unit="pt")):
        Nt, K, p_max_dbm, rician_k_db = channel_params_fn(x_val)
        p_max_watt = float(dbm_to_watt(p_max_dbm))
        noise_watt = cfg.system.noise_power_watt

        for t in range(n_trials):
            ch = generate_rician_channel(Nt, K, cfg.system, rng, rician_k_db=rician_k_db)
            rates, times = _evaluate_all_methods_on_channel(ch.H, p_max_watt, noise_watt, cfg, rng)
            for m in ALL_METHODS:
                raw_rates[m][idx, t] = rates[m]
                raw_times[m][idx, t] = times[m]

    npz_payload: dict[str, np.ndarray] = {
        "x_values": np.asarray(x_values, dtype=np.float64),
        "methods": np.asarray(ALL_METHODS),
    }
    for m in ALL_METHODS:
        key = m.replace("-", "_")
        npz_payload[f"rates_{key}"] = raw_rates[m]
        npz_payload[f"times_{key}"] = raw_times[m]

    np.savez_compressed(data_dir / f"{out_stem}.npz", **npz_payload)

    rows = []
    for idx, x_val in enumerate(x_values):
        row: dict[str, float] = {x_label: float(x_val)}
        for m in ALL_METHODS:
            row[f"{m}_mean_bps_hz"] = float(np.mean(raw_rates[m][idx]))
            row[f"{m}_std_bps_hz"] = float(np.std(raw_rates[m][idx], ddof=1)) if n_trials > 1 else 0.0
            row[f"{m}_runtime_ms"] = float(np.mean(raw_times[m][idx]) * 1e3)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(data_dir / f"{out_stem}.csv", index=False, float_format="%.6f")
    return df


def run_exp1_geometry_and_beampattern(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Exp 1: Representative polar circular deployment and 360-deg ULA beampattern."""
    ensure_dirs(data_dir)
    sys_cfg = cfg.system
    Nt, K = sys_cfg.default_nt, sys_cfg.default_k
    p_max_watt = float(dbm_to_watt(sys_cfg.default_p_max_dbm))
    noise_watt = sys_cfg.noise_power_watt

    # Pick a clean well-separated polar realization for clear illustration
    rng = np.random.default_rng(cfg.seed + 101)
    best_ch = None
    best_score = -1.0
    for _ in range(40):
        cand = generate_rician_channel(Nt, K, sys_cfg, rng)
        if np.min(cand.radii_m) < 15.0:
            continue
        sorted_ang = np.sort(cand.angles_rad)
        diffs = np.diff(np.append(sorted_ang, sorted_ang[0] + 2.0 * np.pi))
        min_sep = float(np.min(diffs))
        if min_sep > best_score:
            best_score = min_sep
            best_ch = cand

    assert best_ch is not None
    order = np.argsort(best_ch.angles_rad)
    H = best_ch.H[:, order]
    radii_m = best_ch.radii_m[order]
    distances_m = best_ch.distances_m[order]
    angles_rad = best_ch.angles_rad[order]
    positions_xy = best_ch.positions_xy[order]
    pl_db = best_ch.pl_db[order]

    res_mrt = mrt_beamformer(H, p_max_watt, noise_watt)
    res_wmmse = wmmse_beamformer(H, p_max_watt, noise_watt, cfg.optimizer.wmmse_max_iter, cfg.optimizer.wmmse_tol)
    res_sga = struct_ga_beamformer(H, p_max_watt, noise_watt, cfg.optimizer, np.random.default_rng(cfg.seed + 102))

    angles_grid_rad = np.linspace(0.0, 2.0 * np.pi, 721)
    pat_mrt = compute_beampattern(res_mrt.W, angles_grid_rad)
    pat_sga = compute_beampattern(res_sga.W, angles_grid_rad)
    pat_wmmse = compute_beampattern(res_wmmse.W, angles_grid_rad)

    np.savez_compressed(
        data_dir / "exp1_geometry_beampattern.npz",
        radii_m=radii_m,
        distances_m=distances_m,
        angles_rad=angles_rad,
        positions_xy=positions_xy,
        pl_db=pl_db,
        angles_grid_rad=angles_grid_rad,
        pat_mrt=pat_mrt,
        pat_sga=pat_sga,
        pat_wmmse=pat_wmmse,
        rate_mrt=np.array([res_mrt.sum_rate]),
        rate_sga=np.array([res_sga.best_sum_rate]),
        rate_wmmse=np.array([res_wmmse.sum_rate]),
    )

    df = pd.DataFrame(
        {
            "UE": [f"UE {k + 1}" for k in range(K)],
            "r_m": radii_m,
            "Phi_deg": np.rad2deg(angles_rad),
            "x_m": positions_xy[:, 0],
            "y_m": positions_xy[:, 1],
            "PL_dB": pl_db,
        }
    )
    df.to_csv(data_dir / "exp1_ue_table.csv", index=False, float_format="%.4f")
    return df


def run_exp2_power_sweep(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Exp 2: Average sum-rate vs. BS transmit power P_max (dBm)."""
    ensure_dirs(data_dir)
    s = cfg.system
    return _run_sweep(
        x_label="P_max_dBm",
        x_values=list(cfg.p_max_dbm_grid),
        channel_params_fn=lambda p_dbm: (s.default_nt, s.default_k, float(p_dbm), s.rician_k_db),
        cfg=cfg,
        seed_offset=2000,
        desc="Exp 2: P_max sweep",
        out_stem="exp2_power_sweep",
        data_dir=data_dir,
    )


def run_exp3_antenna_sweep(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Exp 3: Average sum-rate vs. number of BS antennas N_t."""
    ensure_dirs(data_dir)
    s = cfg.system
    return _run_sweep(
        x_label="Nt_antennas",
        x_values=list(cfg.nt_grid),
        channel_params_fn=lambda nt_val: (int(nt_val), s.default_k, s.default_p_max_dbm, s.rician_k_db),
        cfg=cfg,
        seed_offset=3000,
        desc="Exp 3: N_t antenna sweep",
        out_stem="exp3_antenna_sweep",
        data_dir=data_dir,
    )


def run_exp4_user_sweep(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Exp 4: Average sum-rate vs. number of single-antenna users K."""
    ensure_dirs(data_dir)
    s = cfg.system
    return _run_sweep(
        x_label="K_users",
        x_values=list(cfg.k_grid),
        channel_params_fn=lambda k_val: (s.default_nt, int(k_val), s.default_p_max_dbm, s.rician_k_db),
        cfg=cfg,
        seed_offset=4000,
        desc="Exp 4: K user sweep",
        out_stem="exp4_user_sweep",
        data_dir=data_dir,
    )


def run_exp5_rician_k_sweep(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Exp 5: Impact of Rician K-factor in the direct-sum channel model."""
    ensure_dirs(data_dir)
    s = cfg.system
    return _run_sweep(
        x_label="Rician_K_dB",
        x_values=list(cfg.rician_k_db_grid),
        channel_params_fn=lambda k_db: (s.default_nt, s.default_k, s.default_p_max_dbm, float(k_db)),
        cfg=cfg,
        seed_offset=5000,
        desc="Exp 5: Rician K sweep",
        out_stem="exp5_rician_k_sweep",
        data_dir=data_dir,
    )


def run_exp6_convergence_stability_runtime(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Exp 6: GA convergence envelopes, 30-run stability analysis, and runtime benchmark."""
    ensure_dirs(data_dir)
    s = cfg.system
    Nt, K = s.default_nt, s.default_k
    p_max_watt = float(dbm_to_watt(s.default_p_max_dbm))
    noise_watt = s.noise_power_watt

    ch = generate_rician_channel(Nt, K, s, np.random.default_rng(cfg.seed + 6000))
    H = ch.H

    res_mrt = mrt_beamformer(H, p_max_watt, noise_watt)
    res_zf = zf_beamformer(H, p_max_watt, noise_watt)
    res_rzf = rzf_beamformer(H, p_max_watt, noise_watt)
    res_wmmse = wmmse_beamformer(H, p_max_watt, noise_watt, cfg.optimizer.wmmse_max_iter, cfg.optimizer.wmmse_tol)

    base_times_ms: dict[str, float] = {}
    for b_name, b_call in (
        ("MRT", lambda: mrt_beamformer(H, p_max_watt, noise_watt)),
        ("ZF", lambda: zf_beamformer(H, p_max_watt, noise_watt)),
        ("RZF", lambda: rzf_beamformer(H, p_max_watt, noise_watt)),
        ("WMMSE", lambda: wmmse_beamformer(H, p_max_watt, noise_watt, cfg.optimizer.wmmse_max_iter, cfg.optimizer.wmmse_tol)),
    ):
        t0 = time.perf_counter()
        n_rep = 100
        for _ in range(n_rep):
            b_call()
        base_times_ms[b_name] = (time.perf_counter() - t0) * 1e3 / n_rep

    n_runs = cfg.stability_runs
    direct_iters = cfg.optimizer.direct_max_iter
    struct_iters = cfg.optimizer.struct_max_iter

    histories: dict[str, np.ndarray] = {
        "Direct-GA": np.zeros((n_runs, direct_iters + 1), dtype=np.float64),
        "Hybrid-GA": np.zeros((n_runs, direct_iters + 1), dtype=np.float64),
        "Struct-GA": np.zeros((n_runs, struct_iters + 1), dtype=np.float64),
    }
    final_rates: dict[str, np.ndarray] = {name: np.zeros(n_runs, dtype=np.float64) for name in histories}
    runtimes_ms: dict[str, np.ndarray] = {name: np.zeros(n_runs, dtype=np.float64) for name in histories}
    n_evals_map: dict[str, int] = {}

    solver_specs = (
        ("Direct-GA", direct_ga_beamformer),
        ("Hybrid-GA", hybrid_ga_beamformer),
        ("Struct-GA", struct_ga_beamformer),
    )

    for run_idx in tqdm(range(n_runs), desc="Exp 6: GA convergence & stability", unit="run"):
        for s_idx, (name, solver_fn) in enumerate(solver_specs):
            run_rng = np.random.default_rng(cfg.seed + 6100 + run_idx * 17 + s_idx)
            res = solver_fn(H, p_max_watt, noise_watt, cfg.optimizer, run_rng)
            histories[name][run_idx] = res.history
            final_rates[name][run_idx] = res.best_sum_rate
            runtimes_ms[name][run_idx] = res.runtime_sec * 1e3
            n_evals_map[name] = res.n_evals

    npz_out: dict[str, np.ndarray] = {
        "mrt_rate": np.array([res_mrt.sum_rate]),
        "zf_rate": np.array([res_zf.sum_rate]),
        "rzf_rate": np.array([res_rzf.sum_rate]),
        "wmmse_rate": np.array([res_wmmse.sum_rate]),
        "wmmse_history": res_wmmse.history,
    }
    for name in histories:
        key = name.replace("-", "_")
        npz_out[f"hist_{key}"] = histories[name]
        npz_out[f"final_{key}"] = final_rates[name]
        npz_out[f"time_ms_{key}"] = runtimes_ms[name]

    np.savez_compressed(data_dir / "exp6_convergence_stability.npz", **npz_out)

    summary_rows = [
        {
            "Method": "MRT",
            "Encoding_Dim": 0,
            "Mean_SumRate_bps_Hz": res_mrt.sum_rate,
            "Std_SumRate_bps_Hz": 0.0,
            "Min_SumRate_bps_Hz": res_mrt.sum_rate,
            "Max_SumRate_bps_Hz": res_mrt.sum_rate,
            "CV_percent": 0.0,
            "Mean_Runtime_ms": base_times_ms["MRT"],
            "Function_Evals": 1,
        },
        {
            "Method": "ZF",
            "Encoding_Dim": 0,
            "Mean_SumRate_bps_Hz": res_zf.sum_rate,
            "Std_SumRate_bps_Hz": 0.0,
            "Min_SumRate_bps_Hz": res_zf.sum_rate,
            "Max_SumRate_bps_Hz": res_zf.sum_rate,
            "CV_percent": 0.0,
            "Mean_Runtime_ms": base_times_ms["ZF"],
            "Function_Evals": 1,
        },
        {
            "Method": "RZF",
            "Encoding_Dim": 0,
            "Mean_SumRate_bps_Hz": res_rzf.sum_rate,
            "Std_SumRate_bps_Hz": 0.0,
            "Min_SumRate_bps_Hz": res_rzf.sum_rate,
            "Max_SumRate_bps_Hz": res_rzf.sum_rate,
            "CV_percent": 0.0,
            "Mean_Runtime_ms": base_times_ms["RZF"],
            "Function_Evals": 1,
        },
        {
            "Method": "WMMSE",
            "Encoding_Dim": 0,
            "Mean_SumRate_bps_Hz": res_wmmse.sum_rate,
            "Std_SumRate_bps_Hz": 0.0,
            "Min_SumRate_bps_Hz": res_wmmse.sum_rate,
            "Max_SumRate_bps_Hz": res_wmmse.sum_rate,
            "CV_percent": 0.0,
            "Mean_Runtime_ms": base_times_ms["WMMSE"],
            "Function_Evals": int(len(res_wmmse.history)),
        },
    ]

    for name in histories:
        vals = final_rates[name]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals, ddof=1)) if n_runs > 1 else 0.0
        dim = 2 * Nt * K if name in ("Direct-GA", "Hybrid-GA") else 2 * K
        summary_rows.append(
            {
                "Method": name,
                "Encoding_Dim": dim,
                "Mean_SumRate_bps_Hz": mean_v,
                "Std_SumRate_bps_Hz": std_v,
                "Min_SumRate_bps_Hz": float(np.min(vals)),
                "Max_SumRate_bps_Hz": float(np.max(vals)),
                "CV_percent": 100.0 * std_v / max(mean_v, 1e-12),
                "Mean_Runtime_ms": float(np.mean(runtimes_ms[name])),
                "Function_Evals": n_evals_map[name],
            }
        )

    df = pd.DataFrame(summary_rows)
    df.to_csv(data_dir / "exp6_stability_runtime.csv", index=False, float_format="%.6f")
    return df


def run_all_experiments(
    cfg: ExperimentConfig = DEFAULT_CONFIG,
    data_dir: Path = DATA_DIR,
) -> None:
    """Execute all six experiments sequentially and save outputs to data_dir."""
    t_start = time.perf_counter()
    ensure_dirs(data_dir)

    print(
        f"[BTL2-GA] Running experiments (seed={cfg.seed}, num_trials={cfg.num_trials}, "
        f"stability_runs={cfg.stability_runs})..."
    )
    run_exp1_geometry_and_beampattern(cfg, data_dir)
    run_exp2_power_sweep(cfg, data_dir)
    run_exp3_antenna_sweep(cfg, data_dir)
    run_exp4_user_sweep(cfg, data_dir)
    run_exp5_rician_k_sweep(cfg, data_dir)
    run_exp6_convergence_stability_runtime(cfg, data_dir)

    total_s = time.perf_counter() - t_start
    print(f"[BTL2-GA] All experiments completed in {total_s:.2f} s. Data saved to: {data_dir}")
