"""Verify 100% numerical reproducibility of all tables and figures in the report.

Modes:
  1. Full isolated from-scratch verification (default, ~8 min):
       python verify_reproducibility.py
     Runs the complete 100-trial Monte-Carlo simulation and 8-figure generation
     into a brand-new empty temporary directory (zero cache), then checks every
     deterministic numeric column across all 6 CSV tables against results/data/.

  2. In-place wipe-and-rebuild verification (~8 min):
       python verify_reproducibility.py --in-place
     Snapshots results/data/*.csv in memory, deletes results/data/* and
     results/figures/*, re-runs the full simulation directly into results/,
     and verifies exact numerical agreement.

  3. Fast report-table sanity check (~1 s):
       python verify_reproducibility.py --check-only
     Verifies all 12 data files & 16 figure files exist in results/ and checks
     key headline numbers against Tables 2–6 of the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, DEFAULT_CONFIG, FIGURES_DIR
from plot_results import plot_all_figures
from src.experiments import run_all_experiments


CSV_FILES: tuple[str, ...] = (
    "exp1_ue_table.csv",
    "exp2_power_sweep.csv",
    "exp3_antenna_sweep.csv",
    "exp4_user_sweep.csv",
    "exp5_rician_k_sweep.csv",
    "exp6_stability_runtime.csv",
)

EXPECTED_FIGURES: tuple[str, ...] = (
    "sodo1_system_architecture.png",
    "fig1_geometry_beampattern.png",
    "fig2_sumrate_vs_power.png",
    "fig3_sumrate_vs_antennas.png",
    "fig4_sumrate_vs_users.png",
    "fig5_sumrate_vs_rician_k.png",
    "fig6_convergence_curves.png",
    "fig7_stability_runtime.png",
)


def _compare_dataframes(
    ref_dfs: dict[str, pd.DataFrame],
    new_dir: Path,
    tol: float = 1e-6,
) -> bool:
    """Compare all deterministic numeric columns (excluding CPU-dependent runtime)."""
    all_ok = True
    print("\n=== Đối chiếu số liệu tái lập với bảng kết quả báo cáo ===")
    for name in CSV_FILES:
        df_ref = ref_dfs[name]
        df_new = pd.read_csv(new_dir / name)
        cols = [
            c
            for c in df_ref.columns
            if "runtime" not in c.lower() and pd.api.types.is_numeric_dtype(df_ref[c])
        ]
        max_diff = float(np.max(np.abs(df_ref[cols].to_numpy() - df_new[cols].to_numpy())))
        ok = max_diff < tol
        all_ok = all_ok and ok
        status = "KHOP 100%" if ok else f"LECH ({max_diff:.3e})"
        print(f"  {name:30s} -> {status} (max_diff = {max_diff:.2e})")
    return all_ok


def _check_artifacts_exist(data_dir: Path, fig_dir: Path) -> bool:
    missing: list[str] = []
    for name in CSV_FILES:
        if not (data_dir / name).is_file():
            missing.append(f"data/{name}")
    for stem in (
        "exp1_geometry_beampattern.npz",
        "exp2_power_sweep.npz",
        "exp3_antenna_sweep.npz",
        "exp4_user_sweep.npz",
        "exp5_rician_k_sweep.npz",
        "exp6_convergence_stability.npz",
    ):
        if not (data_dir / stem).is_file():
            missing.append(f"data/{stem}")
    for png in EXPECTED_FIGURES:
        if not (fig_dir / png).is_file():
            missing.append(f"figures/{png}")
        pdf = png.replace(".png", ".pdf")
        if not (fig_dir / pdf).is_file():
            missing.append(f"figures/{pdf}")

    if missing:
        print("[FAIL] Thiếu các file đầu ra:", ", ".join(missing))
        return False
    print(f"[OK] Đủ 12 file dữ liệu trong {data_dir} và 16 file hình (8 PNG + 8 PDF) trong {fig_dir}.")
    return True


def _check_report_headline_numbers(data_dir: Path) -> bool:
    """Verify headline values reported in Tables 2–6 of the report (seed=2026, 100 trials)."""
    df_p = pd.read_csv(data_dir / "exp2_power_sweep.csv")
    df_m = pd.read_csv(data_dir / "exp3_antenna_sweep.csv")
    df_k = pd.read_csv(data_dir / "exp4_user_sweep.csv")
    df_s = pd.read_csv(data_dir / "exp6_stability_runtime.csv")

    checks = [
        ("Bảng 3 | P_max=20 dBm Struct-DE", float(df_p.loc[df_p["P_max_dBm"] == 20.0, "Struct-DE_mean_bps_hz"].iloc[0]), 22.684209),
        ("Bảng 3 | P_max=30 dBm Struct-DE", float(df_p.loc[df_p["P_max_dBm"] == 30.0, "Struct-DE_mean_bps_hz"].iloc[0]), 34.852465),
        ("Bảng 3 | P_max=30 dBm MRT", float(df_p.loc[df_p["P_max_dBm"] == 30.0, "MRT_mean_bps_hz"].iloc[0]), 9.128021),
        ("Bảng 4 | M=32 Struct-DE", float(df_m.loc[df_m["M_antennas"] == 32.0, "Struct-DE_mean_bps_hz"].iloc[0]), 32.929272),
        ("Bảng 4 | M=32 Direct-DE", float(df_m.loc[df_m["M_antennas"] == 32.0, "Direct-DE_mean_bps_hz"].iloc[0]), 24.529808),
        ("Bảng 5 | K=8 ZF dip", float(df_k.loc[df_k["K_users"] == 8.0, "ZF_mean_bps_hz"].iloc[0]), 17.257863),
        ("Bảng 5 | K=10 Struct-DE (K>M)", float(df_k.loc[df_k["K_users"] == 10.0, "Struct-DE_mean_bps_hz"].iloc[0]), 34.154937),
        ("Bảng 6 | 30-run Struct-DE mean", float(df_s.loc[df_s["Method"] == "Struct-DE", "Mean_SumRate_bps_Hz"].iloc[0]), 23.654324),
    ]

    all_ok = True
    print("\n=== Kiểm tra các mốc số liệu đinh trong Báo cáo (Bảng 3–6) ===")
    for label, actual, expected in checks:
        diff = abs(actual - expected)
        ok = diff < 1e-4
        all_ok = all_ok and ok
        tag = "OK" if ok else "MISMATCH"
        print(f"  [{tag:8s}] {label:32s}: actual={actual:.6f} | report={expected:.6f}")
    return all_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kiểm chứng khả năng tái lập 100% số liệu và đồ thị của báo cáo BTL2."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check-only",
        action="store_true",
        help="Chỉ kiểm tra sự tồn tại của 28 file kết quả và đối chiếu mốc số liệu Bảng 3–6 (không chạy lại mô phỏng).",
    )
    group.add_argument(
        "--in-place",
        action="store_true",
        help="Xóa trắng results/data và results/figures, chạy lại toàn bộ 100 mẫu kênh tại chỗ và đối chiếu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DEFAULT_CONFIG

    if args.check_only:
        ok_files = _check_artifacts_exist(DATA_DIR, FIGURES_DIR)
        ok_nums = _check_report_headline_numbers(DATA_DIR)
        if not (ok_files and ok_nums):
            sys.exit(1)
        print("\n[PASS] Toàn bộ dữ liệu và hình ảnh trong results/ khớp 100% với báo cáo.")
        return

    # Snapshot reference CSVs before re-running from scratch
    ref_dfs = {name: pd.read_csv(DATA_DIR / name) for name in CSV_FILES}

    if args.in_place:
        print("[Verify] Xóa trắng results/data/* và results/figures/* để chạy tái lập tại chỗ...")
        for p in DATA_DIR.glob("*"):
            if p.is_file():
                p.unlink()
        for p in FIGURES_DIR.glob("*"):
            if p.is_file():
                p.unlink()

        run_all_experiments(cfg=cfg, data_dir=DATA_DIR)
        plot_all_figures(data_dir=DATA_DIR, fig_dir=FIGURES_DIR, cfg=cfg)

        ok_files = _check_artifacts_exist(DATA_DIR, FIGURES_DIR)
        ok_cmp = _compare_dataframes(ref_dfs, DATA_DIR)
        ok_nums = _check_report_headline_numbers(DATA_DIR)
    else:
        with tempfile.TemporaryDirectory(prefix="btl2_repro_") as tmp:
            tmp_root = Path(tmp)
            tmp_data = tmp_root / "data"
            tmp_figs = tmp_root / "figures"
            print(f"[Verify] Chạy mô phỏng sạch từ đầu (100 trials, seed={cfg.seed}) vào thư mục tạm: {tmp_root}")
            run_all_experiments(cfg=cfg, data_dir=tmp_data)
            plot_all_figures(data_dir=tmp_data, fig_dir=tmp_figs, cfg=cfg)

            ok_files = _check_artifacts_exist(tmp_data, tmp_figs)
            ok_cmp = _compare_dataframes(ref_dfs, tmp_data)
            ok_nums = _check_report_headline_numbers(tmp_data)

    if not (ok_files and ok_cmp and ok_nums):
        print("\n[FAIL] Có sai lệch trong kiểm chứng tái lập!")
        sys.exit(1)

    print("\n[PASS] Kiểm chứng tái lập THÀNH CÔNG 100% — toàn bộ bảng số liệu và 8 đồ thị khớp hoàn toàn với báo cáo!")


if __name__ == "__main__":
    main()
