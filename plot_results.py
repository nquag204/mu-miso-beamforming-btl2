"""Generate 300-DPI IEEE-publication-ready plots from cached Monte-Carlo data.

Can be executed standalone after main.py:
    $env:MPLBACKEND = "Agg"; python "btl2\\plot_results.py"
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Default to non-interactive backend when run headlessly
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, DEFAULT_CONFIG, FIGURES_DIR, ExperimentConfig
from src.utils import ensure_dirs


# Consistent colorblind-friendly style registry across all figures
METHOD_STYLES: dict[str, dict] = {
    "WMMSE": {"color": "#111111", "linestyle": "-", "marker": "s", "linewidth": 2.1, "markersize": 6.0, "zorder": 9},
    "Struct-DE": {"color": "#D62728", "linestyle": "--", "marker": "o", "linewidth": 2.2, "markersize": 6.0, "zorder": 10},
    "Struct-PSO": {"color": "#FF7F0E", "linestyle": "-.", "marker": "^", "linewidth": 1.9, "markersize": 5.8, "zorder": 8},
    "Struct-GA": {"color": "#9467BD", "linestyle": ":", "marker": "D", "linewidth": 1.9, "markersize": 5.2, "zorder": 7},
    "RZF": {"color": "#1F77B4", "linestyle": "-", "marker": "v", "linewidth": 1.8, "markersize": 5.5, "zorder": 6},
    "ZF": {"color": "#2CA02C", "linestyle": "--", "marker": "P", "linewidth": 1.7, "markersize": 5.5, "zorder": 5},
    "Direct-DE": {"color": "#E377C2", "linestyle": "-", "marker": "x", "linewidth": 1.6, "markersize": 6.0, "zorder": 4},
    "Direct-PSO": {"color": "#17BECF", "linestyle": "-.", "marker": "+", "linewidth": 1.6, "markersize": 6.5, "zorder": 3},
    "Direct-GA": {"color": "#8C564B", "linestyle": ":", "marker": "1", "linewidth": 1.6, "markersize": 7.0, "zorder": 2},
    "MRT": {"color": "#7F7F7F", "linestyle": "--", "marker": "*", "linewidth": 1.7, "markersize": 6.5, "zorder": 1},
}

SWEEP_PLOT_ORDER: tuple[str, ...] = (
    "WMMSE",
    "Struct-DE",
    "Struct-PSO",
    "Struct-GA",
    "RZF",
    "ZF",
    "Direct-DE",
    "Direct-PSO",
    "Direct-GA",
    "MRT",
)


def _configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.65,
            "savefig.bbox": "tight",
        }
    )


def _save_fig(fig: plt.Figure, stem: str, fig_dir: Path, cfg: ExperimentConfig) -> None:
    png_path = fig_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=cfg.figure_dpi)
    if cfg.save_pdf:
        fig.savefig(fig_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  [Saved] {png_path.name}")


def plot_exp1_geometry_and_beampattern(
    data_dir: Path = DATA_DIR,
    fig_dir: Path = FIGURES_DIR,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> None:
    """Fig 1: Two-panel system scenario (UE spatial sector) + ULA normalized beampattern."""
    d = np.load(data_dir / "exp1_geometry_beampattern.npz")
    positions_xy = d["positions_xy"]
    angles_rad = d["angles_rad"]
    distances_m = d["distances_m"]
    angles_grid_deg = d["angles_grid_deg"]
    pat_sde = d["pat_sde"]
    rate_mrt = float(d["rate_mrt"][0])
    rate_sde = float(d["rate_sde"][0])
    rate_wmmse = float(d["rate_wmmse"][0])

    s = cfg.system
    K = positions_xy.shape[0]
    ue_colors = ["#1F77B4", "#D62728", "#2CA02C", "#FF7F0E", "#9467BD", "#17BECF"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))

    # Left panel: 2D annular sector deployment
    half_rad = np.deg2rad(s.sector_angle_deg) / 2.0
    arc = np.linspace(-half_rad, half_rad, 200)
    ax1.plot(s.r_min_m * np.cos(arc), s.r_min_m * np.sin(arc), "k--", lw=1.1, alpha=0.6)
    ax1.plot(s.r_max_m * np.cos(arc), s.r_max_m * np.sin(arc), "k-", lw=1.3, alpha=0.8)
    for boundary in (-half_rad, half_rad):
        ax1.plot(
            [s.r_min_m * np.cos(boundary), s.r_max_m * np.cos(boundary)],
            [s.r_min_m * np.sin(boundary), s.r_max_m * np.sin(boundary)],
            "k-",
            lw=1.3,
            alpha=0.8,
        )
    # Fill annular sector
    outer_x, outer_y = s.r_max_m * np.cos(arc), s.r_max_m * np.sin(arc)
    inner_x, inner_y = s.r_min_m * np.cos(arc[::-1]), s.r_min_m * np.sin(arc[::-1])
    ax1.fill(np.concatenate([outer_x, inner_x]), np.concatenate([outer_y, inner_y]), color="#E8F1FA", alpha=0.55)

    ax1.scatter([0.0], [0.0], marker="^", s=160, color="#111111", zorder=10, label=f"BS (ULA, M={s.default_m})")
    for k in range(K):
        xk, yk = positions_xy[k]
        c = ue_colors[k % len(ue_colors)]
        ax1.plot([0.0, xk], [0.0, yk], linestyle=":", color=c, alpha=0.75, lw=1.1)
        ax1.scatter([xk], [yk], s=95, color=c, edgecolor="black", linewidth=0.8, zorder=9,
                    label=f"UE {k+1} (d={distances_m[k]:.1f} m, θ={np.rad2deg(angles_rad[k]):+.1f}°)")
        ax1.annotate(f"UE {k+1}", (xk, yk), xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold", color=c)

    ax1.set_title("(a) Kịch bản phủ sóng MU-MISO (Sector 120°)")
    ax1.set_xlabel("Tọa độ x (m)")
    ax1.set_ylabel("Tọa độ y (m)")
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlim(-10, s.r_max_m + 12)
    ax1.set_ylim(-s.r_max_m * np.sin(half_rad) - 12, s.r_max_m * np.sin(half_rad) + 12)
    ax1.legend(loc="lower left", fontsize=8.0, framealpha=0.92)

    # Right panel: Angular beampattern (dB normalized per beam peak) of Struct-DE
    max_pat = float(np.max(pat_sde))
    for k in range(K):
        c = ue_colors[k % len(ue_colors)]
        pat_db = 10.0 * np.log10(np.maximum(pat_sde[k] / max(max_pat, 1e-30), 1e-6))
        ax2.plot(angles_grid_deg, pat_db, color=c, lw=1.8, label=f"Búp sóng UE {k+1}")
        ax2.axvline(np.rad2deg(angles_rad[k]), color=c, linestyle="--", lw=1.0, alpha=0.7)

    ax2.set_title(
        f"(b) Giản đồ búp sóng Struct-DE (R={rate_sde:.2f} vs WMMSE {rate_wmmse:.2f}, MRT {rate_mrt:.2f} bps/Hz)"
    )
    ax2.set_xlabel("Góc phương vị θ (độ)")
    ax2.set_ylabel("Độ lợi bức xạ chuẩn hóa (dB)")
    ax2.set_xlim(-75, 75)
    ax2.set_ylim(-35, 2)
    ax2.legend(loc="lower center", ncol=2, fontsize=8.5, framealpha=0.92)

    fig.tight_layout()
    _save_fig(fig, "fig1_geometry_beampattern", fig_dir, cfg)


def _plot_sweep_figure(
    npz_path: Path,
    stem: str,
    title: str,
    xlabel: str,
    fig_dir: Path,
    cfg: ExperimentConfig,
    legend_loc: str = "best",
    vertical_note: tuple[float, str] | None = None,
) -> None:
    d = np.load(npz_path)
    x_vals = d["x_values"]

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for m in SWEEP_PLOT_ORDER:
        key = f"rates_{m.replace('-', '_')}"
        if key not in d:
            continue
        mean_y = np.mean(d[key], axis=1)
        ax.plot(x_vals, mean_y, label=m, **METHOD_STYLES[m])

    if vertical_note is not None:
        xv, note = vertical_note
        ax.axvline(xv, color="#333333", linestyle=":", linewidth=1.3, alpha=0.8)
        ylim = ax.get_ylim()
        ax.text(
            xv + 0.15,
            ylim[0] + 0.06 * (ylim[1] - ylim[0]),
            note,
            fontsize=8.8,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF8E1", edgecolor="#888888", alpha=0.9),
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tổng tốc độ bit trung bình (bps/Hz)")
    ax.set_xticks(x_vals)
    ax.legend(loc=legend_loc, ncol=2, framealpha=0.93)
    fig.tight_layout()
    _save_fig(fig, stem, fig_dir, cfg)


def plot_exp2_power_sweep(
    data_dir: Path = DATA_DIR,
    fig_dir: Path = FIGURES_DIR,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> None:
    s = cfg.system
    _plot_sweep_figure(
        npz_path=data_dir / "exp2_power_sweep.npz",
        stem="fig2_sumrate_vs_power",
        title=f"Tổng tốc độ bit trung bình theo công suất phát BS (M={s.default_m}, K={s.default_k}, K_Rician={s.rician_k_db:.0f} dB)",
        xlabel="Công suất phát tối đa của trạm gốc P_max (dBm)",
        fig_dir=fig_dir,
        cfg=cfg,
        legend_loc="upper left",
    )


def plot_exp3_antenna_sweep(
    data_dir: Path = DATA_DIR,
    fig_dir: Path = FIGURES_DIR,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> None:
    s = cfg.system
    _plot_sweep_figure(
        npz_path=data_dir / "exp3_antenna_sweep.npz",
        stem="fig3_sumrate_vs_antennas",
        title=f"Ảnh hưởng của số anten trạm gốc M đến tổng tốc độ bit (K={s.default_k}, P_max={s.default_p_max_dbm:.0f} dBm)",
        xlabel="Số lượng anten trạm gốc M",
        fig_dir=fig_dir,
        cfg=cfg,
        legend_loc="upper left",
    )


def plot_exp4_user_sweep(
    data_dir: Path = DATA_DIR,
    fig_dir: Path = FIGURES_DIR,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> None:
    s = cfg.system
    _plot_sweep_figure(
        npz_path=data_dir / "exp4_user_sweep.npz",
        stem="fig4_sumrate_vs_users",
        title=f"Ảnh hưởng của số người dùng K đến tổng tốc độ bit (M={s.default_m}, P_max={s.default_p_max_dbm:.0f} dBm)",
        xlabel="Số lượng người dùng đơn anten K",
        fig_dir=fig_dir,
        cfg=cfg,
        legend_loc="upper left",
        vertical_note=(float(s.default_m), "Ranh giới quá tải\n(K = M = 8)"),
    )


def plot_exp5_rician_k_sweep(
    data_dir: Path = DATA_DIR,
    fig_dir: Path = FIGURES_DIR,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> None:
    s = cfg.system
    _plot_sweep_figure(
        npz_path=data_dir / "exp5_rician_k_sweep.npz",
        stem="fig5_sumrate_vs_rician_k",
        title=f"Ảnh hưởng của hệ số Rician K-factor (M={s.default_m}, K={s.default_k}, P_max={s.default_p_max_dbm:.0f} dBm)",
        xlabel="Hệ số Rician K-factor (dB)  [Rayleigh NLoS  →  LoS mạnh]",
        fig_dir=fig_dir,
        cfg=cfg,
        legend_loc="center right",
    )


def plot_exp6_convergence_and_stability(
    data_dir: Path = DATA_DIR,
    fig_dir: Path = FIGURES_DIR,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> None:
    """Fig 6 (Convergence envelopes) & Fig 7 (Stability boxplot + Runtime log-bar)."""
    d = np.load(data_dir / "exp6_convergence_stability.npz")
    mrt_rate = float(d["mrt_rate"][0])
    zf_rate = float(d["zf_rate"][0])
    rzf_rate = float(d["rzf_rate"][0])
    wmmse_rate = float(d["wmmse_rate"][0])

    # --- Figure 6: Convergence Curves (Direct vs Structure-Guided) ---
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.2, 4.9), sharey=True)

    for ax, group, title_sub in (
        (ax_l, ("Direct-DE", "Direct-PSO", "Direct-GA"), "(a) Biểu diễn trực tiếp (2MK = 64 biến)"),
        (ax_r, ("Struct-DE", "Struct-PSO", "Struct-GA"), "(b) Biểu diễn theo cấu trúc tối ưu (2K = 8 biến)"),
    ):
        for name in group:
            hist = d[f"hist_{name.replace('-', '_')}"]   # (n_runs, iters+1)
            iters = np.arange(hist.shape[1])
            mean_h = np.mean(hist, axis=0)
            std_h = np.std(hist, axis=0)
            style = METHOD_STYLES[name]
            ax.plot(iters, mean_h, color=style["color"], linestyle=style["linestyle"], linewidth=2.1, label=f"{name} (mean ± 1σ)")
            ax.fill_between(iters, mean_h - std_h, mean_h + std_h, color=style["color"], alpha=0.18)

        ax.axhline(wmmse_rate, color="#111111", linestyle="-", lw=1.5, label=f"WMMSE ({wmmse_rate:.2f})")
        ax.axhline(rzf_rate, color="#1F77B4", linestyle="--", lw=1.3, label=f"RZF ({rzf_rate:.2f})")
        ax.axhline(zf_rate, color="#2CA02C", linestyle=":", lw=1.3, label=f"ZF ({zf_rate:.2f})")
        ax.axhline(mrt_rate, color="#7F7F7F", linestyle="-.", lw=1.3, label=f"MRT ({mrt_rate:.2f})")

        ax.set_title(title_sub)
        ax.set_xlabel("Thế hệ / Vòng lặp (Iteration)")
        ax.legend(loc="lower right", fontsize=8.2, framealpha=0.92)

    ax_l.set_ylabel("Tổng tốc độ bit tốt nhất (bps/Hz)")

    # Add zoomed-in inset inside panel (b) placed cleanly above the legend
    inset_ax = ax_r.inset_axes([0.11, 0.45, 0.55, 0.36])
    for name in ("Struct-DE", "Struct-PSO", "Struct-GA"):
        hist = d[f"hist_{name.replace('-', '_')}"]
        iters_z = np.arange(min(26, hist.shape[1]))
        mean_z = np.mean(hist[:, : len(iters_z)], axis=0)
        std_z = np.std(hist[:, : len(iters_z)], axis=0)
        style = METHOD_STYLES[name]
        inset_ax.plot(iters_z, mean_z, color=style["color"], linestyle=style["linestyle"], linewidth=1.8)
        inset_ax.fill_between(iters_z, mean_z - std_z, mean_z + std_z, color=style["color"], alpha=0.2)
    inset_ax.set_xlim(0, 25)
    y_min_z = min(float(np.min(np.mean(d[f"hist_{n.replace('-', '_')}"], axis=0))) for n in ("Struct-DE", "Struct-PSO", "Struct-GA")) - 0.45
    y_max_z = max(float(np.max(np.mean(d[f"hist_{n.replace('-', '_')}"], axis=0))) for n in ("Struct-DE", "Struct-PSO", "Struct-GA")) + 0.18
    inset_ax.set_ylim(y_min_z, y_max_z)
    inset_ax.set_title("Phóng to 25 vòng đầu", fontsize=8.5, fontweight="bold")
    inset_ax.tick_params(labelsize=7.8)
    inset_ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(f"Đường cong hội tụ trung bình trên {cfg.stability_runs} lần chạy độc lập (M={cfg.system.default_m}, K={cfg.system.default_k}, P_max={cfg.system.default_p_max_dbm:.0f} dBm)", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "fig6_convergence_curves", fig_dir, cfg)

    # --- Figure 7: Stability Boxplot & Computational Runtime ---
    meta_names = ["Direct-GA", "Direct-PSO", "Direct-DE", "Struct-GA", "Struct-PSO", "Struct-DE"]
    box_data = [d[f"final_{m.replace('-', '_')}"] for m in meta_names]
    mean_ms = [float(np.mean(d[f"time_ms_{m.replace('-', '_')}"])) for m in meta_names]

    # Also load baseline runtimes from CSV for complete runtime panel
    csv_path = data_dir / "exp6_stability_runtime.csv"
    import pandas as pd

    df_rt = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))

    bp = ax1.boxplot(box_data, tick_labels=meta_names, patch_artist=True, widths=0.55)
    for patch, m in zip(bp["boxes"], meta_names):
        patch.set_facecolor(METHOD_STYLES[m]["color"])
        patch.set_alpha(0.55)
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.8)

    ax1.axhline(wmmse_rate, color="#111111", linestyle="--", lw=1.4, label=f"WMMSE ({wmmse_rate:.2f} bps/Hz)")
    ax1.axhline(rzf_rate, color="#1F77B4", linestyle=":", lw=1.3, label=f"RZF ({rzf_rate:.2f} bps/Hz)")
    ax1.set_title(f"(a) Độ ổn định nghiệm qua {cfg.stability_runs} lần chạy độc lập")
    ax1.set_ylabel("Tổng tốc độ bit cuối cùng (bps/Hz)")
    ax1.tick_params(axis="x", rotation=22)
    ax1.legend(loc="lower right", fontsize=8.6)

    # Runtime horizontal bar chart (log scale)
    rt_methods = df_rt["Method"].tolist()
    rt_vals = df_rt["Mean_Runtime_ms"].tolist()
    bar_colors = [METHOD_STYLES.get(m, {"color": "#555555"})["color"] for m in rt_methods]

    y_pos = np.arange(len(rt_methods))
    bars = ax2.barh(y_pos, rt_vals, color=bar_colors, alpha=0.82, edgecolor="#222222", linewidth=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(rt_methods)
    ax2.set_xscale("log")
    ax2.set_xlabel("Thời gian thực thi trung bình mỗi kênh (ms, thang log)")
    ax2.set_title("(b) So sánh chi phí tính toán (Runtime)")
    ax2.invert_yaxis()

    for bar, val in zip(bars, rt_vals):
        ax2.text(
            val * 1.18,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.2f} ms" if val >= 0.1 else f"{val:.3f} ms",
            va="center",
            fontsize=8.3,
        )
    ax2.set_xlim(right=max(rt_vals) * 4.5)

    fig.tight_layout()
    _save_fig(fig, "fig7_stability_runtime", fig_dir, cfg)


def plot_system_architecture_diagram(
    fig_dir: Path = FIGURES_DIR,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> None:
    """Sơ đồ 1: MU-MISO downlink signal chain and metaheuristic optimization loop."""
    s = cfg.system
    fig, ax = plt.subplots(figsize=(12.2, 5.1))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def draw_box(
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str,
        fc: str,
        ec: str,
        title_color: str = "#0F2942",
    ) -> None:
        box = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.14",
            linewidth=1.5,
            edgecolor=ec,
            facecolor=fc,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2.0,
            y + h - 0.30,
            title,
            ha="center",
            va="top",
            fontsize=9.8,
            fontweight="bold",
            color=title_color,
            zorder=4,
        )
        ax.text(
            x + w / 2.0,
            y + h / 2.0 - 0.16,
            body,
            ha="center",
            va="center",
            fontsize=8.6,
            color="#222222",
            linespacing=1.32,
            zorder=4,
        )

    def draw_arrow(x1: float, y1: float, x2: float, y2: float, label: str = "", lx: float = 0.0, ly: float = 0.0, color: str = "#1F4E79", ls: str = "-") -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.32", lw=1.6, color=color, linestyle=ls),
            zorder=5,
        )
        if label:
            ax.text(lx, ly, label, ha="center", va="center", fontsize=8.4, fontweight="bold", color=color, zorder=6)

    # Outer grouping frame: Base Station (BS)
    bs_frame = patches.FancyBboxPatch(
        (0.25, 2.55),
        6.15,
        2.35,
        boxstyle="round,pad=0.06,rounding_size=0.16",
        linewidth=1.3,
        linestyle="--",
        edgecolor="#1F4E79",
        facecolor="#F4F8FC",
        zorder=1,
    )
    ax.add_patch(bs_frame)
    ax.text(0.45, 4.77, "TRẠM GỐC (BS – M anten ULA, f_c = 3.5 GHz, P_max)", fontsize=9.2, fontweight="bold", color="#1F4E79", va="top")

    # Top row: Signal chain
    draw_box(
        0.45, 2.80, 1.80, 1.60,
        "Nguồn dữ liệu",
        "K luồng độc lập\ns = [s₁, ..., s_K]ᵀ\nE[s sᴴ] = I_K",
        fc="#FFFFFF", ec="#1F4E79",
    )
    draw_box(
        2.65, 2.80, 2.05, 1.60,
        "Bộ tạo búp sóng W",
        "W = [w₁, ..., w_K] ∈ ℂ^(M×K)\nx = Σ w_k s_k\n||W||_F² = P_max",
        fc="#E3F0FB", ec="#1F4E79",
    )
    draw_box(
        5.05, 2.80, 1.20, 1.60,
        "Mảng ULA",
        "M phần tử\nd_a = λ/2\nSector 120°",
        fc="#FFFFFF", ec="#1F4E79",
    )
    draw_box(
        6.80, 2.80, 2.45, 1.60,
        "Kênh truyền Rician H",
        "d_k ∈ [20, 100] m, α = 3.0\nβ_k = β₀ (d_k / d₀)^(-α)\nh_k = √β_k (LoS + NLoS)",
        fc="#FFF6E5", ec="#D97706", title_color="#92400E",
    )
    draw_box(
        9.75, 2.80, 2.40, 1.60,
        "K Người dùng (UE)",
        "y_k = h_kᴴ w_k s_k + MUI + n_k\nn_k ~ CN(0, σ² = -94 dBm)\nSINR_k → R_sum = Σ log₂(1+SINR_k)",
        fc="#EAF7EE", ec="#2E7D32", title_color="#1B5E20",
    )

    draw_arrow(2.25, 3.60, 2.65, 3.60, "s", 2.45, 3.84)
    draw_arrow(4.70, 3.60, 5.05, 3.60, "x", 4.88, 3.84)
    draw_arrow(6.25, 3.60, 6.80, 3.60, "Phát", 6.53, 3.86)
    draw_arrow(9.25, 3.60, 9.75, 3.60, "Hᴴx", 9.50, 3.84)

    # Bottom row: Evolutionary Optimization Engine
    opt_frame = patches.FancyBboxPatch(
        (1.65, 0.25),
        8.65,
        1.80,
        boxstyle="round,pad=0.06,rounding_size=0.16",
        linewidth=1.5,
        edgecolor="#9B1C1C",
        facecolor="#FDF2F2",
        zorder=1,
    )
    ax.add_patch(opt_frame)
    ax.text(1.85, 1.94, "KHỐI TỐI ƯU HÓA BEAMFORMING (HEURISTIC / TIẾN HÓA & ĐỐI CHIẾU)", fontsize=9.2, fontweight="bold", color="#9B1C1C", va="top")

    draw_box(
        1.90, 0.42, 2.45, 1.22,
        "1. Mã hóa nghiệm",
        "• Direct: x ∈ [-1, 1]^(2MK)\n• Struct: z ∈ [-0.05, 1]^(2K)\n  (đối ngẫu p, λ độc lập M)",
        fc="#FFFFFF", ec="#B91C1C", title_color="#9B1C1C",
    )
    draw_box(
        4.75, 0.42, 2.55, 1.22,
        "2. Tiến hóa / Bầy đàn",
        "DE (dither F, CR=0.85)\nPSO (w: 0.9→0.4) | GA (BLX-α)\nChiếu công suất ||W||_F² = P_max",
        fc="#FFFFFF", ec="#B91C1C", title_color="#9B1C1C",
    )
    draw_box(
        7.70, 0.42, 2.35, 1.22,
        "3. Đánh giá & Chọn lọc",
        "Vector hóa NumPy / Gram O(K³)\nmax R_sum(W) qua T_max vòng\nTham chiếu: MRT, ZF, RZF, WMMSE",
        fc="#FFFFFF", ec="#B91C1C", title_color="#9B1C1C",
    )

    draw_arrow(4.35, 1.03, 4.75, 1.03, color="#9B1C1C")
    draw_arrow(7.30, 1.03, 7.70, 1.03, color="#9B1C1C")

    # Feedback CSI H from Channel straight down to Optimizer, and optimal W* straight up to Beamformer
    ax.annotate(
        "",
        xy=(8.02, 2.11),
        xytext=(8.02, 2.74),
        arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.32", lw=1.8, color="#B45309"),
        zorder=5,
    )
    ax.text(8.18, 2.42, "CSI hoàn hảo H", ha="left", va="center", fontsize=8.7, fontweight="bold", color="#B45309")

    ax.annotate(
        "",
        xy=(3.68, 2.74),
        xytext=(3.68, 2.11),
        arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.32", lw=1.8, color="#9B1C1C"),
        zorder=5,
    )
    ax.text(3.84, 2.38, "Nghiệm tối ưu W*", ha="left", va="center", fontsize=8.7, fontweight="bold", color="#9B1C1C")

    fig.suptitle(
        f"Sơ đồ kiến trúc hệ thống MU-MISO đường xuống và vòng lặp tối ưu Beamforming (M={s.default_m}, K={s.default_k})",
        fontsize=11.5,
        fontweight="bold",
        color="#0F2942",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig, "sodo1_system_architecture", fig_dir, cfg)


def plot_all_figures(
    data_dir: Path = DATA_DIR,
    fig_dir: Path = FIGURES_DIR,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> None:
    """Regenerate every figure in the report from saved .npz/.csv artifacts."""
    _configure_matplotlib()
    ensure_dirs(fig_dir)
    print(f"[BTL2] Generating 300-DPI IEEE figures into: {fig_dir}")
    plot_system_architecture_diagram(fig_dir, cfg)
    plot_exp1_geometry_and_beampattern(data_dir, fig_dir, cfg)
    plot_exp2_power_sweep(data_dir, fig_dir, cfg)
    plot_exp3_antenna_sweep(data_dir, fig_dir, cfg)
    plot_exp4_user_sweep(data_dir, fig_dir, cfg)
    plot_exp5_rician_k_sweep(data_dir, fig_dir, cfg)
    plot_exp6_convergence_and_stability(data_dir, fig_dir, cfg)
    print("[BTL2] All figures generated successfully.")


if __name__ == "__main__":
    plot_all_figures()
