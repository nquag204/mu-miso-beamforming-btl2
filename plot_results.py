"""Generate 300-DPI IEEE-publication-ready plots from cached Monte-Carlo data (GA focus).

Can be executed standalone after main.py:
    $env:MPLBACKEND = "Agg"; python plot_results.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, DEFAULT_CONFIG, FIGURES_DIR, ExperimentConfig
from src.utils import ensure_dirs


METHOD_STYLES: dict[str, dict] = {
    "Struct-GA": {"color": "#D62728", "linestyle": "-", "marker": "o", "linewidth": 2.3, "markersize": 6.2, "zorder": 10},
    "WMMSE": {"color": "#111111", "linestyle": "--", "marker": "s", "linewidth": 2.0, "markersize": 5.8, "zorder": 9},
    "Hybrid-GA": {"color": "#FF7F0E", "linestyle": "-.", "marker": "^", "linewidth": 2.0, "markersize": 6.0, "zorder": 8},
    "ZF": {"color": "#2CA02C", "linestyle": "--", "marker": "P", "linewidth": 1.8, "markersize": 5.6, "zorder": 6},
    "RZF": {"color": "#1F77B4", "linestyle": "-", "marker": "v", "linewidth": 1.8, "markersize": 5.6, "zorder": 5},
    "Direct-GA": {"color": "#9467BD", "linestyle": ":", "marker": "D", "linewidth": 2.0, "markersize": 5.4, "zorder": 4},
    "MRT": {"color": "#7F7F7F", "linestyle": "--", "marker": "*", "linewidth": 1.7, "markersize": 6.8, "zorder": 2},
}

SWEEP_PLOT_ORDER: tuple[str, ...] = (
    "Struct-GA",
    "WMMSE",
    "Hybrid-GA",
    "ZF",
    "RZF",
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
            "legend.fontsize": 9.0,
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


def plot_system_architecture_diagram(
    fig_dir: Path = FIGURES_DIR,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> None:
    """Sơ đồ 1: MU-MISO downlink polar circular model & Genetic Algorithm optimization loop."""
    s = cfg.system
    fig, ax = plt.subplots(figsize=(12.2, 5.15))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5.25)
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
            y + h - 0.28,
            title,
            ha="center",
            va="top",
            fontsize=9.7,
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
            fontsize=8.5,
            color="#222222",
            linespacing=1.32,
            zorder=4,
        )

    def draw_arrow(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        label: str = "",
        lx: float = 0.0,
        ly: float = 0.0,
        color: str = "#1F4E79",
    ) -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.32", lw=1.6, color=color),
            zorder=5,
        )
        if label:
            ax.text(lx, ly, label, ha="center", va="center", fontsize=8.4, fontweight="bold", color=color, zorder=6)

    bs_frame = patches.FancyBboxPatch(
        (0.25, 2.58),
        6.10,
        2.36,
        boxstyle="round,pad=0.06,rounding_size=0.16",
        linewidth=1.3,
        linestyle="--",
        edgecolor="#1F4E79",
        facecolor="#F4F8FC",
        zorder=1,
    )
    ax.add_patch(bs_frame)
    ax.text(
        0.45,
        4.82,
        f"TRẠM GỐC BS TẠI TÂM O(0,0) (N_t anten ULA, bán kính R = {s.cell_radius_m:.0f} m, P_max)",
        fontsize=9.1,
        fontweight="bold",
        color="#1F4E79",
        va="top",
    )

    draw_box(
        0.45, 2.82, 1.78, 1.60,
        "Nguồn dữ liệu",
        "K luồng độc lập\ns = [s₁, ..., s_K]ᵀ\nE[s sᴴ] = I_K",
        fc="#FFFFFF", ec="#1F4E79",
    )
    draw_box(
        2.63, 2.82, 2.05, 1.60,
        "Bộ tạo búp sóng W",
        "W = [w₁, ..., w_K] ∈ ℂ^(Nt×K)\nx = Σ w_k s_k\n||W||_F² = P_max",
        fc="#E3F0FB", ec="#1F4E79",
    )
    draw_box(
        5.03, 2.82, 1.18, 1.60,
        "Mảng ULA",
        "N_t anten\nd_a = λ/2\nPhủ 360°",
        fc="#FFFFFF", ec="#1F4E79",
    )
    draw_box(
        6.75, 2.82, 2.55, 1.60,
        "Kênh Rician cộng trực tiếp",
        f"r_k=R·rand(), Φ_k=2π·rand()\nPL_k = {s.pl0_db:.0f} + 10·{s.path_loss_exponent:.1f}·log₁₀(d_k/d₀)\nh_k = √PL_k · (h_NLoS + h_LoS)",
        fc="#FFF6E5", ec="#D97706", title_color="#92400E",
    )
    draw_box(
        9.75, 2.82, 2.40, 1.60,
        "K Người dùng (UE)",
        f"y_k = h_kᴴ w_k s_k + MUI + n_k\nn_k ~ CN(0, σ² = {s.noise_power_dbm:.0f} dBm)\nSINR_k → R_sum = Σ log₂(1+SINR_k)",
        fc="#EAF7EE", ec="#2E7D32", title_color="#1B5E20",
    )

    draw_arrow(2.23, 3.62, 2.63, 3.62, "s", 2.43, 3.85)
    draw_arrow(4.68, 3.62, 5.03, 3.62, "x", 4.86, 3.85)
    draw_arrow(6.21, 3.62, 6.75, 3.62, "Phát", 6.48, 3.87)
    draw_arrow(9.30, 3.62, 9.75, 3.62, "Hᴴx", 9.52, 3.85)

    opt_frame = patches.FancyBboxPatch(
        (1.60, 0.25),
        8.75,
        1.82,
        boxstyle="round,pad=0.06,rounding_size=0.16",
        linewidth=1.5,
        edgecolor="#9B1C1C",
        facecolor="#FDF2F2",
        zorder=1,
    )
    ax.add_patch(opt_frame)
    ax.text(
        1.80,
        1.96,
        "KHỐI TỐI ƯU BEAMFORMING BẰNG GIẢI THUẬT DI TRUYỀN (GENETIC ALGORITHM – GA)",
        fontsize=9.2,
        fontweight="bold",
        color="#9B1C1C",
        va="top",
    )

    draw_box(
        1.85, 0.42, 2.55, 1.24,
        "1. Mã hóa NST (Chromosome)",
        "• Direct-GA: 2·N_t·K gen thực\n• Hybrid-GA: Khởi tạo MRT/ZF/RZF\n• Struct-GA: 2K gen (p, λ) đối ngẫu",
        fc="#FFFFFF", ec="#B91C1C", title_color="#9B1C1C",
    )
    draw_box(
        4.75, 0.42, 2.60, 1.24,
        "2. Toán tử Tiến hóa GA",
        "Chọn lọc Tournament (cỡ 3)\nLai ghép BLX-α (p_c=0.9, α=0.3)\nĐột biến Gaussian + Elitism (2)",
        fc="#FFFFFF", ec="#B91C1C", title_color="#9B1C1C",
    )
    draw_box(
        7.70, 0.42, 2.40, 1.24,
        "3. Chiếu công suất & Fitness",
        "Chiếu chuẩn ||W||_F² = P_max\nĐánh giá lô R_sum / Gram O(K³)\nĐối chiếu: MRT, ZF, RZF, WMMSE",
        fc="#FFFFFF", ec="#B91C1C", title_color="#9B1C1C",
    )

    draw_arrow(4.40, 1.04, 4.75, 1.04, color="#9B1C1C")
    draw_arrow(7.35, 1.04, 7.70, 1.04, color="#9B1C1C")

    ax.annotate(
        "",
        xy=(8.02, 2.13),
        xytext=(8.02, 2.76),
        arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.32", lw=1.8, color="#B45309"),
        zorder=5,
    )
    ax.text(8.18, 2.44, "CSI hoàn hảo H (N_t×K)", ha="left", va="center", fontsize=8.6, fontweight="bold", color="#B45309")

    ax.annotate(
        "",
        xy=(3.65, 2.76),
        xytext=(3.65, 2.13),
        arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.32", lw=1.8, color="#9B1C1C"),
        zorder=5,
    )
    ax.text(3.82, 2.40, "Nghiệm tối ưu W*", ha="left", va="center", fontsize=8.7, fontweight="bold", color="#9B1C1C")

    fig.suptitle(
        f"Sơ đồ kiến trúc hệ thống MU-MISO tọa độ cực và vòng lặp tối ưu GA (N_t={s.default_nt}, K={s.default_k})",
        fontsize=11.5,
        fontweight="bold",
        color="#0F2942",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig, "sodo1_system_architecture", fig_dir, cfg)


def plot_exp1_geometry_and_beampattern(
    data_dir: Path = DATA_DIR,
    fig_dir: Path = FIGURES_DIR,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> None:
    """Fig 1: Polar circular coverage cell O(0,0) radius R + 360-deg polar beampattern of Struct-GA."""
    d = np.load(data_dir / "exp1_geometry_beampattern.npz")
    positions_xy = d["positions_xy"]
    radii_m = d["radii_m"]
    angles_rad = d["angles_rad"]
    angles_grid_rad = d["angles_grid_rad"]
    pat_sga = d["pat_sga"]
    rate_mrt = float(d["rate_mrt"][0])
    rate_sga = float(d["rate_sga"][0])
    rate_wmmse = float(d["rate_wmmse"][0])

    s = cfg.system
    K = positions_xy.shape[0]
    ue_colors = ["#1F77B4", "#D62728", "#2CA02C", "#FF7F0E", "#9467BD", "#17BECF"]

    fig = plt.figure(figsize=(12.2, 5.3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")

    # Left panel: Full circular cell centered at O(0, 0) with radius R
    circle_ang = np.linspace(0.0, 2.0 * np.pi, 360)
    cx, cy = s.cell_radius_m * np.cos(circle_ang), s.cell_radius_m * np.sin(circle_ang)
    ax1.fill(cx, cy, color="#EBF4FC", alpha=0.55)
    ax1.plot(cx, cy, "k-", lw=1.4, label=f"Vùng phủ (R = {s.cell_radius_m:.0f} m)")
    for rad_ring in (s.cell_radius_m * 0.25, s.cell_radius_m * 0.5, s.cell_radius_m * 0.75):
        ax1.plot(rad_ring * np.cos(circle_ang), rad_ring * np.sin(circle_ang), "k:", lw=0.7, alpha=0.4)

    ax1.scatter([0.0], [0.0], marker="^", s=165, color="#111111", zorder=10, label=f"BS tại O(0,0) (N_t={s.default_nt})")
    for k in range(K):
        xk, yk = positions_xy[k]
        c = ue_colors[k % len(ue_colors)]
        ax1.plot([0.0, xk], [0.0, yk], linestyle=":", color=c, alpha=0.8, lw=1.2)
        ax1.scatter(
            [xk],
            [yk],
            s=95,
            color=c,
            edgecolor="black",
            linewidth=0.8,
            zorder=9,
            label=f"UE {k+1} (r={radii_m[k]:.1f} m, Φ={np.rad2deg(angles_rad[k]):.1f}°)",
        )
        ax1.annotate(f"UE {k+1}", (xk, yk), xytext=(6, 5), textcoords="offset points", fontsize=9, fontweight="bold", color=c)

    ax1.set_title(f"(a) Mô hình tọa độ cực tâm O(0,0), bán kính R = {s.cell_radius_m:.0f} m")
    ax1.set_xlabel("Tọa độ x = r·cos(Φ) (m)")
    ax1.set_ylabel("Tọa độ y = r·sin(Φ) (m)")
    ax1.set_aspect("equal", adjustable="box")
    lim = s.cell_radius_m * 1.15
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.legend(loc="lower left", fontsize=7.8, framealpha=0.92)

    # Right panel: 360-deg polar beampattern of Struct-GA
    max_pat = float(np.max(pat_sga))
    for k in range(K):
        c = ue_colors[k % len(ue_colors)]
        pat_norm = np.sqrt(np.maximum(pat_sga[k] / max(max_pat, 1e-30), 0.0))
        ax2.plot(angles_grid_rad, pat_norm, color=c, lw=1.9, label=f"Búp sóng UE {k+1}")
        ax2.plot([angles_rad[k], angles_rad[k]], [0.0, 1.05], color=c, linestyle="--", lw=1.1, alpha=0.75)

    ax2.set_title(
        f"(b) Giản đồ búp sóng cực Struct-GA\n(R_GA={rate_sga:.2f} vs WMMSE {rate_wmmse:.2f}, MRT {rate_mrt:.2f} bps/Hz)",
        pad=14,
    )
    ax2.set_ylim(0.0, 1.08)
    ax2.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax2.legend(loc="upper right", bbox_to_anchor=(1.22, 1.08), fontsize=8.3, framealpha=0.92)

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
        title=f"Tổng tốc độ bit trung bình theo công suất phát P_max (N_t={s.default_nt}, K={s.default_k}, n={s.path_loss_exponent:.1f})",
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
        title=f"Ảnh hưởng của số anten trạm gốc N_t đến tổng tốc độ bit (K={s.default_k}, P_max={s.default_p_max_dbm:.0f} dBm)",
        xlabel="Số lượng anten phát tại trạm gốc N_t",
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
        title=f"Ảnh hưởng của số người dùng K đến tổng tốc độ bit (N_t={s.default_nt}, P_max={s.default_p_max_dbm:.0f} dBm)",
        xlabel="Số lượng người dùng đơn anten K",
        fig_dir=fig_dir,
        cfg=cfg,
        legend_loc="upper left",
        vertical_note=(float(s.default_nt), f"Ranh giới quá tải\n(K = N_t = {s.default_nt})"),
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
        title=f"Ảnh hưởng của thành phần LoS (Rician K-factor) trong mô hình cộng trực tiếp (N_t={s.default_nt}, K={s.default_k})",
        xlabel="Hệ số Rician K-factor (dB)  [NLoS Rayleigh  →  LoS mạnh]",
        fig_dir=fig_dir,
        cfg=cfg,
        legend_loc="upper left",
    )


def plot_exp6_convergence_and_stability(
    data_dir: Path = DATA_DIR,
    fig_dir: Path = FIGURES_DIR,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> None:
    """Fig 6 (GA convergence envelopes) & Fig 7 (30-run GA stability boxplot + Runtime)."""
    d = np.load(data_dir / "exp6_convergence_stability.npz")
    mrt_rate = float(d["mrt_rate"][0])
    zf_rate = float(d["zf_rate"][0])
    rzf_rate = float(d["rzf_rate"][0])
    wmmse_rate = float(d["wmmse_rate"][0])

    s = cfg.system
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.2, 4.9), sharey=True)

    for ax, group, title_sub in (
        (ax_l, ("Direct-GA", "Hybrid-GA"), f"(a) Biểu diễn trực tiếp (2·N_t·K = {2*s.default_nt*s.default_k} gen)"),
        (ax_r, ("Struct-GA",), f"(b) Biểu diễn cấu trúc tối ưu (2K = {2*s.default_k} gen)"),
    ):
        for name in group:
            hist = d[f"hist_{name.replace('-', '_')}"]
            iters = np.arange(hist.shape[1])
            mean_h = np.mean(hist, axis=0)
            std_h = np.std(hist, axis=0)
            style = METHOD_STYLES[name]
            ax.plot(iters, mean_h, color=style["color"], linestyle=style["linestyle"], linewidth=2.2, label=f"{name} (mean ± 1σ)")
            ax.fill_between(iters, mean_h - std_h, mean_h + std_h, color=style["color"], alpha=0.18)

        ax.axhline(wmmse_rate, color="#111111", linestyle="--", lw=1.5, label=f"WMMSE ({wmmse_rate:.2f})")
        ax.axhline(zf_rate, color="#2CA02C", linestyle=":", lw=1.4, label=f"ZF ({zf_rate:.2f})")
        ax.axhline(rzf_rate, color="#1F77B4", linestyle="-.", lw=1.3, label=f"RZF ({rzf_rate:.2f})")
        ax.axhline(mrt_rate, color="#7F7F7F", linestyle="--", lw=1.3, label=f"MRT ({mrt_rate:.2f})")

        ax.set_title(title_sub)
        ax.set_xlabel("Thế hệ tiến hóa (Generation)")
        ax.legend(loc="lower right", fontsize=8.4, framealpha=0.92)

    ax_l.set_ylabel("Tổng tốc độ bit tốt nhất (bps/Hz)")

    # Zoomed inset for Struct-GA first 25 generations
    hist_sga = d["hist_Struct_GA"]
    iters_z = np.arange(min(26, hist_sga.shape[1]))
    mean_z = np.mean(hist_sga[:, : len(iters_z)], axis=0)
    std_z = np.std(hist_sga[:, : len(iters_z)], axis=0)
    inset_ax = ax_r.inset_axes([0.11, 0.44, 0.56, 0.36])
    inset_ax.plot(iters_z, mean_z, color=METHOD_STYLES["Struct-GA"]["color"], lw=2.0)
    inset_ax.fill_between(iters_z, mean_z - std_z, mean_z + std_z, color=METHOD_STYLES["Struct-GA"]["color"], alpha=0.22)
    inset_ax.set_xlim(0, 25)
    inset_ax.set_ylim(float(np.min(mean_z - std_z)) - 0.25, float(np.max(mean_z + std_z)) + 0.20)
    inset_ax.set_title("Phóng to 25 thế hệ đầu (Struct-GA)", fontsize=8.3, fontweight="bold")
    inset_ax.tick_params(labelsize=7.8)
    inset_ax.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(
        f"Đường cong hội tụ trung bình của GA trên {cfg.stability_runs} lần chạy độc lập (N_t={s.default_nt}, K={s.default_k}, P_max={s.default_p_max_dbm:.0f} dBm)",
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _save_fig(fig, "fig6_convergence_curves", fig_dir, cfg)

    # --- Figure 7: Stability Boxplot & Computational Runtime ---
    ga_names = ["Direct-GA", "Hybrid-GA", "Struct-GA"]
    box_data = [d[f"final_{m.replace('-', '_')}"] for m in ga_names]
    df_rt = pd.read_csv(data_dir / "exp6_stability_runtime.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))

    bp = ax1.boxplot(box_data, tick_labels=ga_names, patch_artist=True, widths=0.45)
    for patch, m in zip(bp["boxes"], ga_names):
        patch.set_facecolor(METHOD_STYLES[m]["color"])
        patch.set_alpha(0.58)
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.9)

    ax1.axhline(wmmse_rate, color="#111111", linestyle="--", lw=1.4, label=f"WMMSE ({wmmse_rate:.2f} bps/Hz)")
    ax1.axhline(zf_rate, color="#2CA02C", linestyle=":", lw=1.4, label=f"ZF ({zf_rate:.2f} bps/Hz)")
    ax1.axhline(rzf_rate, color="#1F77B4", linestyle="-.", lw=1.3, label=f"RZF ({rzf_rate:.2f} bps/Hz)")
    ax1.set_title(f"(a) Độ ổn định nghiệm GA qua {cfg.stability_runs} lần chạy độc lập")
    ax1.set_ylabel("Tổng tốc độ bit cuối cùng (bps/Hz)")
    ax1.legend(loc="lower right", fontsize=8.6)

    rt_methods = df_rt["Method"].tolist()
    rt_vals = df_rt["Mean_Runtime_ms"].tolist()
    bar_colors = [METHOD_STYLES.get(m, {"color": "#555555"})["color"] for m in rt_methods]

    y_pos = np.arange(len(rt_methods))
    bars = ax2.barh(y_pos, rt_vals, color=bar_colors, alpha=0.84, edgecolor="#222222", linewidth=0.7)
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
            fontsize=8.5,
        )
    ax2.set_xlim(right=max(rt_vals) * 4.5)

    fig.tight_layout()
    _save_fig(fig, "fig7_stability_runtime", fig_dir, cfg)


def plot_all_figures(
    data_dir: Path = DATA_DIR,
    fig_dir: Path = FIGURES_DIR,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> None:
    """Regenerate every figure in the report from saved .npz/.csv artifacts."""
    _configure_matplotlib()
    ensure_dirs(fig_dir)
    print(f"[BTL2-GA] Generating 300-DPI IEEE figures into: {fig_dir}")
    plot_system_architecture_diagram(fig_dir, cfg)
    plot_exp1_geometry_and_beampattern(data_dir, fig_dir, cfg)
    plot_exp2_power_sweep(data_dir, fig_dir, cfg)
    plot_exp3_antenna_sweep(data_dir, fig_dir, cfg)
    plot_exp4_user_sweep(data_dir, fig_dir, cfg)
    plot_exp5_rician_k_sweep(data_dir, fig_dir, cfg)
    plot_exp6_convergence_and_stability(data_dir, fig_dir, cfg)
    print("[BTL2-GA] All 8 figures generated successfully.")


if __name__ == "__main__":
    plot_all_figures()
