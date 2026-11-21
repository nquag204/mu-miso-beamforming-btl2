# BTL2 – Đề tài 1: Tối ưu Beamforming cho Hệ thống MU-MISO bằng Giải thuật Heuristic / Tiến hóa

Mã nguồn mô phỏng Monte-Carlo, bộ kiểm thử tự động (`pytest` + `hypothesis`), dữ liệu kết quả và báo cáo học thuật chuẩn IEEE cho **Bài tập lớn 02 – Môn Thông tin Di động (EE5019 – HK261, Nhóm 10, ĐH Bách Khoa TP.HCM)**.

---

## 1. Tổng quan bài toán

Hệ thống đường xuống **MU-MISO (Multi-User Multiple-Input Single-Output)** gồm 01 Trạm gốc (BS) trang bị mảng anten thẳng đều ULA gồm $M$ phần tử ($d_a = \lambda/2$) phục vụ đồng thời $K$ người dùng (UE) đơn anten trong vùng phủ hình vành khăn sector $120^\circ$ ($d_k \in [20, 100]\,\text{m}$) trên kênh pha-đinh **Rician** (băng tần 5G NR n78, $f_c = 3.5\,\text{GHz}$, $B = 10\,\text{MHz}$, suy hao log-distance $\alpha = 3.0$, tạp âm $\sigma^2 = -94\,\text{dBm}$).

Mục tiêu là thiết kế ma trận tạo búp sóng $\mathbf{W} = [\mathbf{w}_1, \dots, \mathbf{w}_K] \in \mathbb{C}^{M \times K}$ tối đa hóa **tổng tốc độ bit (Sum-Rate)** dưới ràng buộc tổng công suất phát tại BS:
$$\max_{\mathbf{W} \in \mathbb{C}^{M \times K}} \sum_{k=1}^{K} \log_2 \!\left(1 + \frac{|\mathbf{h}_k^H \mathbf{w}_k|^2}{\sum_{j \neq k}^{K} |\mathbf{h}_k^H \mathbf{w}_j|^2 + \sigma^2}\right) \quad \text{s.t.} \quad \|\mathbf{W}\|_F^2 \le P_{\max}.$$

### Các phương pháp đã cài đặt
- **Heuristic / Tiến hóa – Biểu diễn trực tiếp ($\mathbb{R}^{2MK}$ + chuẩn hóa Frobenius):**
  - `Direct-DE`: Differential Evolution (`DE/current-to-best/1/bin`, vector hóa NumPy).
  - `Direct-PSO`: Global-Best Particle Swarm Optimization (quán tính giảm tuyến tính $w: 0.9 \to 0.4$).
  - `Direct-GA`: Real-Coded Genetic Algorithm (Tournament + BLX-$\alpha$ + Gaussian mutation + Elitism).
- **Heuristic / Tiến hóa – Biểu diễn theo cấu trúc nghiệm tối ưu ($\mathbb{R}^{2K}$, đối ngẫu đường lên–đường xuống [Björnson et al., 2014]):**
  - `Struct-DE`, `Struct-PSO`, `Struct-GA`: rút gọn số chiều từ $2MK$ xuống $2K$ (độc lập với $M$), đánh giá lô theo ma trận Gram $\mathcal{O}(K^3)$.
  - `SciPy-DE`: đối chiếu chéo với `scipy.optimize.differential_evolution`.
- **Các phương pháp tham chiếu (Baselines):**
  - `MRT` (Maximum Ratio Transmission), `ZF` (Zero-Forcing), `RZF` (Regularized Zero-Forcing / MMSE), và `WMMSE` (Weighted MMSE [Shi et al., 2011]).

---

## 2. Cấu trúc thư mục

```text
mu-miso-beamforming-btl2/
├── config.py                                  # Cấu hình tập trung mọi tham số hệ thống, kênh & giải thuật
├── main.py                                    # Script chính chạy toàn bộ 6 thực nghiệm & xuất 8 hình
├── plot_results.py                            # Vẽ lại toàn bộ 8 đồ thị 300 DPI từ dữ liệu đã lưu
├── verify_reproducibility.py                  # Kiểm chứng tái lập 100% số liệu & đồ thị báo cáo từ con số 0
├── run_all.ps1                                # Script PowerShell chạy 1 chạm (pytest -> main.py)
├── requirements.txt                           # Thư viện Python cần thiết (Python >= 3.10)
├── README.md                                  # Hướng dẫn cài đặt và tái lập mô phỏng (file này)
├── src/
│   ├── utils.py                               # Đổi đơn vị dBm <-> W, dB <-> linear
│   ├── channel.py                             # Sinh vị trí UE, Path Loss, kênh Rician ULA
│   ├── metrics.py                             # Chuẩn hóa công suất, SINR, Sum-Rate (vector hóa), Beampattern
│   ├── baselines.py                           # MRT, ZF, RZF (MMSE), WMMSE
│   ├── optimizers.py                          # Direct/Struct DE, PSO, GA & SciPy DE cross-check
│   └── experiments.py                         # Bộ điều phối Monte-Carlo, lưu .npz & .csv
├── tests/
│   ├── test_channel.py                        # Kiểm chứng vị trí, suy hao và E[||h_k||^2] ≈ M * beta_k
│   ├── test_metrics.py                        # Kiểm chứng ràng buộc công suất, bất biến pha (Hypothesis)
│   └── test_algorithms.py                     # Kiểm chứng triệt nhiễu ZF, hội tụ đơn điệu, khớp SciPy DE
├── results/
│   ├── data/                                  # 6 bộ dữ liệu Monte-Carlo (.npz) & 6 bảng tóm tắt (.csv)
│   └── figures/                               # 8 hình chuẩn IEEE 300 DPI (.png & .pdf)
└── report/
    ├── baocao_btl2.tex                        # Mã nguồn báo cáo LaTeX tiếng Việt chuẩn IEEE
    ├── references.bib                         # 12 tài liệu tham khảo BibTeX chuẩn IEEE
    ├── assets/                                # Logo & khung bìa HCMUT cho LaTeX
    ├── Bao_cao_BTL02_DeTai1_updated.docx      # File báo cáo Word hoàn chỉnh
    └── Bao_cao_BTL02_DeTai1_updated.pdf       # File báo cáo PDF hoàn chỉnh
```

---

## 3. Hướng dẫn cài đặt & Tái lập mô phỏng

**Yêu cầu môi trường:** `Python >= 3.10` và `Git`. Toàn bộ lệnh dưới đây được thực thi **ngay tại thư mục gốc của repo sau khi clone**.

### Bước 1: Clone repository
```powershell
git clone https://github.com/nquag204/mu-miso-beamforming-btl2.git
cd mu-miso-beamforming-btl2
```

### Bước 2: Tạo môi trường ảo và cài đặt thư viện

**Trên Windows PowerShell:**
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Trên Linux / macOS (hoặc Git Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Bước 3: Chạy bộ kiểm thử tự động (Verification Suite)
```powershell
python -m pytest tests -v
```
> **Kết quả kỳ vọng:** `21 passed` (thời gian chạy ~1–3 giây), xác nhận toàn bộ các bất biến vật lý và toán học (thống kê kênh Rician, ràng buộc công suất, bất biến xoay pha, triệt nhiễu ZF, hội tụ đơn điệu và độ khớp với `SciPy-DE`).

### Bước 4: Tái tạo toàn bộ đồ thị 300 DPI từ dữ liệu đã lưu (~5 giây)
```powershell
# Windows PowerShell
$env:MPLBACKEND = "Agg"; python plot_results.py

# Linux / macOS
MPLBACKEND=Agg python plot_results.py
```
Lệnh trên đọc các file `.npz`/`.csv` trong `results/data/` và xuất đủ 8 hình (`.png` 300 DPI và `.pdf`) vào `results/figures/`.

### Bước 5: Chạy lại toàn bộ mô phỏng Monte-Carlo từ đầu

- **Chạy kiểm tra nhanh (15 mẫu kênh/điểm, 10 lần ổn định — mất ~1 phút):**
  ```powershell
  $env:MPLBACKEND = "Agg"; python main.py --quick
  ```
- **Chạy tái lập đầy đủ giống báo cáo (100 mẫu kênh/điểm, 30 lần ổn định, `seed=2026` — mất ~8 phút):**
  ```powershell
  $env:MPLBACKEND = "Agg"; python main.py
  ```
- **Hoặc chạy 1 chạm toàn bộ kiểm thử + mô phỏng trên Windows PowerShell:**
  ```powershell
  .\run_all.ps1 -Quick   # Kiểm tra nhanh (pytest + 15 trials)
  .\run_all.ps1          # Tái lập đầy đủ (pytest + 100 trials)
  ```
- **Tùy chỉnh tham số mô phỏng:** Mọi tham số hệ thống ($M, K, P_{\max}, R_{\min}, R_{\max}, \alpha, K_{\text{Rician}}, \sigma^2$), siêu tham số DE/PSO/GA và hạt giống `seed` đều tập trung tại `config.py`. Có thể ghi đè nhanh số mẫu kênh qua tham số dòng lệnh, ví dụ `python main.py --trials 50`.

---

## 4. Danh mục đồ thị & dữ liệu xuất ra

| Ký hiệu trong báo cáo | File đầu ra trong `results/figures/` | Dữ liệu nguồn (`results/data/`) | Nội dung |
| :--- | :--- | :--- | :--- |
| Sơ đồ 1 | `sodo1_system_architecture.png` | `config.py` | Sơ đồ khối kiến trúc MU-MISO đường xuống và vòng lặp tối ưu |
| Hình 1 | `fig1_geometry_beampattern.png` | `exp1_geometry_beampattern.npz`, `exp1_ue_table.csv` | Kịch bản phủ sóng Sector $120^\circ$ và giản đồ búp sóng ULA của `Struct-DE` |
| Hình 2 | `fig2_sumrate_vs_power.png` | `exp2_power_sweep.npz`, `exp2_power_sweep.csv` | Tổng tốc độ bit trung bình theo công suất phát $P_{\max} \in [0, 30]\,\text{dBm}$ |
| Hình 3 | `fig3_sumrate_vs_antennas.png` | `exp3_antenna_sweep.npz`, `exp3_antenna_sweep.csv` | Ảnh hưởng của số anten trạm gốc $M \in \{4, 8, 12, 16, 24, 32\}$ |
| Hình 4 | `fig4_sumrate_vs_users.png` | `exp4_user_sweep.npz`, `exp4_user_sweep.csv` | Ảnh hưởng của số người dùng $K \in \{2, 4, 6, 8, 10\}$ (gồm vùng quá tải $K > M$) |
| Hình 5 | `fig5_sumrate_vs_rician_k.png` | `exp5_rician_k_sweep.npz`, `exp5_rician_k_sweep.csv` | Ảnh hưởng của hệ số Rician $K$-factor ($\kappa \in [-20, 15]\,\text{dB}$) |
| Hình 6 | `fig6_convergence_curves.png` | `exp6_convergence_stability.npz` | Đường cong hội tụ trung bình $\pm 1\sigma$ trên 30 lần chạy độc lập |
| Hình 7 | `fig7_stability_runtime.png` | `exp6_convergence_stability.npz`, `exp6_stability_runtime.csv` | Boxplot độ ổn định nghiệm và so sánh thời gian tính toán (thang log) |

---

## 5. Kiểm chứng tái lập 100% kết quả báo cáo (Dành cho người chấm / Tester)

Để xác minh độc lập rằng toàn bộ bảng số liệu (**Bảng 2–6**) và **8 đồ thị** trong báo cáo được sinh ra từ mã nguồn thật và có thể tái lập chính xác tuyệt đối từ con số 0 (`seed = 2026`, `100` hiện thực kênh độc lập/điểm, `30` lần chạy ổn định):

### Cách 1: Kiểm chứng tự động bằng 1 lệnh (`verify_reproducibility.py`)
```powershell
# (A) Kiểm tra nhanh đủ 28 file kết quả & đối chiếu các mốc số liệu Bảng 3–6 (~1 giây)
python verify_reproducibility.py --check-only

# (B) Chạy sạch từ con số 0 vào thư mục tạm (100 trials, ~8 phút) và tự động đối chiếu từng bảng CSV
$env:MPLBACKEND = "Agg"; python verify_reproducibility.py

# (C) Hoặc xóa trắng results/data/* & results/figures/*, chạy tái lập tại chỗ rồi đối chiếu
$env:MPLBACKEND = "Agg"; python verify_reproducibility.py --in-place
```
> **Kết quả kỳ vọng:** Cả 6 bảng CSV in `KHOP 100% (max_diff = 0.00e+00)` (loại trừ cột `runtime_ms` phụ thuộc tải CPU), sinh đủ 12 file dữ liệu và 16 file hình (8 PNG + 8 PDF).

### Cách 2: Tự kiểm chứng thủ công từng bước (Clean-Room Test)
```powershell
# 1. Lưu riêng bản results gốc để đối chiếu, sau đó xóa trắng toàn bộ dữ liệu & hình cũ
Copy-Item -Recurse "results" "..\results_origin"
Remove-Item "results\data\*", "results\figures\*" -Force

# 2. Chạy lại toàn bộ mô phỏng đầy đủ (KHÔNG thêm --quick để khớp đúng 100 kênh của báo cáo)
$env:MPLBACKEND = "Agg"; python main.py

# 3. Kiểm tra mốc số liệu Bảng 3–6 và mở 8 hình mới sinh ra để đối chiếu với báo cáo PDF/Word
python verify_reproducibility.py --check-only
explorer "results\figures"
```

### Một số mốc số liệu đối chứng chính trong báo cáo (`seed=2026`, 100 mẫu kênh)
- **Bảng 3 (`exp2_power_sweep.csv`):** Tại $P_{\max}=20\,\text{dBm}$ → `Struct-DE = 22.68`, `WMMSE = 22.22`, `ZF = 22.31`, `MRT = 8.44 bps/Hz`; tại $P_{\max}=30\,\text{dBm}$ → `Struct-DE = 34.85`, `MRT = 9.13 bps/Hz`.
- **Bảng 4 (`exp3_antenna_sweep.csv`):** $M = 4 \to 32$ → `Struct-DE: 16.64 → 32.93 bps/Hz`, `Direct-DE: 15.00 → 24.53 bps/Hz`.
- **Bảng 5 (`exp4_user_sweep.csv`):** $K = 8$ → `ZF = 17.26`, `Struct-DE = 31.44 bps/Hz`; $K = 10$ ($K > M$) → `Struct-DE = 34.15 bps/Hz`.
- **Bảng 6 (`exp6_stability_runtime.csv`):** Qua 30 lần chạy độc lập, `Struct-DE/PSO/GA` đều đạt `23.654 bps/Hz` (`Std = 0.0000`, `CV = 0.000%`), khớp `SciPy-DE = 23.654 bps/Hz`.
