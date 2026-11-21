# BTL2 – Đề tài 1: Tối ưu Beamforming cho Hệ thống MU-MISO bằng Giải thuật Heuristic / Tiến hóa

Mã nguồn mô phỏng Monte-Carlo, bộ kiểm thử tự động (`pytest` + `hypothesis`), và báo cáo học thuật chuẩn IEEE cho **Bài tập lớn 02 – Môn Thông tin di động (HK261)**.

## 1. Tổng quan bài toán

Hệ thống đường xuống **MU-MISO (Multi-User Multiple-Input Single-Output)** gồm 01 Trạm gốc (BS) trang bị mảng anten thẳng đều ULA gồm $M$ phần tử phục vụ đồng thời $K$ thiết bị người dùng (UE) đơn anten trong vùng phủ hình vành khăn sector $120^\circ$ ($d_k \in [20, 100]\,\text{m}$) trên kênh pha-đinh **Rician** (băng tần 5G NR n78, $f_c = 3.5\,\text{GHz}$, suy hao log-distance $\alpha = 3.0$, tạp âm $\sigma^2 = -94\,\text{dBm}$).

Mục tiêu là tìm ma trận tạo búp sóng $\mathbf{W} = [\mathbf{w}_1, \dots, \mathbf{w}_K] \in \mathbb{C}^{M \times K}$ nhằm tối đa hóa **tổng tốc độ bit (Sum-Rate)** dưới ràng buộc tổng công suất phát tại BS:
$$\max_{\mathbf{W}} \sum_{k=1}^{K} \log_2 \!\left(1 + \frac{|\mathbf{h}_k^H \mathbf{w}_k|^2}{\sum_{j \neq k}^{K} |\mathbf{h}_k^H \mathbf{w}_j|^2 + \sigma^2}\right) \quad \text{s.t.} \quad \|\mathbf{W}\|_F^2 \le P_{\max}.$$

### Các phương pháp cài đặt
- **Giải thuật Heuristic / Tiến hóa – Biểu diễn trực tiếp ($\mathbb{R}^{2MK}$ + chuẩn hóa Frobenius):**
  - `Direct-DE`: Differential Evolution (`DE/current-to-best/1/bin` vector hóa).
  - `Direct-PSO`: Global-Best Particle Swarm Optimization (giảm dần quán tính $w: 0.9 \to 0.4$).
  - `Direct-GA`: Real-Coded Genetic Algorithm (Tournament + BLX-$\alpha$ + Gaussian mutation + Elitism).
- **Giải thuật Heuristic / Tiến hóa – Biểu diễn theo cấu trúc nghiệm tối ưu ($\mathbb{R}^{2K}$, đối ngẫu đường lên–đường xuống [Björnson et al., 2014]):**
  - `Struct-DE`, `Struct-PSO`, `Struct-GA`: giảm số chiều tìm kiếm từ $2MK$ xuống $2K$ (độc lập với $M$), đánh giá hàm mục tiêu theo ma trận Gram $\mathcal{O}(K^3)$ vector hóa bằng NumPy.
  - `SciPy-DE`: tích hợp `scipy.optimize.differential_evolution` để đối chiếu chéo với mã nguồn mở chuẩn.
- **Các phương pháp tham chiếu (Baselines):**
  - `MRT` (Maximum Ratio Transmission), `ZF` (Zero-Forcing), `RZF` (Regularized Zero-Forcing / MMSE), và `WMMSE` (Weighted MMSE [Shi et al., 2011]).

## 2. Cấu trúc thư mục

```text
btl2/
├── config.py                     # Cấu hình tập trung mọi tham số hệ thống, kênh & giải thuật
├── main.py                       # Script chính chạy toàn bộ 6 thực nghiệm & xuất 7 hình
├── plot_results.py               # Vẽ lại toàn bộ đồ thị 300 DPI từ dữ liệu đã lưu
├── requirements.txt              # Danh sách thư viện Python cần thiết
├── README.md                     # Tài liệu hướng dẫn tái lập mô phỏng (file này)
├── src/
│   ├── utils.py                  # Đổi đơn vị dBm <-> W, dB <-> linear
│   ├── channel.py                # Sinh vị trí UE, Path Loss, kênh Rician ULA
│   ├── metrics.py                # Chuẩn hóa công suất, SINR, Sum-Rate (vector hóa), Beampattern
│   ├── baselines.py              # MRT, ZF, RZF (MMSE), WMMSE
│   ├── optimizers.py             # Direct/Struct DE, PSO, GA & SciPy DE cross-check
│   └── experiments.py            # Bộ điều phối Monte-Carlo, xuất .npz & .csv
├── tests/
│   ├── test_channel.py           # Kiểm chứng vị trí, suy hao và E[||h_k||^2] ≈ M * beta_k
│   ├── test_metrics.py           # Kiểm chứng ràng buộc công suất, bất biến pha (Hypothesis)
│   └── test_algorithms.py        # Kiểm chứng triệt nhiễu ZF, hội tụ đơn điệu, đối chiếu SciPy DE
├── results/
│   ├── data/                     # Dữ liệu Monte-Carlo thô (.npz) và bảng tóm tắt (.csv)
│   └── figures/                  # 7 hình đồ thị chuẩn IEEE 300 DPI (.png & .pdf)
└── report/
    ├── baocao_btl2.tex           # Mã nguồn báo cáo LaTeX tiếng Việt chuẩn IEEE
    ├── references.bib            # File trích dẫn BibTeX chuẩn IEEE
    ├── build_docx.py             # Script sinh tự động báo cáo Word từ số liệu & hình đã chạy
    └── Bao_cao_BTL02_DeTai1.docx # File báo cáo Word hoàn chỉnh
```

## 3. Cài đặt môi trường & Thứ tự chạy (Windows PowerShell)

```powershell
# 1. (Khuyến nghị) Tạo và kích hoạt môi trường ảo Python
python -m venv "btl2\.venv"
& "btl2\.venv\Scripts\Activate.ps1"
pip install -r "btl2\requirements.txt"

# 2. Chạy toàn bộ bộ kiểm thử tự động (21 test cases + Hypothesis property tests)
python -m pytest "btl2\tests" -v

# 3. Chạy toàn bộ mô phỏng Monte-Carlo (100 mẫu kênh/điểm, 30 lần chạy ổn định) & xuất hình
$env:MPLBACKEND = "Agg"; python "btl2\main.py"

# (Tùy chọn) Chạy thử nhanh với 15 mẫu kênh để kiểm tra trong vài giây:
$env:MPLBACKEND = "Agg"; python "btl2\main.py" --quick

# 4. Vẽ lại toàn bộ đồ thị 300 DPI từ dữ liệu đã lưu trong results/data/ (không cần chạy lại Monte-Carlo)
$env:MPLBACKEND = "Agg"; python "btl2\plot_results.py"

# 5. (Tùy chọn) Tái tạo file báo cáo Word (.docx) từ bảng CSV và hình 300 DPI mới nhất
python "btl2\report\build_docx.py"
```

## 4. Danh mục đồ thị & dữ liệu xuất ra

| Hình | File đầu ra | Nội dung |
| :--- | :--- | :--- |
| Hình 1 | `results/figures/fig1_geometry_beampattern.png` | Kịch bản phủ sóng Sector $120^\circ$ và giản đồ búp sóng ULA của Struct-DE |
| Hình 2 | `results/figures/fig2_sumrate_vs_power.png` | Tổng tốc độ bit trung bình theo công suất phát $P_{\max} \in [0, 30]\,\text{dBm}$ |
| Hình 3 | `results/figures/fig3_sumrate_vs_antennas.png` | Ảnh hưởng của số anten trạm gốc $M \in \{4, 8, 12, 16, 24, 32\}$ |
| Hình 4 | `results/figures/fig4_sumrate_vs_users.png` | Ảnh hưởng của số người dùng $K \in \{2, 4, 6, 8, 10\}$ (gồm cả vùng quá tải $K > M$) |
| Hình 5 | `results/figures/fig5_sumrate_vs_rician_k.png` | Ảnh hưởng của hệ số Rician $K$-factor ($\kappa \in [-20, 15]\,\text{dB}$) |
| Hình 6 | `results/figures/fig6_convergence_curves.png` | Đường cong hội tụ trung bình $\pm 1\sigma$ trên 30 lần chạy độc lập |
| Hình 7 | `results/figures/fig7_stability_runtime.png` | Boxplot độ ổn định nghiệm và so sánh thời gian tính toán (thang log) |
