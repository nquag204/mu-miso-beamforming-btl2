# Tài liệu Chuẩn bị Báo cáo Trực tiếp bằng Đối thoại (Vấn đáp BTL02 – Đề tài 1)

**Môn học:** Thông tin Di động (EE5019) – HK 261  
**Giảng viên hướng dẫn:** PGS.TS Hà Hoàng Kha  
**Nhóm thực hiện:** Nhóm 10 – Trần Quốc Nam (2212161) & Nguyễn Nhật Quang (2212746)  
**Đề tài 1:** Tối ưu Beamforming cho hệ thống MU-MISO đường xuống bằng giải thuật Heuristic / Tiến hóa

---

## Phần 1: Kịch bản Trình bày Mở đầu (3–5 phút, chia 2 thành viên)

> **Mẹo đối thoại:** Nói ngắn gọn theo mạch **Vấn đề → Mô hình → Ý tưởng đột phá → Kết quả định lượng → Giới hạn thực tế**. Không đọc thuộc lòng công thức dài, mà chỉ vào hình/bảng tương ứng trong báo cáo.

### Thành viên 1 (1.5–2 phút) – Bối cảnh, Mô hình Hệ thống, Kênh Rician & Bài toán Tối ưu
1. **Đặt vấn đề:** Nhóm thực hiện Đề tài 1 về thiết kế bộ tạo búp sóng đường xuống (Downlink Transmit Beamforming) cho hệ thống MU-MISO gồm 1 trạm gốc mảng thẳng đều ULA có $M$ anten phục vụ đồng thời $K$ người dùng đơn anten trên cùng tài nguyên tần số–thời gian.
2. **Kịch bản & Kênh truyền (chỉ vào Hình 1a & Bảng 1–2):**
   - Mô phỏng microcell 5G NR n78 ($f_c = 3.5\,\text{GHz}$, $B = 10\,\text{MHz}$, nhiễu $\sigma^2 = -94\,\text{dBm}$).
   - Người dùng phân bố đều theo diện tích trong hình vành khăn $[20, 100]\,\text{m}$, sector $120^\circ$ (tránh đối xứng gương trước–sau của ULA).
   - Suy hao Log-distance với tham chiếu Friis ở $1\,\text{m}$ ($\beta_0 \approx -43.32\,\text{dB}$, $\alpha = 3.0$) kết hợp kênh pha-đinh **Rician** (LoS ULA nửa bước sóng + NLoS Rayleigh, $K_{\text{Rician}} = 6\,\text{dB}$).
3. **Bài toán tối ưu (chỉ vào mục 3):**
   - Tối đa hóa tổng tốc độ bit (Sum-Rate) $R_{\text{sum}}(\mathbf{W}) = \sum_{k=1}^K \log_2(1 + \text{SINR}_k(\mathbf{W}))$ dưới ràng buộc tổng công suất $\|\mathbf{W}\|_F^2 \le P_{\max}$.
   - Điểm khó: hàm mục tiêu **không lồi** và **NP-hard** do $\mathbf{W}$ nằm ở cả tử số và mẫu số của SINR.

### Thành viên 2 (2–2.5 phút) – Giải thuật Tiến hóa (Direct vs Struct), Kiểm thử & Kết quả chính
1. **Hai hướng biểu diễn nghiệm cho DE / PSO / GA (chỉ vào mục 4):**
   - **Cách 1 (`Direct-DE/PSO/GA`):** Tối ưu trực tiếp $2MK$ biến thực (phần thực + ảo của $\mathbf{W}$), xử lý ràng buộc bằng phép chiếu chuẩn hóa Frobenius ($\|\mathbf{W}\|_F^2 = P_{\max}$), 100% nghiệm hợp lệ không cần hàm phạt.
   - **Cách 2 (`Struct-DE/PSO/GA` – điểm nhấn kỹ thuật):** Khai thác Định lý cấu trúc beamforming tối ưu theo đối ngẫu đường lên–đường xuống (Björnson et al., 2014), rút không gian tìm kiếm từ $2MK$ xuống chỉ còn **$2K$ biến** $(\mathbf{p}, \boldsymbol{\lambda})$ độc lập với $M$, kết hợp đồng nhất thức Woodbury trên ma trận Gram $K\times K$ để đánh giá quần thể với độ phức tạp $\mathcal{O}(K^3)$.
2. **Kiểm chứng & Kết quả nổi bật (chỉ lần lượt Hình 1b → Hình 2 → Hình 3 → Hình 4 → Hình 6–7):**
   - Toàn bộ code Python được vector hóa bằng NumPy, kiểm chứng bởi **21/21 unit test (`pytest` + `hypothesis`)** và đối chiếu chéo với `scipy.optimize.differential_evolution`, chạy trung bình **100 kênh độc lập/điểm**.
   - **Công suất phát (Hình 2):** Tại $P_{\max} = 30\,\text{dBm}$, `Struct-DE` đạt **34.85 bps/Hz**, vượt MRT bão hòa ở 9.13 bps/Hz (+281.8%) và vượt cả WMMSE bị giới hạn 100 vòng lặp (33.35 bps/Hz).
   - **Số anten $M$ (Hình 3):** Khi $M$ tăng từ $4 \to 32$, `Direct-DE` gặp *lời nguyền số chiều* ($32 \to 256$ biến, chững lại ở 24.53 bps/Hz), trong khi `Struct-DE` tăng mạnh lên **32.93 bps/Hz** với thời gian chạy giữ nguyên **hằng số ~15.6–16.2 ms**.
   - **Số người dùng $K$ (Hình 4):** Tại $K=M=8$, `ZF` tụt mạnh xuống 17.26 bps/Hz do khuếch đại tạp âm; sang vùng quá tải $K=10 > M=8$, `Struct-DE` đạt **34.15 bps/Hz** nhờ tự động lập lịch công suất ngầm.
   - **Hội tụ & Ổn định (Hình 6–7):** Qua 30 lần chạy độc lập, `Struct-DE/PSO/GA` hội tụ sau ~10–15 vòng về cùng cực trị **23.654 bps/Hz** ($\text{CV} < 0.001\%$), khớp `SciPy-DE` và nhanh hơn `WMMSE` (13.1–15.6 ms vs 21.4 ms).
3. **Kết luận & Giới hạn:** Giải thuật tiến hóa rất mạnh về tối ưu toàn cục và dễ mở rộng ràng buộc phi tuyến/rời rạc, nhưng độ trễ ~13–18 ms trên CPU phù hợp cho kênh biến thiên chậm hoặc tối ưu ngoại tuyến/tạo nhãn AI hơn là lịch biểu thời gian thực 1 ms của 5G tốc độ cao.

---

## Phần 2: Sơ đồ Khối Hệ thống & Vòng lặp Tối ưu (Dùng để vẽ bảng / giải thích nhanh)

```text
  K luồng dữ liệu s = [s_1,...,s_K]^T
                 │
                 ▼
      ┌─────────────────────┐        CSI hoàn hảo H = [h_1,...,h_K]
      │  Bộ tạo búp sóng W  │◄────────────────────────────────────────────┐
      │   (M x K, ||W||_F²  │                                             │
      │       = P_max)      │◄─────── Bộ tối ưu Heuristic/Tiến hóa        │
      └──────────┬──────────┘         (DE / PSO / GA: Direct 2MK          │
                 │ x = W s             hoặc Struct 2K theo đối ngẫu)      │
                 ▼                                                        │
      ┌─────────────────────┐                                             │
      │ Mảng ULA M anten BS │  (d_a = λ/2, sector 120°, f_c = 3.5 GHz)    │
      └──────────┬──────────┘                                             │
                 │ Kênh Rician H = sqrt(β) * (LoS + NLoS)                 │
                 ▼                                                        │
      ┌─────────────────────┐                                             │
      │  K người dùng (UE)  │───► y_k = h_k^H w_k s_k + MUI + n_k ────────┘
      │  đơn anten + AWGN   │     Tính SINR_k & Sum-Rate R_sum(W)
      └─────────────────────┘
```

---

## Phần 3: Bảng "Số liệu Vàng" Cần Nhớ Khi Đối thoại

| Hạng mục | Giá trị chính xác từ mô phỏng (100 kênh, `seed=2026`) | Ý nghĩa để chốt câu trả lời |
| :--- | :--- | :--- |
| Link budget mặc định | $f_c=3.5\,\text{GHz}$, $B=10\,\text{MHz}$, $\text{PL}_0=43.32\,\text{dB}$, $\alpha=3.0$, $\sigma^2=-94\,\text{dBm}$ | Tạo đủ 2 vùng SNR thấp và cao trên dải $P_{\max} \in [0, 30]\,\text{dBm}$ |
| $P_{\max} = 0\,\text{dBm}$ ($M=8, K=4$) | `MRT`=3.42, `ZF`=3.60, `RZF`=4.19, `Direct-DE`=4.86, `WMMSE`=5.08, `Struct-DE`=**5.08** | Ở SNR thấp, phân bổ công suất cho UE gần giúp DE/WMMSE vượt MRT/ZF/RZF |
| $P_{\max} = 20\,\text{dBm}$ ($M=8, K=4$) | `MRT`=8.44, `Direct-DE`=19.59, `RZF`=20.21, `WMMSE`=22.22, `ZF`=22.31, `Struct-DE`=**22.68** | Cấu hình cơ sở; `Struct-DE` cao nhất |
| $P_{\max} = 30\,\text{dBm}$ ($M=8, K=4$) | `MRT`=9.13, `Direct-DE`=26.58, `RZF`=32.40, `WMMSE`=33.35, `ZF`=34.83, `Struct-DE`=**34.85** | `MRT` bão hòa MUI; `Struct-DE` tăng **+281.8%** so với `MRT` |
| Quét anten $M = 4 \to 32$ ($K=4$) | `Struct-DE`: **16.64 → 32.93 bps/Hz** (runtime **16.20 → 15.57 ms**)<br>`Direct-DE`: 15.00 → 24.53 bps/Hz (runtime 29.0 → 54.7 ms) | Chứng minh thực nghiệm độ phức tạp $\mathcal{O}(K^3)$ độc lập $M$ và *lời nguyền số chiều* của Direct ($32 \to 256$ biến) |
| Quét user $K = 2 \to 10$ ($M=8$) | $K=8$: `ZF` tụt xuống **17.26**, `Struct-DE`=**31.44**<br>$K=10$: `ZF`=22.92, `WMMSE`=33.65, `Struct-DE`=**34.15** | `Struct-DE` tự lập lịch người dùng (cắt nguồn UE yếu) nên mạnh vượt trội khi quá tải $K > M$ |
| Ổn định 30 lần chạy ($M=8, K=4$) | `Struct-DE/PSO/GA`: **23.654 bps/Hz** ($\text{CV} = 0.000\%$, **13.1–18.3 ms**)<br>`SciPy-DE`: **23.654 bps/Hz** (27.88 ms)<br>`WMMSE`: 21.510 bps/Hz (21.40 ms)<br>`Direct-DE`: 20.481 ($\text{CV}=4.91\%$, 34.5 ms), `Direct-PSO`: 19.999 ($\text{CV}=10.45\%$) | `Struct` vừa ổn định tuyệt đối, vừa khớp chuẩn mã nguồn mở SciPy, vừa nhanh hơn WMMSE |

---

## Phần 4: Ngân hàng 18 Câu hỏi Vấn đáp Xoáy sâu & Cách Trả lời Chuẩn

### Nhóm A: Mô hình Hệ thống & Kênh truyền Rician

**Câu 1: Tại sao khi sinh vị trí người dùng lại dùng công thức căn bậc hai $d_k = \sqrt{R_{\min}^2 + u_k(R_{\max}^2 - R_{\min}^2)}$ mà không lấy phân bố đều $d_k \sim \mathcal{U}(R_{\min}, R_{\max})$? Và tại sao lấy $R_{\min} = 20\,\text{m}$?**
- **Trả lời:**
  - Diện tích hình tròn tăng tỷ lệ với $r^2$ ($dA = r\,dr\,d\theta$). Nếu sinh $d_k$ đều tuyến tính, mật độ người dùng sẽ bị dồn dày đặc ở gần trạm gốc và thưa ở rìa ô mạng. Biến đổi biến đổi ngược $d_k = \sqrt{R_{\min}^2 + u(R_{\max}^2 - R_{\min}^2)}$ đảm bảo phân bố **đều theo diện tích**.
  - Chọn $R_{\min} = 20\,\text{m}$ để tránh điểm kỳ dị suy hao $d \to 0$ (mô hình Log-distance không hợp lệ trong trường gần) và đảm bảo điều kiện trường xa ($d \gg 2D^2/\lambda$ với mảng 8–32 anten ở 3.5 GHz).

**Câu 2: Tại sao nhóm chọn sector $120^\circ$ ($[-60^\circ, +60^\circ]$) thay vì cả vòng tròn $360^\circ$?**
- **Trả lời:** Thứ nhất, trong thực tế mạng di động 4G/5G, mỗi trạm gốc thường chia thành 3 sector $120^\circ$. Thứ hai, về vật lý anten, mảng thẳng đều ULA có đáp ứng phụ thuộc vào $\sin\theta_k$, nên góc $\theta$ và $180^\circ - \theta$ cho cùng một vector lái (đối xứng gương trước–sau). Giới hạn trong $[-60^\circ, +60^\circ]$ giúp ánh xạ $\theta_k \mapsto \sin\theta_k$ là đơn ánh (1–1), loại bỏ trùng búp sóng gương và làm Hình 1(b) phản ánh đúng thực tế.

**Câu 3: Làm sao kiểm chứng kênh Rician sinh ra là đúng về mặt thống kê?**
- **Trả lời:** Trong công thức $\mathbf{h}_k = \sqrt{\beta_k}\left(\sqrt{\frac{\kappa}{\kappa+1}}\bar{\mathbf{h}}_k^{\text{LoS}} + \sqrt{\frac{1}{\kappa+1}}\tilde{\mathbf{h}}_k^{\text{NLoS}}\right)$:
  - Thành phần LoS có chuẩn $\|\bar{\mathbf{h}}_k^{\text{LoS}}\|_2^2 = M$.
  - Thành phần NLoS $\mathcal{CN}(\mathbf{0}, \mathbf{I}_M)$ có $\mathbb{E}[\|\tilde{\mathbf{h}}_k^{\text{NLoS}}\|_2^2] = M$ và trung bình bằng 0 (độc lập với LoS).
  - Suy ra $\mathbb{E}[\|\mathbf{h}_k\|_2^2] = \beta_k\left(\frac{\kappa}{\kappa+1}M + \frac{1}{\kappa+1}M\right) = M\beta_k$.
  - Nhóm đã viết unit test tự động `test_rician_channel_expected_power_invariant` chạy trên 2.000 mẫu kênh độc lập và xác nhận tỷ số thực nghiệm $\mathbb{E}[\|\mathbf{h}_k\|^2]/(M\beta_k)$ nằm trong sai số $< 2\%$ quanh 1.

**Câu 4: Nhóm đặt những giả thiết gì trong mô hình? Nếu bỏ giả thiết Perfect CSI thì sao?**
- **Trả lời:** Các giả thiết gồm: (1) Trạm gốc biết hoàn hảo CSI (Perfect CSI); (2) Kênh pha-đinh phẳng theo tần số (flat-fading, tương ứng từng sóng mang con OFDM); (3) Hệ thống đơn tế bào (không có nhiễu liên tế bào); (4) Đồng bộ hoàn hảo và nhiễu AWGN đồng nhất $\sigma^2$ tại mọi UE.
- Nếu có sai số ước lượng kênh ($\hat{\mathbf{H}} = \mathbf{H} + \mathbf{E}$), các điểm null của ZF/Struct-DE sẽ bị lệch góc so với vị trí thực của UE gây rò rỉ nhiễu MUI và tạo trần hiệu năng ở SNR cao; khi đó cần chuyển sang tối ưu bền vững (Robust Beamforming — tối đa hóa kỳ vọng hoặc trường hợp xấu nhất Sum-Rate), và ưu điểm của DE/PSO là dễ dàng thay đổi hàm mục tiêu sang dạng trung bình Monte-Carlo trên sai số CSI mà không cần đạo hàm.

**Câu 5: Tại sao chọn $f_c = 3.5\,\text{GHz}$, $B = 10\,\text{MHz}$, $\sigma^2 = -94\,\text{dBm}$ và $P_{\max} \in [0, 30]\,\text{dBm}$?**
- **Trả lời:** Đây là băng tần chuẩn 5G NR n78 cho microcell đô thị. Với tạp âm nhiệt $-174\,\text{dBm/Hz}$, băng thông $10\,\text{MHz}$ (+70 dB) và hệ số tạp âm $\text{NF} = 10\,\text{dB}$, ta có $\sigma^2 = -174 + 70 + 10 = -94\,\text{dBm}$. Với cự ly trung bình $\sim 70\,\text{m}$, suy hao $\text{PL} \approx 99\,\text{dB}$, giúp dải $P_{\max} \in [0, 30]\,\text{dBm}$ quét trọn vẹn từ miền công suất thấp (SNR nhận quanh $0\,\text{dB}$) đến miền giới hạn nhiễu cao (SNR $\approx 25\text{--}30\,\text{dB}$), quan sát rõ toàn bộ đặc tính của mọi thuật toán.

---

### Nhóm B: Toán học Bài toán & Giải thuật Heuristic / Tiến hóa

**Câu 6: Tại sao bài toán tối đa hóa Sum-Rate là không lồi và tại sao luôn ép dấu bằng $\|\mathbf{W}\|_F^2 = P_{\max}$?**
- **Trả lời:**
  - Không lồi vì $\text{SINR}_k(\mathbf{W}) = \frac{|\mathbf{h}_k^H \mathbf{w}_k|^2}{\sum_{j\neq k} |\mathbf{h}_k^H \mathbf{w}_j|^2 + \sigma^2}$ là phân thức của hai dạng toàn phương theo $\mathbf{W}$, dẫn tới nhiều cực trị địa phương (đã được chứng minh NP-hard trong [6], [7]).
  - Luôn đạt dấu bằng tại biên: Giả sử nghiệm $\mathbf{W}$ có $\|\mathbf{W}\|_F^2 < P_{\max}$. Nhân toàn bộ $\mathbf{W}$ với hệ số $c = \sqrt{P_{\max}} / \|\mathbf{W}\|_F > 1$. Khi đó $\text{SINR}_k(c\mathbf{W}) = \frac{c^2 |\mathbf{h}_k^H \mathbf{w}_k|^2}{c^2 \sum_{j\neq k}|\mathbf{h}_k^H \mathbf{w}_j|^2 + \sigma^2} = \frac{|\mathbf{h}_k^H \mathbf{w}_k|^2}{\sum_{j\neq k}|\mathbf{h}_k^H \mathbf{w}_j|^2 + \sigma^2 / c^2} > \text{SINR}_k(\mathbf{W})$ với mọi user đang hoạt động. Vì vậy $R_{\text{sum}}$ tăng chặt và nghiệm tối ưu bắt buộc thỏa $\|\mathbf{W}^*\|_F^2 = P_{\max}$.

**Câu 7: Cách xử lý ràng buộc công suất của nhóm có ưu điểm gì so với dùng hàm phạt (Penalty Function)?**
- **Trả lời:** Nếu dùng hàm phạt $f_{\text{pen}} = R_{\text{sum}} - \rho \max(0, \|\mathbf{W}\|_F^2 - P_{\max})$, ta phải dò siêu tham số $\rho$ và phần lớn cá thể trong quần thể sẽ rơi vào vùng không hợp lệ làm lãng phí số lần đánh giá. Phép chiếu chuẩn hóa Frobenius $\mathbf{W} = \sqrt{P_{\max}}\tilde{\mathbf{W}}/\|\tilde{\mathbf{W}}\|_F$ (cho Direct) và chuẩn hóa simplex $\sum p_k = P_{\max}$ (cho Struct) đảm bảo **100% cá thể ở mọi thế hệ đều nằm chính xác trên mặt biên khả thi**, không cần bất kỳ tham số phạt nào.

**Câu 8: Giải thích ý tưởng `Struct-DE`! Tại sao rút từ $2MK$ xuống $2K$ biến mà không mất nghiệm tối ưu toàn cục, và tại sao thời gian chạy độc lập với $M$?**
- **Trả lời:**
  - Theo Định lý 2 của Björnson và cộng sự (*IEEE SPM*, 2014 [7]) về đối ngẫu đường lên–đường xuống (Uplink–Downlink Duality), bất kỳ điểm tối ưu Pareto nào của bài toán MU-MISO với ràng buộc tổng công suất đều có hướng beamformer chính là bộ lọc thu MMSE của kênh đường lên ảo: $\mathbf{u}_k(\boldsymbol{\lambda}) = (\mathbf{I}_M + \sum_j \frac{\lambda_j}{\sigma^2}\mathbf{h}_j\mathbf{h}_j^H)^{-1}\mathbf{h}_k$ với $\sum_k \lambda_k = P_{\max}$, sau đó gán công suất đường xuống $p_k \ge 0$ ($\sum_k p_k = P_{\max}$). Vì vậy không gian $2K$ chiều $(\mathbf{p}, \boldsymbol{\lambda})$ **chứa trọn nghiệm tối ưu toàn cục**!
  - Nhờ đồng nhất thức đẩy qua (Woodbury/push-through): $(\mathbf{I}_M + \tilde{\mathbf{H}}\mathbf{D}\tilde{\mathbf{H}}^H)^{-1}\tilde{\mathbf{H}} = \tilde{\mathbf{H}}(\mathbf{I}_K + \mathbf{D}\mathbf{G}_H)^{-1}$ với $\mathbf{G}_H = \tilde{\mathbf{H}}^H\tilde{\mathbf{H}} \in \mathbb{C}^{K\times K}$ chỉ tính **1 lần duy nhất** trước vòng lặp tiến hóa. Trong toàn bộ 100 thế hệ, ta chỉ giải hệ tuyến tính $K\times K$ (ví dụ $4\times 4$) mà **không hề nhân với chiều $M$**, nên thời gian chạy gần như giữ nguyên ~15.6 ms dù $M = 4$ hay $M = 32$!

**Câu 9: Trong code `optimizers.py`, tại sao miền tìm kiếm của `Struct-DE/PSO/GA` là `[-0.05, 1.0]` thay vì `[0.0, 1.0]`?**
- **Trả lời:** Vì ta áp dụng hàm cắt ngưỡng $\tilde{p}_k = \max(z_k, 0)$. Nếu đặt miền $[0, 1]$, xác suất một biến thực liên tục chạm chính xác giá trị $0$ là bằng 0, khiến thuật toán khó tắt hẳn một user yếu. Mở rộng nhẹ biên dưới sang `-0.05` tạo một khoảng có độ đo dương để giải thuật dễ dàng gán $p_k = 0$ — đây chính là chìa khóa giúp `Struct-DE` tự lập lịch người dùng và đạt 34.15 bps/Hz ở chế độ quá tải $K = 10 > M = 8$.

**Câu 10: Trong 3 giải thuật DE, PSO và GA, giải thuật nào tốt nhất cho bài toán này và vì sao?**
- **Trả lời:**
  - Ở không gian thấp $2K = 8$ chiều (`Struct`), cả ba đều đạt cùng cực trị toàn cục 23.654 bps/Hz ($\text{CV} = 0\%$), trong đó `Struct-PSO` nhanh nhất (13.09 ms) còn `Struct-DE` (15.62 ms) cân bằng nhất khi $K$ lớn.
  - Ở không gian cao $2MK = 64$ chiều (`Direct`), **`Direct-DE` ổn định và đạt trung bình tốt nhất** (20.48 bps/Hz, $\text{CV} = 4.91\%$) so với `Direct-GA` ($\text{CV} = 6.84\%$) và `Direct-PSO` ($\text{CV} = 10.45\%$). Lý do: toán tử đột biến vi sai $\mathbf{x}_{r_1} - \mathbf{x}_{r_2}$ của DE tự động điều chỉnh bước nhảy theo hình dạng phân bố quần thể, tránh hiện tượng cả đàn bị kéo sớm vào một cực trị địa phương như `gbest` trong PSO (như 2 điểm ngoại lai ~14 bps/Hz ở Hình 7a).

**Câu 11 (Câu hỏi mở rộng nâng cao – Ăn điểm 10): Nếu đề bài thêm ràng buộc công suất riêng cho từng anten (Per-Antenna Power Constraint) hoặc pha rời rạc 2-bit thì `Struct-DE` còn áp dụng nguyên xi được không?**
- **Trả lời:** Rất chính xác — Định lý cấu trúc đối ngẫu với $\mathbf{I}_M$ ở phương trình (10) chỉ đúng chính xác cho ràng buộc *tổng công suất* $\|\mathbf{W}\|_F^2 \le P_{\max}$. Nếu có ràng buộc công suất riêng từng anten, ma trận $\mathbf{I}_M$ phải thay bằng ma trận đường chéo biến đối ngẫu $\operatorname{diag}(\mathbf{q})$ (tăng thêm $M$ biến thực). Còn với ràng buộc phần cứng phi tuyến/rời rạc (như lượng tử hóa pha 2-bit hoặc bộ khuếch đại phi tuyến), biểu diễn trực tiếp `Direct-DE/GA` (hoặc lai khởi tạo từ `Struct-DE` rồi tinh chỉnh bằng `Direct-DE`) sẽ phát huy thế mạnh tối đa vì hoàn toàn không cần công thức đạo hàm hay cấu trúc đối ngẫu!

---

### Nhóm C: Giải thích Các Hiện tượng "Lạ" & Chuyên sâu trên Đồ thị

**Câu 12: Tại sao trên Hình 2 & Bảng 6, `Struct-DE` lại CAO HƠN cả `WMMSE` (34.85 vs 33.35 ở 30 dBm; 23.65 vs 21.51 ở Exp 6) và lại chạy NHANH HƠN cả `WMMSE` (15.6 ms vs 21.4 ms)?**
- **Trả lời:** Có 2 nguyên nhân hoàn toàn rõ ràng về mặt thuật toán và triển khai:
  1. *Về Sum-Rate:* WMMSE chỉ đảm bảo hội tụ về điểm dừng KKT địa phương và nổi tiếng là **hội tụ chậm (dưới tuyến tính)** ở vùng SNR cao khi chênh lệch suy hao gần–xa giữa các người dùng lớn (lên tới 15–20 dB trong vành khăn 20–100 m). Với giới hạn `wmmse_max_iter = 100` (khởi tạo từ RZF), WMMSE vẫn đang trên đà tăng chậm chưa chạm tới cực trị toàn cục, trong khi `Struct-DE/PSO/GA` tìm trực tiếp trên không gian $2K=8$ chiều và nhảy thẳng tới cực trị toàn cục chỉ sau ~10–15 thế hệ.
  2. *Về thời gian chạy:* Mỗi vòng lặp WMMSE phải phân rã trị riêng `eigh` ma trận $M \times M$ và chạy 60 bước chia đôi tuần tự tìm nhân tử Lagrange $\mu$ (lặp Python 100 lần). Ngược lại, `Struct-DE` tính ma trận Gram $K\times K$ một lần duy nhất rồi dùng `np.linalg.solve` **vector hóa theo lô (batch)** cho cả 36 cá thể trên mảng $4\times 4$ trong C/LAPACK nên thực thi rất nhanh.

**Câu 13: Tại sao trên Hình 2 ở vùng $P_{\max} = 20\text{--}30\,\text{dBm}$, `RZF` (32.40 bps/Hz) lại THẤP HƠN `ZF` (34.83 bps/Hz)? Bình thường người ta hay nói RZF luôn hơn ZF?**
- **Trả lời:** Rất tinh tế ạ! Nhận định "RZF luôn hơn ZF" chỉ đúng khi các người dùng có **suy hao đường truyền bằng nhau** (hoặc khi cùng dùng chung một kiểu phân bổ công suất). Trong cài đặt kinh điển của Peel et al. [5], `RZF` chuẩn hóa **toàn ma trận** $\mathbf{W}_{\text{RZF}} = \sqrt{P_{\max}}\frac{\mathbf{V}}{\|\mathbf{V}\|_F}$, mà chuẩn mỗi cột của ma trận nghịch đảo tỷ lệ nghịch với độ lợi kênh ($\|\mathbf{v}_k\| \propto 1/\sqrt{\beta_k}$). Do đó, khi có người dùng ở 25 m và người dùng ở 95 m (chênh lệch path loss > 17 dB), RZF chuẩn hóa toàn ma trận tự động đổ phần lớn công suất cho người dùng ở xa (kiểu cân bằng công bằng), làm giảm **tổng** tốc độ bit so với `ZF` chuẩn hóa từng cột rồi chia đều $P_{\max}/K$. Ngược lại ở 0–5 dBm, hạng tử điều chuẩn $\frac{K\sigma^2}{P_{\max}}\mathbf{I}_K$ phát huy tác dụng chống khuếch đại tạp âm nên RZF (4.19) lại vượt ZF (3.60).

**Câu 14: Trên Hình 4, tại sao đường `ZF` sụt rất mạnh tại $K = M = 8$ (17.26 bps/Hz) nhưng sang $K = 10$ lại nảy ngược lên 22.92 bps/Hz?**
- **Trả lời:** Đây là hiện tượng kinh điển của **lý thuyết ma trận ngẫu nhiên (luật Marcenko–Pastur)**: khi tỷ lệ $K/M \to 1$ (ma trận vuông $8 \times 8$), giá trị kỳ dị nhỏ nhất của $\mathbf{H}$ tiến gần về 0 khiến hệ số điều kiện của $(\mathbf{H}^H\mathbf{H})^{-1}$ tăng vọt → ZF phải dùng gần như toàn bộ công suất để "bơm" chống nhiễu nên tạp âm bị khuếch đại cực mạnh (tụt xuống 17.26 bps/Hz). Khi sang $K = 10 > M = 8$, ma trận trở nên "béo" không còn nghiệm triệt nhiễu 0 tuyệt đối, hàm `pinv` trả về nghiệm giả nghịch đảo chuẩn nhỏ nhất (tự bỏ ép null hoàn toàn) nên hệ số điều kiện bớt xấu hơn điểm kỳ dị $K=M=8$, làm Sum-Rate hồi nhẹ lên 22.92 bps/Hz — nhưng vẫn thua rất xa `Struct-DE` (34.15 bps/Hz).

**Câu 15: Trên Hình 5, tại sao khi $K_{\text{Rician}}$ tăng từ $-20\,\text{dB}$ (Rayleigh) lên $+15\,\text{dB}$ (LoS mạnh), `ZF` giảm nhẹ còn `MRT` lại tăng từ 7.96 lên 10.65 bps/Hz?**
- **Trả lời:** Khi $K_{\text{Rician}}$ tăng, kênh ít pha-đinh ngẫu nhiên sâu hơn và hội tụ về vector lái ULA thuần túy $\mathbf{a}(\theta_k)$. Với `ZF`, nếu ngẫu nhiên có 2 UE rơi vào góc gần nhau trong sector $120^\circ$, hai vector lái LoS gần như đồng tuyến làm ma trận kênh suy biến mạnh hơn so với Rayleigh tán xạ giàu (rich scattering), nên hiệu năng ZF giảm nhẹ. Với `MRT`, do tránh được các hố pha-đinh sâu Rayleigh và búp sóng ULA tập trung năng lượng ổn định vào từng UE tách góc, tổng tốc độ tăng nhẹ từ 7.96 lên 10.65 bps/Hz (nhưng vẫn bị giới hạn nặng bởi búp phụ gây MUI so với 22.23 bps/Hz của `Struct-DE`).

**Câu 16: Trong Bảng 6, tại sao `SciPy-DE (Struct)` có cột `Function_Evals = 40` rất nhỏ mà thời gian chạy lại là `27.88 ms`?**
- **Trả lời:** Vì nhóm bật chế độ `vectorized=True` trong `scipy.optimize.differential_evolution`. Ở chế độ này, thuộc tính `res.nfev` của SciPy đếm số lần gọi hàm mục tiêu theo **lô quần thể (batch calls)** (40 lần gọi × kích thước quần thể `popsize * dim = 10 * 8 = 80` cá thể ≈ **3.200 đánh giá cá thể**, tương đương 3.636 đánh giá của `Struct-DE` tự cài đặt!).

---

### Nhóm D: Câu hỏi Kiểm chứng Code & Demo Trực tiếp

**Câu 17: Nhóm kiểm chứng độ đúng đắn của code mô phỏng như thế nào?**
- **Trả lời:** Nhóm áp dụng kiểm thử phần mềm chuẩn bằng `pytest` kết hợp **Property-Based Testing (`hypothesis`)** với 21 test cases trong `btl2/tests/`, kiểm tra tự động:
  1. Hình học & thống kê kênh Rician ($\mathbb{E}[\|\mathbf{h}_k\|^2] \approx M\beta_k$ trên 2.000 mẫu).
  2. Ràng buộc công suất $\|\mathbf{W}\|_F^2 \le P_{\max} + 10^{-9}$ cho mọi thuật toán.
  3. Tính bất biến xoay pha cột $\mathbf{w}_k \mapsto e^{j\phi_k}\mathbf{w}_k$ và bất biến tỷ lệ $(\mathbf{H}, \sigma^2)$ trên 40 mẫu ngẫu nhiên của `hypothesis`.
  4. Triệt nhiễu ZF ($|\mathbf{h}_k^H \mathbf{w}_j| < 10^{-10}$ khi $M \ge K$).
  5. Độ khớp giữa bộ đánh giá nhanh ma trận Gram $\mathcal{O}(K^3)$ và công thức tính Sum-Rate tường minh (sai số $< 10^{-10}$).
  6. Đường cong hội tụ đơn điệu không giảm và khớp kết quả với `scipy.optimize.differential_evolution` sai lệch $< 2\%$.

**Câu 18: Nếu thầy muốn đổi tham số (ví dụ $M=16, K=6$) để kiểm tra code có chạy thật không thì làm thế nào?**
- **Trả lời:** Toàn bộ tham số tập trung tại `btl2/config.py`, không ghi cứng rải rác. Em có thể sửa `default_m` / `default_k` trong `config.py` rồi chạy `python btl2\main.py --quick` (chạy nhanh 15 mẫu kênh) hoặc `python btl2\plot_results.py` để tái tạo toàn bộ đồ thị ngay lập tức.

---

## Phần 5: Kịch bản Demo Live 2 phút trên Terminal (Copy-Paste Sẵn)

Khi giảng viên yêu cầu xem code và chạy kiểm chứng tại chỗ, mở PowerShell tại `C:\Quang\Thông tin di động` và chạy lần lượt:

```powershell
# 1. Chạy toàn bộ 21 test kiểm chứng toán học & vật lý (~1.3 giây)
python -m pytest "btl2\tests" -v

# 2. Mở nhanh file cấu hình tập trung config.py để chứng minh không hardcode
Get-Content "btl2\config.py" -TotalCount 65

# 3. (Nếu thầy yêu cầu chạy thử tái lập mô phỏng nhanh ~1 phút mà không đè dữ liệu lâu)
# Hoặc vẽ lại toàn bộ 7 hình 300 DPI từ dữ liệu .npz đã lưu (~5 giây):
$env:MPLBACKEND = "Agg"; python "btl2\plot_results.py"
```
