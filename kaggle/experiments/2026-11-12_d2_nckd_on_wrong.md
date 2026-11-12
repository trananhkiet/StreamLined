# D2 — Error-aware decoupled KD: bỏ TCKD trên mẫu teacher sai (hướng 2)

Người dùng chọn (2026-11-12): "Thử hướng 2 đi. Train đi. 6h sau vô check lại." Nằm trong hướng nghiên cứu
[2026-11-12_research_direction_f1_mechanism.md](2026-11-12_research_direction_f1_mechanism.md). Một run, **seed 2022**, tài khoản Kaggle 1.

## Ý tưởng
KL của KD tách đúng thành hai phần (đẳng thức DKD):  KL = TCKD + (1 − p_T[y]) · NCKD,
với TCKD (Target-Class KD) = KL nhị phân "lớp đúng vs phần còn lại" và NCKD (Non-target-Class KD) = KL trên phân bố các lớp còn lại đã chuẩn hóa lại.
Khi **teacher sai**, TCKD bảo student "lớp đúng có xác suất thấp" — đối nghịch trực tiếp với CE; NCKD chỉ nói về quan hệ giữa các lớp sai nên có thể vẫn hữu ích.
- Teacher **đúng**: giữ nguyên KL đầy đủ (như F1).
- Teacher **sai**: KD_i = (1 − p_T[y]) · NCKD_i (bỏ TCKD). Vẫn nhân T² và vẫn dùng trọng số CE/KD của F1 (0.6/0.4) và mask scalar.
$$L=\\frac1B\\sum_i(1-w_i)CE_i+\\frac{m}{B}\\sum_i w_i\\,KD_i+0.5L_{BC},\\quad KD_i=\\begin{cases}KL_i & \\text{teacher đúng}\\\\(1-p_T[y])\\,NCKD_i & \\text{teacher sai}\\end{cases}$$

## Cài đặt (`kaggle/streamlined_kd_cifar100.py`)
- Env mới `WRONG_KD_MODE`: `full` (mặc định = F1) hoặc `nckd_only`; chỉ dùng với `SAMPLE_WEIGHT_MODE=teacher_correct`.
- `nckd_per_sample`: mask lớp đúng bằng −1e9 trước log_softmax/softmax để lấy phân bố non-target chuẩn hóa; hệ số (1 − p_T[y]) lấy từ softmax đầy đủ ở T=4.
- Log mỗi epoch: `wrong=…% tckd_removed=…%` (tỉ lệ teacher sai và **phần KL bị bỏ** trên các mẫu teacher sai) — cho biết hiệu ứng thực sự lớn cỡ nào.
- Notebook `kaggle/experiments/d2_nckd_on_wrong_seed2022.ipynb` (diff với F1 cùng bản build chỉ đúng dòng `WRONG_KD_MODE`); slug `streamlined-kd-cifar100-d2-nckd-wrong-seed2022`.
- **Kiểm tra CPU (2026-11-12):** (1) chế độ mặc định/F1/F3 bit-exact với script trước; (2) `WRONG_KD_MODE=full` ≡ F1 bit-exact; (3) đẳng thức KL = TCKD + (1−p_T[y])·NCKD đúng (sai số tối đa 7.6e-6, TCKD tính độc lập); (4) KD của D2 khớp tính tay, CE và BC không đổi; (5) chế độ lạ báo lỗi; (6) smoke test 1 epoch chạy được.
  (Trên dữ liệu giả, TCKD chỉ chiếm 3.8% KL của mẫu sai; trên dữ liệu thật có thể khác — xem log `tckd_removed`.)

## Cấu hình
Giống F1 seed 2022: teacher mdistiller 72.41, 300 epoch, LR 0.1 ×0.1 ở 100/188/250, SGD momentum 0.9 wd 5e-4 batch 64, T=4, CE/KD/BC nền 0.5, mask scalar, trọng số 0.6/0.4. Quota tài khoản 1 trước run: 24.52h/30h (còn 5.48h; run ~4.7–5.4h → sát giới hạn). Refresh 2026-11-14 00:00 UTC.

## Đối chứng và quy tắc kết luận (best top-1; ghi thêm epoch cuối và TB 10 cuối)
Cùng seed 2022: **F1** = 72.45 / 72.34 / 72.27 (best / cuối / TB 10); **E4** = 72.11 / 71.84 / 71.92; J2 = 72.39 / 72.02 / 72.14; J1 = 71.51 / 71.24 / 71.24. Trung bình hai seed: F1 72.39 / 72.24 / 72.21; E4 72.11 / 71.77 / 71.83. Paper SKD best 72.50.
Δ = best(D2) − best(F1 seed 2022). "tương đương" nếu |Δ| < 0.20; D2 "hơn" nếu Δ ≥ +0.20; "kém" nếu Δ ≤ −0.20. Xem thêm epoch cuối và TB 10 cuối (ổn định hơn best).
**Giới hạn:** một seed; nhiễu giữa run cùng cấu hình tới ~0.4; best lạc quan ~0.2–0.4; **chỉ sửa ~31% mẫu (nhánh teacher sai)** nên hiệu ứng kỳ vọng nhỏ (có thể < 0.2 → khó phân biệt nhiễu). Không có F1a/F1b nên chưa biết lợi ích của F1 đến từ nhánh mẫu sai hay mẫu đúng. Kết luận chỉ SƠ BỘ.

## Kết quả D2 — đã xác minh từ log Kaggle (tài khoản 1, version 1), 2026-11-12
Config xác nhận trong `summary.json`: `seed=2022`, `sample_weight_mode=teacher_correct` (0.6/0.4), `wrong_kd_mode=nckd_only`, `epochs=300`, `lr_milestones=[100,188,250]`, `teacher_source=mdistiller`, CE/KD 0.5/0.5, `kd_mask_mode=scalar`.
Log: `device: cuda`, **teacher 72.41 / 92.50**. Runtime 17.299 s (~4,81h). Log cơ chế: teacher sai **30.8–30.9%** ảnh train; **phần KL bị bỏ (TCKD) trên mẫu sai chỉ ~4.1% trung bình** (3.4–5.3%) giá trị KL của các mẫu đó.

| Seed 2022 | Best top-1 | Epoch cuối | TB 10 cuối | Best top-5 | Best test loss | Train top-1 cuối |
|---|---:|---:|---:|---:|---:|---:|
| **D2** (bỏ TCKD khi teacher sai) | **72.05** (ep 261) | 71.70 | 71.73 | 92.96 | 1.0523 | 67.58 |
| F1 (đối chứng) | 72.45 | 72.34 | 72.27 | 92.98 | 1.0267 | 67.52 |
| E4 | 72.11 | 71.84 | 71.92 | 92.86 | 1.0371 | 67.12 |
| J2 (xáo trộn) | 72.39 | 72.02 | 72.14 | 93.20 | 1.0377 | 66.61 |
| Δ D2 − F1 | **−0.40** | **−0.64** | **−0.54** | −0.02 | +0.026 | +0.06 |
| Δ D2 − E4 | −0.06 | −0.14 | −0.19 | +0.10 | +0.015 | +0.46 |

Đường cong D2: ep100 44.66 → ep101 63.21 → ep188 63.92 → ep189 69.90 → ep250 70.76 → ep251 71.56 → ep260 71.47 → ep300 71.70. Top-5 epoch: 261 (72.05), 295 (71.99), 289 (71.97), 285 (71.93), 294 (71.93).

### KẾT LUẬN SƠ BỘ (một seed; best top-1; quy tắc của plan)
- **Δ = 72.05 − 72.45 = −0.40 ≤ −0.20 → D2 KÉM hơn F1** (epoch cuối −0.64, TB 10 cuối −0.54: cùng hướng, lớn hơn).
- **D2 ≈ E4** (−0.06 best, −0.14 cuối, −0.19 TB 10): bỏ TCKD trên mẫu sai **xóa mất phần lợi ích của F1 so với E4** (F1 − E4 = +0.34 best / +0.50 cuối ở seed 2022).
- Giả thuyết của hướng 2 ("TCKD của mẫu teacher sai gây hại, NCKD còn hữu ích") **không được ủng hộ**; bằng chứng nghiêng ngược lại: tri thức lớp đích trên mẫu teacher sai (giữ ở trọng số KD 0.4 như F1) có thể **hữu ích**, ví dụ như tín hiệu làm mềm nhãn ("mẫu này mơ hồ")
  chứ không chỉ là nhiễu đối nghịch với CE. Đây là giả thuyết giải thích, chưa kiểm chứng.

**Giới hạn:**
1. **Một seed**; nhiễu giữa hai run cùng cấu hình tới ~0.4 (J2: 71.99 vs 72.39). Δ = −0.40 best bằng cỡ nhiễu; nhưng hướng nhất quán trên cả ba chỉ số (best/cuối/TB 10) và D2 rơi đúng mức E4.
2. **Phần KL bị bỏ chỉ ~4%** giá trị loss trên mẫu sai, nhưng ảnh hưởng gradient có thể lớn hơn giá trị loss (TCKD có gradient tập trung vào xác suất lớp đúng). Chưa đo gradient của TCKD riêng.
3. D2 chỉ đổi ~31% mẫu (nhánh teacher sai), nên không thể suy ra lợi ích của F1 đến từ nhánh nào (F1a/F1b chưa chạy).
4. Best top-1 lạc quan ~0.2–0.4.

**Gợi ý bước sau (chỉ nêu, chưa chạy; chờ người dùng quyết định):**
- Nếu muốn chắc về D2: lặp seed 2021 (đối chứng F1 seed 2021 = 72.32 best).
- F1a/F1b (tách phía mẫu sai vs mẫu đúng của F1) vẫn là cách định vị nguồn lợi ích rẻ nhất.
- Hướng đảo ngược (suy đoán): trên mẫu teacher sai **giữ TCKD, bỏ NCKD** để kiểm tra giả thuyết "tín hiệu lớp đích trên mẫu sai mới là phần hữu ích".
- Quota tài khoản 1: ~29,3h/30h đã dùng (ước tính; còn ~0,7h); refresh 2026-11-14 00:00 UTC → chưa chạy được gì trước đó.

## Trạng thái
**COMPLETE** (2026-11-12). Kết luận sơ bộ: D2 kém hơn F1 (−0.40 best), ≈ E4. Scheduler đã tắt; không còn run nào đang chạy.
