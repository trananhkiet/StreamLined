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

## Trạng thái
Code + test + notebook xong (2026-11-12). Xem HANDOFF cho trạng thái submit.
