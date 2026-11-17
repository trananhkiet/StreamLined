# VW1 — Trọng số KD cố định theo view (seed 2022): view hay dao động trọng số?

Ngày: 2026-11-17. Nền: F1b seed 2022 (E4 + trọng số mẫu CE 1−w / KD w). VW1: `SAMPLE_WEIGHT_MODE=view_fixed`,
`SW_VIEW_WEAK=0.598`, `SW_VIEW_STRONG=0.540` (= trọng số TB theo view của F1b đo trong log VS1). Mọi mẫu weak nhận 0.598, mọi mẫu
strong nhận 0.540; **không** dùng teacher đúng/sai, **không** có dao động ngẫu nhiên. Còn lại giữ nguyên F1b (mask scalar, BC, kiến trúc,
augmentation, optimizer, LR, seed). `view_id` lấy từ `ViewTagged` như VS1.

## Câu hỏi (bảng 2×2, seed 2022, best / cuối / TB10)
| | Trọng số cố định | Trọng số dao động (0.5/0.6 trên mẫu) |
|---|---|---|
| Không phân biệt view | E4 (0.5) 72.11/71.84/71.92 · J1 (0.538) 71.51/71.24/71.24 | J2 (xáo toàn batch) 72.39/72.02/72.14 |
| Có phân biệt view | **VW1: ?** | VS1 72.55/72.22/72.37 · F1b 72.60/72.28/72.43 |

- **H2 (view):** chỉ cần mức KD khác nhau giữa weak/strong.
- **H3 (dao động):** trọng số dao động giữa các mẫu đóng vai trò điều chuẩn.

## Tiêu chí diễn giải (ghi trước; ưu tiên cuối và TB10; 1 seed = thăm dò, nhiễu ~0.4)
- VW1 ≈ F1b/VS1 (chênh < 0.2) → **H2 được ủng hộ**: chỉ cần trọng số theo view.
- VW1 thấp hơn F1b ≥ 0.3 nhưng vẫn cao hơn E4 ≥ 0.2 → view và dao động đều góp một phần.
- VW1 ≈ E4 hoặc thấp hơn (≤ E4 + 0.2) → **H3 nghiêng mạnh**: lợi ích đến từ dao động trọng số, không phải view.
- Lưu ý: J1 (hằng 0.538) kém E4 0.6 chưa giải thích; nếu đó là nhiễu thì so sánh "cố định vs dao động" yếu đi.
- Log mỗi epoch: `vs weak=..% Tacc w/s=.. w w/s=0.5980/0.5400 ...` (`same`/`trust_on_correct` = 0 trong mode này, không có ý nghĩa).

## Preflight (CPU) — PASS
`kaggle/test_vw1_view_fixed.py`: F1b và VS1 bit-exact so với bản trước khi thêm VW1; trọng số VW1 đúng 0.598/0.540 cho mọi mẫu theo view,
không phụ thuộc teacher đúng/sai; CE/KD đúng công thức, BC giống hệt; thiếu view_id → lỗi; VW1 với hai trọng số bằng nhau ==
F1b với SW_TRUST=SW_DISTRUST (3 bước train, bit-exact); VW1 0.598/0.540 đổi cập nhật thật; thống kê epoch đúng 0.598/0.540.
`kaggle/test_vs1_view_shuffle.py` chạy lại trên code mới: PASS.

## Train
Tài khoản 2 `kitit5512trn` (quota 5.04/30h trước run). Notebook `kaggle/experiments/vw1_view_fixed_kd_seed2022.ipynb`, metadata
`kaggle/experiments/vw1_kernel-metadata.json` (slug `kitit5512trn/streamlined-kd-cifar100-vw1-seed2022`, T4, Docker ghim).
Lịch: submit 18:01 UTC 2026-11-17; kiểm tra kết quả 23:56 UTC (theo yêu cầu người dùng: 2 lần).

## Kết quả
(điền sau)
