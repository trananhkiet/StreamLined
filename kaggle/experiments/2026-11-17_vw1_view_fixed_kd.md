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

## Kết quả — VW1 COMPLETE (đọc 2026-11-17 23:56 UTC)
Kaggle `kitit5512trn/streamlined-kd-cifar100-vw1-seed2022` v1 (tài khoản 2), 300 epoch, teacher mdistiller 72.41/92.50,
runtime 17.615 s (~4,89h). Log lấy bằng `kaggle kernels logs` (đủ 300 dòng epoch).

**Hợp lệ:** `vs weak` = 50.0% mọi epoch; `w w/s` = 0.5980/0.5400 ở **mọi** epoch (min = max); `Tacc w/s` TB 97.88/40.46 — trùng khớp tới
từng số với VS1 (cùng seed → cùng thứ tự batch và augmentation), nên VW1 và VS1/F1b chỉ khác nhau ở cách gán trọng số.

| Run (seed 2022) | Best | Cuối | TB 10 cuối | Train top-1 | Best top-5 |
|---|---:|---:|---:|---:|---:|
| E4 | 72.11 | 71.84 | 71.92 | 67.12 | — |
| F1b | 72.60 | 72.28 | 72.43 | 66.91 | — |
| VS1 | 72.55 | 72.22 | 72.37 | 66.54 | 93.21 |
| **VW1** | **72.34** (ep 294) | **71.81** | **71.90** | 66.47 | 92.98 |
| VW1 − F1b | −0.26 | **−0.47** | **−0.53** | | |
| VW1 − E4 | +0.23 | −0.03 | −0.02 | | |

### Bảng 2×2 đã đủ (seed 2022, best / cuối / TB10)
| | Trọng số cố định | Trọng số dao động (0.5/0.6 trên mẫu) |
|---|---|---|
| Không phân biệt view | E4 72.11/71.84/71.92 · J1 71.51/71.24/71.24 | J2 72.39/72.02/72.14 |
| Có phân biệt view | **VW1 72.34/71.81/71.90** | VS1 72.55/72.22/72.37 · F1b 72.60/72.28/72.43 |

### Diễn giải (theo tiêu chí ghi trước; ưu tiên cuối và TB10)
- Cuối 71.81 ≤ E4 + 0.2 (72.04) và TB10 71.90 ≤ E4 + 0.2 (72.12) → **rơi vào nhánh H3**: chỉ phân bổ KD theo view (đúng trung bình
  của F1b) **không** tái tạo được lợi ích của F1b; VW1 ≈ E4 ở cuối/TB10 (chỉ best +0.23).
- Ghép với VS1 ≈ F1b: lợi ích của F1b không đến từ việc biết teacher đúng ở mẫu strong nào (VS1), cũng không đến từ mức KD trung bình
  khác nhau giữa view (VW1). Thứ còn lại chung cho F1b, VS1 và J2 là **trọng số dao động giữa các mẫu trong batch** (0.5/0.6).
- Mẫu hình 2×2 (cuối/TB10): mọi run trọng số **cố định** (E4, J1, VW1) ở 71.2–71.9; mọi run trọng số **dao động** (J2, VS1, F1b) ở
  72.0–72.4. Phân biệt view chỉ thêm ~0.2 khi đã có dao động (VS1 so J2), và ≈ 0 khi không có dao động (VW1 so E4).
- **Kết luận thăm dò:** nghiêng mạnh về **H3 — dao động trọng số CE/KD giữa các mẫu hoạt động như một bộ điều chuẩn**; H2 (view) đơn lẻ
  không được ủng hộ.

### Giới hạn
1. Một seed (2022); nhiễu giữa run tới ~0.4 — chênh VW1 − F1b ở cuối/TB10 (~0.5) vượt mức này, nhưng các so sánh khác (VS1 − J2 ~0.2) thì không.
2. J1 (hằng 0.538) kém E4 0.6 vẫn chưa giải thích; nếu cả E4 lẫn J1 là "cố định" thì J1 thấp bất thường.
3. "Dao động" ở đây là trọng số nhị phân 0.5/0.6 với tỷ lệ ~69/31 (theo teacher đúng) hoặc hoán vị của nó; chưa biết phân phối khác
   (ví dụ ngẫu nhiên thuần, không liên quan teacher) có cho hiệu ứng tương tự không.
4. Phép thử trực tiếp H3 (chưa chạy, cần người dùng quyết định): trọng số ngẫu nhiên thuần, độc lập teacher, cùng trung bình/phương sai
  với F1b (ví dụ w ~ Bernoulli: 0.6 với xác suất 0.69, 0.5 còn lại, rút mới mỗi batch); và/hoặc thêm seed 2021 cho VW1/VS1.
