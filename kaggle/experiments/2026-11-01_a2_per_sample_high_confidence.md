# A2 — per-sample KD mask, chọn mẫu high-confidence

Ghi chú: session tự động (scheduler v5, lần chạy 19:00 giờ Sài Gòn 2026-11-01); không hỏi lại. Đối chứng chính = A1 (cùng recipe, chỉ khác hướng chọn mẫu);
đối chứng phụ = run B (CE/KD 0.5/0.5, 70.26% best / 70.04% final).

## Giả thuyết
A1 (distill chỉ từ nửa mẫu teacher *kém* tự tin) kém B khoảng 1 điểm (69.28% best / 68.97% final). Giả thuyết A2: KD nên lấy từ nửa mẫu teacher *tự tin*,
vì tín hiệu ở đó đáng tin hơn; kỳ vọng A2 > A1 và có thể tiến gần hoặc vượt B. Chưa biết hướng thật sự, đây là giả thuyết cần kiểm chứng.

## Thay đổi (một yếu tố so với A1)
`KD_MASK_SELECT=high`: `confidence_mask = confidence >= median` thay vì `<=`. Giữ `KD_MASK_MODE=per_sample`, `L_KD = mean(mask_i * KL_i)` chia cho batch size
(không chia `sum(mask)` nên không cần bù hệ số). Mask BC (class mask) giữ nguyên. Code: `kaggle/streamlined_kd_cifar100.py` (default vẫn `low`),
notebook `kaggle/experiments/a2_per_sample_high_confidence.ipynb` (default `per_sample` + `high`).

## Log thêm (theo plan)
Mỗi epoch ghi vào `history.csv`: `teacher_top1_selected` (độ chính xác teacher trên nhóm mẫu được KD chọn), `teacher_top1_rest` (nhóm còn lại), `kd_selected_fraction`.
Dùng để xem chọn mẫu có thực sự đáng tin không, hay mẫu khó mới hữu ích.

## Config giữ như B/A1
CE/KD 0.5/0.5, T=4, tik 0.1, seed 2021, 240 epoch, SGD 0.1 (momentum 0.9, wd 5e-4), milestones 150/180/210, batch 64, T4, dataset `fedesoriano/cifar100` v1.

## Kiểm tra trước khi chạy
Unit test CPU (dữ liệu ngẫu nhiên, B=64, C=100): loss và gradient hữu hạn ở cả 3 chế độ (scalar/low, per_sample/low, per_sample/high);
chế độ `low` giữ nguyên hành vi A1; chế độ `high` chọn đúng nửa mẫu teacher tự tin (teacher top-1 trên nhóm chọn cao hơn hẳn nhóm còn lại; fraction 0.5).
Diff notebook A2 so với A1 chỉ gồm đổi mask + logging. Smoke test full pipeline không chạy được ở cloud session (không có CIFAR-100 cục bộ).

## Kết quả (đã xác minh từ log Kaggle, notebook version 1, 2026-11-02)
Notebook: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-a2-per-sample-high-conf (v1, T4, dataset attach OK).

| | Run B | A1 (low-conf) | **A2 (high-conf)** | A2 − A1 | A2 − B |
|---|---:|---:|---:|---:|---:|
| Best test top-1 | 70.26% (ep 235) | 69.28% (ep 223) | **69.76%** (ep 234) | +0.48 | **−0.50** |
| Final test top-1 (ep 240) | 70.04% | 68.97% | **69.57%** | +0.60 | **−0.47** |
| Best test top-5 | 92.32% | 92.25% | 92.04% | −0.21 | −0.28 |
| Best test loss | 1.0730 | 1.0563 | 1.1067 | +0.050 | +0.034 |
| Runtime | — | 13.160 s | 14.173 s (~3,94 giờ) | | |

Teacher 71.59% / 92.51% (khớp). Train top-1 cuối 66.00%.

## Bài học
- Giả thuyết được ủng hộ **một phần**: distill từ mẫu teacher tự tin (A2) tốt hơn từ mẫu kém tự tin (A1) khoảng +0,5 đến +0,6 điểm top-1.
- Nhưng **cả hai biến thể per-sample đều kém run B** (scalar, mask thực tế không chọn mẫu): A2 vẫn −0,5. Vậy việc ép KD chỉ lấy một nửa mẫu (theo thứ tự nào cũng vậy) đang làm mất tín hiệu hữu ích; A1/A2 không phải hướng cải thiện so với B.
- A2 có top-5 và test loss kém B, còn A1 có test loss tốt hơn B nhưng top-1 kém hơn: các chỉ số không cùng chiều, nên chưa nên đọc một chỉ số riêng lẻ.
- Cảnh báo: 1 seed (2021); phương sai giữa seed chưa đo, chênh ~0,5 điểm chưa đủ để kết luận mạnh.
- Không lấy được `history.csv` (host `www.kaggleusercontent.com` bị chặn trong environment) nên chưa có số `teacher_top1_selected/rest` theo epoch; cần mở host này hoặc tải file thủ công để phân tích phần này.

## Bước tiếp
Theo luật "chỉ kết hợp hướng đã cải thiện độc lập" và vì A1/A2 chưa vượt B, A3 lấy **run B làm nền**: [2026-11-02_a3_lambda_bc_sweep.md](2026-11-02_a3_lambda_bc_sweep.md).
