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

## Trạng thái
Xem EXPERIMENT_HANDOFF.md (mục "A2").
