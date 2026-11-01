# A1 — per-sample KD mask

Ghi chú: session tự động; không hỏi lại. Đối chứng = run B (CE/KD 0.5/0.5, 70.26% best / 70.04% final).

## Giả thuyết
Ở run B, `mean(KL_scalar * mask)` = `KL * mean(mask)` nên mask không chọn mẫu. Tính KL theo từng mẫu rồi `mean(mask_i * KL_i)` (mask low-confidence 50% như repo) làm KD chỉ distill từ mẫu teacher kém tự tin; kỳ vọng thay đổi accuracy (chưa biết hướng).

## Thay đổi (một yếu tố)
`KD_MASK_MODE=per_sample` (`kd_loss(..., per_sample=True)`); BC giữ nguyên. Code: `kaggle/streamlined_kd_cifar100.py` (default vẫn `scalar` = hành vi B). Notebook: `a1_per_sample_kd_mask.ipynb` (đổi default thành per_sample).
Config khác giữ như B: CE/KD 0.5/0.5, T=4, tik 0.1, seed 2021, 240 epoch, SGD 0.1, milestones 150/180/210, batch 64, T4.

## Kiểm tra trước khi chạy
Unit test CPU: loss hữu hạn, gradient hữu hạn (ce 2.63, kd 1.17, bc 0.009 trên dữ liệu ngẫu nhiên). Smoke test full pipeline không chạy được ở cloud session (không có CIFAR-100 cục bộ); code path thay đổi chỉ nằm trong `distillation_losses`.

## Trạng thái
v5 submit qua REST push lúc ~2026-11-01 03:50 UTC, dataset attach đã xác minh, đang RUNNING. Xem EXPERIMENT_HANDOFF.md (mục "A1 ĐÃ SUBMIT THÀNH CÔNG").
