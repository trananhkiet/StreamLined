# A1 — per-sample KD mask

Ghi chú: session tự động; không hỏi lại. Đối chứng = run B (CE/KD 0.5/0.5, 70.26% best / 70.04% final).

## Giả thuyết
Ở run B, `mean(KL_scalar * mask)` = `KL * mean(mask)` nên mask không chọn mẫu. Tính KL theo từng mẫu rồi `mean(mask_i * KL_i)` (mask low-confidence 50% như repo) làm KD chỉ distill từ mẫu teacher kém tự tin; kỳ vọng thay đổi accuracy (chưa biết hướng).

## Thay đổi (một yếu tố)
`KD_MASK_MODE=per_sample` (`kd_loss(..., per_sample=True)`); BC giữ nguyên. Code: `kaggle/streamlined_kd_cifar100.py` (default vẫn `scalar` = hành vi B). Notebook: `a1_per_sample_kd_mask.ipynb` (đổi default thành per_sample).
Config khác giữ như B: CE/KD 0.5/0.5, T=4, tik 0.1, seed 2021, 240 epoch, SGD 0.1, milestones 150/180/210, batch 64, T4.

## Kiểm tra trước khi chạy
Unit test CPU: loss hữu hạn, gradient hữu hạn (ce 2.63, kd 1.17, bc 0.009 trên dữ liệu ngẫu nhiên). Smoke test full pipeline không chạy được ở cloud session (không có CIFAR-100 cục bộ); code path thay đổi chỉ nằm trong `distillation_losses`.

## Kết quả (đã xác minh từ log và summary.json của Kaggle, version 5, 2026-11-01)
Notebook: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-a1-per-sample-mask (v5, T4, dataset `fedesoriano/cifar100` attach OK).

| | Run B (đối chứng) | **A1** | Chênh lệch |
|---|---:|---:|---:|
| Best test top-1 | 70.26% (epoch 235) | **69.28%** (epoch 223) | **−0.98** |
| Final test top-1 (epoch 240) | 70.04% | **68.97%** | **−1.07** |
| Best test top-5 | 92.32% | 92.25% | −0.07 |
| Best test loss | 1.0730 | 1.0563 | −0.0167 |
| Runtime | — | 13.160 s (~3,66 giờ) | |

Teacher 71.59% / 92.51% (khớp run B). Train top-1 cuối: 65.97%. LR drop ở epoch 150/180/210 hoạt động như cũ
(best: 45.60% @ep41 → 62.86% @ep175 → 68.57% @ep205 → 69.28% @ep223).

## Bài học
- Chỉ distill KL từ nửa mẫu mà teacher **kém tự tin** (mask low-confidence thật sự có tác dụng) làm student **kém hơn** run B khoảng 1 điểm
  cả best lẫn final. Top-5 gần như không đổi, nên mất mát nằm ở top-1 chứ không phải chất lượng xếp hạng chung.
- Cảnh báo: mới 1 seed (2021). Chưa có phương sai giữa các seed nên chưa kết luận mạnh; chênh ~1 điểm là tín hiệu đáng chú ý nhưng chưa phải bằng chứng cuối.
- Gợi ý cho A2: nếu mẫu teacher tự tin mới là mẫu hữu ích, mask high-confidence nên cải thiện so với A1.

## Bước tiếp
A2 (high-confidence, cùng recipe): xem [2026-11-01_a2_per_sample_high_confidence.md](2026-11-01_a2_per_sample_high_confidence.md).
