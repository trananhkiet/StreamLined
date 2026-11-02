# A3 — tách lambda_bc khỏi kd_weight (sweep 0 / 0.5 / 2.0)

Ghi chú: session tự động (scheduler v5, lần chạy 03:00 giờ Sài Gòn 2026-11-02); không hỏi lại.
**Cách hiểu đã chọn:** plan viết "trên cùng cấu hình". A1 và A2 đều chưa vượt run B, nên theo luật "chỉ kết hợp hướng đã cải thiện độc lập",
nền của A3 là **run B** (KD mask scalar như repo, `KD_MASK_MODE=scalar`, `KD_MASK_SELECT=low`), chỉ đổi hệ số BC.
Mỗi lần trigger chỉ submit một run nên sweep chạy lần lượt: **A3-bc0 (lambda_bc=0)** → A3-bc2 (lambda_bc=2.0). `lambda_bc=0.5` chính là run B (`kd_weight=0.5`),
đã có kết quả 70.26% / 70.04% nên không chạy lại (trừ khi cần đo phương sai seed).

## Giả thuyết
BC (Gram-matrix/Tikhonov, quan hệ giữa các mẫu) dùng chung hệ số với KD. Tách riêng để xem BC đóng góp bao nhiêu:
- `lambda_bc=0`: bỏ BC. Nếu kết quả ≈ B thì BC gần như không đóng góp; nếu kém hơn thì BC có ích.
- `lambda_bc=2.0`: tăng BC 4x so với B. BC có giá trị nhỏ (~0,009 trên dữ liệu ngẫu nhiên) nhưng giá trị nhỏ không đồng nghĩa gradient yếu, nên cần đo.
Chưa biết hướng; đây là giả thuyết cần kiểm chứng.

## Thay đổi (một yếu tố so với B)
`LAMBDA_BC` (mới): hệ số riêng của BC. Mặc định `-1` = dùng `kd_weight` (giữ nguyên hành vi A/B/A1/A2). A3-bc0 đặt `LAMBDA_BC=0.0`.
BC vẫn là BC toàn batch nhân class mask scalar như cũ (không đổi cách mask, để đối chứng). CE/KD weight giữ 0.5/0.5.
Code: `kaggle/streamlined_kd_cifar100.py`; notebook `kaggle/experiments/a3_lambda_bc0.ipynb` (default `LAMBDA_BC=0.0`).

## Log thêm (theo plan)
Mỗi 100 batch ghi norm L2 của gradient từng thành phần **chưa nhân hệ số** theo trọng số lớp cuối (`fc.weight`) và lấy trung bình theo epoch vào `history.csv`:
`gradnorm_ce`, `gradnorm_kd`, `gradnorm_bc`. Không thay đổi backward thật (dùng `autograd.grad` riêng, `retain_graph=True`).
Vẫn còn `teacher_top1_selected/rest`, `kd_selected_fraction` từ A2.

## Config giữ như B
CE/KD 0.5/0.5, T=4, tik 0.1, seed 2021, 240 epoch, SGD 0.1 (momentum 0.9, wd 5e-4), milestones 150/180/210, batch 64, T4, dataset `fedesoriano/cifar100` v1.

## Kiểm tra trước khi chạy
Unit test CPU (B=64, C=100): (1) loss mới **giống hệt bit** bản cũ (commit f6a09a5) ở cả 3 chế độ scalar/low, per_sample/low, per_sample/high (sai lệch 0);
(2) `lambda_bc` = -1/0/0.5/2.0 cho đúng trọng số BC (0 → BC bằng đúng 0), loss và gradient hữu hạn; (3) probe gradient lớp cuối trên ResNet20 thật cho giá trị hữu hạn dương,
reset sau mỗi epoch, backward thật vẫn chạy. Mẫu: gradnorm CE 1.07, KD 0.26, BC 0.006 (dữ liệu ngẫu nhiên, chỉ để minh họa thang).
Smoke test full pipeline không chạy được ở cloud session (không có CIFAR-100 cục bộ).

## Kết quả A3-bc0 (`LAMBDA_BC=0`) — đã xác minh từ log Kaggle, notebook version 1, 2026-11-02
Notebook: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-a3-lambda-bc0 (v1, T4, dataset attach OK).

| | Run B (`lambda_bc=0.5`) | **A3-bc0 (`lambda_bc=0`)** | Chênh lệch |
|---|---:|---:|---:|
| Best test top-1 | 70.26% (ep 235) | **70.03%** (ep 236) | −0.23 |
| Final test top-1 (ep 240) | 70.04% | **69.85%** | −0.19 |
| Best test top-5 | 92.32% | **92.63%** | +0.31 |
| Best test loss | 1.0730 | **1.0603** | −0.013 |
| Runtime | — | 13.830 s (~3,84 giờ) | |

Teacher 71.59% / 92.51% (khớp). Train top-1 cuối 66.66%. Config xác nhận trong `summary.json`: `kd_mask_mode=scalar`, `kd_mask_select=low`, `lambda_bc=0.0`.

### Bài học (A3-bc0)
- Bỏ hẳn BC chỉ làm top-1 giảm ~0,2 điểm (best và final), còn top-5 (+0,31) và test loss (−0,013) thậm chí tốt hơn B. Chênh lệch top-1 này nằm trong mức dao động seed thường gặp
  (A1 đến A2 đã lệch 0,5 chỉ vì đổi mask), nên **chưa chứng minh được BC có ích**; kết quả cho thấy đóng góp của BC trong recipe này rất nhỏ.
- Quan trọng hơn: bản thân kết quả A3-bc0 (không BC) vẫn cao hơn cả A1 lẫn A2, nghĩa là mất ~1 điểm ở A1/A2 do chọn mẫu per-sample, không do BC.
- Mới 1 seed (2021); cần nhiều seed trước khi kết luận. Không lấy được `history.csv` (host `www.kaggleusercontent.com` bị chặn) nên chưa có `gradnorm_ce/kd/bc` theo epoch để đối chiếu giả thuyết "BC nhỏ về giá trị chưa chắc yếu về gradient".

## Kế hoạch còn lại của A3
- **A3-bc2 (`LAMBDA_BC=2.0`)**: notebook `kaggle/experiments/a3_lambda_bc2.ipynb` (khác `a3_lambda_bc0.ipynb` đúng một dòng: default `LAMBDA_BC`). Đối chứng = run B (0.5) và A3-bc0 (0).
- `lambda_bc=0.5` = run B, không chạy lại.

## Trạng thái
A3-bc0 xong (ở trên). A3-bc2 submit 2026-11-02 ~04:08 UTC: notebook `streamlined-kd-cifar100-a3-lambda-bc2` version 1, dataset attach đã xác minh, RUNNING (>2 phút). Chưa có kết quả. Xem EXPERIMENT_HANDOFF.md (mục "TRẠNG THÁI HIỆN TẠI").
