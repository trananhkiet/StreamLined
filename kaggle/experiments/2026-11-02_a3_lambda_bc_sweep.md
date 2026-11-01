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

## Trạng thái
A3-bc0 submit 2026-11-01 ~20:08 UTC: notebook `streamlined-kd-cifar100-a3-lambda-bc0` version 1, dataset attach đã xác minh, RUNNING (>2 phút). Chưa có kết quả. A3-bc2 chờ sau. Xem EXPERIMENT_HANDOFF.md (mục "TRẠNG THÁI HIỆN TẠI").
