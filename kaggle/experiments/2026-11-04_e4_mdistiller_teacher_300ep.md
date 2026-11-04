# E4 — teacher chuẩn CRD/mdistiller (72.34%) + lịch E1 kéo dài 300 epoch

Người dùng chỉ định (2026-11-04): chạy **một run gộp**, không tách thành 2 run. Chạy sau khi E3 hoàn tất (cả 2 run seed 2022) và kết luận E đã ghi.

## Giả thuyết
Teacher hiện tại (checkpoint phát hành của repo SKD) chỉ đạt 71.59%, thấp hơn paper (72.40) 0.81 điểm. Dùng teacher ResNet56 chuẩn của CRD/mdistiller
(72.34% trong CRD, checkpoint tự ghi 72.41%), và train lâu hơn với lịch của E1 (run tốt nhất), có thể kéo student lên gần số paper (72.50).

## Thay đổi so với E1 (2 yếu tố, cố ý theo yêu cầu người dùng)
| | E1 | **E4** |
|---|---|---|
| Teacher | SKD release, 71.59% | **mdistiller `resnet56_vanilla/ckpt_epoch_240.pth`, 72.34%** |
| Epoch | 240 | **300** |
| Milestones (giảm LR ×0.1) | 80/150/200 | **100/188/250** (E1 nhân tỉ lệ 300/240 = 1.25; 187.5 làm tròn 188) |
| Còn lại | CE/KD/BC 0.5/0.5/0.5, T=4, SGD 0.1, momentum 0.9, wd 5e-4, batch 64, seed 2021 | giống E1 |

Bảng LR: epoch 1–100 = 0.1, 101–188 = 0.01, 189–250 = 0.001, 251–300 = 0.0001 (cùng tỉ lệ thời gian mỗi giai đoạn như E1).

**Giới hạn diễn giải:** đổi 2 yếu tố cùng lúc nên không tách được phần đóng góp của teacher và của số epoch.
Paper và các bảng KD dùng 240 epoch, nên E4 không phải so sánh trực tiếp với paper; là thử nghiệm "đẩy hết mức".

## Nguồn teacher
- Tar: `https://github.com/megvii-research/mdistiller/releases/download/checkpoints/cifar_teachers.tar` (364 MB; script stream và chỉ giải nén một file).
- File: `cifar_teachers/resnet56_vanilla/ckpt_epoch_240.pth`, SHA-256 `4143895c82105da0a73236230db11aee5be39b660fe21e1323037c774f994155`.
- Checkpoint là dict `{epoch: 240, accuracy: 72.41, optimizer, model}`; trọng số ở khóa `model`.

## Implement (`kaggle/streamlined_kd_cifar100.py`)
- Env mới `TEACHER_SOURCE` (`skd` mặc định = hành vi cũ; `mdistiller`). `EPOCHS`, `LR_MILESTONES` đã có.
- Load checkpoint bằng `weights_only=True`, chỉ cho phép thêm kiểu numpy scalar (accuracy được lưu dạng numpy).
- Checkpoint thiếu buffer `num_batches_tracked` của BN (bộ đếm bước, không dùng khi eval) → điền 0, rồi `load_state_dict(strict=True)`.
- Chốt an toàn: với `mdistiller`, nếu teacher top-1 trên test < 72.0 thì dừng run ngay (ánh xạ trọng số sai).
- Notebook `kaggle/experiments/e4_mdteacher_300ep.ipynb`: khác notebook nền đúng 3 dòng mặc định (`TEACHER_SOURCE`, `EPOCHS`, `LR_MILESTONES`).

## Kiểm tra trước khi chạy (CPU, 2026-11-04)
1. Mặc định (`skd`): `load_state_dict` mới trả về tensor giống hệt bản cũ cho checkpoint SKD (SHA khớp) → các run cũ / run B seed 2022 không đổi.
2. `mdistiller`: tải thật qua hàm của script, SHA khớp; 287 khóa trùng tên và shape với ResNet56 của ta, chỉ thiếu 57 `num_batches_tracked`; `strict=True` OK.
3. Forward (lan truyền xuôi) trên 8 ảnh ngẫu nhiên: logits của ResNet56 của ta = logits của model ResNet56 gốc trong repo mdistiller, chênh lệch tối đa **0.0**.
4. Normalization của mdistiller giống ta: mean (0.5071, 0.4867, 0.4408), std (0.2675, 0.2565, 0.2761).
5. Chưa đo được accuracy teacher trên CIFAR-100 ở local (không có dữ liệu); sẽ đọc dòng `Teacher ResNet56 (mdistiller): top1=...` trong log Kaggle (kỳ vọng ~72.3–72.4).

## Chạy
- Slug: `streamlined-kd-cifar100-e4-mdteacher-300ep` (tài khoản 2 `kitit5512trn`), T4, dataset `fedesoriano/cifar100`, internet bật (để tải teacher).
- Thời gian ước tính: ~4.9 giờ GPU (E1 13.975 s × 1.25).
- Log có dạng `epoch xxx/300 | ...`.

## Đánh giá
- Chỉ số chính: test top-1 epoch 300; phụ: trung bình 10 epoch cuối, best top-1 (có selection bias).
- So với E1 (seed 2021): 70.75 final / 70.68 trung bình 10 cuối; với paper SKD 72.50.

## Trạng thái
Chuẩn bị xong (code + notebook + test). **Chờ E3 xong** rồi scheduler v5 mới submit.
