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
Đã submit 2026-11-05 ~00:28 UTC sau khi E3 hoàn tất (E1 thắng run B ở cả 2 seed: +0.71 / +0.66).
- Kaggle (tài khoản 2): https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-e4-mdteacher-300ep — **version 1**; notebook `kaggle/experiments/e4_mdteacher_300ep.ipynb` không đổi từ commit `1d6054a`.
- `kaggle kernels pull -m` xác nhận dataset `fedesoriano/cifar100`, Docker image ghim, T4, internet bật; RUNNING.
- Quota tài khoản 2 trước khi submit: 15,87h/30h dùng (còn 14,1h ≥ 5,5h).
- COMPLETE (xem kết quả bên dưới).

## Kết quả E4 — đã xác minh từ log Kaggle (tài khoản 2, version 1), 2026-11-05
Config xác nhận trong `summary.json`: `seed=2021`, `epochs=300`, `teacher_source=mdistiller`, `learning_rate=0.1`, `lr_milestones=[100,188,250]`, CE/KD 0.5/0.5, `kd_mask_mode=scalar`, `lambda_bc=-1`, `loss_mode=skd`, `optim_mode=sgd`, T=4.
Log: `device: cuda`, dataset attach OK, tải teacher từ mdistiller OK (SHA-256 khớp), **teacher đo trên test: top-1 72.41%, top-5 92.50%** (khớp số ghi trong checkpoint; qua chốt ≥ 72.0).

| | Run B (s2021) | E1 (s2021) | **E4** | E4 − E1 | Paper SKD |
|---|---:|---:|---:|---:|---:|
| Teacher top-1 | 71.59 | 71.59 | **72.41** | +0.82 | 72.40 |
| Số epoch | 240 | 240 | **300** | | 240 |
| **Test top-1 epoch cuối (chỉ số chính)** | 70.04 | 70.75 | **71.70** | **+0.95** | 72.50 |
| Trung bình 10 epoch cuối | 70.07 | 70.68 | **71.73** | +1.05 | |
| Best test top-1 (có selection bias) | 70.26 | 70.79 | **72.10** (ep 291) | +1.31 | |
| Best test top-5 | 92.32 | 92.65 | **93.02** | +0.37 | |
| Best test loss | 1.0730 | 1.0405 | 1.0421 | +0.002 | |
| Train top-1 cuối | 66.73 | 68.26 | 67.05 | −1.21 | |
| Runtime | — | 13.975 s | 16.959 s (~4,71 giờ) | | |

Đường cong: ep100 44.71% → ep101 62.82% → ep188 63.84% → ep189 69.31% → ep200 70.67% → ep250 70.76% → ep251 71.44% → ep300 71.70%.
Ở epoch 240 (cùng mốc với paper nhưng đang giữa giai đoạn LR 0.001, chưa giảm lần cuối): 70.40%.

### Bài học (E4, 1 seed)
- **Kết quả tốt nhất của dự án**: 71.70% epoch cuối (+0.95 so với E1, +1.66 so với run B), best 72.10%. Khoảng cách tới paper (72.50) thu hẹp từ ~1.75 xuống **~0.80 điểm** (epoch cuối) / 0.40 điểm (best).
- Teacher mdistiller đo đúng 72.41% trên pipeline của ta → xác nhận ánh xạ trọng số đúng và teacher này tương đương số teacher của paper (72.40).
- **Không tách được đóng góp** của teacher (+0.82 điểm teacher) và của 300 epoch, vì E4 đổi 2 yếu tố cùng lúc theo chỉ định của người dùng.
  Gợi ý nếu muốn tách: chạy teacher mdistiller với lịch E1 240 epoch (80/150/200), một yếu tố so với E1.
- Train top-1 cuối thấp hơn E1 (67.05 vs 68.26) trong khi test cao hơn: teacher tốt hơn cho tín hiệu KD khó khớp hơn nhưng tổng quát tốt hơn.
- So với paper không hoàn toàn công bằng: paper dùng 240 epoch; mới 1 seed.
