# H1 / H2 — RW-DIR: trọng số độ tin cậy trong ma trận Gram của BC

Người dùng đề xuất (2026-11-09). Chạy tuần tự trên **tài khoản Kaggle 2** (quota ~15,2h/30h đã dùng, refresh 2026-11-14 00:00 UTC):
H1 trước, H2 sau. Scheduler: mỗi 6 giờ, lần đầu **01:00 giờ Sài Gòn 10/11** (cron `CRON_TZ=Asia/Saigon 0 1,7,13,19 * * *`).

## 1. Ý tưởng nghiên cứu
**Giả thuyết:** khi teacher dự đoán sai một ảnh, quan hệ giữa ảnh đó với những ảnh khác trong ma trận Gram có thể kém tin cậy hơn.

SKD hiện tại dùng D = G_S − G_T. Đề xuất thay bằng D' = W ⊙ (G_S − G_T) (⊙ = nhân từng phần tử), với

    r_i = 1.0 nếu teacher dự đoán đúng ảnh i;  r_i = r_wrong nếu teacher dự đoán sai
    W_ij = sqrt(r_i · r_j) / mean_{k,l} sqrt(r_k · r_l)

D' được đưa vào Mahalanobis loss gốc của SKD, giữ nguyên Tikhonov (tik = 0.1) và Cholesky.
Khác F1: F1 đổi trọng số CE/KD theo từng mẫu và để BC nguyên; H chỉ đổi **BC** (CE, KL, LR, teacher, augmentation không đổi).

## 2. Cài đặt (`kaggle/streamlined_kd_cifar100.py`)
- Env mới: `GRAM_WEIGHT_MODE` (`none` mặc định = hành vi cũ; `teacher_correct`), `GRAM_R_WRONG` (mặc định 0.5).
- `gram_reliability_weights(teacher_logits, labels, r_wrong)`: ma trận W [B,B] đối xứng, trung bình đúng bằng 1 (chuẩn hóa), tính trong `torch.no_grad`.
  Teacher đúng/sai lấy từ `argmax(logit teacher) == nhãn` của chính ảnh train (sau augment), như F1.
- `paper_skd_loss(..., gram_weights)`: nhân `difference` (đã chia `num_classes`, là [B,B]) với W ngay sau khi tính, rồi covariance + Tikhonov + Cholesky + khoảng cách Mahalanobis như cũ.
  Class mask của BC giữ nguyên.
- Log mỗi epoch thêm: `gram_wrong=...% W[min,max]` (% ảnh teacher sai, min/max của W) và `gradnorm ce/kd/bc=...` (norm gradient từng thành phần ở lớp cuối, **chưa nhân hệ số**,
  đo cách 100 batch; BC có tính W). Dùng để kiểm tra thang gradient, vì chuẩn hóa W không đảm bảo gradient bằng nhau.
- Notebook (build từ cùng script; khác cấu hình E4 đúng các dòng mặc định RW-DIR):
  - `kaggle/experiments/h1_rw_dir_05.ipynb`: `GRAM_WEIGHT_MODE=teacher_correct`, `GRAM_R_WRONG=0.5`. Slug `streamlined-kd-cifar100-h1-rw-dir-05`.
  - `kaggle/experiments/h2_rw_dir_025.ipynb`: như H1 với `GRAM_R_WRONG=0.25`. Slug `streamlined-kd-cifar100-h2-rw-dir-025`.

## 3. Cấu hình chạy (giống E4)
Teacher mdistiller 72.41% (`TEACHER_SOURCE=mdistiller`), ResNet56 → ResNet20, 300 epoch, LR 0.1 giảm ×0.1 ở 100/188/250, SGD momentum 0.9, wd 5e-4, batch 64, T=4,
CE/KD/BC nền 0.5/0.5/0.5, mask `scalar`, seed 2021. **Không** dùng trọng số CE/KD theo mẫu của F1.

| Run | Thay đổi | Mục đích |
|---|---|---|
| E4 | baseline 71.70 epoch cuối / 71.73 TB 10 cuối / 72.10 best | đối chứng (đã có) |
| H1 | RW-DIR, r_wrong = 0.5 | kiểm tra ý tưởng chính |
| H2 | RW-DIR, r_wrong = 0.25 | kiểm tra độ nhạy |

Ước tính mỗi run ~4,8 giờ GPU (như E4/F1). Quota còn ~14,8h: H1 + H2 ≈ 9,6h, đủ.

## 4. Kiểm tra trước khi chạy (CPU, 2026-11-09) — đạt
1. Mặc định (`none`) và chế độ F1: loss và gradient **bit-exact** so với script trước (dữ liệu giả 64 mẫu × 100 lớp).
2. `GRAM_R_WRONG=1.0`: W ≡ 1 → loss và gradient chênh lệch đúng **0.0** so với E4 (bit-exact).
3. `r_wrong=0.5`: W khớp tính tay (đối xứng, trung bình 1; với 32/64 ảnh teacher đúng, W ∈ {0.6863, 0.9706, 1.3726}).
4. CE và KD không đổi, chỉ BC đổi (0.008745 → 0.008900 trên dữ liệu giả); gradient hữu hạn với r_wrong 0.5 và 0.25; chế độ lạ báo lỗi.
5. Smoke test 1 epoch (2 batch, ResNet20/ResNet56, dữ liệu giả): chạy được, log `gram_teacher_wrong_pct`, `gram_w_*`, `gradnorm_bc`.

## 5. Rủi ro đã biết (nêu trước khi chạy)
- **BC đóng góp rất nhỏ trong recipe này.** Thí nghiệm A3 (run B, 240 epoch): bỏ BC (`lambda_bc=0`) chỉ kém ~0,2 điểm, `lambda_bc=2.0` cũng không khác (±0,2) — trong mức nhiễu seed.
  Đổi trọng số bên trong BC vì thế có thể cho kết quả **không phân biệt được với E4**; đó sẽ là kết quả hợp lệ, không phải lỗi cài đặt.
- Khoảng cách Mahalanobis dùng covariance của chính D', nên khi nhân W vào D thì covariance cũng co giãn theo → một phần tác dụng của W bị khử (chỉ còn phần W không đồng đều giữa các hàng/cột).
  Vì vậy cần xem `gradnorm_bc` giữa H1/H2 và E4 (E4 chưa có `gradnorm_bc` trong stdout; so với giá trị nhỏ ~0,003–0,006 đã đo trên dữ liệu giả).
- Tỉ lệ ảnh teacher sai trên ảnh train đã augment ≈ 31% (G1 đo 30.8%).
- Mới 1 seed; nhiễu epoch-to-epoch ~0,2–0,3 nên ngưỡng sàng lọc 0,2 không phải bằng chứng ý nghĩa thống kê.

## 6. Quy tắc quyết định (người dùng đặt)
Chỉ số theo QUY ƯỚC CHỈ SỐ: so giữa các run bằng top-1 **epoch cuối** và TB 10 epoch cuối; best chỉ để so với paper (72.50).
- Ngưỡng sàng lọc: H1 hoặc H2 cải thiện **khoảng 0.2–0.3 điểm** epoch cuối so với E4 (71.70) → epoch cuối ≳ **71.90** (cũng ghi khoảng cách tới F1 72.13).
- Nếu có run đạt ngưỡng: chạy lại **ứng viên tốt nhất với seed 2022**, đồng thời cần **E4 seed 2022** để so cặp; sau đó mở rộng 3–5 seed (bước mở rộng này cần người dùng xác nhận).
  Quota sau H1+H2 còn ~5,2h (< 5,5h cần để submit) → seed 2022 chỉ chạy được sau khi quota refresh 2026-11-14 00:00 UTC; ghi blocker "chờ quota" nếu đến lúc đó chưa đủ.
- Nếu không run nào đạt ngưỡng: ghi kết luận (RW-DIR không thêm giá trị so với E4), dừng, tắt scheduler, chờ người dùng.

## Kết quả H1 (r_wrong = 0.5) — đã xác minh từ log Kaggle (tài khoản 2, version 2), 2026-11-10
Version 1 của notebook kẹt QUEUED ~6h không bắt đầu; với sự cho phép của người dùng đã push lại thành version 2 (chạy ngay). Config xác nhận trong `summary.json`: `seed=2021`, `epochs=300`,
`teacher_source=mdistiller`, `lr_milestones=[100,188,250]`, CE/KD 0.5/0.5, `kd_mask_mode=scalar`, `lambda_bc=-1`, `loss_mode=skd`, `optim_mode=sgd`, T=4.
Log: `device: cuda`, dataset OK, **teacher 72.41% / 92.50%** (giống E4). Quota tăng 19.385 s (~5,38h) cho một phiên — không có phiên chạy trùng.

| | E4 (đối chứng) | **H1** | H1 − E4 | F1 |
|---|---:|---:|---:|---:|
| **Test top-1 epoch 300 (chỉ số quyết định)** | 71.70 | **71.97** | **+0.27** | 72.13 |
| Trung bình 10 epoch cuối (291–300) | 71.73 | **71.95** | +0.22 | 72.14 |
| Best test top-1 (so với paper) | 72.10 | **72.20** (ep 272) | +0.10 | 72.32 |
| Best test top-5 | 93.02 | 93.06 | +0.04 | 92.83 |
| Best test loss | 1.0421 | 1.0374 | −0.005 | 1.0498 |
| Train top-1 cuối | 67.05 | 67.34 | +0.29 | 67.81 |
| Runtime | 16.959 s | 19.334 s (~5,37h) | | 17.332 s |

Đường cong H1: ep100 42.43 → ep101 63.39 → ep188 59.05 → ep189 70.01 → ep250 70.54 → ep251 71.79 → ep260 72.07 → ep300 71.97.
Số đo RW-DIR: teacher sai trên ảnh train đã augment **30.8–31.0%** mỗi epoch (khớp kỳ vọng ~31%); W ∈ [0.605, 1.21] (trung bình theo batch).
**gradnorm ce/kd/bc** (lớp cuối, chưa nhân hệ số): ~0.97 / 0.52 / **0.0053** ở cuối huấn luyện (ep1: 0.805/0.397/0.0064; ep100: 0.851/0.471/0.0036) — gradient BC chỉ ~1% gradient KD.

### Đánh giá H1
- **Đạt ngưỡng sàng lọc** (epoch cuối 71.97 ≥ 71.90 = E4 + 0.2): +0.27 epoch cuối, +0.22 TB 10 cuối, +0.10 best. Thấp hơn F1 (72.13) 0.16.
- **Cần thận trọng khi diễn giải:** gradient của BC chỉ ~1% gradient KD (đo trực tiếp), nên một thay đổi bên trong BC khó là nguyên nhân thực của +0.27; nhiều khả năng một phần đáng kể là nhiễu quỹ đạo (đổi bất kỳ thứ gì cũng làm lệch đường huấn luyện).
  Nhiễu epoch-to-epoch ~0.2–0.3 và mới 1 seed → mức này không phải bằng chứng ý nghĩa thống kê. Cần H2 và seed 2022 (kèm E4 seed 2022) để phân biệt.

## Trạng thái
Code + notebook + test CPU xong (2026-11-09). **H1 COMPLETE** (xem trên).
**H2 ĐÃ SUBMIT** 2026-11-10 ~07:40 UTC, tài khoản 2 `kitit5512trn`, version 1, commit `48b47c1`:
https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-h2-rw-dir-025 — `kaggle kernels pull -m` xác nhận dataset `fedesoriano/cifar100`, Docker image ghim, T4, GPU + internet bật; RUNNING.
Quota tài khoản 2 trước H2: 20,65h/30h (còn 9,35h). Dự kiến xong sau ~5,4h (~13:00 UTC). Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.
