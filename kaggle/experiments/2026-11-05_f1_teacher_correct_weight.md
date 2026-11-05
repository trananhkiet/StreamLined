# F1 — trọng số CE/KD theo từng mẫu dựa vào việc teacher đoán đúng hay sai

Ý tưởng của người dùng (2026-11-05). Người dùng chọn phiên bản đúng/sai 0.6/0.4 (phương án 1); bản mềm kiểu CCKD để sau.

## Giả thuyết
Khi teacher đoán sai một ảnh, "đáp án mềm" của nó dễ dạy sai cho student. Tin teacher nhiều hơn khi nó đúng và tin nhãn thật nhiều hơn khi nó sai
sẽ giúp student tốt hơn. Nghiên cứu liên quan: CCKD-L (Mishra & Sundaram 2021, trọng số KD/CE theo xác suất teacher gán cho nhãn đúng),
WSL (Zhou et al., ICLR 2021), AID, Teacher Difficulty-Aware Weighting, BiKD. Chưa thấy ai áp vào SKD (mask theo độ tự tin + thành phần BC).

## Thay đổi (một yếu tố so với E4)
Với mỗi ảnh i: w_i = 0.6 nếu argmax(teacher) == nhãn, ngược lại 0.4.

    L = mean_i[(1 − w_i) · CE_i] + mean(mask_kd) · mean_i[w_i · KL_i] + 0.5 · BC

- Teacher đúng → CE 0.4 / KD 0.6; teacher sai → CE 0.6 / KD 0.4.
- BC giữ 0.5 (tính trên ma trận Gram của cả batch, không tách theo mẫu được).
- Giữ hệ số mask kiểu `scalar` của code gốc (KD nhân tỉ lệ mẫu được mask chọn, ~0.5), như run B/E4.
- Env: `SAMPLE_WEIGHT_MODE=teacher_correct` (mặc định `none` = hành vi cũ), `SW_TRUST=0.6`, `SW_DISTRUST=0.4`.
- Log thêm `teacher_train_top1_sw`: % ảnh train (sau augment) teacher đoán đúng = % mẫu nhận KD 0.6.

Còn lại giống hệt **E4**: teacher mdistiller (72.41%), 300 epoch, LR 0.1 giảm ×0.1 ở 100/188/250, SGD momentum 0.9, wd 5e-4, batch 64, T=4, seed 2021.
Notebook `kaggle/experiments/f1_teacher_correct_weight.ipynb`: khác cấu hình E4 (build từ cùng script) đúng một dòng `SAMPLE_WEIGHT_MODE`.

## Kiểm tra trước khi chạy (CPU, 2026-11-05)
1. Mặc định (`none`): loss và gradient **bit-exact** so với script trước (dữ liệu giả 64 mẫu).
2. `teacher_correct` với 0.5/0.5: loss chênh 0.0, gradient chênh tối đa 9e-10 so với baseline → công thức đúng là mở rộng của run B/E4.
3. `teacher_correct` 0.6/0.4: CE, KD khớp tính tay; BC giữ nguyên; gradient hữu hạn.
4. Smoke test 1 epoch (2 batch) với ResNet20/ResNet56 trên dữ liệu giả: chạy được, log `teacher_train_top1_sw` và `gradnorm_*` hoạt động.

## Lưu ý diễn giải
- Teacher đúng phần lớn ảnh train, nên đa số mẫu nhận KD 0.6: F1 vừa "tăng KD chung" vừa "giảm KD ở mẫu teacher sai"; hai hiệu ứng không tách được.
- Mức lệch 0.6/0.4 nhỏ → hiệu ứng có thể nhỏ.

## Đánh giá (quy ước chỉ số trong HANDOFF)
- Quyết định: top-1 epoch cuối và trung bình 10 epoch cuối so với E4 (71.70 / 71.73, seed 2021). Ngưỡng: F1 epoch cuối ≥ 72.20 (E4 + 0.5).
- So với paper: best top-1 (E4 72.10; paper 72.50).
- Nếu đạt ngưỡng: sau khi quota refresh (2026-11-07 00:00 UTC) chạy **F1 seed 2022** và **E4 seed 2022** (mỗi trigger một run) để xác nhận ở seed thứ hai.
  Nếu không đạt: ghi kết luận, dừng.

## Chạy
- Slug `streamlined-kd-cifar100-f1-teacher-weight` (tài khoản 2 `kitit5512trn`), T4, dataset `fedesoriano/cifar100`, internet bật (tải teacher mdistiller). ~4,7–4,9 giờ GPU.

## Trạng thái
Submit 2026-11-05 ~10:43 UTC (https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-f1-teacher-weight, version 1, commit `9090d82`). COMPLETE — xem kết quả bên dưới.

## Kết quả F1 — đã xác minh từ log Kaggle (tài khoản 2, version 1), 2026-11-05
Config xác nhận trong `summary.json`: `sample_weight_mode=teacher_correct`, `sw_trust=0.6`, `sw_distrust=0.4`, `teacher_source=mdistiller`, `epochs=300`,
`lr_milestones=[100,188,250]`, `seed=2021`, `kd_mask_mode=scalar`, `loss_mode=skd`, T=4. Log: `device: cuda`, dataset OK, **teacher 72.41% / 92.50%** (giống E4).

| | E4 (đối chứng) | **F1** | F1 − E4 | Paper SKD |
|---|---:|---:|---:|---:|
| **Test top-1 epoch 300 (chỉ số quyết định)** | 71.70 | **72.13** | **+0.43** | |
| Trung bình 10 epoch cuối (291–300) | 71.73 | **72.14** | +0.41 | |
| Best test top-1 (so với paper) | 72.10 (ep 291) | **72.32** (ep 293) | +0.22 | 72.50 (F1 −0.18) |
| Best test top-5 | 93.02 | 92.83 | −0.19 | |
| Best test loss | 1.0421 | 1.0498 | +0.008 | |
| Train top-1 cuối | 67.05 | 67.81 | +0.76 | |
| Runtime | 16.959 s | 17.332 s (~4,81 giờ) | | |

Đường cong F1: ep100 47.28% → ep101 63.14% → ep188 64.32% → ep189 70.08% → ep250 70.63% → ep251 71.86% → ep260 72.21% → ep300 72.13%.
So với E4 ở cùng mốc: ep101 62.82 / ep189 69.31 / ep251 71.44 / ep300 71.70 → F1 cao hơn E4 ở **mọi** mốc giảm LR (+0.3 đến +0.8).
Top-5 epoch theo test top-1: 293 (72.32), 275/288/289 (72.30), 270 (72.28) → phần cuối F1 ổn định quanh 72.1–72.3, không có một epoch "may" đột biến như E4 ep 291.
`teacher_train_top1_sw` không in ra stdout (chỉ có trong `history.csv`, host `www.kaggleusercontent.com` bị chặn) → chưa lấy được tỉ lệ mẫu nhận KD 0.6.

### Kết luận theo luật quyết định
- F1 epoch cuối 72.13 < ngưỡng 72.20 (E4 + 0.5) → **KHÔNG ĐẠT** (thiếu 0.07 điểm). Theo luật: không chạy cặp seed 2022 tự động, dừng kế hoạch F, chờ người dùng.
- Tuy không đạt ngưỡng, tín hiệu **nhất quán theo hướng tốt**: +0.43 epoch cuối, +0.41 TB 10 cuối, +0.22 best, cao hơn E4 ở mọi mốc giảm LR.
  Nhiễu seed đã đo ở E3 chỉ ~0.01–0.1 điểm (240 epoch, teacher cũ), nên +0.4 có thể là hiệu ứng thật — cần seed 2022 để xác nhận.
- **Best top-1 72.32: gần paper nhất từ trước tới nay** (−0.18), dù vẫn train 300 epoch (paper 240).
- Bài học: giảm tin teacher ở mẫu nó sai (và tăng nhẹ KD ở mẫu nó đúng) giúp student khớp train tốt hơn (+0.76 train top-1) và tổng quát tốt hơn;
  test loss gần như không đổi, top-5 giảm nhẹ.

### Gợi ý bước sau (chờ người dùng quyết định)
1. Chạy **F1 seed 2022 + E4 seed 2022** dù chưa qua ngưỡng, vì +0.43 sát ngưỡng và có lý do tin là thật (~9,6h GPU, sau khi quota refresh 2026-11-07).
2. Mức mạnh hơn **0.7/0.3** (một yếu tố so với F1) xem hiệu ứng có tăng theo mức lệch không.
3. Bản mềm kiểu CCKD: w_i = 0.4 + 0.2·p_teacher(nhãn đúng).
