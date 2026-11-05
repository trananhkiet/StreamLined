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
Chuẩn bị xong; submit ngay theo yêu cầu người dùng (xem HANDOFF).
