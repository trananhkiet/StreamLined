# G1 / F4 — sửa nhãn mềm teacher khi sai, và chuyển dần sang học nhãn ở giai đoạn LR cuối

Người dùng chọn (2026-11-07), sau kế hoạch F (F1 tốt nhất: 72.13 epoch cuối / 72.32 best). Chạy tuần tự trên **tài khoản Kaggle 2**
(quota ~5,4h/30h đã dùng, refresh 2026-11-14 00:00 UTC): G1 trước, F4 sau. Scheduler: lần chạy đầu 14:00 giờ Sài Gòn 07/11, sau đó mỗi 5 giờ.

## Đối chứng (seed 2021, teacher mdistiller 72.41%, 300 epoch, milestones 100/188/250, CE/KD/BC nền 0.5, T=4, mask scalar)
| Run | Final | TB 10 cuối | Best |
|---|---:|---:|---:|
| E4 (không trọng số) | 71.70 | 71.73 | 72.10 |
| F1 (hard 0.6/0.4) | 72.13 | 72.14 | 72.32 |

## G1 — sửa nhãn mềm của teacher ở ảnh nó đoán sai (một yếu tố so với E4)
- Với ảnh train mà argmax(teacher) ≠ nhãn: **hoán đổi logit của lớp teacher chọn nhầm với logit của lớp đúng** → nhãn mềm giữ nguyên hình dạng phân phối
  (cùng tập giá trị logit) nhưng lớp đúng giữ xác suất cao nhất. Teacher sau sửa đúng 100% ảnh train.
- Áp cho mọi thứ dùng logit teacher trong loss: KD, BC và hai mask độ tự tin (xác suất max không đổi; chỉ `class_confidence` đổi nhẹ). Đánh giá teacher trên test không đổi.
- Trọng số CE/KD/BC giữ 0.5/0.5/0.5 như E4 (không có trọng số theo mẫu) → G1 so sánh trực tiếp với **E4** và **F1**.
- Env: `TEACHER_CORRECTION=swap` (mặc định `none` = hành vi cũ). Log mỗi epoch in thêm `teacher_swapped=...` (% ảnh bị sửa, kỳ vọng ~31%).
- Giả thuyết: F1 giảm tin teacher ở ảnh nó sai; G1 giữ thông tin tương quan giữa các lớp của teacher nhưng bỏ phần dạy sai → có thể ≥ F1.
  Ý tưởng do Claude đề xuất, chưa tra kỹ nghiên cứu tương tự.
- Notebook `kaggle/experiments/g1_teacher_swap.ipynb`: khác cấu hình E4 (build từ cùng script) đúng dòng `TEACHER_CORRECTION`. Slug `streamlined-kd-cifar100-g1-teacher-swap`.

## F4 — chuyển dần sang học nhãn ở giai đoạn LR cuối (một yếu tố so với F1)
- Giả thuyết của người dùng: khi student đã gần bằng teacher thì nên học theo nhãn tiếp. Cài đặt theo **giai đoạn**: từ epoch 251 (LR 0.0001, giai đoạn cuối)
  nhân w_i (trọng số KD của F1: 0.6/0.4) và trọng số BC với 0.5 → KD 0.3 (teacher đúng) / 0.2 (sai), CE 0.7 / 0.8, BC 0.25. Epoch 1–250 giống hệt F1.
- Lý do dùng epoch thay vì điều kiện theo dữ liệu: student train top-1 (~68%) không bao giờ vượt teacher (~69% trên ảnh train) nên điều kiện "student ≥ teacher" không kích hoạt;
  giai đoạn LR cuối là lúc student đã hội tụ. Biến thể theo từng ảnh (F4b: p_student(nhãn) ≥ p_teacher(nhãn)) để dành cho sau.
- Env: `LATE_KD_START=251`, `LATE_KD_SCALE=0.5` (mặc định `LATE_KD_START=0` = tắt); chỉ dùng kèm `SAMPLE_WEIGHT_MODE=teacher_correct/teacher_prob`.
- Notebook `kaggle/experiments/f4_late_ce.ipynb`: khác cấu hình F1 (build từ cùng script) đúng dòng `LATE_KD_START`. Slug `streamlined-kd-cifar100-f4-late-ce`.

## Kiểm tra trước khi chạy (CPU, 2026-11-07)
1. Các chế độ cũ (mặc định, F1, F2, F3) cho loss và gradient **bit-exact** so với script trước.
2. G1: hàm hoán đổi khớp tính tay từng hàng; mọi hàng sau sửa có argmax = nhãn; tập giá trị logit mỗi hàng không đổi; loss trên logit đã hoán đổi giống hệt giữa hai phiên bản script.
3. F4: trước epoch 251 bit-exact F1; scale 1.0 sau epoch 251 bit-exact F1; sau epoch 251 khớp tính tay (CE 0.7/0.8, KD 0.3/0.2, BC ×0.5); báo lỗi nếu dùng mà không có trọng số theo mẫu.
4. Smoke test 1 epoch CPU (dữ liệu giả) G1 + F4: chạy được, log `teacher_swapped_pct`.

## Đánh giá (QUY ƯỚC CHỈ SỐ trong HANDOFF)
- Quyết định: top-1 epoch cuối và TB 10 cuối; so với E4 (71.70 / 71.73) và F1 (72.13 / 72.14). Ngưỡng ứng viên: ≥ E4 + 0.5 = 72.20 ở epoch cuối.
- So với paper: best top-1 (72.50). Mới 1 seed (2021); ứng viên qua ngưỡng sẽ được xác nhận bằng seed 2022 (cùng E4 seed 2022).

## Chạy
- Tài khoản 2 (`kitit5512trn`), T4, dataset `fedesoriano/cifar100`, internet bật. ~5 giờ mỗi run. Mỗi lần trigger submit tối đa một run.

## Trạng thái
Chuẩn bị xong (code + notebook + test). Chờ scheduler lần 14:00 giờ Sài Gòn 07/11 submit G1.
