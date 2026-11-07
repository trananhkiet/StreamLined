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
Chuẩn bị xong (code + notebook + test). G1 và F4 đều xong (xem kết quả bên dưới).

### G1 đã submit (2026-11-07 07:00 UTC = 14:00 giờ Sài Gòn)
- https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-g1-teacher-swap — **version 1**; notebook `kaggle/experiments/g1_teacher_swap.ipynb` (commit `03bfda1`).
- `kaggle kernels pull -m` xác nhận dataset `fedesoriano/cifar100`, Docker image ghim, T4, internet bật; RUNNING sau ~3,5 phút (đã qua bước tải và kiểm tra teacher).
- Quota tài khoản 2 trước khi submit: 19.428 s / 108.000 s (5,4h) → còn ~24,6h. Dự kiến xong ~12:00 UTC (19:00 giờ Sài Gòn), có thể muộn đến ~12:30 UTC.
- **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.** F4 submit sau khi G1 đọc xong.

## Kết quả G1 (hoán đổi logit teacher khi sai) — đã xác minh từ log Kaggle (tài khoản 2, version 1), 2026-11-07
Config xác nhận: `teacher_correction=swap`, `sample_weight_mode=none`, `teacher_source=mdistiller`, `epochs=300`, `lr_milestones=[100,188,250]`, `seed=2021`.
Log: `device: cuda`, teacher 72.41% / 92.50%. Runtime 18.380 s (~5,1 giờ).

| | E4 | F1 | **G1** | G1 − E4 | G1 − F1 |
|---|---:|---:|---:|---:|---:|
| **Test top-1 epoch 300** | 71.70 | 72.13 | **71.44** | −0.26 | −0.69 |
| TB 10 epoch cuối | 71.73 | 72.14 | **71.77** | +0.04 | −0.37 |
| Best top-1 (so với paper 72.50) | 72.10 | 72.32 | **72.09** (ep 285) | −0.01 | −0.23 |
| Best top-5 | 93.02 | 92.83 | **93.16** | +0.14 | +0.33 |
| Best test loss | 1.0421 | 1.0498 | 1.0347 | −0.007 | −0.015 |
| Train top-1 cuối | 67.05 | 67.81 | 67.91 | +0.86 | +0.10 |

Đường cong G1: ep101 62.47 → ep189 69.70 → ep200 70.53 → ep251 71.67 → ep260 71.76 → ep280 71.59 → ep300 71.44.
`teacher_swapped` (% ảnh train có teacher sai, được sửa nhãn): TB **30.83%** (30.50–31.17) → teacher đúng ~69.2% ảnh train, khớp số đo ở F1/F2.
Top-5 epoch theo test top-1: 285 (72.09), 292 (72.03), 278/281 (71.98), 269 (71.96).

### Bài học (1 seed)
- **G1 ≈ E4**, kém F1: TB 10 cuối +0.04, best −0.01, epoch cuối −0.26 (epoch 300 rơi xuống dưới mức TB 10 cuối 0.33 → nhiễu giữa các epoch cuối, nên TB 10 cuối đáng tin hơn). Không đạt ngưỡng ứng viên (< 72.20).
- Sửa nhãn mềm của teacher **không thay được** việc giảm trọng số KD ở ảnh teacher sai (F1). Top-5 và test loss của G1 tốt hơn E4 nhẹ (93.16 vs 93.02; 1.035 vs 1.042) → xếp hạng lớp tốt hơn chút nhưng top-1 không tăng.
- Giả thuyết giải thích (chưa kiểm chứng): lợi ích của F1 đến từ việc **bớt ép student theo teacher ở 31% ảnh khó** (nhiều ảnh trong đó cũng khó với student), chứ không phải vì nhãn mềm bị "lệch lớp"; hoán đổi logit vẫn buộc student khớp một phân phối được dựng tay trên những ảnh teacher không nắm vững.

## F4 đã submit (2026-11-07 13:05 UTC = 20:05 giờ Sài Gòn)
- https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-f4-late-ce — **version 1**; notebook `kaggle/experiments/f4_late_ce.ipynb` (không đổi từ commit `03bfda1`).
- Quota tài khoản 2 trước khi submit: 37.854 s / 108.000 s (10,5h) → còn ~19,5h. Dự kiến xong ~18:05–18:30 UTC (01:05–01:30 giờ Sài Gòn 08/11).
- **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.**

## Kết quả F4 (giảm KD, tăng CE từ epoch 251) — đã xác minh từ log Kaggle (tài khoản 2, version 1), 2026-11-08
Config xác nhận: `sample_weight_mode=teacher_correct`, `sw_trust=0.6`, `sw_distrust=0.4`, `late_kd_start=251`, `late_kd_scale=0.5`, `teacher_correction=none`,
`teacher_source=mdistiller`, `epochs=300`, `lr_milestones=[100,188,250]`, `seed=2021`. Log: `device: cuda`, teacher 72.41% / 92.50%. Runtime 17.046 s (~4,73 giờ).

**Kiểm tra tính tất định:** F4 trùng F1 *từng số* ở epoch 100, 101, 188, 189, 200, 250 (ví dụ ep250 test 70.63%, train 65.50%) → hai run tách nhau đúng tại epoch 251 như thiết kế.

| | E4 | F1 | **F4** | F4 − F1 |
|---|---:|---:|---:|---:|
| **Test top-1 epoch 300** | 71.70 | **72.13** | **71.83** | −0.30 |
| TB 10 epoch cuối | 71.73 | **72.14** | **72.03** | −0.11 |
| Best top-1 (so với paper 72.50) | 72.10 | **72.32** | **72.32** (ep 297) | 0.00 |
| Best top-5 | 93.02 | 92.83 | 92.64 | −0.19 |
| Best test loss | 1.0421 | 1.0498 | 1.0353 | −0.015 |
| **Train top-1 cuối** | 67.05 | 67.81 | **69.36** | **+1.55** |
| Runtime | 16.959 s | 17.332 s | 17.046 s | |

Giai đoạn cuối (sau khi tách ở epoch 251): ep251 F1 71.86 vs F4 71.80; ep260 72.21 vs 72.20; ep280 71.93 vs 71.98; ep300 72.13 vs 71.83. Top-5 epoch: 297 (72.32), 293 (72.24), 270/278/289 (72.23).
`teacher_train_top1` TB 69.17% (như F1/F2).

### Bài học (1 seed)
- **F4 không cải thiện F1**: epoch cuối −0.30, TB 10 cuối −0.11, best bằng nhau (72.32). Chênh lệch nằm trong biên nhiễu giữa các epoch cuối (~0.2–0.3). Không đạt ngưỡng ứng viên (< 72.20).
- **Train top-1 tăng +1.55 (67.81 → 69.36) nhưng test không tăng**: dịch trọng số sang CE khiến student khớp nhãn train tốt hơn (bắt đầu tách teacher) mà không tổng quát tốt hơn → khoảng cách train–test rộng ra, dấu hiệu overfit nhẹ; KD vẫn có vai trò regularization ở LR cuối.
- Giả thuyết "student bằng teacher thì nên học theo nhãn" **chưa được ủng hộ** ở dạng chuyển theo giai đoạn (scale 0.5, từ epoch 251). Chưa thử dạng theo từng ảnh (F4b) hay mức khác.

## TỔNG KẾT KẾ HOẠCH G1 / F4 (seed 2021, nền E4: teacher mdistiller, 300 epoch, 100/188/250)
| Run | Thay đổi | Epoch cuối | TB 10 cuối | Best |
|---|---|---:|---:|---:|
| E4 | không | 71.70 | 71.73 | 72.10 |
| **F1** | trọng số theo teacher đúng/sai 0.6/0.4 | **72.13** | **72.14** | **72.32** |
| F2 | 0.7/0.3 | 71.90 | 72.04 | 72.23 |
| F3 | mềm 0.4 + 0.2·p | 71.73 | 71.77 | 71.97 |
| G1 | hoán đổi logit khi teacher sai | 71.44 | 71.77 | 72.09 |
| F4 | F1 + giảm KD/BC ×0.5 từ epoch 251 | 71.83 | 72.03 | 72.32 |

- **F1 vẫn tốt nhất**; G1 và F4 không vượt F1 và không đạt ngưỡng ứng viên (≥ 72.20 epoch cuối). Mọi biến thể quanh F1 (F2, F3, G1, F4) đều không hơn → F1 nằm ở vùng khá phẳng quanh 72.0–72.3.
- Nhiễu giữa các epoch cuối ~0.2–0.3 điểm (ví dụ G1: epoch 300 thấp hơn TB 10 cuối 0.33) — ngang cỡ chênh lệch giữa nhiều biến thể; cần ≥ 2 seed để phân biệt các biến thể sát nhau.
- Gợi ý (chờ người dùng): (1) **F1 seed 2022 + E4 seed 2022** (~9,7h, quota còn ~14,8h tới 14/11) để kiểm chứng F1 hơn E4 thật; (2) chỉ khi F1 được xác nhận mới tinh chỉnh thêm (mức 0.55/0.45, F4b theo từng ảnh);
  (3) hoặc chuyển sang thay đổi ngoài loss (ví dụ cấu hình huấn luyện) vì các biến thể loss quanh F1 đã bão hòa.
