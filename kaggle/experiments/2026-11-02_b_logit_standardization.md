# B — Logit Standardization (plan item B; khác với "run B" baseline cũ)

Ghi chú: session tự động (scheduler v5, lần chạy 19:17 giờ Sài Gòn 2026-11-02); không hỏi lại. Để tránh nhầm, run này gọi là **LS** (Logit Standardization).
Nền = run B (baseline CE/KD 0.5/0.5, KD mask scalar như repo, `LAMBDA_BC` mặc định = `kd_weight`), vì A1/A2/A3 chưa cho hướng nào vượt rõ run B.
Đối chứng = run B (70.26% best / 70.04% final).

## Giả thuyết
Theo [Sun et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html),
KD cổ điển ép student khớp **biên độ** logits của teacher. Student nhỏ (ResNet20) không thể có cùng biên độ với teacher lớn (ResNet56), nên ràng buộc đó có hại.
Chuẩn hóa z-score theo từng mẫu (`z_hat = (z - mean(z)) / (std(z) + 1e-7)`) trước softmax/KL cho student và teacher giúp student học **quan hệ giữa các class** thay vì biên độ.
Kỳ vọng LS > run B. Chưa biết hướng thật sự.

## Thay đổi (một yếu tố so với run B)
`LOGIT_STD=1`: trong `kd_loss`, cả `student_logits` và `teacher_logits` được `standardize_logits` trước `log_softmax/softmax(·/T)`. Công thức khớp mã gốc
([sunshangquan/logit-standardization-KD](https://github.com/sunshangquan/logit-standardization-KD/blob/master/mdistiller/distillers/KD.py)); kiểm tra số học: KL khớp công thức chính thức đến 6 chữ số.
**Chỉ nhánh KL** thay đổi. CE, Gram/BC (`paper_skd_loss`, vốn đã L2-normalize logits) và hai mask (độ tự tin của teacher, class mask) vẫn dùng logits gốc.
T=4 trước (đúng plan); thử riêng T=1,2 sau nếu cần. `kd_mask_mode=scalar`, `kd_mask_select=low`, `lambda_bc=-1` (→ 0.5) giữ như run B.
Code: `kaggle/streamlined_kd_cifar100.py` (default `LOGIT_STD=0` = hành vi cũ), notebook `kaggle/experiments/b_logit_standardization.ipynb` (default `LOGIT_STD=1`).

## Log thêm (theo plan)
Mỗi epoch trong `history.csv`: `student_logit_std`, `teacher_logit_std` (std logits gốc theo mẫu, trung bình), và `gradnorm_ce/kd/bc` (norm gradient lớp cuối của từng thành phần **chưa nhân hệ số**).
Dùng để phân biệt cải thiện thật với việc đổi thang loss: sau chuẩn hóa, thành phần KD ở T=4 nhỏ hơn nhiều (thử trên dữ liệu ngẫu nhiên: KD 1.42 → 0.24, gradnorm_kd ~0.26 → ~0.14) nên cần so `gradnorm_kd` giữa LS và run B.
**Lưu ý cảnh báo trước:** nếu LS kém B thì có thể do thang KD nhỏ (tương đương giảm `kd_weight`), không nhất thiết do chuẩn hóa; lúc đó cân nhắc đổi T hoặc `kd_weight` (mỗi lần một yếu tố).

## Config giữ như run B
CE/KD 0.5/0.5, T=4, tik 0.1, seed 2021, 240 epoch, SGD 0.1 (momentum 0.9, wd 5e-4), milestones 150/180/210, batch 64, T4, dataset `fedesoriano/cifar100` v1.

## Kiểm tra trước khi chạy
Unit test CPU: (1) với `LOGIT_STD=0` loss **giống hệt bit** commit trước (`5765f2d`) ở cả 5 cấu hình thử (scalar/per_sample, low/high, `lambda_bc` -1/0/2.0);
(2) `standardize_logits` bằng đúng công thức chính thức, mean≈0, std≈1; (3) với `LOGIT_STD=1` chỉ KD thay đổi (CE, BC giữ nguyên), loss và gradient hữu hạn;
(4) probe gradient lớp cuối và thống kê std logits chạy trên ResNet20 thật. Smoke test full pipeline không chạy được ở cloud session (không có CIFAR-100 cục bộ).

## Kết quả LS (T=4) — đã xác minh từ log Kaggle, notebook version 1, 2026-11-03
Notebook: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-b-logit-std (v1, T4, dataset attach OK). Config xác nhận trong `summary.json`: `logit_std=true`, `temperature=4.0`, `kd_mask_mode=scalar`, `lambda_bc=-1` (→0.5).

| | Run B | **LS (T=4)** | Chênh lệch |
|---|---:|---:|---:|
| Best test top-1 | 70.26% (ep 235) | **68.87%** (ep 228) | **−1.39** |
| Final test top-1 (ep 240) | 70.04% | **68.60%** | **−1.44** |
| Best test top-5 | 92.32% | **91.84%** | −0.48 |
| Best test loss | 1.0730 | 1.0742 | +0.001 |
| Runtime | — | 13.820 s (~3,84 giờ) | |

Teacher 71.59% / 92.51% (khớp). Train top-1 cuối 67.01%.
Đường cong: LS thấp hơn A2/A3 ngay sau LR drop đầu tiên (epoch 151: 59.96% so với 62.09% ở A2 và 62.52–62.61% ở A3; A1 là 60.29%).

### Bài học
- LS ở T=4 với recipe CE/KD 0.5/0.5 **kém run B ~1,4 điểm** (top-1 và top-5). Đây là kết quả âm rõ nhất cho đến nay, lớn hơn nhiều so với nhiễu seed ước tính ~0,3 điểm.
- **Nghi ngờ chính là thang loss**, không phải bản thân ý tưởng chuẩn hóa: sau z-score logits có std 1 nên ở T=4 phân phối rất phẳng và KL nhỏ đi nhiều (thử trên dữ liệu ngẫu nhiên: KD 1.42 → 0.24, ~6 lần), tức tín hiệu distill yếu đi gần như giảm `kd_weight`.
  Repo gốc của LS cho CIFAR-100 (Res56→Res20) cũng bù bằng `kd_weight` rất lớn (khoảng 9) so với CE nhỏ; recipe của ta (0.5/0.5) không có phần bù đó.
  Train top-1 cuối 67.01% cao hơn A1/A2/A3 (65.97–66.69%) trong khi test thấp hơn: student khớp train nhiều hơn nhưng tổng quát kém hơn, có thể vì nhận ít regularization từ teacher hơn, phù hợp với giả thuyết KD bị yếu đi (chỉ là gợi ý, chưa kiểm chứng riêng; chưa có train top-1 của run B để so trực tiếp).
- Chưa lấy được `history.csv` (host `www.kaggleusercontent.com` bị chặn) nên chưa có `gradnorm_kd` / `student_logit_std` để xác nhận trực tiếp giả thuyết thang loss.

## Kết quả LS-T2 (T=2) — đã xác minh từ log Kaggle, notebook version 1, 2026-11-03
Notebook: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-b-logit-std-t2 (v1, T4, dataset attach OK). Config xác nhận trong `summary.json`: `logit_std=true`, `temperature=2.0`, còn lại như run B.

| | Run B | LS (T=4) | **LS-T2 (T=2)** | LS-T2 − B |
|---|---:|---:|---:|---:|
| Best test top-1 | 70.26% (ep 235) | 68.87% (ep 228) | **69.30%** (ep 230) | **−0.96** |
| Final test top-1 (ep 240) | 70.04% | 68.60% | **68.96%** | **−1.08** |
| Best test top-5 | 92.32% | 91.84% | 92.16% | −0.16 |
| Best test loss | 1.0730 | 1.0742 | 1.0571 | −0.016 |
| Runtime | — | 13.820 s | 14.006 s (~3,89 giờ) | |

Teacher 71.59% / 92.51% (khớp). Train top-1 cuối 67.15%.

### Kết luận B (Logit Standardization) trong recipe CE/KD 0.5/0.5
- Hạ T từ 4 xuống 2 thu hồi ~0,4 điểm (best +0,43, final +0,36) và đưa top-5/test loss gần hoặc tốt hơn B, phù hợp giả thuyết "KD bị yếu đi sau chuẩn hóa ở T=4".
  Nhưng vẫn **kém run B gần 1 điểm top-1**; chuẩn hóa logits **không** cải thiện recipe này.
- Chưa thử: T=1, hoặc tăng `kd_weight` để bù thang (repo gốc dùng ~9) — đó là yếu tố thứ hai/khác, ngoài phạm vi "một yếu tố mỗi lần" của run này; ghi lại như hướng chưa khám phá, ưu tiên thấp vì quota hạn chế.
- Mới 1 seed (2021). Chưa có `history.csv` (host `www.kaggleusercontent.com` bị chặn) nên chưa kiểm chứng trực tiếp bằng `gradnorm_kd`.

## Bước tiếp
Chuyển sang plan item **C (DKD)**: [2026-11-03_c_dkd.md](2026-11-03_c_dkd.md).

## Trạng thái
B hoàn tất (LS T=4 và T=2). Xem EXPERIMENT_HANDOFF.md (mục "TRẠNG THÁI HIỆN TẠI").
