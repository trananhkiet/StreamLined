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

## Trạng thái
Xem EXPERIMENT_HANDOFF.md (mục "TRẠNG THÁI HIỆN TẠI").
