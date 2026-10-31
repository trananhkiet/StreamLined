# SKD experiment handoff

Updated: 2026-10-31.

## Mục tiêu và trạng thái

Đang reproduce Streamlined Knowledge Distillation trên CIFAR-100 với teacher
ResNet56 và student ResNet20. Pipeline hiện giữ các phần chính của repo: 240
epoch, SGD, weak + strong augmentation (100.000 samples/epoch), temperature 4,
Tikhonov `0.1`, chạy trên Kaggle T4.

| Run | CE/KD | Teacher top-1 | Best student | Final student |
|---|---:|---:|---:|---:|
| A | 0.1/0.9 | 71.59% | 68.49% (epoch 235) | 68.16% |
| B — tốt nhất | 0.5/0.5 | 71.59% | **70.26% (epoch 235)** | 70.04% |
| Paper | — | 72.40% | 72.50% | — |

Run B cải thiện `+1.77` điểm so với A nhưng còn thấp hơn paper `2.24` điểm.
Teacher checkpoint được phát hành chỉ đạt 71.59%, thấp hơn số paper 0.81 điểm.

Chi tiết và config đầy đủ:

- [Run A](experiments/2026-10-18_resnet56_resnet20_skd_cifar100.md)
- [So sánh A/B](experiments/2026-10-18_skd_loss_weight_comparison.md)

Lưu ý: code hiện đánh giá và chọn best checkpoint bằng CIFAR-100 test set mỗi
epoch; `best top-1` vì vậy có selection bias. Luôn báo cáo cả best và epoch 240.

## Quy trình cho mỗi experiment

1. Copy notebook từ run B và chỉ thay đổi **một yếu tố**.
2. Ghi trước hypothesis, config và tên notebook vào file `experiments/*.md`.
3. Trước khi train, kiểm tra dataset attached, teacher SHA-256/accuracy, seed,
   batch size, LR schedule và không có batch limit.
4. Chạy notebook private trên Kaggle; theo dõi trạng thái và log các mốc LR.
5. Khi hoàn tất, lưu `summary.json`, `history.csv`, best/last checkpoint; ghi best
   epoch, final epoch, runtime và chênh lệch với run B vào experiment log.

## Plan cải thiện kết quả

Giữ teacher hiện tại, ResNet20 và recipe của run B làm mốc. Các hướng dưới đây
**chưa chạy**; đây là giả thuyết cải tiến, không bảo đảm đạt 72.5%. Mỗi run chỉ
đổi một yếu tố; chỉ kết hợp các hướng đã cải thiện độc lập.

### A. Điều chỉnh chọn mẫu và sức mạnh loss SKD

- Code hiện reduce KD/BC thành scalar rồi nhân mask: `mean(loss * mask)` chỉ
  bằng `loss * mean(mask)`, không thực sự chọn mẫu/class để distill.
- A1: giữ KL theo từng mẫu `k_i = T² KL(p_teacher || p_student)` rồi tính
  `L_KD = mean(mask_i * k_i)`, giữ mask low-confidence 50% như repo. So với B
  để đo riêng tác động chọn mẫu; không đồng thời đổi sang high-confidence.
- A2: từ A1, thử mask high-confidence; log accuracy của teacher trên nhóm được
  chọn để xem chọn mẫu đáng tin hay mẫu khó hữu ích hơn. Nếu chia cho
  `sum(mask)` thay vì batch size thì phải bù hệ số để tránh tăng gấp đôi loss.
- A3: tách `lambda_bc` khỏi `kd_weight`, thử `0 / 0.5 / 2.0` trên cùng cấu hình.
  BC là quan hệ giữa **samples** (ma trận B×B), nên class mask C chiều không
  thể gán trực tiếp lên các hàng BC; dùng BC toàn batch cho sweep này và giữ
  hệ số mask scalar cũ để đối chứng. Log norm gradient CE/KD/BC ở lớp cuối;
  BC nhỏ về giá trị không đồng nghĩa gradient yếu.

### B. Logit Standardization: giảm ràng buộc về biên độ logits

Theo [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html),
chuẩn hóa riêng logits mỗi mẫu: `z_hat = (z - mean(z)) / (std(z) + 1e-7)` trước
softmax/KL, để student học quan hệ giữa classes thay vì phải khớp biên độ teacher.
Chỉ thay nhánh KL; CE và Gram/BC vẫn dùng logits gốc. Chạy bật/tắt tại `T=4`
trước, sau đó thử riêng `T=1,2` nếu cần. Log std logits và KL gradient norm để
phân biệt cải thiện representation với thay đổi thang loss.
[Code gốc](https://github.com/sunshangquan/logit-standardization-KD/blob/master/mdistiller/distillers/KD.py).

### C. DKD: tách kiến thức target và non-target

[DKD (CVPR 2022)](https://arxiv.org/abs/2203.08679) tách KL thành TCKD
(đúng class vs phần còn lại) và NCKD (phân bố giữa các class không phải nhãn).
Điều này tránh NCKD bị suy giảm khi teacher quá tự tin ở class đúng.
Đầu tiên chạy `L = CE + w(t)*(alpha*TCKD + beta*NCKD)`, bỏ KL cũ và BC;
`alpha=1, beta=2, T=4, w(t)=min(epoch/20,1)` theo
[config Res56→Res20](https://github.com/megvii-research/mdistiller/blob/master/configs/cifar100/dkd/res56_res20.yaml)
và [defaults](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/engine/cfg.py).
Giữ LR 0.1 của B cho ablation; config gốc dùng 0.05 nên đây chưa phải reproduce
DKD đầy đủ. Nếu tốt, chạy DKD+BC để đo phần đóng góp thêm của SKD.
Không dùng nguyên defaults `Distiller/DKD.py` local: bản đó có clamp logits và
CE/alpha/beta khác upstream. Log TCKD/NCKD riêng để biết nhánh nào hữu ích.

### D. DOT: cải tiến optimizer khi CE và distillation cạnh tranh

[DOT (ICCV 2023), code chính thức](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/engine/dot.py)
giữ buffer momentum riêng cho task loss và distillation loss. Áp dụng cho
`L_task=CE`, `L_distill=KD+BC`; backward riêng rồi cập nhật theo DOT, không cộng
loss trước backward như SGD hiện tại. Dùng implementation gốc, lưu cả hai
momentum buffer để resume. So với cùng loss dùng SGD; log cosine giữa hai
gradient, accuracy và thời gian/epoch. Đây là hướng sau cùng vì cần sửa trainer
và tăng chi phí backward, trong khi B/C chỉ cần thay loss.

Ưu tiên **B trước** vì ít sửa code và có cơ sở từ paper; sau đó A, C, D theo
ngân sách GPU. Sweep hệ số/nhiệt độ là đề xuất của dự án, không phải cấu hình
đã được paper xác nhận tối ưu cho SKD. Mỗi run ghi vào `experiments/*.md`:
`giả thuyết → thay đổi → config → kết quả → bài học → bước tiếp`.
Chọn hyperparameter bằng validation tách từ train, giữ cùng split cho các ứng
viên; run B dùng full train chỉ là mốc lịch sử. Chốt recipe rồi train full train
với ít nhất 3 seed, báo cáo mean/std và accuracy cuối; chỉ test sau khi chốt.

## Kinh nghiệm đã rút ra

- `0.5/0.5` phù hợp implementation hiện tại hơn `0.1/0.9`.
- Không dùng download CIFAR-100 bên ngoài; phải attach dataset Kaggle để tránh
  mất hàng giờ vì mạng chậm.
- Không kết luận từ một seed hoặc so trực tiếp với baseline 65.79% chưa matched.
- Giữ từng run ở notebook/version riêng để không mất output và checkpoint cũ.
