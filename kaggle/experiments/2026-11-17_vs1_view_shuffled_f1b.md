# VS1 — View-Stratified Shuffled F1b (seed 2022): F1b cải thiện vì đâu?

Ngày: 2026-11-17. Nền: F1b seed 2022 (`SEED=2022 EPOCHS=300 TEACHER_SOURCE=mdistiller LR_MILESTONES=100,188,250
SAMPLE_WEIGHT_MODE=teacher_correct SW_TRUST=0.6 SW_DISTRUST=0.5`). VS1 chỉ đổi `SAMPLE_WEIGHT_MODE=teacher_correct_view_shuffled`.

## Giả thuyết
- **H1 (mức mẫu):** F1b tốt vì biết chính xác *mẫu nào* teacher đúng để tăng KD cho mẫu đó.
- **H2 (mức view):** F1b tốt chủ yếu vì phân bổ KD khác nhau giữa weak/strong (trọng số TB weak ≈ 0.598, strong ≈ 0.540),
  không cần biết teacher đúng ở mẫu nào.

## Thiết kế VS1
1. Tính trọng số F1b như cũ: w_i = 0.6 nếu teacher đúng, 0.5 nếu sai (CE dùng 1−w_i, KD dùng w_i).
2. Trong **từng batch**, hoán vị ngẫu nhiên w_i **chỉ giữa các mẫu cùng view** có mặt trong batch đó (view nào < 2 mẫu thì giữ nguyên).
3. Multiset trọng số của mỗi view trong batch giữ nguyên; trọng số mới (w̃) dùng chung cho CE và KD như F1b.
- `view_id`: loader là `ConcatDataset([weak, strong])` có shuffle, nên số weak/strong mỗi batch thay đổi và không theo nửa batch.
  Mỗi dataset được bọc bởi `ViewTagged` (trả thêm `view_id` 0 = weak, 1 = strong); dataset bên trong được gọi y như cũ, nên
  ảnh/augmentation/RNG không đổi. Thứ tự batch dùng generator riêng của DataLoader, không đổi.
- RNG hoán vị: `torch.Generator` CPU riêng (seed 2022 + 54321), không chạm RNG toàn cục.
- Không đổi: cách kết hợp CE/KD, mask scalar KD, BC (Gram/Mahalanobis), kiến trúc, augmentation, dataset, teacher, optimizer,
  batch size, epoch, LR schedule, seed.
- Log mỗi epoch: `vs weak=..% Tacc w/s=.. w w/s=.. same=..% trust_on_correct=..% skips=..`.

## Kiểm tra trước khi train — PASS

### Cơ chế loss của F1b (đọc code + test)
F1b áp trọng số **theo từng mẫu, trước khi lấy trung bình**: `CE = mean((1−w_i)·CE_i)`, `KD = mean(w_i·KL_i) × mask_fraction`
(mask_fraction là scalar ~0.5 tính từ confidence teacher), BC không đổi. Test gradient: ∂CE/∂s_i = (1−w_i)(p_i − e_y)/B khớp.
→ Hoán vị w_i thật sự thay đổi loss; không có vấn đề "reduce thành scalar trước rồi mới nhân trọng số".

### Unit test CPU (`kaggle/test_vs1_view_shuffle.py`, dữ liệu tổng hợp, teacher đúng weak 97.7% / strong 39.4%)
| Kiểm tra | Kết quả |
|---|---|
| P1 Tắt VS1 → loss F1b khớp bit-exact bản trước khi thêm VS1 | PASS |
| P2 Trọng số F1b áp theo từng mẫu trước khi lấy mean (kiểm gradient) | PASS |
| P3 400 batch: multiset, tổng, số lượng 0.5/0.6 của từng view giữ nguyên; RNG toàn cục không bị chạm | PASS |
| P3 P(teacher đúng \| w=0.6) sau hoán vị: weak 0.977 (kỳ vọng ngẫu nhiên 0.977), strong 0.413 (kỳ vọng 0.394); F1b = 1.000 | PASS |
| P4 View chỉ có 1 mẫu giữ nguyên trọng số; view_id không liên tục | PASS |
| P5 BC giống hệt F1b; CE/KD đúng công thức với trọng số đã hoán vị | PASS |
| P6 Tỷ lệ VS1/F1b trung bình 400 batch: CE 0.937, KD 1.000, ‖grad‖ 0.967 (trong [0.9, 1.1]) | PASS |
| P7 `ViewTagged`: ảnh/nhãn giống hệt với cùng RNG; ConcatDataset cho view 0/1 đúng; collate 3-tuple | PASS |
| P8 3 bước train: F1b bản mới == bản trước (bit-exact); VS1 với trọng số bằng nhau == F1b (bit-exact); VS1 0.6/0.5 khác F1b (1.1e-2) | PASS |

### Kiểm tra trên batch augment thật (Kaggle CPU, tài khoản 1, `trananhkiet/streamlined-kd-cifar100-vs1-preflight`)
Logits thật của student F1b (checkpoint cuối) và teacher trên 20.000 mẫu augment (10k weak + 10k strong của cùng ảnh train),
1.500 batch 64 mẫu rút ngẫu nhiên (trộn view như ConcatDataset). Teacher đúng: weak 97.84%, strong 39.86%.

| Chỉ số trên cùng batch | F1b | VS1 | Tỷ lệ VS1/F1b |
|---|---:|---:|---:|
| CE (đã nhân trọng số) | 0.56865 | 0.54052 | 0.951 |
| KD (đã nhân trọng số × mask) | 0.60664 | 0.61036 | 1.006 |
| BC | 0.00979 | 0.00979 | 1.000 |
| ‖gradient tổng tại logits‖ | 0.04310 | 0.04179 | 0.970 |

| View | Trọng số KD trung bình | P(teacher đúng \| w = 0.6): F1b → VS1 |
|---|---:|---|
| weak | 0.5979 | 1.000 → **0.980** |
| strong | 0.5395 | 1.000 → **0.411** |

Trọng số không đổi sau hoán vị: 74.7% mẫu.

### Nhận xét quan trọng trước khi train (ảnh hưởng cách diễn giải)
- Vì teacher đúng 97.8% trên view weak, hoán vị trong view weak gần như **không đổi gì** (0.98 vs 1.00). VS1 thực chất kiểm tra
  thông tin mức mẫu **trong view strong** (1.00 → 0.41). Nếu VS1 ≈ F1b thì thông tin mức mẫu ở view strong không quan trọng;
  nhưng không loại trừ được vai trò mức mẫu ở view weak (vì ở đó gần như không có gì để xáo).
- CE của VS1 thấp hơn ~5%: F1b đặt CE 0.5 lên chính các mẫu teacher sai (CE cao), VS1 rải đều nên tổng CE thấp hơn một chút.
  Đây là hệ quả tất yếu của việc bỏ liên kết mẫu–trọng số, không phải lỗi scaling; cần nhớ khi diễn giải.

## Train VS1
- Tài khoản Kaggle 2 `kitit5512trn` (người dùng yêu cầu dùng tài khoản khác cho bước train; tài khoản 1 hết quota 34.09h/30h;
  tài khoản 2 và 3 đều 0/30h — chọn tài khoản 2 vì đã chạy nhiều run 300 epoch thành công với cùng dataset/Docker).
- Notebook `kaggle/experiments/vs1_view_shuffled_f1b_seed2022.ipynb` (6 dòng mặc định khác script chung: 4 của E4 +
  `SAMPLE_WEIGHT_MODE=teacher_correct_view_shuffled` + `SW_DISTRUST=0.5`), metadata `kaggle/experiments/vs1_kernel-metadata.json`
  (slug `kitit5512trn/streamlined-kd-cifar100-vs1-seed2022`, T4, Docker ghim, dataset `fedesoriano/cifar100`).

## Tiêu chí diễn giải (ghi trước)
Ưu tiên epoch cuối và TB 10 epoch cuối; một seed chỉ là thăm dò; nhiễu giữa run tới ~0.4.
- VS1 kém F1b rõ (≥ 0.3 ở cuối và TB10): nghiêng về **H1** (thông tin teacher đúng/sai từng mẫu, ở view strong, có ích).
- VS1 ≈ F1b (chênh < 0.2): **H2** hoặc ngẫu nhiên giải thích phần lớn hiệu quả.
- VS1 vẫn hơn E4: giữ phân bổ trọng số theo view có thể có ích, nhưng chưa chứng minh nguyên nhân.

| Run (seed 2022) | Best | Cuối | TB 10 |
|---|---:|---:|---:|
| E4 | 72.11 | 71.84 | 71.92 |
| F1b | 72.60 | 72.28 | 72.43 |
| VS1 | TBD | TBD | TBD |

## Kết quả
(điền khi run xong)
