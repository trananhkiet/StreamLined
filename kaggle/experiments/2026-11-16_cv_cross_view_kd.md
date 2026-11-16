# Cross-View KD: D0 (chẩn đoán) → CV1 (train 300 epoch), seed 2022

Ngày lập kế hoạch: 2026-11-16. Nền: E4 seed 2022 (`SEED=2022 EPOCHS=300 TEACHER_SOURCE=mdistiller LR_MILESTONES=100,188,250`,
mọi thứ khác là mặc định của `streamlined_kd_cifar100.py`). Teacher mdistiller ResNet56 (72.41%), T=4.

## Phát hiện khi đọc code (ảnh hưởng thiết kế)
- Loader train là `ConcatDataset([weak_set, strong_set])` với `shuffle=True` → mỗi epoch có 100k mẫu; view weak và view strong của
  cùng một ảnh là **hai mẫu độc lập, nằm ở hai batch khác nhau**. Trong E4 **không có cặp weak/strong trong batch**, nên công thức
  `[KL(T_weak‖S_weak) + KL(T_strong‖S_strong)]/2` chỉ mang tính minh họa: E4 thực ra là KD từng mẫu, cùng view.
- Hệ quả: CV1 **phải** tạo thêm một view weak phụ của **đúng ảnh gốc** cho mỗi mẫu strong để lấy `T_weak`. Thiết kế chọn
  (giữ batch, thứ tự shuffle, batch size, augmentation của student không đổi):
  - Mẫu weak: giữ nguyên, teacher target = teacher(x_weak) như E4.
  - Mẫu strong: student vẫn thấy x_strong như E4; dataset trả thêm `x_weak_aux = weak_transform(cùng ảnh gốc)` (crop/flip **độc lập**
    với crop của view strong); KD target = teacher(x_weak_aux).
  - Mask scalar (median confidence), CE và BC dùng **teacher logits gốc của batch** (teacher(x) như E4) → chỉ có target KD của mẫu strong đổi.
  - Hệ quả không tránh được: việc rút thêm số ngẫu nhiên cho view phụ làm chuỗi augmentation khác E4 (thứ tự mẫu không đổi vì
    sampler dùng generator riêng). Chi phí: thêm một lần forward teacher cho ~50% batch.
- D0 dùng đúng cấu trúc cặp này: tensor augment đã có từ run `gap-groups-aug` (10k ảnh train cố định, mỗi ảnh 1 view weak + 1 view
  strong, crop độc lập, seed 12345), checkpoint E4 seed 2022 (cuối = chính; best = phụ).

## D0 — tiêu chí ghi trước (trên các mẫu strong, cùng S_strong, cùng T=4, T², reduction)
- Original: `KL(softmax(T_strong/4) ‖ softmax(S_strong/4))·T²`; Cross: `KL(softmax(T_weak/4) ‖ softmax(S_strong/4))·T²`.
- **C1 (xung đột):** tỷ lệ cos(g_CE, g_KD) < 0 của Cross ≤ 0.75 × Original **và** giảm tuyệt đối ≥ 10 điểm phần trăm.
- **C2 (độ lớn gradient):** mean và p99 của ‖g_KD‖ Cross / Original nằm trong [0.67, 1.5]; không NaN/Inf.
- **C3 (ghép cặp):** idx và nhãn của cặp weak/strong khớp 100%; teacher đúng trên view weak ≥ 95%.
- **C4 (ngữ nghĩa, cờ đỏ):** tỷ lệ view strong "mất ngữ nghĩa" (p_T_strong[y] < 0.01 **và** p_S_strong[y] < 0.01) ≤ 25%.
  Báo cáo thêm: argmax T_weak == argmax T_strong, JS(T_weak, T_strong), tỷ lệ T_weak đúng & T_strong sai.
- **PASS** khi C1–C4 đều đạt (đánh giá trên checkpoint cuối). Nếu không: dừng, báo cáo, **không train**.
- Lưu ý ghi trước: C1 gần như chắc đạt vì teacher-weak đúng ~98% trên ảnh train (teacher đã nhớ train) và CE luôn kéo về y;
  nên PASS chỉ là điều kiện cần, không phải bằng chứng CV1 sẽ tăng accuracy. Rủi ro: target T_weak gần one-hot (p_T[y]≈0.96
  trên view weak) → ít "dark knowledge" hơn, có thể giống label smoothing.

## CV1 — nếu D0 PASS
`CROSS_VIEW_KD=1` (mặc định 0). Test hồi quy: tắt mode → loss khớp bit-exact code hiện tại; bật mode nhưng cho x_weak_aux = x →
loss khớp bit-exact (chứng minh chỉ đổi target). 300 epoch, seed 2022, Kaggle T4, tài khoản 1. So sánh best/final/TB10 với
E4 (72.11/71.84/71.92) và F1b (72.60/72.28/72.43). +0.2 so với E4 chỉ là tín hiệu ban đầu. Không chạy CV2/seed khác/đổi hyperparameter.

## Kết quả D0 — **FAIL** (rớt C2), dừng, KHÔNG train CV1

Run: Kaggle `trananhkiet/streamlined-kd-cifar100-cv-d0` v1 (đọc `aug/e4_last.npz`, `aug/e4_best.npz` từ output của
`gap-groups-aug`), notebook `kaggle/experiments/cv_d0_cross_view.ipynb`. Chỉ tính toán trên logits có sẵn, ~30 giây.
10.000 cặp (weak, strong) của cùng ảnh train; cùng S_strong, T=4, T², reduction như nhau.

### Bảng D0 — checkpoint cuối E4 seed 2022 (best-epoch cho số gần như trùng)
| Chỉ số trên mẫu strong | Original (T_strong) | Cross (T_weak) |
|---|---:|---:|
| Tỷ lệ xung đột cos(g_CE, g_KD) < 0 | 54.67% | **3.87%** |
| cos trung bình [CI95] | +0.042 [+0.033,+0.050] | +0.742 [+0.736,+0.748] |
| KD trung bình (×T²) | 2.440 | 6.903 |
| ‖g_KD‖ trung bình / p99 | 0.355 / 1.093 | **0.807 / 2.349** |
| ‖g_CE‖ trung bình | 0.632 | 0.632 |
| Tỷ lệ ‖g_KD‖ Cross/Original (mean / p99) | — | **2.27 / 2.15** |

| Teacher / ngữ nghĩa | Giá trị |
|---|---:|
| Teacher đúng: weak / strong | 97.84% / 39.86% |
| p_T[y]: weak / strong | 0.926 / 0.359 |
| max prob teacher: weak / strong | 0.933 / 0.658 |
| T_weak đúng & T_strong sai | **58.36%** |
| T_weak sai & T_strong đúng | 0.38% |
| argmax T_weak == argmax T_strong | 39.80% |
| JS(T_weak, T_strong) trung bình | 0.381 |
| Student strong đúng; argmax S_strong == argmax T_weak | 54.09%; 54.25% |
| View strong "mất ngữ nghĩa" (p_T_strong[y] < 0.01 và p_S_strong[y] < 0.01) | 9.82% |
| Ghép cặp: nhãn weak/strong khớp | 100% (assert) |

Theo trường hợp teacher (Original → Cross): T_w đúng & T_s đúng (39.5%): xung đột 27.8% → 8.7%, ‖g_KD‖ 0.384 → 0.701;
T_w đúng & T_s sai (58.4%): xung đột 73.1% → 0.3%, ‖g_KD‖ 0.336 → 0.890; T_w sai & T_s đúng (0.4%): 5.3% → 23.7%;
cả hai sai (1.8%): 56.2% → 9.6%.

### Đánh giá tiêu chí (ghi trước)
- **C1 xung đột: ĐẠT** (54.67% → 3.87%; giảm 50.8 điểm, còn 7% mức cũ). Như đã dự đoán trước, gần như do cấu trúc: T_weak đúng 98%.
- **C2 độ lớn gradient: TRƯỢT** — ‖g_KD‖ tăng **2.27×** (mean), 2.15× (p99), ngoài khoảng [0.67, 1.5]. KD loss tăng 2.8×.
  Nguyên nhân: T_weak rất tự tin vào y (max prob 0.93) trong khi student trên view strong còn mơ hồ (đúng 54%), nên khoảng cách
  KL lớn hơn nhiều. Ở mẫu strong, ‖g_KD‖ Cross (0.81) vượt cả ‖g_CE‖ (0.63).
- **C3 ghép cặp: ĐẠT** (nhãn khớp 100%, teacher-weak đúng 97.84%).
- **C4 ngữ nghĩa: ĐẠT** (9.82% ≤ 25%), nhưng cần lưu ý: chỉ 39.8% cặp có teacher đồng ý argmax giữa hai view.
- **Quyết định: FAIL → không train CV1.**

### Vì sao C2 quan trọng (không phải trượt "kỹ thuật")
CV1 không chỉ đổi **hướng** target mà còn tăng **gấp ~2.3 lần lực KD** trên ~50% dữ liệu (các mẫu strong). Nếu train, kết quả
sẽ lẫn hai hiệu ứng: target tốt hơn và KD mạnh hơn. Các thí nghiệm trước cho thấy tăng khối lượng KD là có hại (J1 tăng đều
71.51; F1c 0.7 giảm 0.40 so với F1b), nên CV1 nguyên bản có rủi ro cao và khó diễn giải.

### Gợi ý (KHÔNG chạy; mỗi cái là CV2/hyperparameter mới, cần người dùng quyết định)
- Cross-view có chuẩn hóa độ lớn: nhân KD của mẫu strong với ~0.44 (=1/2.27) để ‖g_KD‖ khớp E4 (cần D0 lại để kiểm C2).
- Target trộn: α·p(T_weak) + (1−α)·p(T_strong), chọn α sao cho ‖g_KD‖ ≈ E4.
- Chỉ dùng T_weak khi T_strong sai (58% mẫu strong), giữ T_strong khi đúng.

## Code đã chuẩn bị (mặc định tắt, chưa dùng)
`CROSS_VIEW_KD` trong `streamlined_kd_cifar100.py`: `CrossViewCIFAR` trả (x_student, y, x_teacher_kd, index, is_strong); với mẫu strong,
teacher forward thêm trên view weak phụ của cùng ảnh; mask, CE và BC vẫn dùng teacher logits gốc; thống kê ghép cặp in mỗi epoch.
Test `kaggle/test_cross_view_kd.py` (CPU): tắt mode khớp bit-exact HEAD (loss và 3 bước train); bật mode với view phụ = x khớp
(sai lệch 1.2e-7 do forward teacher trên tập con); đổi view phụ thì cập nhật đổi thật (8.6e-3); ghép cặp đúng 20/20 ảnh; ConcatDataset
collate đúng. Tất cả PASS.
