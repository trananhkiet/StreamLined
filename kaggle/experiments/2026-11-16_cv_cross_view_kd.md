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

## Kết quả
(điền sau)
