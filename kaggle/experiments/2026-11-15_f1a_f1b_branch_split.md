# F1a / F1b — tách hai phía của F1: nhánh mẫu teacher SAI và nhánh mẫu teacher ĐÚNG (một seed 2022)

Người dùng yêu cầu (2026-11-15): chạy **song song F1a và F1b**, rồi lập scheduler kiểm tra lại sau ~6 giờ. Nằm trong hướng nghiên cứu
[2026-11-12_research_direction_f1_mechanism.md](2026-11-12_research_direction_f1_mechanism.md); lý do: D2 ([2026-11-12_d2_nckd_on_wrong.md](2026-11-12_d2_nckd_on_wrong.md)) kém F1 −0.40 nhưng chưa biết lợi ích của F1 nằm ở nhánh nào. Tài khoản Kaggle 1.

## Thiết kế (không cần code mới: dùng `SAMPLE_WEIGHT_MODE=teacher_correct` với hai hệ số khác nhau)
F1 đổi hai thứ cùng lúc so với E4 (KD 0.5 cho mọi mẫu): tăng KD ở mẫu teacher đúng (0.5→0.6) và giảm KD ở mẫu sai (0.5→0.4). F1a/F1b mỗi cái chỉ đổi MỘT phía:

| Run | KD khi teacher đúng | KD khi teacher sai | (CE đúng / CE sai) | `SW_TRUST` / `SW_DISTRUST` | Đổi so với E4 | KD trung bình* |
|---|---:|---:|---|---|---|---:|
| E4 (đối chứng) | 0.5 | 0.5 | 0.5 / 0.5 | — | — | 0.500 |
| **F1a** | 0.5 | **0.4** | 0.5 / 0.6 | 0.5 / 0.4 | chỉ **giảm KD ở mẫu sai** | 0.469 |
| **F1b** | **0.6** | 0.5 | 0.4 / 0.5 | 0.6 / 0.5 | chỉ **tăng KD ở mẫu đúng** | 0.569 |
| F1 (đối chứng) | 0.6 | 0.4 | 0.4 / 0.6 | 0.6 / 0.4 | cả hai | 0.538 |
*Trọng số KD trung bình khi teacher đúng ~69.2% mẫu train.

Thành một bảng 2×2 (nhánh sai × nhánh đúng): E4 (0.5, 0.5) · F1a (0.5, 0.4) · F1b (0.6, 0.5) · F1 (0.6, 0.4) với cặp = (KD đúng, KD sai).
Hiệu ứng nhánh SAI (giảm KD ở mẫu sai): F1a − E4 và F1 − F1b. Hiệu ứng nhánh ĐÚNG (tăng KD ở mẫu đúng): F1b − E4 và F1 − F1a.

## Cấu hình
Giống F1 seed 2022: teacher mdistiller 72.41, 300 epoch, LR 0.1 ×0.1 ở 100/188/250, SGD momentum 0.9 wd 5e-4 batch 64, T=4, BC 0.5, mask scalar, **seed 2022**. Hai run chạy **song song** (~4.7–5.4h GPU mỗi run).
Notebook (sinh bằng `make_variant_notebook.py`; diff với F1 cùng bản build đúng một dòng mặc định):
- `kaggle/experiments/f1a_kd05_correct_kd04_wrong_seed2022.ipynb` — slug `streamlined-kd-cifar100-f1a-seed2022`
- `kaggle/experiments/f1b_kd06_correct_kd05_wrong_seed2022.ipynb` — slug `streamlined-kd-cifar100-f1b-seed2022`
Kiểm tra CPU: loss CE/KD của cả hai khớp tính tay với trọng số tương ứng; gradient hữu hạn; chế độ mặc định không đổi.
Quota tài khoản 1 trước run: 0/30h đã dùng (đã refresh; lần refresh tiếp 2026-11-21 00:00 UTC).

## Đối chứng (cùng seed 2022; best / epoch cuối / TB 10 cuối) và quy ước
QUY ƯỚC: best top-1 (ghi thêm epoch cuối và TB 10 cuối). F1 = 72.45 / 72.34 / 72.27; E4 = 72.11 / 71.84 / 71.92; J2 (xáo trộn) = 72.39 / 72.02 / 72.14; J1 (hằng 0.5384) = 71.51 / 71.24 / 71.24; D2 (NCKD-only trên mẫu sai) = 72.05 / 71.70 / 71.73. Trung bình hai seed: F1 72.39 / 72.24 / 72.21; E4 72.11 / 71.77 / 71.83.

## Cách đọc kết quả (ngưỡng 0.20 chỉ là sàng lọc)
- **F1a ≈ F1 (và F1a > E4):** lợi ích nằm ở nhánh mẫu SAI → hướng "xử lý mẫu teacher sai" có ý nghĩa.
- **F1b ≈ F1 (và F1b > E4):** lợi ích nằm ở nhánh mẫu ĐÚNG → các hướng chỉ sửa mẫu sai (như D2) không thể giúp.
- Cả hai ≈ E4 hoặc cả hai gần F1: lợi ích cần cả hai phía cùng lúc hoặc trong nhiễu.
**Giới hạn:** một seed; nhiễu cùng cấu hình tới ~0.4; best lạc quan ~0.2–0.4. **Trọng số KD trung bình thay đổi cùng lúc** (F1a 0.469 thấp hơn E4, F1b 0.569 cao hơn; J1 cho thấy tăng KD đồng đều lên 0.538 làm tệ đi −0.60), nên kết quả F1a/F1b có thể trộn lẫn hiệu ứng "mức KD trung bình" với hiệu ứng "nhánh". Kết luận chỉ SƠ BỘ.

## Trạng thái
Notebook sẵn sàng (2026-11-15). Xem HANDOFF cho trạng thái submit.
