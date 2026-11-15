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

## Kết quả — đã xác minh từ log Kaggle (tài khoản 1, version 1), 2026-11-15
Config xác nhận trong `summary.json`: F1a `sw_trust=0.5, sw_distrust=0.4`; F1b `sw_trust=0.6, sw_distrust=0.5`; cả hai `seed=2022`, `sample_weight_mode=teacher_correct`, 300 epoch, milestones 100/188/250, teacher mdistiller **72.41 / 92.50**, CE/KD 0.5/0.5, mask scalar. `teacher_train_top1` TB 69.17% (cả hai). Runtime F1a 17.120 s (~4,76h), F1b 17.673 s (~4,91h).

| Seed 2022 (KD đúng, KD sai) | Best top-1 | Epoch cuối | TB 10 cuối | Best top-5 | Best loss | Train top-1 cuối |
|---|---:|---:|---:|---:|---:|---:|
| E4 (0.5, 0.5) | 72.11 | 71.84 | 71.92 | 92.86 | 1.0371 | 67.12 |
| **F1a** (0.5, **0.4**) | **72.02** (ep 285) | 71.75 | 71.80 | 93.10 | 1.0317 | 68.29 |
| **F1b** (**0.6**, 0.5) | **72.60** (ep 265) | 72.28 | 72.43 | 92.85 | 1.0385 | 66.91 |
| F1 (0.6, 0.4) | 72.45 | 72.34 | 72.27 | 92.98 | 1.0267 | 67.52 |

Đường cong F1a: ep100 44.02 → ep101 62.74 → ep188 64.66 → ep189 69.62 → ep250 70.53 → ep251 71.56 → ep260 71.49 → ep300 71.75.
Đường cong F1b: ep100 44.09 → ep101 63.32 → ep188 65.25 → ep189 70.32 → ep250 71.09 → ep251 72.32 → ep260 72.18 → ep300 72.28. Top-5 epoch F1b: 265 (72.60), 298 (72.60), 292 (72.58), 294 (72.54), 299 (72.53) — phần cuối ổn định ở 72.5+.

### Bảng hiệu ứng (best / epoch cuối / TB 10 cuối)
| Hiệu ứng | Ở hàng KD-sai thấp / cao | Giá trị | Trung bình hai cách đo |
|---|---|---|---|
| **Giảm KD ở mẫu SAI** (0.5 → 0.4) | F1a − E4 (KD đúng = 0.5) | −0.09 / −0.09 / −0.12 | **−0.12 / −0.02 / −0.14 ≈ 0** |
| | F1 − F1b (KD đúng = 0.6) | −0.15 / +0.06 / −0.16 | |
| **Tăng KD ở mẫu ĐÚNG** (0.5 → 0.6) | F1b − E4 (KD sai = 0.5) | **+0.49 / +0.44 / +0.51** | **+0.46 / +0.52 / +0.49** |
| | F1 − F1a (KD sai = 0.4) | **+0.43 / +0.59 / +0.47** | |

### KẾT LUẬN SƠ BỘ (một seed 2022; best top-1; ngưỡng 0.20)
- **Toàn bộ lợi ích của F1 so với E4 đến từ việc TĂNG KD trên mẫu teacher ĐÚNG (0.5 → 0.6)**, không phải từ việc giảm KD trên mẫu sai: hiệu ứng nhánh đúng ≈ **+0.46 đến +0.52** và **nhất quán ở cả hai hàng** (KD sai = 0.5 và 0.4, cả ba chỉ số); hiệu ứng nhánh sai ≈ **0** (−0.1 đến 0, hơi âm).
- **F1b (0.6, 0.5) ≥ F1:** best **72.60** (cao hơn F1 +0.15, **cao hơn paper SKD 72.50**), TB 10 cuối 72.43 (+0.16), epoch cuối 72.28 (−0.06). Tức tương đương F1 (|Δ| < 0.20) và đơn giản hơn: chỉ đổi một phía. F1b là **kết quả best top-1 cao nhất của dự án** (72.60) — nhưng chỉ một seed.
- **F1a ≈ E4** (−0.09): chỉ giảm KD khi teacher sai không giúp (train top-1 cao hơn 68.29 mà test không tăng).
- Hệ quả: giả thuyết **A được tinh chỉnh**: thông tin "teacher đúng" có ích theo hướng **"tin teacher nhiều hơn khi nó đúng"**, không phải "tin teacher ít hơn khi nó sai". Điều này khớp: D2 (chỉnh nhánh sai) không giúp; J2 (cùng 0.6 nhưng gán ngẫu nhiên, nên đa số 0.6 rơi vào mẫu sai/khó) ≈ E4; F1 ≈ F1b.
  → Hướng "xử lý mẫu teacher sai" (D2, Error-Aware DKD phía mẫu sai) **không phải đòn bẩy**; hướng đáng khai thác hơn là **mẫu teacher đúng**.

**Giới hạn:** (1) một seed; nhiễu cùng cấu hình tới ~0.4; best lạc quan ~0.2–0.4 (nhưng epoch cuối và TB 10 cuối đồng hướng: F1b TB 10 cuối 72.43 cũng cao nhất). Dấu của hiệu ứng nhánh đúng nhất quán (6/6 dương trong bảng trên), độ lớn ~0.5 vượt nhiễu thông thường. (2) KD trung bình đổi cùng lúc (F1a 0.469, F1b 0.569): đã bổ sung J1 (hằng 0.5384 → −0.60 so với E4) mâu thuẫn với "tăng KD chung có lợi" nên J1 hoặc là một run xấu bất thường (một seed) hoặc phản ứng phi tuyến; **chưa giải thích được**. (3) chưa kiểm tra seed 2021.
Quota tài khoản 1: ~9,7h/30h đã dùng; còn ~20,3h; refresh tiếp theo 2026-11-21 00:00 UTC.

**Gợi ý bước sau (chỉ nêu, chưa chạy):** (a) **F1b seed 2021** (đối chứng F1 seed 2021 = 72.32 / 72.13 / 72.14, E4 = 72.10 / 71.70 / 71.73) để xác nhận F1b ≥ F1 và vượt 72.5; (b) **F1c (0.7, 0.5)** — tăng thêm KD ở mẫu đúng, xem hiệu ứng có tăng theo liều không (F2 0.7/0.3 theo đúng/sai chỉ 72.23 ở seed 2021 nhưng KD sai ở đó là 0.3); (c) làm rõ J1 bằng một run hằng 0.5 vs 0.538 ở seed khác nếu cần.

## Trạng thái
**COMPLETE** (2026-11-15). Scheduler đã tắt; không còn run nào đang chạy.
