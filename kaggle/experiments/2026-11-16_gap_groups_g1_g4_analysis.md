# Phân tích nhóm G1–G4 (teacher/student đúng-sai): E4 vs F1b, seed 2022 — KẾT QUẢ

Ngày: 2026-11-16. Chỉ inference trên checkpoint đã có; không train, không scheduler. Notebook: `kaggle/experiments/gap_groups_g1_g4_analysis.ipynb`,
Kaggle `trananhkiet/streamlined-kd-cifar100-gap-groups-g1-g4` **version 2** (v1 lỗi: `extract` khởi tạo lại CIFAR100 cho từng mẫu, chạy ~30 phút vô ích; đã sửa, commit `ed88745`+fix). Run v2 mất ~2 phút. Quota tài khoản 1: 20,31h/30h đã dùng (còn ~9,7h).

## Cách đo
- Checkpoint: `last_checkpoint.pt` (epoch 300, **cùng epoch** cho hai run) là phân tích chính; `resnet20_skd_best.pth` là phụ. Teacher mdistiller dùng chung (logit teacher của hai run khớp nhau).
- Dữ liệu phân tích: 10.000 ảnh **train** cố định (rng seed 0), transform không augment, **một view dùng chung**. Test chỉ để mô tả/kiểm tra, không dùng để chọn công thức.
- **Kiểm tra toàn vẹn (test) PASS, tái tạo đúng số đã ghi:** teacher 72.41; E4 last 71.84 / best 72.11; F1b last 72.28 / best 72.60.
- Gradient: dạng đóng theo logits student, đã so với autograd (sai số ≤1e-10): g_CE = softmax(s)−onehot(y); g_KD = T(softmax(s/T)−softmax(t/T)), T=4. Scalar mask ~0.5 và BC không nằm trong phép đo (cos không phụ thuộc scale).

## Bảng 1 — Train subset (10k), checkpoint cuối. Teacher đúng 98.33% (đã nhớ train)
| Run | Nhóm | n | % | CE | KD | p_T[y] | p_S[y] | cos(g_CE,g_KD) [CI95] | cos<0 % | ‖g_CE‖ | ‖g_KD‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E4 | G1 | 8524 | 85.24 | 0.132 | 2.219 | 0.972 | 0.896 | +0.403 [+0.390,+0.416] | 25.3 | 0.125 | 0.521 |
| E4 | G2 | 1309 | 13.09 | 2.312 | 3.503 | 0.864 | 0.172 | **+0.776** [+0.768,+0.783] | 0.0 | 1.028 | 0.584 |
| E4 | G3 | 33 | 0.33 | 0.462 | 2.365 | 0.329 | 0.660 | −0.210 [−0.315,−0.104] | 75.8 | 0.421 | 0.440 |
| E4 | G4 | 134 | 1.34 | 3.542 | 2.405 | 0.247 | 0.090 | +0.391 [+0.337,+0.444] | 10.4 | 1.140 | 0.397 |
| F1b | G1 | 8427 | 84.27 | 0.127 | 2.091 | 0.972 | 0.900 | +0.384 [+0.370,+0.397] | 26.5 | 0.121 | 0.503 |
| F1b | G2 | 1406 | 14.06 | 2.412 | 3.522 | 0.866 | 0.160 | **+0.786** [+0.779,+0.794] | 0.0 | 1.047 | 0.595 |
| F1b | G3 | 28 | 0.28 | 0.433 | 2.618 | 0.343 | 0.692 | −0.234 [−0.373,−0.083] | 78.6 | 0.369 | 0.446 |
| F1b | G4 | 139 | 1.39 | 3.400 | 2.296 | 0.247 | 0.101 | +0.379 [+0.319,+0.434] | 14.4 | 1.132 | 0.393 |

Student top-1 trên subset: E4 85.57, F1b 84.55. Chuyển nhóm E4→F1b: G1→G1 7970, G1→G2 554, G2→G1 457, G2→G2 852, G3→G3 20, G3→G4 13, G4→G3 8, G4→G4 126. Checkpoint best-epoch cho kết quả tương tự (G2 cos +0.775/+0.787; xem log notebook).

## Bảng 2 — Test (CHỈ MÔ TẢ), checkpoint cuối. Teacher đúng 72.41%
| Run | Nhóm | n | % | CE | KD | p_T[y] | p_S[y] | cos [CI95] | cos<0 % | ‖g_CE‖ | ‖g_KD‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E4 | G1 | 6388 | 63.88 | 0.113 | 2.210 | 0.939 | 0.911 | +0.267 [+0.251,+0.283] | 34.6 | 0.108 | 0.517 |
| E4 | G2 | 853 | 8.53 | 2.200 | 3.206 | 0.760 | 0.184 | +0.685 [+0.672,+0.697] | 0.1 | 1.028 | 0.542 |
| E4 | G3 | 796 | 7.96 | 0.434 | 2.972 | 0.163 | 0.686 | **−0.390** [−0.411,−0.369] | 87.8 | 0.372 | 0.504 |
| E4 | G4 | 1963 | 19.63 | 3.828 | 2.714 | 0.071 | 0.081 | +0.049 [+0.032,+0.065] | 41.6 | 1.137 | 0.425 |
| F1b | G1 | 6421 | 64.21 | 0.110 | 2.133 | 0.938 | 0.913 | +0.241 [+0.226,+0.256] | 35.9 | 0.106 | 0.509 |
| F1b | G2 | 820 | 8.20 | 2.135 | 3.049 | 0.758 | 0.185 | +0.691 [+0.678,+0.705] | 0.0 | 1.029 | 0.534 |
| F1b | G3 | 807 | 8.07 | 0.448 | 2.973 | 0.154 | 0.677 | −0.380 [−0.401,−0.359] | 89.2 | 0.380 | 0.495 |
| F1b | G4 | 1952 | 19.52 | 3.889 | 2.655 | 0.074 | 0.085 | +0.060 [+0.044,+0.076] | 39.5 | 1.139 | 0.423 |

Chuyển nhóm test E4→F1b: G1→G1 6009, G1→G2 379, G2→G1 412, G2→G2 441, G3→G3 513, G3→G4 283, G4→G3 294, G4→G4 1669.
Net: G2→G1 +33 (412−379), G4→G3 +11 (294−283); tổng +44 mẫu đúng (+0.44 điểm). Churn hai chiều ~1370 mẫu; McNemar χ²=(706−662)²/1368=1.41, **p≈0.23 (không có ý nghĩa thống kê trên một cặp seed)**.

## Trả lời các câu hỏi
1. **G2 có CE/KD hỗ trợ nhau nhiều hơn các nhóm khác không?** Có: cos +0.78 (train) / +0.69 (test), gần như 0% mẫu xung đột, CI không giao với G1 (+0.40/+0.27), G3, G4. **Nhưng phần lớn là hệ quả cấu trúc**: teacher đúng nên phân phối KD dồn khối lượng vào y, cùng hướng với CE; không cần "học" thêm điều gì. Ngoài ra ở G2 ‖g_CE‖ (1.03) lớn gấp ~1.8 lần ‖g_KD‖ (0.58): CE đã chi phối, KD là phần bổ trợ.
2. **G3 có xung đột không?** Có, rõ nhất: cos −0.21 (train, n nhỏ) và **−0.39 (test, n≈800)**, 76–89% mẫu có cos<0. Đây là nhóm KD kéo ngược CE (teacher p_T[y]≈0.16). Hạn chế: trên train subset G3 chỉ 28–33 mẫu vì teacher đúng 98%.
3. **F1b có đổi gradient/confidence theo nhóm so với E4 không?** Rất ít. cos nhóm gần như giữ nguyên (±0.03); p_S[y] ở G2 không tăng (train 0.172→0.160, test 0.184→0.185); KD trung bình giảm ở G1/G2 (student bám teacher hơn: 2.219→2.091, 3.203→3.049 test). Trên train, F1b có **nhiều** G2 hơn (1406 vs 1309; train acc 84.55 vs 85.57): F1b khớp train kém hơn, nhất quán với CE bị giảm trọng số (0.4 thay vì 0.5). Lợi ích test (+33 ở G2→G1) không đi cùng việc G2 được tin cậy cao hơn.
4. **Dữ liệu có ủng hộ tăng KD riêng cho G2 thay vì toàn bộ mẫu teacher đúng không?** **Chưa đủ dữ liệu để ủng hộ, và có dấu hiệu bất lợi.**
   - Khối lượng: G2 chỉ 13–14% mẫu train (8.5% test), G1 là 84%. K1 (+0.1 chỉ ở G2) chỉ thêm ~1/7 lượng KD phụ so với F1b (+0.1 ở mọi mẫu teacher đúng). Thí nghiệm liều trước đó cho thấy khối lượng KD quan trọng (F1c 0.7 làm giảm 0.40; J1 tăng đều làm giảm), nên K1 khó tái tạo mức tăng của F1b trừ khi cơ chế thật sự nằm ở G2.
   - Cơ chế gradient không cho biết nhóm nào tạo ra lợi ích; ở G1 KD thậm chí chiếm ưu thế (‖g_KD‖≈4×‖g_CE‖).
   - Lợi ích test +44 mẫu nằm trong nhiễu (p≈0.23), khoảng 3/4 đến từ G2→G1.

## Kết luận về giả thuyết G2: **CHƯA ĐỦ DỮ LIỆU** (không bác bỏ, không được ủng hộ)
- Bằng chứng ủng hộ yếu: G2 có CE–KD đồng hướng nhất; phần lớn lợi ích ròng của F1b trên test nằm ở G2→G1 (+33/+44).
- Bằng chứng chống: confidence G2 không tăng; G2 trên train còn tăng; hiệu ứng ròng không có ý nghĩa; một cặp seed.
- Giới hạn: (i) một seed (2022); (ii) checkpoint cuối, không phải quỹ đạo huấn luyện; (iii) phân tích trên ảnh không augment, trong khi huấn luyện dùng 2 view, view mạnh làm teacher đúng chỉ ~69% (nhóm G3/G4 ở lúc train lớn hơn nhiều so với train subset sạch này); (iv) cos tại logits, không phải trong không gian tham số; (v) train subset là dữ liệu student đã thấy.

## Về K1 (w = 0.5 + 0.1·1[teacher đúng]·1[student sai])
- **Khuyến nghị: chưa train K1.** Lý do ở mục 4 (lượng KD phụ quá nhỏ, G2 không phải nơi F1b dịch chuyển rõ).
- Ưu: gọn, tăng KD ở nơi CE–KD đồng hướng, có thể đo được. Hạn chế: "student sai" đổi theo epoch nên trọng số động, nhiễu vì student sai ở view augment; không gradient qua phép chọn (argmax). Rủi ro: gần như WSL/ADA-KD/TGeo-KD (trọng số theo S–T–GT) nên tính mới thấp; giả định tăng KD ở chỗ student kém đã có (WSL).
- Nếu muốn kiểm tra thêm trước khi tốn quota train (đều là inference, vài phút): (a) lặp lại đúng phân tích này cho cặp **seed 2021** (E4 s2021 và F1b s2021 đã có output); (b) đo trên view augment 2-view giống lúc train; (c) tính nhóm theo train-augmented để có G3/G4 đủ lớn. Chưa chạy, chờ người dùng quyết định.

---

# PHỤ LỤC — Phân tích trên view AUGMENT (phân bố lúc train)

Notebook `kaggle/experiments/gap_groups_aug_g1_g4.ipynb` (Kaggle `trananhkiet/streamlined-kd-cifar100-gap-groups-aug` v1, ~70 giây). Cùng 10k ảnh train cố định, mỗi ảnh có **1 view weak** (crop+flip) và **1 view strong** (crop+flip+RandAugment), sinh **một lần** với seed 12345; teacher và cả 4 student chạy trên đúng cùng tensor → so E4/F1b sạch. Tổng 20.000 mẫu. Mọi phép đo như trên.

**Kiểm tra khớp với lúc train:** teacher đúng **68.85%** trên 20k mẫu (weak 97.84%, strong 39.86%), khớp mức ~69.2% ghi trong log train. Nghĩa là phân tích này đại diện cho tín hiệu mà student thực sự nhận.

## Bảng 3 — Gộp hai view (n=20.000), checkpoint cuối. Student: E4 69.17, F1b 68.06
| Run | Nhóm | n | % | CE | KD | p_T[y] | p_S[y] | cos [CI95] | cos<0 % | ‖g_CE‖ | ‖g_KD‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E4 | G1 | 11787 | 58.94 | 0.182 | 1.901 | 0.930 | 0.863 | +0.349 [+0.339,+0.359] | 27.3 | 0.160 | 0.430 |
| E4 | G2 | 1983 | 9.91 | 2.226 | 2.650 | 0.776 | 0.171 | **+0.700** [+0.693,+0.708] | 0.0 | 0.998 | 0.437 |
| E4 | G3 | 2047 | 10.23 | 0.679 | 2.842 | 0.094 | 0.575 | **−0.292** [−0.303,−0.282] | 88.2 | 0.464 | 0.369 |
| E4 | G4 | 4183 | 20.91 | 3.700 | 2.610 | 0.036 | 0.065 | **−0.034** [−0.041,−0.027] | 62.7 | 1.035 | 0.320 |
| F1b | G1 | 11697 | 58.48 | 0.177 | 1.826 | 0.931 | 0.867 | +0.338 [+0.328,+0.349] | 28.2 | 0.156 | 0.421 |
| F1b | G2 | 2073 | 10.37 | 2.276 | 2.617 | 0.777 | 0.166 | +0.707 [+0.700,+0.714] | 0.0 | 1.012 | 0.440 |
| F1b | G3 | 1915 | 9.57 | 0.691 | 2.818 | 0.093 | 0.574 | −0.299 [−0.310,−0.288] | 89.4 | 0.464 | 0.367 |
| F1b | G4 | 4315 | 21.57 | 3.679 | 2.551 | 0.038 | 0.068 | −0.039 [−0.046,−0.032] | 63.2 | 1.032 | 0.317 |

Chuyển nhóm E4→F1b (gộp): G1→G1 10962, G1→G2 825, G2→G1 735, G2→G2 1248, G3→G3 1412, G3→G4 635, G4→G3 503, G4→G4 3680.

## Bảng 4 — Theo view (checkpoint cuối)
| View | Teacher đúng | Student E4 / F1b | Nhóm | % (E4) | cos (E4) | cos<0 % (E4) | cos (F1b) |
|---|---|---|---|---|---|---|---|
| weak | 97.84% | 84.25 / 83.74 | G1 | 83.69 | +0.388 | 25.3 | +0.381 |
| weak | | | G2 | 14.15 | +0.742 | 0.0 | +0.749 |
| weak | | | G3 | 0.56 | −0.237 | 73.2 | −0.165 |
| weak | | | G4 | 1.60 | +0.358 | 11.2 | +0.324 |
| strong | 39.86% | 54.09 / 52.38 | G1 | 34.18 | +0.252 | 32.1 | +0.232 |
| strong | | | G2 | 5.68 | +0.596 | 0.0 | +0.609 |
| strong | | | G3 | 19.91 | **−0.293** | 88.6 | −0.302 |
| strong | | | G4 | 40.23 | **−0.050** | 64.8 | −0.054 |

Checkpoint best-epoch cho cùng bức tranh (gộp: G2 +0.701/+0.709, G3 −0.279/−0.298, G4 −0.038/−0.039; student 69.15/67.80).

## Cái gì đổi so với phân tích ảnh sạch
1. **Nhóm teacher-sai không còn là "hiếm".** G3+G4 = **31%** mẫu (sạch: 1.7%); riêng view strong 60%. Ở đây **KD ngược hướng CE**: G3 cos −0.29 (88% mẫu xung đột); **G4 cos ≈ −0.03 (63% xung đột)**, tức KD gần như vuông góc/ngược CE ở chỗ cả hai cùng sai (trên ảnh sạch G4 là +0.39 chỉ vì n nhỏ). Đây là chỗ KD không giúp hướng đúng nhãn.
2. **G2 nhỏ hơn nữa:** 9.9% (strong 5.7%). Cos vẫn cao (+0.70, 0% xung đột) nhưng đúng như trước là hệ quả cấu trúc.
3. **F1b khớp train kém hơn E4** (68.06 vs 69.17, −1.1 điểm) và mất mẫu đúng ở **cả hai** phía: vùng teacher đúng (G1 −90, G2 tăng +90) và vùng teacher sai (G3 −132), dù trọng số vùng teacher sai không đổi (cùng 0.5/0.5). Tức F1b là kiểu điều chuẩn (fit train ít hơn) chứ không phải học thêm ở G2; test +0.44 đi cùng train −1.1.
4. cos và confidence theo nhóm gần như **không đổi** giữa E4 và F1b (|Δcos| ≤ 0.02 ở gộp); p_S[y] ở G2 không tăng (0.171→0.166).

## Hệ quả cho giả thuyết và K1 (sau khi có số augment)
- **G2: vẫn "chưa đủ dữ liệu"**, không có thêm bằng chứng ủng hộ; và G2 chỉ ~10% mẫu → K1 (+0.1 chỉ ở G2) thêm khoảng **14%** lượng KD phụ của F1b (9.91/68.85 ≈ 0.14). **Vẫn khuyến nghị chưa train K1.**
- **Điểm mới đáng theo dõi:** ở vùng teacher sai (31% tín hiệu lúc train) KD xung đột CE rõ (G3) hoặc vô hướng (G4). Giả thuyết "giảm KD ở vùng teacher sai" đúng cơ chế nhưng đã thử: F1a (KD 0.4 ở vùng sai) ≈ E4, D2 (bỏ TCKD) làm tệ hơn, nên thay đổi nhẹ không đổi kết quả và thay đổi mạnh có hại. Gradient cos chỉ là bằng chứng cơ chế, không thay thế thí nghiệm.
- **Không dùng test để chọn công thức** trong toàn bộ phụ lục này; các con số trên là train-augmented.

## Giới hạn bổ sung
Một seed (2022); một lần bốc augment (seed 12345, mỗi ảnh một view mỗi loại, nên CI chỉ phản ánh nhiễu lấy mẫu mẫu, không phản ánh nhiễu augment); student đã thấy các ảnh gốc này; phân tích ở checkpoint cuối, không theo quỹ đạo. Quota tài khoản 1 sau run: 22,34h/30h đã dùng (còn ~7,7h).
