# L1–L4: bốn biến thể loss mới trên nền F1b (R56→R20, seed 2022)

Ngày: 2026-11-27. Mọi run: EPOCHS=300, TEACHER_SOURCE=mdistiller, LR_MILESTONES=100,188,250, SAMPLE_WEIGHT_MODE=teacher_correct, SW_TRUST=0.6, SW_DISTRUST=0.5, SEED=2022.
Đối chứng cùng seed: E4 = 72.11 / 71.84 / 71.92; F1b = 72.60 / 72.28 / 72.43 (best / final / TB10). Nhiễu giữa các run ≈ 0.3–0.5, nên 1 seed chỉ để sàng lọc.

## Động cơ (từ các thí nghiệm trước)
- F1b chỉ hơn E4 ~0.3–0.5 (R56→R20) và không hơn ở R110→R32. J1 (trọng số hằng) và J2 (xáo trọng số) cho thấy một phần lợi ích không đến từ "độ tin cậy" của teacher.
- BC (loss Gram) gần như vô hiệu (~1% gradient KD), nên phần đáng chỉnh là CE/KD theo mẫu.
- D2 (chỉ NCKD khi teacher sai) và G1 (sửa logit) không rõ lợi → thử hướng khác: *tăng KD ở đâu*, *tin ai*, *CE có cần bù không*.

## Bốn biến thể (mặc định mọi env mới = F1b, bit-exact — xem `test_l1_l4_losses.py`)
| Id | Env | Công thức | Câu hỏi |
|---|---|---|---|
| L1 F1b-TCKD | `CORRECT_BOOST_PART=tckd` | KD_i = 0.5·KL_i + (w_i−0.5)·TCKD_i | Phần KD tăng thêm cho mẫu teacher đúng có lợi ở thành phần nhị phân target-vs-rest? |
| L2 F1b-NCKD | `CORRECT_BOOST_PART=nckd` | KD_i = 0.5·KL_i + (w_i−0.5)·(1−p_T[y])·NCKD_i | …hay ở "dark knowledge" (non-target)? |
| L3 F1b-Top2 | `TRUST_TOPK=2` | tin teacher nếu nhãn nằm trong top-2 của teacher | Teacher "gần đúng" (nhãn hạng 2) có đáng tin không? |
| L4 F1b-CE0.5 | `CE_WEIGHT_MODE=fixed` | CE_i luôn nhân 0.5 (không phải 1−w_i) | Giảm CE ở mẫu tin cậy có cần thiết, hay chỉ KD tăng là đủ? |

Ghi chú: KL = TCKD + (1−p_T[y])·NCKD (đẳng thức DKD, T²). L1 + L2 chia đôi đúng phần boost của F1b. L4 làm tổng trọng số CE+KD > 1 ở mẫu tin cậy (0.5+0.6).

## Chạy
| Id | Tài khoản | Kernel |
|---|---|---|
| L1 | 1 (trananhkiet) | `streamlined-kd-cifar100-l1-f1b-tckd-seed2022` |
| L2 | 1 (trananhkiet) | `streamlined-kd-cifar100-l2-f1b-nckd-seed2022` |
| L3 | 2 (kitit5512trn) | `streamlined-kd-cifar100-l3-f1b-top2-seed2022` |
| L4 | 2 (kitit5512trn) | `streamlined-kd-cifar100-l4-f1b-ce05-seed2022` |

Thời gian dự kiến ~4.9h/run. Notebook: `l{1..4}_*_seed2022.ipynb` (mỗi cái khác F1b đúng 1 dòng mặc định). Test: `python3 test_l1_l4_losses.py OLD NEW` (OLD = script ở commit e77acde).

## Tiêu chí đọc kết quả
Chỉ coi là có tín hiệu nếu best/TB10 vượt F1b ≥ ~0.4 *và* cùng chiều ở TB10; nếu không, ghi nhận "không tách khỏi nhiễu". Chọn cấu hình không dựa trên test set — bất kỳ biến thể thắng nào phải xác nhận thêm seed 2023/2024 trước khi kết luận.

## Kết quả
(chưa có — đang chạy)
