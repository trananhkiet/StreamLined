# E — Lịch learning rate (nền run B) — KẾ HOẠCH

Ghi chú: kế hoạch do người dùng duyệt ngày 2026-11-03 (bỏ phương án cosine + warmup). **D (DOT) tạm dừng**: code đã sẵn sàng ở commit `3c9fea4`
(`OPTIM_MODE=dot`, notebook `kaggle/experiments/d_dot.ipynb`), chạy lại sau nếu người dùng yêu cầu.

## Bối cảnh
A1, A2, A3, B (Logit Standardization), C (DKD) đều chưa vượt run B (70.26% best / 70.04% final, 1 seed). Câu hỏi tiếp theo: lịch LR có phải điểm nghẽn không?
Quan sát từ log: ở LR 0.1 (epoch 1–150) test top-1 dao động ~38–50%, rồi nhảy lên ~62–64% sau lần giảm LR đầu tiên. Đây là hình dạng bình thường của step LR
(LR cao giúp khám phá, giảm LR giúp hội tụ), **chưa chứng minh** LR 0.1 quá cao; lịch 150/180/210 cũng là lịch chuẩn của KD trên CIFAR-100. Vì vậy E là phép thử, chưa phải kỳ vọng chắc chắn.

## Nền cố định (giống run B)
CE/KD/BC 0.5/0.5/0.5 (`LAMBDA_BC` mặc định = `kd_weight`), KD mask scalar, T=4, tik 0.1, SGD momentum 0.9, weight decay 5e-4, batch 64, 240 epoch,
weak + strong augmentation, teacher ResNet56 71.59%, dataset `fedesoriano/cifar100` v1, GPU T4, Docker image như run B. **Mỗi run chỉ đổi một yếu tố.**

## Các run
| Run | Thay đổi so với run B | Câu hỏi |
|---|---|---|
| **E1** | Step LR 0.1, giảm ×0.1 ở epoch **80 / 150 / 200** (thay vì 150 / 180 / 210) | Giữ LR cao 150 epoch có quá lâu không? |
| **E2** | LR khởi đầu **0.05**, giữ lịch 150 / 180 / 210 | Bản thân LR 0.1 có quá cao không? (0.05 là LR chuẩn của CRD/DKD/mdistiller cho student ResNet20) |
| **E3** | Chạy lại **ứng viên tốt nhất (nếu đạt ngưỡng) và run B** với **seed 2022** | Chênh lệch có vượt nhiễu seed không? |

Phương án đã cân nhắc và bỏ: cosine annealing + warmup 5 epoch (người dùng quyết định bỏ).

## Chỉ số và luật quyết định
- Chỉ số chính: **test top-1 ở epoch 240**. Phụ: best top-1 (có selection bias vì chọn trên test set) và **trung bình 10 epoch cuối** (ổn định hơn, tính từ log).
- Một ứng viên chỉ được coi là "tốt hơn B" nếu epoch-240 top-1 **cao hơn B ≥ 0,5 điểm** (lớn hơn nhiễu ~0,3 của 1 seed) **và** giữ được ở E3 với seed 2022.
- Nếu không ứng viên nào đạt ngưỡng: E3 vẫn chạy **run B seed 2022** để đo nhiễu seed thực tế (chỉ 1 run), rồi kết luận "lịch LR không phải điểm nghẽn".

```mermaid
flowchart LR
    E1[E1: step 80/150/200] --> E2[E2: LR 0.05, lịch cũ]
    E2 --> Q{Có run nào hơn B ≥ 0,5 điểm ở epoch 240?}
    Q -- Có --> E3a[E3: seed 2022 cho ứng viên + run B]
    Q -- Không --> E3b[E3: chỉ run B seed 2022, đo nhiễu]
```

## Thay đổi code dự kiến (mặc định giữ nguyên bit như run B)
Biến môi trường mới: `LEARNING_RATE` (mặc định `0.1`), `LR_MILESTONES` (mặc định `150,180,210`), `SEED` (mặc định `2021`). Scheduler vẫn cập nhật theo epoch (MultiStepLR, gamma 0.1).
Test CPU trước khi submit: (1) mặc định cho trọng số bit-exact so với commit trước; (2) in bảng LR từng epoch của E1/E2 và so với bảng tính tay
(E1: epoch 1–80 = 0.1, 81–150 = 0.01, 151–200 = 0.001, 201–240 = 0.0001; E2: 0.05 / 0.005 / 0.0005 / 0.00005 tại các mốc 150/180/210).

## Tài khoản Kaggle và ngân sách
- Chạy trên **tài khoản 2** (`kitit5512trn`): CLI dùng `export KAGGLE_API_TOKEN="$KAGGLE_KEY2"`; Kaggle MCP trong session đã chuyển sang tài khoản 2.
  Notebook slug dạng `kitit5512trn/streamlined-kd-cifar100-e<n>-<tên>`. Kết quả cũ của tài khoản 1 (`trananhkiet`) đọc bằng CLI với `KAGGLE_KEY`.
- Đọc kết quả bằng `kaggle kernels logs <slug>` (lọc trong shell) hoặc MCP `list_notebook_session_output`; host `www.kaggleusercontent.com` vẫn bị chặn nên chưa tải được `history.csv`.
- Lần submit đầu trên tài khoản 2 phải xác minh: dataset attach đúng và Docker image ghim của run B dùng được. Nếu image bị từ chối: dừng, ghi blocker, **không tự đổi image**.
- Mỗi run ~3,9h GPU. E1 + E2 ≈ 8h, E3 ≈ 4–8h → tổng ≈ 12–16h, nằm trong 30h/tuần của tài khoản 2 (refresh 2026-11-07 00:00 UTC).

## Trạng thái
Kế hoạch đã duyệt; chưa implement, chưa submit. Bước tiếp theo: thêm `LEARNING_RATE`/`LR_MILESTONES`/`SEED` + test, submit E1 trên tài khoản 2.
