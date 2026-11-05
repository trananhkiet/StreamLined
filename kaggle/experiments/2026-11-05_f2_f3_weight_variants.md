# F2 / F3 — biến thể trọng số CE/KD theo teacher (tiếp nối F1)

Người dùng chỉ định (2026-11-05), sau F1 (+0.43 so với E4, sát ngưỡng +0.5): thử tiếp 2 biến thể. Chạy trên **tài khoản Kaggle 3**
(`KAGGLE_USERNAME3` / `KAGGLE_KEY3`, quota mới 30h, refresh 2026-11-07 00:00 UTC).

## Đối chứng
- **F1** (seed 2021): hard 0.6/0.4 — epoch cuối 72.13, TB 10 cuối 72.14, best 72.32.
- **E4** (seed 2021): không trọng số theo mẫu — 71.70 / 71.73 / 72.10.
- Nền chung: teacher mdistiller 72.41%, 300 epoch, milestones 100/188/250, seed 2021, CE/KD/BC nền 0.5, T=4, mask `scalar`.

## F2 — hard 0.7/0.3 (một yếu tố so với F1: mức lệch)
- w_i = 0.7 nếu teacher đoán đúng ảnh i, 0.3 nếu sai. L = mean[(1 − w_i)·CE_i] + mask·mean[w_i·KL_i] + 0.5·BC.
- Env: `SAMPLE_WEIGHT_MODE=teacher_correct`, `SW_TRUST=0.7`, `SW_DISTRUST=0.3`.
- Câu hỏi: hiệu ứng có tăng khi lệch mạnh hơn không? Nếu F2 > F1 → hướng "tin teacher theo đúng/sai" còn dư địa; nếu F2 < F1 → 0.6/0.4 gần tối ưu hoặc lệch mạnh làm hại.

## F3 — bản mềm kiểu CCKD (một yếu tố so với F1: cách tính trọng số)
- w_i = 0.4 + 0.2 · p_teacher(nhãn đúng của ảnh i) (softmax ở T=1). Trọng số chạy liên tục trong [0.4, 0.6], cùng khoảng với F1.
- Teacher càng tự tin vào nhãn đúng → càng tin KD; teacher đúng nhưng do dự → trọng số giữa; teacher sai và tự tin vào lớp khác → gần 0.4.
- Env: `SAMPLE_WEIGHT_MODE=teacher_prob`, `SW_TRUST=0.6`, `SW_DISTRUST=0.4`.
- Ý tưởng gần CCKD-L (Mishra & Sundaram 2021). Câu hỏi: trọng số mềm có tốt hơn trọng số 2 mức không?

## Implement (`kaggle/streamlined_kd_cifar100.py`)
- Thêm chế độ `teacher_prob` vào `teacher_correct_weights`; log thêm `teacher_p_label_mean`.
- Dòng log mỗi epoch in thêm `teacher_train_top1=...` (và `p_label=...` với F3) khi bật trọng số theo mẫu, để đọc được từ stdout
  (history.csv không tải được do host `www.kaggleusercontent.com` bị chặn). Chế độ mặc định: dòng log không đổi.

## Kiểm tra (CPU, 2026-11-05)
1. Bit-exact so với script trước ở 3 chế độ: mặc định, F1 (0.6/0.4), hard 0.7/0.3 (loss và gradient).
2. `teacher_prob` với 0.5/0.5: loss chênh 0.0, gradient chênh 9e-10 so với baseline.
3. `teacher_prob` 0.6/0.4 và hard 0.7/0.3: CE/KD khớp tính tay; BC giữ nguyên; gradient hữu hạn; w nằm trong [0.4, 0.6].
4. Smoke test 1 epoch CPU (dữ liệu giả) chế độ `teacher_prob`: chạy được, có `teacher_p_label_mean`.
5. Smoke test GPU trên tài khoản 3 (2 epoch × 30 batch, teacher mdistiller, chế độ `teacher_prob`): xem mục Trạng thái.

## Đánh giá (QUY ƯỚC CHỈ SỐ trong HANDOFF)
- So với F1 và E4 bằng top-1 epoch cuối và TB 10 cuối; best top-1 để so với paper (72.50).
- Mới 1 seed (2021) mỗi run; biến thể nào hơn E4 ≥ +0.5 (≥ 72.20) là ứng viên xác nhận seed 2022 (cùng với E4 seed 2022).

## Chạy
- F2 slug `streamlined-kd-cifar100-f2-hard-07-03`, F3 slug `streamlined-kd-cifar100-f3-soft-cckd` (tài khoản 3), T4, ~4,8h mỗi run.

## Trạng thái
Plan + code + test CPU xong. **BLOCKER tài khoản 3: chưa có GPU.**
- Smoke test `trananhkiet5551/streamlined-kd-account3-smoke` (version 1, 2026-11-05 16:51 UTC): metadata trên server đúng (`enable_gpu: true`,
  `machine_shape: NvidiaTeslaT4`, dataset `fedesoriano/cifar100`, Docker image ghim, internet bật) nhưng log in **`device: cpu`** và script tự dừng
  (kiểm tra GPU) sau ~1 phút → ERROR. Giống hệt tình trạng tài khoản 2 lúc đầu (2026-11-03): tài khoản chưa được bật GPU, thường do chưa xác minh số điện thoại.
- Đã xác nhận phần còn lại của tài khoản 3 hoạt động: CLI xác thực bằng `KAGGLE_KEY3` (token KGAT_), đọc được dataset, push được kernel;
  Kaggle MCP đã ở tài khoản 3 (quota GPU 0s/30h).
- Cách gỡ: người dùng xác minh số điện thoại cho tài khoản `trananhkiet5551` trên Kaggle (Settings → Phone verification), rồi chạy lại smoke test.
  Sau đó submit F2 và F3 (có thể song song: 2 session, mỗi run ~4,8h quota).

## Cập nhật 2026-11-05 17:35 UTC — F2 chạy trên tài khoản 2 (người dùng chỉ định), chấp nhận bị cắt do quota
- Tài khoản 3 chưa có GPU (smoke test lần 2 vẫn `device: cpu`) → người dùng chọn dùng lại tài khoản 2 (`kitit5512trn`), Kaggle MCP đã kết nối lại tài khoản 2.
- Quota tài khoản 2 lúc submit: 91.540 s / 108.000 s dùng (còn 16.460 s ≈ 4,57h). Một run 300 epoch cần ~17.000–17.400 s →
  **dự kiến bị Kaggle dừng ở ~epoch 284–290** (57–58 s/epoch). Người dùng chấp nhận chạy luôn.
  Hệ quả: có thể không có top-1 epoch 300 / TB 10 epoch cuối / summary.json; đọc kết quả từ log (best và các epoch đã chạy).
  So sánh công bằng với F1 sẽ dùng **cùng mốc epoch cuối cùng mà F2 đạt được** (top-1 tại epoch đó và TB 10 epoch trước đó của cả F1 và F2).
- F2: https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-f2-hard-07-03 — version 1, submit ~17:32 UTC;
  notebook `kaggle/experiments/f2_hard_07_03.ipynb` (khác cấu hình F1 build từ cùng script đúng 2 dòng `SW_TRUST`/`SW_DISTRUST`), commit `db45c42`.
- F3 (bản mềm) chạy sau khi quota refresh (2026-11-07 00:00 UTC).
