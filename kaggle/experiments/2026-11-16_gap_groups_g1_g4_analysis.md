# Phân tích nhóm G1–G4 (teacher/student đúng-sai) cho E4 vs F1b — TRẠNG THÁI: BỊ CHẶN, CHƯA CÓ KẾT QUẢ

Ngày: 2026-11-16. Chỉ inference trên checkpoint đã có, không train, không submit, không scheduler.

## Trạng thái
- **Chưa có số liệu nào.** Không thể tải checkpoint student của E4/F1b seed 2022 về môi trường này.
- Checkpoint **có tồn tại** trên Kaggle (tài khoản 1): mỗi run có `resnet20_skd_best.pth`, `last_checkpoint.pt` (epoch 300) và `mdistiller_resnet56_vanilla_ckpt_epoch_240.pth`:
  - E4 seed 2022: `trananhkiet/streamlined-kd-cifar100-e4-seed2022`
  - F1b seed 2022: `trananhkiet/streamlined-kd-cifar100-f1b-seed2022`
- Kaggle MCP `download_notebook_output` chỉ trả URL ở host `www.kaggleusercontent.com` (host bị chặn từ trước, xem mục tài khoản trong HANDOFF). Một lệnh tải trực tiếp bị bộ phân loại quyền của auto mode từ chối; **không thử cách vòng**.
- Trong repo chỉ có `.pth` của teacher/student vanilla (`CIFAR-100/*_Pth/`), không phải E4/F1b. Không dùng chúng thay thế.
- Máy này cũng không có dữ liệu CIFAR-100 (script cần `train/test/meta` như trên Kaggle).

## Đã làm xong
- `kaggle/analysis_gap_groups.py` (3 lệnh: `selftest`, `extract`, `report`). `selftest` PASS trên logits tổng hợp: gradient dạng đóng theo logits student khớp autograd (CE và T²·KL, sai số ≤1e-10), KD khớp `kd_loss(per_sample=True)` của repo, phân nhóm đầy đủ. Đây **không** phải kết quả thật.

## Thiết kế (ghi trước khi thấy số)
- Dữ liệu: 10.000 ảnh train cố định (rng seed 0, độc lập seed train), transform không augment, **một view dùng chung** cho teacher và student → không dùng test để chọn công thức. Hạn chế: student đã train trên toàn bộ 50k ảnh nên nhóm G phản ánh độ khớp train, không phải tổng quát hóa. (Test chỉ chạy mô tả nếu muốn: `--split test`.)
- Checkpoint: ưu tiên `last_checkpoint.pt` (cùng epoch 300 cho cả hai); thêm best-epoch (khác epoch giữa hai run) như phân tích phụ.
- Mỗi nhóm đo: n/tỷ lệ, CE, KD (T=4, ×T²), p_teacher[y], p_student[y], cos(g_CE, g_KD) tại logits student (mean, median, CI95 bootstrap, % cos<0), |g_CE|, |g_KD|; ma trận chuyển nhóm E4→F1b.
- Giới hạn gradient: mask scalar (~0.5) và BC không có trong phép đo; cos không phụ thuộc scale nên không đổi.

## Tiêu chí kết luận (ghi trước)
- G2 **được ủng hộ** nếu cos_mean(G2) > cos_mean của G1/G3/G4 với CI95 không giao, **và** F1b làm tăng p_student[y] ở G2 hơn G1, **và** G2 của F1b ít mẫu hơn E4 (chuyển G2→G1).
- G2 **bị bác bỏ** nếu cos(G2) ≈ cos(G1) hoặc F1b không đổi gì ở G2.
- Ngoài hai trường hợp: ghi "chưa đủ dữ liệu". Cosine chỉ là bằng chứng cơ chế tối ưu, không chứng minh tăng KD riêng G2 sẽ tăng accuracy.
- K1 (w = 0.5 + 0.1·1[teacher đúng]·1[student sai]) **chưa được đề xuất train**; quyết định sau khi có số. Lưu ý trước: student sai trên input hiện tại thay đổi theo epoch nên K1 là trọng số động (cần tính trong train loop, không có gradient qua phép chọn); gần với WSL/TGeo-KD (trọng số theo S–T–GT) nên tính mới thấp.

## Cách chạy khi đã có checkpoint (trên Kaggle hoặc máy có file)
```bash
python kaggle/analysis_gap_groups.py extract e4.npz  --student E4/last_checkpoint.pt  --teacher mdistiller_resnet56_vanilla_ckpt_epoch_240.pth --split train
python kaggle/analysis_gap_groups.py extract f1b.npz --student F1b/last_checkpoint.pt --teacher mdistiller_resnet56_vanilla_ckpt_epoch_240.pth --split train
python kaggle/analysis_gap_groups.py report --a E4=e4.npz --b F1b=f1b.npz --out gap_groups.json
```
