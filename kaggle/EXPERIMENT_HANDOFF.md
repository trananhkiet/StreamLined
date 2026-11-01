# SKD experiment handoff

Updated: 2026-10-31 (trigger tự động).

## Mục tiêu và trạng thái

Đang reproduce Streamlined Knowledge Distillation trên CIFAR-100 với teacher
ResNet56 và student ResNet20. Pipeline hiện giữ các phần chính của repo: 240
epoch, SGD, weak + strong augmentation (100.000 samples/epoch), temperature 4,
Tikhonov `0.1`, chạy trên Kaggle T4.

| Run | CE/KD | Teacher top-1 | Best student | Final student |
|---|---:|---:|---:|---:|
| A | 0.1/0.9 | 71.59% | 68.49% (epoch 235) | 68.16% |
| B — tốt nhất | 0.5/0.5 | 71.59% | **70.26% (epoch 235)** | 70.04% |
| Paper | — | 72.40% | 72.50% | — |

Run B cải thiện `+1.77` điểm so với A nhưng còn thấp hơn paper `2.24` điểm.
Teacher checkpoint được phát hành chỉ đạt 71.59%, thấp hơn số paper 0.81 điểm.

Chi tiết và config đầy đủ:

- [Run A](experiments/2026-10-18_resnet56_resnet20_skd_cifar100.md)
- [So sánh A/B](experiments/2026-10-18_skd_loss_weight_comparison.md)

Lưu ý: code hiện đánh giá và chọn best checkpoint bằng CIFAR-100 test set mỗi
epoch; `best top-1` vì vậy có selection bias. Luôn báo cáo cả best và epoch 240.

## Quy trình cho mỗi experiment

1. Copy notebook từ run B và chỉ thay đổi **một yếu tố**.
2. Ghi trước hypothesis, config và tên notebook vào file `experiments/*.md`.
3. Trước khi train, kiểm tra dataset attached, teacher SHA-256/accuracy, seed,
   batch size, LR schedule và không có batch limit.
4. Chạy notebook private trên Kaggle; theo dõi trạng thái và log các mốc LR.
5. Khi hoàn tất, lưu `summary.json`, `history.csv`, best/last checkpoint; ghi best
   epoch, final epoch, runtime và chênh lệch với run B vào experiment log.

## Plan cải thiện kết quả

Giữ teacher hiện tại, ResNet20 và recipe của run B làm mốc. Các hướng dưới đây
**chưa chạy**; đây là giả thuyết cải tiến, không bảo đảm đạt 72.5%. Mỗi run chỉ
đổi một yếu tố; chỉ kết hợp các hướng đã cải thiện độc lập.

### A. Điều chỉnh chọn mẫu và sức mạnh loss SKD

- Code hiện reduce KD/BC thành scalar rồi nhân mask: `mean(loss * mask)` chỉ
  bằng `loss * mean(mask)`, không thực sự chọn mẫu/class để distill.
- A1: giữ KL theo từng mẫu `k_i = T² KL(p_teacher || p_student)` rồi tính
  `L_KD = mean(mask_i * k_i)`, giữ mask low-confidence 50% như repo. So với B
  để đo riêng tác động chọn mẫu; không đồng thời đổi sang high-confidence.
- A2: từ A1, thử mask high-confidence; log accuracy của teacher trên nhóm được
  chọn để xem chọn mẫu đáng tin hay mẫu khó hữu ích hơn. Nếu chia cho
  `sum(mask)` thay vì batch size thì phải bù hệ số để tránh tăng gấp đôi loss.
- A3: tách `lambda_bc` khỏi `kd_weight`, thử `0 / 0.5 / 2.0` trên cùng cấu hình.
  BC là quan hệ giữa **samples** (ma trận B×B), nên class mask C chiều không
  thể gán trực tiếp lên các hàng BC; dùng BC toàn batch cho sweep này và giữ
  hệ số mask scalar cũ để đối chứng. Log norm gradient CE/KD/BC ở lớp cuối;
  BC nhỏ về giá trị không đồng nghĩa gradient yếu.

### B. Logit Standardization: giảm ràng buộc về biên độ logits

Theo [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_Logit_Standardization_in_Knowledge_Distillation_CVPR_2024_paper.html),
chuẩn hóa riêng logits mỗi mẫu: `z_hat = (z - mean(z)) / (std(z) + 1e-7)` trước
softmax/KL, để student học quan hệ giữa classes thay vì phải khớp biên độ teacher.
Chỉ thay nhánh KL; CE và Gram/BC vẫn dùng logits gốc. Chạy bật/tắt tại `T=4`
trước, sau đó thử riêng `T=1,2` nếu cần. Log std logits và KL gradient norm để
phân biệt cải thiện representation với thay đổi thang loss.
[Code gốc](https://github.com/sunshangquan/logit-standardization-KD/blob/master/mdistiller/distillers/KD.py).

### C. DKD: tách kiến thức target và non-target

[DKD (CVPR 2022)](https://arxiv.org/abs/2203.08679) tách KL thành TCKD
(đúng class vs phần còn lại) và NCKD (phân bố giữa các class không phải nhãn).
Điều này tránh NCKD bị suy giảm khi teacher quá tự tin ở class đúng.
Đầu tiên chạy `L = CE + w(t)*(alpha*TCKD + beta*NCKD)`, bỏ KL cũ và BC;
`alpha=1, beta=2, T=4, w(t)=min(epoch/20,1)` theo
[config Res56→Res20](https://github.com/megvii-research/mdistiller/blob/master/configs/cifar100/dkd/res56_res20.yaml)
và [defaults](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/engine/cfg.py).
Giữ LR 0.1 của B cho ablation; config gốc dùng 0.05 nên đây chưa phải reproduce
DKD đầy đủ. Nếu tốt, chạy DKD+BC để đo phần đóng góp thêm của SKD.
Không dùng nguyên defaults `Distiller/DKD.py` local: bản đó có clamp logits và
CE/alpha/beta khác upstream. Log TCKD/NCKD riêng để biết nhánh nào hữu ích.

### D. DOT: cải tiến optimizer khi CE và distillation cạnh tranh

[DOT (ICCV 2023), code chính thức](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/engine/dot.py)
giữ buffer momentum riêng cho task loss và distillation loss. Áp dụng cho
`L_task=CE`, `L_distill=KD+BC`; backward riêng rồi cập nhật theo DOT, không cộng
loss trước backward như SGD hiện tại. Dùng implementation gốc, lưu cả hai
momentum buffer để resume. So với cùng loss dùng SGD; log cosine giữa hai
gradient, accuracy và thời gian/epoch. Đây là hướng sau cùng vì cần sửa trainer
và tăng chi phí backward, trong khi B/C chỉ cần thay loss.

Thứ tự chạy do người dùng chỉ định (duy nhất): **A1 → A2 → A3 → B → C → D**. Sweep hệ số/nhiệt độ là đề xuất của dự án, không phải cấu hình
đã được paper xác nhận tối ưu cho SKD. Mỗi run ghi vào `experiments/*.md`:
`giả thuyết → thay đổi → config → kết quả → bài học → bước tiếp`.
Chọn hyperparameter bằng validation tách từ train, giữ cùng split cho các ứng
viên; run B dùng full train chỉ là mốc lịch sử. Chốt recipe rồi train full train
với ít nhất 3 seed, báo cáo mean/std và accuracy cuối; chỉ test sau khi chốt.

## Kinh nghiệm đã rút ra

- `0.5/0.5` phù hợp implementation hiện tại hơn `0.1/0.9`.
- Không dùng download CIFAR-100 bên ngoài; phải attach dataset Kaggle để tránh
  mất hàng giờ vì mạng chậm.
- Không kết luận từ một seed hoặc so trực tiếp với baseline 65.79% chưa matched.
- Giữ từng run ở notebook/version riêng để không mất output và checkpoint cũ.

## Trạng thái run hiện tại — BLOCKER (2026-10-31)

Experiment tiếp theo theo thứ tự A1 → A2 → A3 → B → C → D là **A1** (per-sample KD mask),
xem [experiments/2026-10-31_a1_per_sample_kd_mask.md](experiments/2026-10-31_a1_per_sample_kd_mask.md).
Code đã commit `03ba69c` (default `KD_MASK_MODE=scalar` = hành vi run B).

- Notebook: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-a1-per-sample-mask
  (kernel id 141928704). v1 là QuickSave placeholder (không train); v2 là SaveAndRunAll,
  session output `364199594`, code tải từ GitHub commit `03ba69c` kèm SHA-256.
- v2 **FAILED sau ~27 giây** (RuntimeError ở `prepare_cifar100`): Kaggle MCP `save_notebook`
  âm thầm bỏ `datasetDataSources=["fedesoriano/cifar100"]` (lỗi MCP đã biết). Không tốn GPU đáng kể,
  **không tính là retry**. `get_notebook_info` không hiển thị data sources nên MCP không xác minh được.
- v3 thử lại với trường `datasetDataSourcesSetter` cũng FAILED sau ~28 giây, cùng lỗi (session output `364202650`): MCP bỏ dataset ở cả hai trường; không thử thêm bằng MCP.
- Không thử được workflow REST: session cloud này không có Kaggle CLI/credentials (`~/.kaggle` không tồn tại)
  và shell không truy cập được kaggle.com. Theo quy tắc, không lặp lại vòng thử.
- **Cách resume**: trong môi trường có credentials, push notebook với dataset attach đúng như run B
  (REST push, xem kaggle/README.md): slug `trananhkiet/streamlined-kd-cifar100-a1-per-sample-mask`,
  `dataset_sources=["fedesoriano/cifar100"]`, GPU T4, docker image như run B
  (`gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`);
  nội dung notebook = file `kaggle/experiments/a1_per_sample_kd_mask.ipynb` (inline) hoặc loader pin commit
  `03ba69c` đã dùng ở v2 (env KD_MASK_MODE=per_sample, CE/KD 0.5/0.5). Sau đó lần trigger kế tiếp kiểm tra run.
- Chưa có kết quả nào của A1; giả thuyết chưa được kiểm chứng.

### Cập nhật trigger 2026-11-01

Blocker vẫn nguyên: kernel A1 ở trạng thái ERROR (v3), shell vẫn không có Kaggle CLI/credentials
(`~/.kaggle` không tồn tại), nên không có đường REST. Không submit lại bằng MCP vì đã xác nhận MCP bỏ
dataset attachment ở cả hai trường (v2, v3). Không tốn thêm GPU. Cách resume như mục BLOCKER ở trên:
cần một môi trường có credentials để push notebook A1 kèm `fedesoriano/cifar100`.

### Cập nhật trigger thủ công 2026-11-01 (kiểm tra môi trường + thử lại)

Kiểm tra trước workflow: KAGGLE_USERNAME KHÔNG, KAGGLE_KEY KHÔNG, `~/.kaggle/kaggle.json` KHÔNG;
`curl https://www.kaggle.com/api/v1/` = 200; `kaggle` CLI đã cài nhưng không có credentials (không dùng được).
Kaggle MCP đã xác thực: GPU quota dùng 69.9s/108000s (refresh 2026-11-07); đọc được notebook A1 và dataset
`fedesoriano/cifar100` v1.
Thử lại một lần qua MCP với cả `datasetDataSources` và `datasetDataSourcesSetter` + đủ cờ `has*`: tạo v4
(SaveAndRunAll, T4), RUNNING lúc đầu nhưng chuyển ERROR trong ~2 phút; `get_notebook_info` không hiển thị
data sources nên không xác minh được attachment, và hành vi trùng v2/v3 (lỗi do thiếu dataset). Không tính retry;
không tốn GPU đáng kể. Kết luận: MCP vẫn không attach được dataset; cần push REST từ môi trường có credentials
(cách resume như mục BLOCKER ở trên). Không thử thêm bằng MCP.

### Cập nhật trigger v5 thủ công 2026-11-01 (03:43 UTC, env mới có KAGGLE_*)

Kiểm tra trước workflow: `KAGGLE_USERNAME` CÓ, `KAGGLE_KEY` CÓ, `~/.kaggle/kaggle.json` KHÔNG; `kaggle` CLI chưa cài.
Repo clone OK (HEAD `cb59dae`, `main`); `git fetch` OK, `git push --dry-run` báo "Everything up-to-date" (chưa kiểm chứng quyền ghi bằng push thật).
Kaggle MCP xác thực được; trạng thái notebook A1 (`streamlined-kd-cifar100-a1-per-sample-mask`) vẫn ERROR (v4), không có run QUEUED/RUNNING.

**Blocker mới (thay cho blocker thiếu credentials):** shell có credentials nhưng `www.kaggle.com` bị network policy của environment chặn
(`curl https://www.kaggle.com/api/v1/` -> `CONNECT tunnel failed, response 403`; proxy log: `connect_rejected`, host `www.kaggle.com:443`).
Vì vậy không chạy được REST push (`kaggle kernels push`) để attach `fedesoriano/cifar100`. Không submit lại bằng MCP vì MCP đã xác nhận bỏ dataset attachment (v2/v3/v4). Không tốn GPU, không tính retry.

**Cách resume:** vào cài đặt environment (menu cloud environment ở thanh tiêu đề session -> Edit), mở Network access rộng hơn hoặc thêm
`www.kaggle.com` (và `kaggle.com`) vào Allowed domains. Sau đó trigger lại: cài `pip install kaggle`, đặt KAGGLE_USERNAME/KAGGLE_KEY, push notebook A1 theo mục BLOCKER ở trên (dataset_sources=["fedesoriano/cifar100"], T4, docker image như run B).

### A1 ĐÃ SUBMIT THÀNH CÔNG — trigger v5 thủ công 2026-11-01 (~03:50 UTC)

Blocker mạng đã được gỡ (đã mở `www.kaggle.com`, `api.kaggle.com`). Lưu ý cho lần sau: `kaggle` CLI 2.x đọc token dạng `KGAT_...`
qua biến `KAGGLE_API_TOKEN`, **không** qua `KAGGLE_KEY`; dùng `export KAGGLE_API_TOKEN="$KAGGLE_KEY"` trước khi gọi CLI (`pip install kaggle`).

- Experiment: **A1** (per-sample KD mask, `KD_MASK_MODE=per_sample`), CE/KD 0.5/0.5, T=4, tik 0.1, seed 2021, 240 epoch, T4. Đối chứng = run B (70.26% best / 70.04% final).
- Code: `kaggle/experiments/a1_per_sample_kd_mask.ipynb` (inline), Git commit `cb59dae`. Diff so với notebook run B chỉ nằm ở `kd_loss` + `distillation_losses` (yếu tố A1).
- Cách push (REST, đúng README): metadata từ run B (`dataset_sources=["fedesoriano/cifar100"]`, docker image như run B, `machine_shape=NvidiaTeslaT4`, internet bật), `kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4`.
- Notebook: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-a1-per-sample-mask — **version 5**, submit lúc ~2026-11-01 03:50 UTC.
- Đã xác minh: `kaggle kernels pull -m` của version mới liệt kê `fedesoriano/cifar100` trong `dataset_sources`; trạng thái RUNNING sau >3 phút (v2/v3/v4 qua MCP đều ERROR trong ~2 phút vì mất dataset). Quota GPU trước khi chạy: 0.03h/30h.
- Việc cần làm ở lần trigger tiếp theo: **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước** (`kaggle kernels status`, hoặc Kaggle MCP). Nếu COMPLETE: tải `summary.json`/`history.csv`/checkpoint, ghi kết quả vào `experiments/2026-10-31_a1_per_sample_kd_mask.md`, so sánh với B, rồi chuyển sang A2. Nếu ERROR: đọc log, retry 1/3.
- Mục "BLOCKER (2026-10-31)" và các cập nhật trước đó trong file này đã **hết hiệu lực** (đã giải quyết bằng REST push).
