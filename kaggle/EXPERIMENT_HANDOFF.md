# SKD experiment handoff

Updated: 2026-11-04 08:33 UTC (trigger v5: E2 xong, E3 run 1 đang chạy).

## Mục tiêu và trạng thái

Đang reproduce Streamlined Knowledge Distillation trên CIFAR-100 với teacher
ResNet56 và student ResNet20. Pipeline hiện giữ các phần chính của repo: 240
epoch, SGD, weak + strong augmentation (100.000 samples/epoch), temperature 4,
Tikhonov `0.1`, chạy trên Kaggle T4.

| Run | CE/KD | Teacher top-1 | Best student | Final student |
|---|---:|---:|---:|---:|
| A | 0.1/0.9 | 71.59% | 68.49% (epoch 235) | 68.16% |
| B — tốt nhất | 0.5/0.5 | 71.59% | **70.26% (epoch 235)** | 70.04% |
| A1 (per-sample KD mask, low-conf) | 0.5/0.5 | 71.59% | 69.28% (epoch 223) | 68.97% |
| A2 (per-sample KD mask, high-conf) | 0.5/0.5 | 71.59% | 69.76% (epoch 234) | 69.57% |
| A3-bc0 (lambda_bc=0, nền B) | 0.5/0.5 | 71.59% | 70.03% (epoch 236) | 69.85% |
| A3-bc2 (lambda_bc=2.0, nền B) | 0.5/0.5 | 71.59% | 70.31% (epoch 239) | 69.86% |
| LS — logit standardization, T=4 (nền B) | 0.5/0.5 | 71.59% | 68.87% (epoch 228) | 68.60% |
| LS-T2 — logit standardization, T=2 (nền B) | 0.5/0.5 | 71.59% | 69.30% (epoch 230) | 68.96% |
| C — DKD (LR 0.1, CE 1.0, alpha 1, beta 2) | — | 71.59% | 69.74% (epoch 234) | 69.51% |
| **E1 — step LR 80/150/200 (nền B)** | 0.5/0.5 | 71.59% | **70.79% (epoch 239)** | **70.75%** |
| E2 — LR 0.05, lịch 150/180/210 (nền B) | 0.5/0.5 | 71.59% | 70.59% (epoch 234) | 70.14% |
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


## (Lịch sử) Trạng thái 2026-11-01 12:10 UTC — đã được thay bằng mục TRẠNG THÁI HIỆN TẠI ở cuối file

**A1 COMPLETE:** best 69.28% (epoch 223), final 68.97% (epoch 240), runtime ~3,66 giờ; thấp hơn run B −0,98 (best) / −1,07 (final). Chi tiết và bài học:
[experiments/2026-10-31_a1_per_sample_kd_mask.md](experiments/2026-10-31_a1_per_sample_kd_mask.md). (Mới 1 seed, chưa kết luận mạnh.) Các mục BLOCKER/"A1 ĐÃ SUBMIT" phía trên đã hết hiệu lực.

**A2 ĐÃ SUBMIT (đang chạy):**
- Experiment: **A2** — per-sample KD mask, chọn mẫu **high-confidence** (`KD_MASK_MODE=per_sample`, `KD_MASK_SELECT=high`), một yếu tố khác A1. Đối chứng chính = A1; phụ = B. Log: [experiments/2026-11-01_a2_per_sample_high_confidence.md](experiments/2026-11-01_a2_per_sample_high_confidence.md).
- Code: Git commit `5fdeb36` (`kaggle/streamlined_kd_cifar100.py`, notebook `kaggle/experiments/a2_per_sample_high_confidence.ipynb`). Có thêm log `teacher_top1_selected/rest`, `kd_selected_fraction` trong `history.csv`.
- Notebook: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-a2-per-sample-high-conf — **version 1**, submit 2026-11-01 ~12:10 UTC, T4, dataset `fedesoriano/cifar100` đã xác minh bằng `kaggle kernels pull -m`, trạng thái RUNNING sau >2,5 phút.
- Quota GPU: A1 đã dùng ~3,7h; còn ~26h/30h trước khi submit A2 (refresh 2026-11-07).
- Việc cần làm ở lần trigger tiếp theo: **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.** Dự kiến xong ~16:00 UTC (≈ 3,7 giờ). Nếu COMPLETE: tải output (Kaggle MCP `list_notebook_session_output`; CLI `kaggle kernels output` có thể bị chặn host `www.kaggleusercontent.com`), ghi kết quả vào file A2, rồi sang **A3** (sweep `lambda_bc` 0/0.5/2.0). Nếu ERROR: đọc log, retry 1/3.

### Ghi chú vận hành (rút ra ở lần chạy 12:00 UTC)
- Lịch v5: đã chuyển sang cron `CRON_TZ=Asia/Saigon 0 19,3,11 * * *` (mỗi 8 giờ). Lần chạy một lần ban đầu tự tắt trigger (`run_once_fired`) nên phải bật lại bằng `update_trigger` (đã làm).
- `kaggle kernels output` bị 403 ở host `www.kaggleusercontent.com`; dùng Kaggle MCP `list_notebook_session_output` (trả log đầy đủ + URL file) hoặc xin mở host này trong Network access nếu cần tải checkpoint.
- `kaggle kernels status` bị rate limit nếu gọi dồn; dùng MCP `get_notebook_session_status`.
- PyTorch không có sẵn trong container; muốn unit test CPU phải `pip install torch` (PyPI hoạt động; index download.pytorch.org bị chặn).
- Bốn file `.pth` báo modified giả (Git LFS): đã `git update-index --skip-worktree`, không commit chúng.


## (Lịch sử) Trạng thái 2026-11-01 20:10 UTC — đã được thay bằng mục TRẠNG THÁI HIỆN TẠI ở cuối file

**A2 COMPLETE:** best 69.76% (epoch 234), final 69.57% (epoch 240), top-5 92.04%, test loss 1.1067, runtime 14.173 s (~3,94 giờ).
So với A1: +0,48 best / +0,60 final (high-conf tốt hơn low-conf). So với run B: **−0,50 best / −0,47 final** — cả A1 và A2 đều chưa vượt B. Chi tiết:
[experiments/2026-11-01_a2_per_sample_high_confidence.md](experiments/2026-11-01_a2_per_sample_high_confidence.md). (1 seed.) Chưa lấy được `history.csv` (host `www.kaggleusercontent.com` bị chặn).

**A3-bc0 ĐÃ SUBMIT (đang chạy):**
- Cách hiểu đã chọn: A1/A2 chưa vượt B nên A3 lấy **run B làm nền** (KD mask scalar), chỉ tách `lambda_bc` khỏi `kd_weight`. Sweep lần lượt: **A3-bc0 (`LAMBDA_BC=0`)** → A3-bc2 (`LAMBDA_BC=2.0`); `lambda_bc=0.5` chính là run B (không chạy lại). Log: [experiments/2026-11-02_a3_lambda_bc_sweep.md](experiments/2026-11-02_a3_lambda_bc_sweep.md).
- Code: Git commit `c2476b2` (`kaggle/streamlined_kd_cifar100.py`: thêm `LAMBDA_BC`, `GRAD_LOG_EVERY`, log `gradnorm_ce/kd/bc` ở `fc.weight` theo epoch; notebook `kaggle/experiments/a3_lambda_bc0.ipynb`). Loss mới giống hệt bit bản cũ ở mọi chế độ cũ (đã test CPU).
- Notebook: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-a3-lambda-bc0 — **version 1**, submit 2026-11-01 ~20:08 UTC, T4, dataset `fedesoriano/cifar100` đã xác minh bằng `kaggle kernels pull -m`, RUNNING sau >2 phút.
- Quota GPU: đã dùng 7,64h trước khi submit A3-bc0; còn ~22,3h/30h (refresh 2026-11-07). Mỗi run ~3,7–3,9 giờ GPU.
- Việc cần làm ở lần trigger tiếp theo: **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.** Dự kiến xong ~00:10 UTC. Nếu COMPLETE: lấy kết quả (MCP `list_notebook_session_output` trả cả log; CLI `kaggle kernels output` bị 403 ở `www.kaggleusercontent.com`), ghi vào file A3, so với B (70.26/70.04), rồi submit **A3-bc2**: build notebook từ `streamlined_kd_cifar100.py` với `LAMBDA_BC` default `2.0` (cùng cách build `a3_lambda_bc0.ipynb`: thay chuỗi `os.getenv("LAMBDA_BC", "-1")` bằng `"2.0"`, dùng `build_notebook.py` với SOURCE/TARGET đổi), slug `streamlined-kd-cifar100-a3-lambda-bc2`, cùng metadata như A3-bc0. Nếu ERROR: đọc log, retry 1/3.

### Ghi chú vận hành bổ sung
- Config `CFG` là frozen dataclass: khi unit test đổi giá trị dùng `object.__setattr__(CFG, name, value)`.
- Unit test CPU cần `pip install torch torchvision tqdm pillow numpy` (PyPI chạy được). Test hồi quy mẫu: so `distillation_losses` của commit trước (git show) với bản mới, sai lệch phải bằng 0.
- Lần chạy 20:04 UTC: trigger đã có cron nên không cần chuyển lịch; `kaggle kernels status` có thể bị rate limit, dùng MCP `get_notebook_session_status`.


## (Lịch sử) Trạng thái 2026-11-02 04:12 UTC — đã được thay bằng mục TRẠNG THÁI HIỆN TẠI ở cuối file

**A3-bc0 COMPLETE** (`LAMBDA_BC=0`, nền run B): best **70.03%** (epoch 236), final **69.85%**, top-5 92.63%, test loss 1.0603, runtime 13.830 s (~3,84 giờ).
So với B: −0,23 best / −0,19 final top-1, nhưng top-5 +0,31 và test loss tốt hơn. Bỏ BC gần như không đổi kết quả; chênh top-1 nhỏ, trong tầm dao động seed (1 seed).
A3-bc0 vẫn cao hơn A1 (69.28) và A2 (69.76): ~1 điểm mất ở A1/A2 là do chọn mẫu per-sample, không phải do BC. Chi tiết: [experiments/2026-11-02_a3_lambda_bc_sweep.md](experiments/2026-11-02_a3_lambda_bc_sweep.md).
Chưa lấy được `history.csv` (host `www.kaggleusercontent.com` bị chặn) nên chưa có `gradnorm_*` và `teacher_top1_selected` theo epoch.

**A3-bc2 ĐÃ SUBMIT (đang chạy):**
- Experiment: **A3-bc2** (`LAMBDA_BC=2.0`, nền run B: KD mask scalar, CE/KD 0.5/0.5). Đối chứng: B (0.5) và A3-bc0 (0). `lambda_bc=0.5` chính là B, không chạy lại.
- Code: Git commit `3619d1a` (notebook `kaggle/experiments/a3_lambda_bc2.ipynb`, khác `a3_lambda_bc0.ipynb` đúng một dòng default `LAMBDA_BC`). Code train đã commit ở `c2476b2`.
- Notebook: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-a3-lambda-bc2 — **version 1**, submit 2026-11-02 ~04:08 UTC, T4, dataset `fedesoriano/cifar100` đã xác minh bằng `kaggle kernels pull -m`, RUNNING sau >2 phút.
- Quota GPU: đã dùng 11,5h trước khi submit; còn ~18,5h/30h (refresh 2026-11-07). Mỗi run ~3,8–3,9 giờ GPU; còn đủ cho khoảng 4 run nữa trong tuần.
- Việc cần làm ở lần trigger tiếp theo (19:00 giờ Sài Gòn): **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.** Dự kiến xong ~08:00 UTC (15:00 giờ Sài Gòn). Nếu COMPLETE: lấy kết quả bằng MCP `list_notebook_session_output` (log đầy đủ), ghi vào file A3, so với B và A3-bc0.
  Sau đó A3 hoàn tất → sang **B** (Logit Standardization): cần implement mới trong `streamlined_kd_cifar100.py`:
  `z_hat = (z - mean(z)) / (std(z) + 1e-7)` theo từng mẫu, chỉ áp cho nhánh KL (CE và Gram/BC dùng logits gốc), bật/tắt tại T=4 trước; log std logits và KL gradient norm.
  Nền của B: run B (KD mask scalar, `LAMBDA_BC` mặc định `-1`), giữ nguyên các phần khác. Nhớ test hồi quy CPU (loss mặc định phải giống hệt bản cũ) như đã làm ở A3. Nếu ERROR: đọc log, retry 1/3.


## (Lịch sử) Trạng thái 2026-11-02 12:25 UTC — đã được thay bằng mục TRẠNG THÁI HIỆN TẠI ở cuối file

**A3 HOÀN TẤT.** `lambda_bc` = 0 / 0.5 (run B) / 2.0 → best 70.03 / 70.26 / 70.31, final 69.85 / 70.04 / 69.86 (1 seed). Biên độ ~0,3 điểm, không đơn điệu:
hệ số BC trong khoảng 0–2 không phân biệt được với nhiễu seed. A3-bc2 chạy 15.188 s (~4,22 giờ). Chi tiết: [experiments/2026-11-02_a3_lambda_bc_sweep.md](experiments/2026-11-02_a3_lambda_bc_sweep.md).
Tổng kết A1/A2/A3: chưa hướng nào vượt rõ run B (70.26/70.04); chọn mẫu per-sample làm tệ hơn ~0,5–1 điểm, BC không ảnh hưởng.

**B (Logit Standardization, gọi tắt LS; khác "run B" baseline) ĐÃ SUBMIT (đang chạy):**
- Nền = run B; chỉ đổi `LOGIT_STD=1` (z-score logits theo mẫu cho cả student và teacher, chỉ nhánh KL, T=4). Log: [experiments/2026-11-02_b_logit_standardization.md](experiments/2026-11-02_b_logit_standardization.md).
- Code: Git commit `fc43972` (`kaggle/streamlined_kd_cifar100.py` thêm `LOGIT_STD`, `standardize_logits`, log `student_logit_std/teacher_logit_std`; notebook `kaggle/experiments/b_logit_standardization.ipynb`). Test CPU: tắt `LOGIT_STD` thì loss giống hệt bit bản cũ; bật thì KL khớp công thức chính thức.
- Notebook: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-b-logit-std — **version 1**, submit 2026-11-02 ~12:21 UTC, T4, dataset `fedesoriano/cifar100` đã xác minh bằng `kaggle kernels pull -m`, RUNNING sau >2 phút.
- Cảnh báo giả thuyết: sau chuẩn hóa ở T=4 thành phần KD nhỏ hơn nhiều (~6x trên dữ liệu ngẫu nhiên). Nếu LS kém run B, có thể do thang KD nhỏ chứ không do chuẩn hóa; khi đó thử đổi T (1 hoặc 2, theo plan) hoặc `kd_weight`, mỗi lần một yếu tố.
- Quota GPU: đã dùng 15,73h trước khi submit LS; còn ~14,3h/30h (refresh 2026-11-07). Ước tính: LS ~4h → còn ~10,3h; C (DKD) ~4h → còn ~6h; D (DOT) cần backward riêng nên có thể ~5–6h. Nếu thiếu quota thì ghi blocker, không tự mở rộng ngân sách.
- Việc cần làm ở lần trigger tiếp theo (03:00 giờ Sài Gòn 2026-11-03): **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.** Dự kiến xong ~16:30 UTC. Nếu COMPLETE: lấy kết quả bằng MCP `list_notebook_session_output`, ghi vào file LS, so với run B.
  Quyết định tiếp: theo plan "chạy bật/tắt tại T=4 trước, sau đó thử riêng T=1,2 nếu cần" — nếu LS ≥ run B hoặc nghi ngờ do thang KD thì chạy LS với T=2 (`TEMPERATURE` hiện chưa có env; cần thêm `temperature: float = float(os.getenv("TEMPERATURE","4.0"))` trong Config và test hồi quy); nếu LS rõ ràng kém và không có lý do thang thì ghi kết luận và sang **C (DKD)**.
  C (DKD): `L = CE + w(t)*(alpha*TCKD + beta*NCKD)`, bỏ KL cũ và BC; `alpha=1, beta=2, T=4, w(t)=min(epoch/20,1)`, LR 0.1; log TCKD/NCKD riêng; không dùng defaults `Distiller/DKD.py` local (có clamp logits). Nếu ERROR: đọc log, retry 1/3.


## (Lịch sử) Trạng thái 2026-11-02 20:15 UTC — đã được thay bằng mục TRẠNG THÁI HIỆN TẠI ở cuối file

**LS (Logit Standardization, T=4) COMPLETE — kết quả âm rõ:** best **68.87%** (epoch 228), final **68.60%**, top-5 91.84%, test loss 1.0742, runtime 13.820 s (~3,84 giờ).
So với run B: **−1,39 best / −1,44 final** (top-5 −0,48), lớn hơn nhiều nhiễu seed (~0,3). Nghi ngờ chính là thang KD nhỏ đi sau chuẩn hóa ở T=4 (KD yếu đi gần như giảm `kd_weight`), không nhất thiết do ý tưởng chuẩn hóa.
Chi tiết: [experiments/2026-11-02_b_logit_standardization.md](experiments/2026-11-02_b_logit_standardization.md). Chưa có `history.csv` (host bị chặn) nên chưa xác nhận trực tiếp bằng `gradnorm_kd`.

**LS-T2 ĐÃ SUBMIT (đang chạy):**
- Experiment: **LS-T2** = LS nhưng `TEMPERATURE=2.0` (theo plan "thử riêng T=1,2 nếu cần"), một yếu tố so với LS(T=4); nền vẫn là run B (CE/KD 0.5/0.5, KD mask scalar, `lambda_bc` mặc định).
- Code: Git commit `203c47a` (`kaggle/streamlined_kd_cifar100.py` thêm env `TEMPERATURE`, mặc định 4.0 giữ nguyên bit; notebook `kaggle/experiments/b_logit_standardization_t2.ipynb`). Test CPU: hồi quy bit-exact, T=2 khớp công thức chính thức, BC không phụ thuộc T.
- Notebook: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-b-logit-std-t2 — **version 1**, submit 2026-11-02 ~20:10 UTC, T4, dataset `fedesoriano/cifar100` đã xác minh bằng `kaggle kernels pull -m`, RUNNING sau >2 phút.
- Quota GPU: đã dùng 19,58h trước khi submit; còn ~10,4h/30h (refresh 2026-11-07 00:00 UTC). Sau LS-T2 (~3,9h) còn ~6,5h; C (DKD, ~3,9h) vừa đủ, còn ~2,6h. **D (DOT) ~5–6h sẽ không đủ quota trước khi refresh** — nếu tới lúc đó thiếu thì ghi blocker "chờ refresh quota 2026-11-07" và không submit (không tự mở rộng ngân sách).
- Việc cần làm ở lần trigger tiếp theo (11:00 giờ Sài Gòn 2026-11-03): **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.** Dự kiến xong ~00:10 UTC 2026-11-03 (tức trước lần trigger 11:00). Nếu COMPLETE: lấy kết quả bằng MCP `list_notebook_session_output`, ghi vào file LS, so với run B và LS(T=4).
  Quyết định tiếp theo (đã suy tính sẵn):
  - Nếu LS-T2 ≥ run B (70.26/70.04) hoặc cải thiện rõ so với LS(T=4): ghi nhận hướng có hiệu quả khi KD đủ mạnh; có thể thử thêm T=1 sau C nếu quota cho phép.
  - Nếu LS-T2 vẫn kém B nhiều (> ~0,5 điểm): kết luận LS không hợp recipe CE/KD 0.5/0.5 (tăng `kd_weight` sẽ là yếu tố thứ hai, ngoài phạm vi "một yếu tố/lần"; ghi vào bài học như hướng chưa thử) rồi sang **C (DKD)**.
  - **C (DKD)** cần implement mới: `L = CE + w(t)*(alpha*TCKD + beta*NCKD)`, bỏ KL cũ và BC; `alpha=1, beta=2, T=4, w(t)=min(epoch/20,1)`, giữ LR 0.1 của B. Theo mdistiller (megvii-research) `DKD` trong `mdistiller/distillers/DKD.py`: `gt_mask` one-hot theo nhãn; `TCKD` = KL giữa phân phối nhị phân [p_target, 1-p_target] của teacher và student (cả hai softmax(·/T)) × T²; `NCKD` = KL giữa phân phối trên các class non-target sau khi trừ logit target (`logits - 1000*gt_mask`) × T²; `loss = w*(alpha*TCKD + beta*NCKD)`. Không dùng defaults `Distiller/DKD.py` local (có clamp logits, CE/alpha/beta khác upstream). Log TCKD/NCKD riêng. Test hồi quy CPU: các chế độ cũ phải giữ bit-exact. Nếu ERROR: đọc log, retry 1/3.


## (Lịch sử) Trạng thái 2026-11-03 04:15 UTC — đã được thay bằng mục TRẠNG THÁI HIỆN TẠI ở cuối file

**B (Logit Standardization) HOÀN TẤT — không cải thiện.** LS-T2 (T=2): best **69.30%** (epoch 230), final **68.96%**, top-5 92.16%, test loss 1.0571, runtime 14.006 s (~3,89 giờ).
So với LS(T=4) (68.87/68.60): +0,43 / +0,36; so với run B (70.26/70.04): **−0,96 / −1,08**. Hạ T thu hồi một phần nhưng LS vẫn kém B gần 1 điểm trong recipe CE/KD 0.5/0.5.
Chưa thử (ưu tiên thấp, ngoài phạm vi một yếu tố/lần): T=1; tăng `kd_weight` để bù thang. Chi tiết: [experiments/2026-11-02_b_logit_standardization.md](experiments/2026-11-02_b_logit_standardization.md).

**C (DKD) ĐÃ SUBMIT (đang chạy):**
- Experiment: **C** = `LOSS_MODE=dkd`: `L = 1.0*CE + min(epoch/20,1)*(1.0*TCKD + 2.0*NCKD)`, bỏ KL+BC, T=4, LR 0.1 (ablation, không phải reproduce đầy đủ với LR 0.05). Đối chứng: run B. Log: [experiments/2026-11-03_c_dkd.md](experiments/2026-11-03_c_dkd.md).
- Code: Git commit `901ffec` (`kaggle/streamlined_kd_cifar100.py`: thêm `LOSS_MODE`, `dkd_terms`, `DKD_*`; notebook `kaggle/experiments/c_dkd.ipynb`). Test CPU: 7 chế độ cũ bit-exact; DKD khớp bản tham chiếu mdistiller đến 6 chữ số; warm-up đúng.
- Notebook: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-c-dkd — **version 1**, submit 2026-11-03 ~04:10 UTC, T4, dataset `fedesoriano/cifar100` đã xác minh bằng `kaggle kernels pull -m`, RUNNING sau >2 phút.
- Quota GPU: đã dùng 23,49h trước khi submit; còn ~6,5h/30h (refresh 2026-11-07 00:00 UTC). Sau C (~3,5–4h) còn khoảng 2,5h.
- Việc cần làm ở lần trigger tiếp theo (19:00 giờ Sài Gòn 2026-11-03): **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.** Dự kiến xong ~08:00 UTC. Nếu COMPLETE: lấy kết quả bằng MCP `list_notebook_session_output`, ghi vào file C, so với run B (70.26/70.04). Nếu loss ra NaN/phân kỳ sớm (test top-1 ~1% kéo dài): ghi lại, cân nhắc LR 0.05 (đúng config gốc) như run thứ hai nếu quota cho phép.
  Nếu DKD tốt: plan nói "chạy DKD+BC để đo phần đóng góp của SKD" (thêm BC vào DKD) — chỉ làm nếu còn quota; ưu tiên D sau khi quota refresh.

**D (DOT) — kế hoạch, KHÔNG đủ quota trước refresh 2026-11-07 00:00 UTC (chạy cần ~5–6h GPU):**
- Trong lúc chờ quota, các lần trigger sau CÓ THỂ chuẩn bị code D trên CPU (không tốn GPU): lấy implementation gốc [`mdistiller/engine/dot.py`](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/engine/dot.py) (raw.githubusercontent.com truy cập được), tích hợp vào trainer: `L_task = CE`, `L_distill = KD + BC` (loss SKD của run B), backward riêng cho từng loss, giữ hai buffer momentum riêng theo DOT, lưu cả hai buffer để resume; so với cùng loss dùng SGD (run B). Log cosine giữa hai gradient, accuracy, thời gian/epoch.
- Test CPU hồi quy: các chế độ cũ phải giữ bit-exact (như đã làm ở A3/LS/C). Chỉ submit D sau khi quota refresh (kiểm tra `kaggle quota`: cần còn ≥ 6h). Trước refresh, nếu không còn việc GPU nào khác: ghi "chờ refresh quota" vào HANDOFF và kết thúc, không tự mở rộng ngân sách.


## (Lịch sử) Trạng thái 2026-11-03 12:45 UTC — đã được thay bằng mục TRẠNG THÁI HIỆN TẠI ở cuối file (D tạm dừng, chuyển sang kế hoạch E)

**C (DKD) COMPLETE:** best **69.74%** (epoch 234), final **69.51%**, top-5 92.22%, test loss 1.0836, runtime 12.176 s (~3,38 giờ, nhanh hơn run SKD ~12%).
So với run B: **−0,52 best / −0,53 final** — không vượt B. Train top-1 cuối chỉ 64.91% (thấp hơn run SKD 1–2 điểm). Không phân kỳ. Chi tiết: [experiments/2026-11-03_c_dkd.md](experiments/2026-11-03_c_dkd.md).

### Tổng kết tạm tất cả run (1 seed = 2021; baseline B vẫn tốt nhất; paper 72.50)
| Run | Thay đổi so với B | Best | Final |
|---|---|---:|---:|
| B (mốc) | — | 70.26 | 70.04 |
| A3-bc2 | `lambda_bc=2.0` | 70.31 | 69.86 |
| A3-bc0 | `lambda_bc=0` | 70.03 | 69.85 |
| A2 | KD per-sample, mask high-confidence | 69.76 | 69.57 |
| C DKD | loss DKD (LR 0.1) | 69.74 | 69.51 |
| LS-T2 | logit std, T=2 | 69.30 | 68.96 |
| A1 | KD per-sample, mask low-confidence | 69.28 | 68.97 |
| LS | logit std, T=4 | 68.87 | 68.60 |
Kết luận tạm: không hướng nào trong A1/A2/A3/B/C vượt run B (khoảng cách tới paper ~2,2 điểm vẫn còn); nhiễu seed ước tính ~0,3 điểm nên A3 (BC) coi như không khác B.

**D (DOT) — code ĐÃ CHUẨN BỊ, CHƯA SUBMIT (chờ quota):**
- Quota GPU còn **3,12h/30h**, refresh **2026-11-07 00:00 UTC** (07:00 giờ Sài Gòn). D cần ~5–6h (hai lần backward mỗi iteration) nên không chạy được trước đó. Không submit run nào khác trong lúc chờ (không tự mở rộng ngân sách).
- Code: Git commit `3c9fea4` — `OPTIM_MODE=dot` (+`DOT_DELTA=0.075`) với optimizer DOT chép nguyên từ mdistiller (MIT); notebook `kaggle/experiments/d_dot.ipynb` (default `OPTIM_MODE=dot`); log kế hoạch/giả thuyết/rủi ro: [experiments/2026-11-03_d_dot.md](experiments/2026-11-03_d_dot.md).
  Test CPU: đường SGD bit-exact; DOT(delta=0) trùng SGD ở float64 (1.1e-15); optimizer state lưu/nạp đủ hai buffer.
- **Việc cần làm ở các lần trigger tiếp theo:**
  1. Chạy `kaggle quota` (và/hoặc MCP `get_accelerator_quota`). Nếu `remaining` < 6h: **không làm gì thêm, không commit**; trả lời ngắn "chờ refresh quota 2026-11-07" rồi kết thúc.
  2. Nếu đã refresh (remaining ≥ 6h): `git fetch` + pull, submit D bằng CLI: slug `streamlined-kd-cifar100-d-dot`, notebook `kaggle/experiments/d_dot.ipynb`, metadata như các run trước (dataset `fedesoriano/cifar100`, docker image như run B, T4); xác minh dataset attach và RUNNING sau ~2 phút; cập nhật file D + HANDOFF; commit/push.
  3. Khi D COMPLETE: ghi kết quả vào `experiments/2026-11-03_d_dot.md` (so với B; log cosine/gradnorm nếu lấy được `history.csv`), cập nhật bảng tổng kết ở trên. Nếu D phân kỳ (test top-1 ~1% kéo dài / NaN): ghi lại, cân nhắc LR 0.05 (cần thêm env `LEARNING_RATE`, mặc định 0.1 giữ nguyên bit) như lần thử thứ hai.
  4. **Plan A1→A2→A3→B→C→D khi đó hoàn tất:** tổng hợp kết quả và bài học vào HANDOFF, rồi gọi `update_trigger` với `enabled=false` cho `trig_01Sh2GBEHeb1grKfyRJfr7nk`. Các hướng mở rộng chưa nằm trong plan (nhiều seed, DKD LR 0.05, DKD+BC, LS với `kd_weight` lớn, T=1) chỉ làm khi người dùng yêu cầu.


## TRẠNG THÁI HIỆN TẠI (cập nhật 2026-11-03, theo quyết định của người dùng) — đọc mục này trước

**Thứ tự chạy mới (người dùng chỉ định, thay cho D): E1 → E2 → E3.** Plan A1→A2→A3→B→C đã xong (bảng tổng kết ở mục trước); **D (DOT) tạm dừng**,
code sẵn sàng ở commit `3c9fea4`, chỉ chạy lại khi người dùng yêu cầu.

**E — lịch learning rate, nền run B** (chi tiết, luật quyết định, ngân sách: [experiments/2026-11-03_e_lr_schedule.md](experiments/2026-11-03_e_lr_schedule.md)):
- **E1**: step LR 0.1, giảm ×0.1 ở epoch 80/150/200.
- **E2**: LR 0.05, lịch cũ 150/180/210.
- **E3**: seed 2022 cho ứng viên tốt nhất (nếu hơn run B ≥ 0,5 điểm ở epoch 240) và cho run B; nếu không có ứng viên đạt ngưỡng thì chỉ chạy run B seed 2022 để đo nhiễu.
- Đã bỏ: cosine annealing + warmup 5 epoch.

**Tài khoản Kaggle:** từ kế hoạch E chạy trên **tài khoản 2 `kitit5512trn`** (`KAGGLE_USERNAME2`/`KAGGLE_KEY2`; CLI: `export KAGGLE_API_TOKEN="$KAGGLE_KEY2"`; quota 30h, refresh 2026-11-07).
Kaggle MCP trong session đã chuyển sang tài khoản 2. Các run cũ ở tài khoản 1 `trananhkiet` đọc bằng CLI với `KAGGLE_KEY`.

**Việc cần làm tiếp theo:** thêm env `LEARNING_RATE`, `LR_MILESTONES`, `SEED` (mặc định giữ nguyên bit như run B) + test CPU, build notebook E1, submit E1 trên tài khoản 2
(xác minh dataset attach và Docker image ghim dùng được; nếu image bị từ chối thì dừng và ghi blocker, không tự đổi image).


### (Đã gỡ) Blocker E1 lúc 2026-11-03 16:35 UTC — tài khoản 2 không có GPU; người dùng đã xử lý, xem mục bên dưới
- Code E đã xong và test (commit `82cbcbd`: env `SEED`, `LEARNING_RATE`, `LR_MILESTONES`; mặc định bit-exact; bảng LR 240 epoch khớp bảng tính tay). Notebook E1: `kaggle/experiments/e1_step80_150_200.ipynb`.
- Submit E1 lên tài khoản 2 (https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-e1-step80-150-200, version 1): dataset + Docker image ghim OK, nhưng kernel chạy **`device: cpu`** và tự dừng sau ~38 giây
  (kiểm tra GPU của script). Không tốn GPU quota.
- Nguyên nhân khả dĩ nhất: tài khoản `kitit5512trn` chưa được bật GPU (thường do chưa xác minh số điện thoại). **Người dùng cần**: xác minh số điện thoại trên Kaggle cho tài khoản 2, hoặc chỉ định dùng lại tài khoản 1 (`trananhkiet`, còn 3,12h tới 2026-11-07).
- Việc ở lần trigger tiếp theo: thử push lại E1 **một lần** (chi phí ~1 phút CPU, không tốn GPU) và kiểm tra log có `device: cuda` không.
  Nếu vẫn `device: cpu` → ghi "vẫn chặn" (không commit lặp lại nếu không có gì mới) và kết thúc; không tự chuyển tài khoản. Sau 3 lần vẫn chặn → dừng thử, chờ người dùng.


### (Lịch sử) E1 đã submit lúc 2026-11-03 16:46 UTC — đã xong, xem mục bên dưới
- Run: **E1** = run B nhưng `LR_MILESTONES=80,150,200` (LR 0.1, gamma 0.1). Notebook `kaggle/experiments/e1_step80_150_200.ipynb`, code commit `82cbcbd`.
- Kaggle (tài khoản 2): https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-e1-step80-150-200 — **version 2**, submit ~16:46 UTC, RUNNING trên GPU
  (quota tài khoản 2 bắt đầu bị trừ). Version 1 trước đó ERROR vì chưa có GPU (blocker quyền, không tính retry).
- Quota tài khoản 2: 0.07h/30h dùng lúc 16:50 UTC (refresh 2026-11-07 00:00 UTC).
- **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.** Dự kiến xong ~20:45 UTC (trước lần trigger 00:27 UTC). Khi COMPLETE: đọc log bằng
  `KAGGLE_API_TOKEN="$KAGGLE_KEY2" kaggle kernels logs kitit5512trn/streamlined-kd-cifar100-e1-step80-150-200` (lọc dòng `epoch ...` và summary JSON), ghi kết quả, rồi build và submit **E2** (`LEARNING_RATE=0.05`).


### (Lịch sử) Trạng thái 2026-11-04 00:35 UTC — đã được thay bằng mục TRẠNG THÁI HIỆN TẠI bên dưới

**E1 COMPLETE — kết quả tốt nhất đến nay:** epoch 240 **70.75%** (+0,71 so với run B 70.04), trung bình 10 epoch cuối **70.68%** (B: 70.07, +0,61), best 70.79% (ep 239),
top-5 92.65%, test loss 1.0405, runtime 13.975 s (~3,88 giờ). **Đạt ngưỡng ≥ +0,5** của luật quyết định → E3 sẽ chạy ứng viên tốt nhất (E1, trừ khi E2 hơn) với seed 2022 và run B seed 2022.
Chi tiết, đường cong và bài học: [experiments/2026-11-03_e_lr_schedule.md](experiments/2026-11-03_e_lr_schedule.md). (1 seed; chờ E3 xác nhận.)

**E2 ĐÃ SUBMIT (đang chạy):**
- Run: **E2** = run B nhưng `LEARNING_RATE=0.05` (lịch 150/180/210). Notebook `kaggle/experiments/e2_lr005.ipynb` (khác nền đúng một dòng), commit `827f5a9` (code train: `82cbcbd`).
- Kaggle (tài khoản 2): https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-e2-lr005 — **version 1**, submit 2026-11-04 ~00:30 UTC; `kaggle kernels pull -m` xác nhận dataset `fedesoriano/cifar100`, Docker image ghim, T4; RUNNING sau >150 giây (không bị lỗi thiếu GPU).
- Quota tài khoản 2: 3,89h/30h dùng trước khi submit E2 (refresh 2026-11-07 00:00 UTC). Sau E2 còn ~22h, đủ cho 2 run E3 (~8h).
- **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.** Dự kiến xong ~04:30 UTC (trước trigger 08:27 UTC). Khi COMPLETE: lọc log bằng
  `kaggle kernels logs` (script mẫu đã dùng: đọc JSON, ghép stdout, regex `epoch (\d+)/240 \| lr=... \| train top1=... \| test top1=...`, khối `summary` JSON cuối), ghi kết quả,
  chọn ứng viên E3 (E1 hoặc E2 — run có epoch-240 cao hơn, cả hai phải ≥ 70.54), rồi build và submit run E3 đầu tiên: **ứng viên seed 2022** (`SEED=2022`, giữ nguyên env của ứng viên).
  Lần trigger sau đó: **run B seed 2022** (slug `streamlined-kd-cifar100-e3-runb-seed2022`).


### (Lịch sử) Trạng thái 2026-11-04 08:33 UTC — đã được thay bằng mục TRẠNG THÁI HIỆN TẠI bên dưới

**E2 COMPLETE (LR 0.05):** epoch 240 **70.14%** (+0,10 so với B), trung bình 10 epoch cuối 70.36% (+0,29), best 70.59% (ep 234), top-5 92.43%, runtime 14.220 s. **Không đạt ngưỡng.**
LR 0.05 làm giai đoạn đầu ổn định hơn nhiều nhưng kết quả cuối ~bằng B; thứ có ích ở E1 là phân bổ lại epoch (giai đoạn LR thấp dài hơn).

**E3 — ứng viên = E1** (70.75 ≥ 70.54; E2 không đạt). Gồm 2 run, mỗi trigger một run:
1. **E3 run 1 ĐÃ SUBMIT (đang chạy): E1 seed 2022** — https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-e3-e1-seed2022, **version 1**, submit 2026-11-04 ~08:29 UTC;
   notebook `kaggle/experiments/e3_e1_seed2022.ipynb` (khác notebook E1 đúng dòng `SEED` = 2022), commit `dc564b1`; dataset `fedesoriano/cifar100`, Docker image ghim, T4 đã xác minh; RUNNING sau >150 giây.
2. Chưa chạy: **run B seed 2022** — slug `streamlined-kd-cifar100-e3-runb-seed2022`, notebook cần build từ `streamlined_kd_cifar100.py` chỉ đổi `SEED` mặc định thành `2022`.
- Quota tài khoản 2: 7,86h/30h dùng trước khi submit E3 run 1; sau 2 run E3 còn ~14h (refresh 2026-11-07 00:00 UTC).
- **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.** Dự kiến xong ~12:30 UTC (trước trigger 16:27 UTC). Khi COMPLETE: ghi kết quả vào file E, rồi build + submit run B seed 2022.
- Khi cả 2 run E3 xong: so (E1 − B) ở seed 2022 với +0.71 của seed 2021; nếu cũng ≥ +0.5 → kết luận E1 tốt hơn B thật; tổng hợp bài học kế hoạch E vào HANDOFF; rồi `update_trigger enabled=false`.

**BỔ SUNG 2026-11-04 ~15:20 UTC (người dùng chỉ định): thêm E4, chạy sau E3.** (vẫn hiệu lực)
- **E4** = lịch E1 kéo dài 300 epoch (milestones 100/188/250) + **teacher chuẩn CRD/mdistiller ResNet56 72.34%** (`TEACHER_SOURCE=mdistiller`).
  Một run gộp 2 yếu tố theo yêu cầu người dùng (không tách được đóng góp từng yếu tố). Chi tiết, nguồn teacher, kiểm tra:
  [experiments/2026-11-04_e4_mdistiller_teacher_300ep.md](experiments/2026-11-04_e4_mdistiller_teacher_300ep.md).
- Code + notebook đã sẵn sàng và test: `kaggle/experiments/e4_mdteacher_300ep.ipynb` (khác notebook nền đúng 3 dòng mặc định). Mặc định của script vẫn là teacher SKD → run B seed 2022 không đổi.
- Slug `streamlined-kd-cifar100-e4-mdteacher-300ep`, ~4,9 giờ GPU; log dạng `epoch xxx/300`. Script tự dừng nếu teacher mdistiller < 72.0% trên test.
- Thứ tự: E3 run 1 → E3 run B seed 2022 → kết luận E → **E4** → tổng kết E4 → tắt trigger.


### (Lịch sử) Trạng thái 2026-11-04 16:32 UTC — đã được thay bằng mục TRẠNG THÁI HIỆN TẠI bên dưới

**E3 run 1 COMPLETE (E1 seed 2022):** epoch 240 **70.71%**, trung bình 10 epoch cuối 70.72%, best 70.88% (ep 223), top-5 92.34%, runtime 14.280 s.
So với E1 seed 2021 (70.75 / 70.68): chênh chỉ ~0,04 điểm → E1 rất ổn định giữa seed. Chi tiết: [experiments/2026-11-03_e_lr_schedule.md](experiments/2026-11-03_e_lr_schedule.md).

**E3 run 2 ĐÃ SUBMIT (đang chạy): run B seed 2022** — https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-e3-runb-seed2022, **version 1**, submit 2026-11-04 ~16:28 UTC;
notebook `kaggle/experiments/e3_runb_seed2022.ipynb` (khác notebook nền đúng dòng `SEED` = 2022), commit `d662170`; `kaggle kernels pull -m` xác nhận dataset `fedesoriano/cifar100`, Docker image ghim, T4, internet bật; RUNNING.
- Quota tài khoản 2: 11,85h/30h dùng lúc 16:30 UTC (trước khi run này tính giờ); sau run này ~15,8h, còn ~14h cho E4 (~4,9h). Refresh 2026-11-07 00:00 UTC.
- **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.** Dự kiến xong ~20:30 UTC (trước trigger 00:27 UTC).
- Khi COMPLETE: ghi kết quả; so (E1 − B) ở seed 2022 (E1 = 70.71) với ngưỡng +0.5 (tức run B seed 2022 ≤ 70.21 thì E1 thắng ở cả 2 seed); ghi kết luận kế hoạch E;
  rồi submit **E4** (`kaggle/experiments/e4_mdteacher_300ep.ipynb`, slug `streamlined-kd-cifar100-e4-mdteacher-300ep`, xem đoạn BỔ SUNG ở trên).


### (Lịch sử) Trạng thái 2026-11-05 00:35 UTC — đã được thay bằng mục TRẠNG THÁI HIỆN TẠI bên dưới

**E3 run 2 COMPLETE (run B seed 2022):** epoch 240 **70.05%**, trung bình 10 epoch cuối 69.98%, best 70.16% (ep 239), top-5 92.51%, runtime 14.479 s.

**KẾ HOẠCH E (E1–E3) HOÀN TẤT — E1 tốt hơn run B ở cả 2 seed:**

| Seed | Run B | E1 (80/150/200) | E1 − B (ep 240) | E1 − B (TB 10 cuối) |
|---|---:|---:|---:|---:|
| 2021 | 70.04 | 70.75 | +0.71 | +0.61 |
| 2022 | 70.05 | 70.71 | +0.66 | +0.74 |

Đạt luật quyết định (≥ +0.5 ở cả 2 seed). Nhiễu seed chỉ ~0.01–0.1 điểm. **Lịch 80/150/200 là nền mới.** E2 (LR 0.05) không đạt (+0.10).
Chi tiết: [experiments/2026-11-03_e_lr_schedule.md](experiments/2026-11-03_e_lr_schedule.md).

**Bảng tổng kết kế hoạch E** (test top-1 epoch 240, teacher SKD 71.59%):
| Run | Seed | Final | TB 10 cuối | Best |
|---|---|---:|---:|---:|
| B (150/180/210) | 2021 | 70.04 | 70.07 | 70.26 |
| B | 2022 | 70.05 | 69.98 | 70.16 |
| E1 (80/150/200) | 2021 | **70.75** | 70.68 | 70.79 |
| E1 | 2022 | **70.71** | 70.72 | 70.88 |
| E2 (LR 0.05) | 2021 | 70.14 | 70.36 | 70.59 |

**E4 ĐÃ SUBMIT (đang chạy):** teacher mdistiller 72.34% + lịch E1 kéo dài 300 epoch (100/188/250), seed 2021.
- https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-e4-mdteacher-300ep, **version 1**, submit 2026-11-05 ~00:28 UTC; notebook `kaggle/experiments/e4_mdteacher_300ep.ipynb` (commit `1d6054a`).
  `kaggle kernels pull -m` xác nhận dataset, Docker image ghim, T4, internet bật; RUNNING.
- Quota tài khoản 2: 15,87h/30h dùng trước khi submit; sau E4 ~20,8h. Refresh 2026-11-07 00:00 UTC.
- **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.** Dự kiến xong ~05:20 UTC (trước trigger 08:27 UTC).
- Khi COMPLETE: kiểm tra teacher top-1 (~72.3–72.4), ghi kết quả vào file E4 + HANDOFF (so với E1 70.75/70.68 và paper 72.50; E4 đổi 2 yếu tố), rồi `update_trigger enabled=false`.
  Nếu ERROR vì teacher < 72.0: không retry, ghi blocker.


### (Lịch sử) Trạng thái 2026-11-05 08:35 UTC — kế hoạch E + E4 hoàn tất; đã được thay bằng mục TRẠNG THÁI HIỆN TẠI bên dưới

**E4 COMPLETE** (teacher mdistiller + lịch E1 kéo dài 300 epoch 100/188/250, seed 2021):
teacher đo được **72.41%**; student epoch 300 **71.70%**, trung bình 10 epoch cuối **71.73%**, best **72.10%** (ep 291), top-5 93.02%, runtime 16.959 s.
So với E1: +0.95 (epoch cuối), +1.05 (TB 10 cuối). Cách paper SKD (72.50): −0.80 (epoch cuối), −0.40 (best). E4 đổi 2 yếu tố → không tách được đóng góp. Chi tiết:
[experiments/2026-11-04_e4_mdistiller_teacher_300ep.md](experiments/2026-11-04_e4_mdistiller_teacher_300ep.md).

**Bảng tổng kết E + E4** (test top-1):
| Run | Teacher | Epoch | Seed | Final | TB 10 cuối | Best |
|---|---|---:|---|---:|---:|---:|
| B (150/180/210) | SKD 71.59 | 240 | 2021 | 70.04 | 70.07 | 70.26 |
| B | SKD 71.59 | 240 | 2022 | 70.05 | 69.98 | 70.16 |
| E1 (80/150/200) | SKD 71.59 | 240 | 2021 | 70.75 | 70.68 | 70.79 |
| E1 | SKD 71.59 | 240 | 2022 | 70.71 | 70.72 | 70.88 |
| E2 (LR 0.05) | SKD 71.59 | 240 | 2021 | 70.14 | 70.36 | 70.59 |
| **E4** (100/188/250) | **mdistiller 72.41** | **300** | 2021 | **71.70** | **71.73** | **72.10** |

- Scheduler v5 (`trig_01Sh2GBEHeb1grKfyRJfr7nk`) đã được tắt (`enabled=false`) theo kế hoạch. Không còn run nào đang chạy.
- Quota tài khoản 2: 20,6h/30h đã dùng lúc 08:35 UTC (còn ~9,4h, đủ ~1 run 300 epoch hoặc 2 run 240 epoch); refresh 2026-11-07 00:00 UTC.
- Việc tiếp theo (chờ người dùng quyết định): ví dụ tách đóng góp của E4 (teacher mdistiller + 240 epoch lịch E1), thêm seed cho E4, hoặc quay lại D (DOT) trên nền mới.


### QUY ƯỚC CHỈ SỐ — CẬP NHẬT 2026-11-11 (người dùng): từ nay so sánh giữa các run bằng **best top-1**
**Thay đổi so với quy ước 2026-11-05 bên dưới:** mọi so sánh và luật quyết định (ngưỡng sàng lọc +0.2 / +0.5 …) dùng **best top-1** (max trên các epoch). Vẫn ghi thêm epoch cuối và TB 10 epoch cuối
trong bảng để tham khảo, nhưng không dùng chúng để ra quyết định. Lưu ý cố hữu của best top-1 (đã nêu với người dùng): CIFAR-100 không có tập validation, best là max của ~300 lần đo trên chính tập test
→ lạc quan ~0.2–0.4 điểm, và chênh lệch giữa các run nhỏ hơn dải này không đáng tin; đây là lựa chọn của người dùng.
Bảng best top-1 (seed 2021, teacher mdistiller, 300 epoch): E4 72.10 · F1 **72.32** · F2 72.23 · F3 71.97 · G1 72.09 · F4 **72.32** · H1 72.20 · paper SKD 72.50.
Hệ quả: theo best, ngưỡng sàng lọc của H = E4 + 0.2 = 72.30 → H1 (72.20, +0.10) **không đạt** (theo epoch cuối trước đây thì đạt: 71.97 ≥ 71.90).

(Quy ước gốc 2026-11-05, giữ để tham chiếu lịch sử:)
#### QUY ƯỚC CHỈ SỐ (người dùng thống nhất 2026-11-05) — áp dụng cho mọi bảng từ nay
Mọi bảng kết quả ghi **cả hai**: best top-1 và top-1 epoch cuối (kèm trung bình 10 epoch cuối).
- **So với paper → dùng best top-1.** Code gốc repo SKD (`CIFAR-100/Utils/train.py`, hàm `test()`) theo dõi `best_acc` trên tập test và lưu checkpoint ở epoch tốt nhất;
  CRD/RepDistiller cũng in "best accuracy". Nên số paper (SKD 72.50) nhiều khả năng là best top-1.
- **So giữa các run của ta (ra quyết định, luật ngưỡng +0.5) → dùng top-1 epoch cuối và trung bình 10 epoch cuối.** CIFAR-100 không có tập validation;
  best top-1 là max của ~240–300 lần đo trên chính tập test nên bị selection bias (lạc quan ~0.2–0.4 điểm, ví dụ E4: best 72.10 ở ep 291 trong khi các epoch quanh đó ~71.6–71.9).

So với paper theo best top-1:
| Run | Teacher | Epoch | Best top-1 | Cách paper (72.50) |
|---|---|---:|---:|---:|
| Run B (s2021) | SKD 71.59 | 240 | 70.26 | −2.24 |
| E1 (s2021) | SKD 71.59 | 240 | 70.79 | −1.71 |
| E1 (s2022) | SKD 71.59 | 240 | 70.88 | −1.62 |
| **E4** (s2021) | mdistiller 72.41 | 300 | **72.10** (ep 291) | **−0.40** |


### (Lịch sử) Trạng thái 2026-11-05 10:50 UTC — F1 đã submit; đã được thay bằng mục TRẠNG THÁI HIỆN TẠI bên dưới

**F1 ĐÃ SUBMIT (đang chạy): trọng số CE/KD theo từng mẫu dựa vào việc teacher đoán đúng/sai** — một yếu tố so với E4.
- Công thức: w_i = 0.6 nếu teacher đoán đúng ảnh i, 0.4 nếu sai; L = mean[(1 − w_i)·CE_i] + mask·mean[w_i·KL_i] + 0.5·BC.
  Env `SAMPLE_WEIGHT_MODE=teacher_correct`, `SW_TRUST=0.6`, `SW_DISTRUST=0.4` (mặc định `none` = hành vi cũ, đã test bit-exact).
- Nền = E4: teacher mdistiller 72.41%, 300 epoch, milestones 100/188/250, seed 2021. Chi tiết, nghiên cứu liên quan, test:
  [experiments/2026-11-05_f1_teacher_correct_weight.md](experiments/2026-11-05_f1_teacher_correct_weight.md).
- Kaggle (tài khoản 2): https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-f1-teacher-weight — **version 1**, submit 2026-11-05 ~10:43 UTC;
  notebook `kaggle/experiments/f1_teacher_correct_weight.ipynb` (khác cấu hình E4 đúng dòng `SAMPLE_WEIGHT_MODE`), commit `9090d82`.
- Quota tài khoản 2: 20,6h/30h dùng trước khi submit; sau F1 ~25,4h. Refresh 2026-11-07 00:00 UTC.
- **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.** Dự kiến xong ~15:35 UTC (trước trigger 16:27 UTC).
- Luật quyết định (so với E4 seed 2021: 71.70 epoch cuối / 71.73 TB 10 cuối / 72.10 best):
  - F1 epoch cuối ≥ **72.20** (E4 + 0.5) → sau khi quota refresh chạy **F1 seed 2022** rồi **E4 seed 2022** (mỗi trigger một run), so cặp cùng seed.
  - Không đạt → ghi kết luận F1, tắt scheduler, chờ người dùng (có thể thử bản mềm kiểu CCKD hoặc mức 0.7/0.3).
- Ghi kèm khi đọc log: `teacher_train_top1_sw` (tỉ lệ mẫu nhận KD 0.6) ở vài epoch đầu/giữa/cuối nếu có trong log; best top-1 so với paper 72.50.


### TRẠNG THÁI HIỆN TẠI — cập nhật 2026-11-05 16:35 UTC (23:35 giờ Sài Gòn) — F1 xong, KHÔNG đạt ngưỡng (sát), scheduler đã tắt

**F1 COMPLETE** (trọng số CE/KD theo teacher đúng/sai 0.6/0.4, nền E4): epoch cuối **72.13%** (+0.43 so với E4 71.70), TB 10 cuối 72.14 (+0.41),
best **72.32%** (ep 293; cách paper 72.50 chỉ −0.18), top-5 92.83, runtime 17.332 s. Teacher 72.41%.
- Luật: cần ≥ 72.20 → **không đạt (thiếu 0.07)**; không tự chạy seed 2022. Tín hiệu nhất quán tốt (cao hơn E4 ở mọi mốc giảm LR). Chi tiết:
  [experiments/2026-11-05_f1_teacher_correct_weight.md](experiments/2026-11-05_f1_teacher_correct_weight.md).

**Bảng tổng kết (test top-1, theo QUY ƯỚC CHỈ SỐ):**
| Run | Teacher | Epoch | Seed | Final | TB 10 cuối | Best |
|---|---|---:|---|---:|---:|---:|
| B | SKD 71.59 | 240 | 2021 | 70.04 | 70.07 | 70.26 |
| E1 (80/150/200) | SKD 71.59 | 240 | 2021 | 70.75 | 70.68 | 70.79 |
| E4 (100/188/250) | mdistiller 72.41 | 300 | 2021 | 71.70 | 71.73 | 72.10 |
| **F1** (E4 + w teacher đúng/sai 0.6/0.4) | mdistiller 72.41 | 300 | 2021 | **72.13** | **72.14** | **72.32** |

- Scheduler v5 (`trig_01Sh2GBEHeb1grKfyRJfr7nk`) đã tắt (`enabled=false`). Không còn run nào đang chạy.
- Quota tài khoản 2: ~25,4h/30h đã dùng; refresh 2026-11-07 00:00 UTC.
- Chờ người dùng chọn: (1) F1 seed 2022 + E4 seed 2022 để xác nhận; (2) mức 0.7/0.3; (3) bản mềm kiểu CCKD.


### BỔ SUNG 2026-11-05 16:55 UTC — kế hoạch F2/F3 trên tài khoản 3 (người dùng chỉ định)
- **Tài khoản Kaggle 3** `trananhkiet5551` (`KAGGLE_USERNAME3`/`KAGGLE_KEY3`; CLI: `export KAGGLE_API_TOKEN="$KAGGLE_KEY3"`). Kaggle MCP đã chuyển sang tài khoản 3.
- **F2**: hard 0.7/0.3 (`SAMPLE_WEIGHT_MODE=teacher_correct`, `SW_TRUST=0.7`, `SW_DISTRUST=0.3`). **F3**: bản mềm kiểu CCKD
  w_i = 0.4 + 0.2·p_teacher(nhãn đúng) (`SAMPLE_WEIGHT_MODE=teacher_prob`). Nền = F1/E4 (teacher mdistiller, 300 epoch, 100/188/250, seed 2021).
  Chi tiết: [experiments/2026-11-05_f2_f3_weight_variants.md](experiments/2026-11-05_f2_f3_weight_variants.md). Code + test CPU xong (bit-exact cho các chế độ cũ).
- **BLOCKER:** smoke test trên tài khoản 3 chạy `device: cpu` dù đã yêu cầu T4 → tài khoản chưa được bật GPU (cần xác minh số điện thoại). Chưa submit F2/F3.


### (Lịch sử) Trạng thái 2026-11-05 17:35 UTC — F2 đã submit; đã được thay bằng mục TRẠNG THÁI HIỆN TẠI bên dưới
- Tài khoản 3 vẫn chưa có GPU → dùng lại **tài khoản 2** (`KAGGLE_KEY2`; Kaggle MCP đã kết nối lại tài khoản 2).
- **F2 ĐÃ SUBMIT (đang chạy):** hard 0.7/0.3 trên nền F1 — https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-f2-hard-07-03,
  version 1, ~17:32 UTC, notebook `kaggle/experiments/f2_hard_07_03.ipynb`, commit `db45c42`.
- Quota còn ~4,57h < ~4,8h cần → **dự kiến bị dừng ở ~epoch 284–290** (~22:10 UTC). Người dùng chấp nhận. So với F1 tại cùng mốc epoch cuối F2 đạt được.
- Sau đó: **F3** (bản mềm CCKD, `SAMPLE_WEIGHT_MODE=teacher_prob`) sau khi quota refresh 2026-11-07 00:00 UTC.
- **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.**


### (Lịch sử) Trạng thái 2026-11-06 01:35 UTC — F2 xong; đã được thay bằng mục TRẠNG THÁI HIỆN TẠI bên dưới
**F2 COMPLETE (đủ 300 epoch, Kaggle cho vượt quota):** epoch cuối **71.90**, TB 10 cuối 72.04, best 72.23 (ep 292), top-5 92.91, runtime 17.457 s.
Kém F1 (−0.23 / −0.10 / −0.09), hơn E4 (+0.20 / +0.31 / +0.13). Lệch mạnh 0.7/0.3 không tốt hơn 0.6/0.4. Không đạt ngưỡng ứng viên (< 72.20).
Teacher đoán đúng ~69.2% ảnh train (đo lần đầu qua log). Chi tiết: [experiments/2026-11-05_f2_f3_weight_variants.md](experiments/2026-11-05_f2_f3_weight_variants.md).

| Run | Trọng số theo mẫu | Final (ep 300) | TB 10 cuối | Best |
|---|---|---:|---:|---:|
| E4 | không | 71.70 | 71.73 | 72.10 |
| **F1** | hard 0.6/0.4 | **72.13** | **72.14** | **72.32** |
| F2 | hard 0.7/0.3 | 71.90 | 72.04 | 72.23 |

- Quota tài khoản 2: 109.041 s / 108.000 s (hết). **F3 (bản mềm CCKD) sẽ submit ở lần trigger 01:27 UTC ngày 07/11** (sau refresh 00:00 UTC).
- **Chờ quota refresh 2026-11-07 00:00 UTC.**


### (Lịch sử) Trạng thái 2026-11-07 00:10 UTC — F3 đã submit; đã được thay bằng mục TRẠNG THÁI HIỆN TẠI bên dưới
- Lịch scheduler v5 (người dùng chỉ định): cron "1 0,5,10,15,20 * * *" UTC = 07:01, 12:01, 17:01, 22:01, 03:01 giờ Sài Gòn.
- Quota tài khoản 2 đã refresh (0 s / 108.000 s; refresh tiếp 2026-11-14 00:00 UTC).
- **F3 ĐÃ SUBMIT (đang chạy):** bản mềm kiểu CCKD w_i = 0.4 + 0.2·p_teacher(nhãn đúng) trên nền F1 —
  https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-f3-soft-cckd, version 1, ~00:03 UTC, notebook `kaggle/experiments/f3_soft_cckd.ipynb`, commit `b7963cf`.
- So với F1 (72.13 / 72.14 / 72.32), F2 (71.90 / 72.04 / 72.23), E4 (71.70 / 71.73 / 72.10). Dự kiến xong ~04:55 UTC.
- **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.** Khi xong: tổng kết F1/F2/F3 rồi tắt scheduler.


### (Lịch sử) Trạng thái 2026-11-07 06:10 UTC — kế hoạch F hoàn tất; đã được thay bằng mục TRẠNG THÁI HIỆN TẠI bên dưới
**F3 COMPLETE** (bản mềm kiểu CCKD): epoch cuối 71.73, TB 10 cuối 71.77, best 71.97 (ep 299), runtime 19.368 s. p_label TB 0.645 → KD TB ≈ 0.53.
≈ E4, kém F1 (−0.40). Chi tiết: [experiments/2026-11-05_f2_f3_weight_variants.md](experiments/2026-11-05_f2_f3_weight_variants.md).

**Tổng kết F (seed 2021, nền E4: teacher mdistiller, 300 epoch, 100/188/250):**
| Run | Trọng số theo mẫu | Final | TB 10 cuối | Best |
|---|---|---:|---:|---:|
| E4 | không | 71.70 | 71.73 | 72.10 |
| **F1** | hard 0.6/0.4 | **72.13** | **72.14** | **72.32** |
| F2 | hard 0.7/0.3 | 71.90 | 72.04 | 72.23 |
| F3 | mềm 0.4 + 0.2·p | 71.73 | 71.77 | 71.97 |

- F1 tốt nhất; không biến thể nào đạt ngưỡng ứng viên ≥ 72.20 ở epoch cuối. Tín hiệu đúng/sai rõ với mức lệch vừa phải có ích; lệch mạnh hoặc làm mềm thì không.
- Scheduler v5 (`trig_01Sh2GBEHeb1grKfyRJfr7nk`) đã tắt (`enabled=false`). Không còn run nào đang chạy.
- Quota tài khoản 2 tuần này: ~5,4h/30h đã dùng (F3); refresh 2026-11-14 00:00 UTC. Tài khoản 3 vẫn chưa có GPU.
- Chờ người dùng chọn bước tiếp: F1 + E4 seed 2022; mức lệch 0.55/0.45; hoặc F4 (chuyển dần sang học nhãn).


### (Lịch sử) Trạng thái 2026-11-07 07:10 UTC — G1 đang chạy; đã được thay bằng mục TRẠNG THÁI HIỆN TẠI bên dưới
- **G1** (nền E4): sửa nhãn mềm teacher ở ảnh nó đoán sai bằng cách hoán đổi logit lớp sai ↔ lớp đúng (`TEACHER_CORRECTION=swap`). Notebook `kaggle/experiments/g1_teacher_swap.ipynb`.
- **F4** (nền F1): từ epoch 251 nhân trọng số KD (w_i) và BC với 0.5, CE nhận phần còn lại (`LATE_KD_START=251`, `LATE_KD_SCALE=0.5`). Notebook `kaggle/experiments/f4_late_ce.ipynb`.
- Chi tiết, kiểm tra, luật quyết định: [experiments/2026-11-07_g1_f4_teacher_fix_and_late_ce.md](experiments/2026-11-07_g1_f4_teacher_fix_and_late_ce.md). Code + test CPU xong (các chế độ cũ bit-exact).
- Tài khoản 2 (quota ~5,4h/30h; refresh 2026-11-14 00:00 UTC). Tài khoản 3 vẫn chưa có GPU.
- Scheduler v5 (người dùng chỉ định): lần đầu **14:00 giờ Sài Gòn 07/11**, sau đó mỗi 5 giờ (chuỗi hẹn một lần). Thứ tự: submit G1 → đọc G1, submit F4 → đọc F4, tổng kết, tắt trigger.
- **G1 ĐÃ SUBMIT (đang chạy):** https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-g1-teacher-swap — version 1, 2026-11-07 07:00 UTC, commit `03bfda1`;
  dataset, Docker image ghim, T4 đã xác minh, RUNNING. Quota tài khoản 2: 5,4h/30h trước run này. Dự kiến xong ~12:00–12:30 UTC.
- Lịch kế tiếp (giờ Sài Gòn, mỗi 5h): 19:00 → 00:00 → 05:00 → 10:00. F4 submit khi G1 đọc xong (khả năng ở lần 00:00 ngày 08/11).
- **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.**


### (Lịch sử) Trạng thái 2026-11-07 13:10 UTC — G1 xong, F4 đang chạy; đã được thay bằng mục TRẠNG THÁI HIỆN TẠI bên dưới
**G1 COMPLETE** (hoán đổi logit teacher khi sai, nền E4): epoch cuối **71.44**, TB 10 cuối **71.77**, best **72.09** (ep 285), top-5 93.16, runtime 18.380 s.
≈ E4 (71.70 / 71.73 / 72.10), kém F1 (72.13 / 72.14 / 72.32); không đạt ngưỡng. Teacher sai ở 30.8% ảnh train. Chi tiết: [experiments/2026-11-07_g1_f4_teacher_fix_and_late_ce.md](experiments/2026-11-07_g1_f4_teacher_fix_and_late_ce.md).

**F4 ĐÃ SUBMIT (đang chạy):** nền F1 + từ epoch 251 nhân KD/BC với 0.5 — https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-f4-late-ce, version 1, 2026-11-07 13:05 UTC,
notebook `kaggle/experiments/f4_late_ce.ipynb` (commit `03bfda1`). Quota tài khoản 2: 10,5h/30h trước F4 (còn ~19,5h). Dự kiến xong ~18:05–18:30 UTC.

| Run | Final | TB 10 cuối | Best |
|---|---:|---:|---:|
| E4 | 71.70 | 71.73 | 72.10 |
| F1 | 72.13 | 72.14 | 72.32 |
| G1 | 71.44 | 71.77 | 72.09 |
| F4 | (đang chạy) | | |

- Scheduler v5: cron "CRON_TZ=Asia/Saigon 0 2,8,14,20 * * *" (mỗi 6h). Lần kế tiếp 02:00 giờ Sài Gòn 08/11: đọc F4, tổng kết E4/F1/G1/F4, tắt scheduler.
- **Đang chờ kết quả; lần trigger tiếp theo kiểm tra run này trước.**


### TRẠNG THÁI HIỆN TẠI — cập nhật 2026-11-07 19:15 UTC (02:15 giờ Sài Gòn 08/11) — KẾ HOẠCH G1 / F4 HOÀN TẤT, scheduler đã tắt
**F4 COMPLETE** (F1 + giảm KD/BC ×0.5 từ epoch 251): epoch cuối **71.83**, TB 10 cuối **72.03**, best **72.32** (ep 297), top-5 92.64, train top-1 cuối 69.36 (F1: 67.81), runtime 17.046 s.
Trùng F1 từng số tới epoch 250 rồi tách đúng tại 251; sau đó không tốt hơn F1 (−0.30 / −0.11 / 0.00) dù train top-1 tăng +1.55 → học nhãn nhiều hơn chỉ làm student khớp train hơn, không tăng test.

**Bảng tổng kết (seed 2021, teacher mdistiller 72.41, 300 epoch, 100/188/250):**
| Run | Thay đổi | Epoch cuối | TB 10 cuối | Best |
|---|---|---:|---:|---:|
| E4 | không | 71.70 | 71.73 | 72.10 |
| **F1** | teacher đúng/sai 0.6/0.4 | **72.13** | **72.14** | **72.32** |
| F2 | 0.7/0.3 | 71.90 | 72.04 | 72.23 |
| F3 | mềm kiểu CCKD | 71.73 | 71.77 | 71.97 |
| G1 | hoán đổi logit khi teacher sai | 71.44 | 71.77 | 72.09 |
| F4 | F1 + giảm KD/BC từ ep 251 | 71.83 | 72.03 | 72.32 |

- **F1 vẫn tốt nhất**; không biến thể nào (F2, F3, G1, F4) vượt F1 hay đạt ngưỡng ≥ 72.20. Nhiễu giữa epoch cuối ~0.2–0.3 → cần ≥ 2 seed để phân biệt. Chi tiết: [experiments/2026-11-07_g1_f4_teacher_fix_and_late_ce.md](experiments/2026-11-07_g1_f4_teacher_fix_and_late_ce.md).
- Scheduler v5 (`trig_01Sh2GBEHeb1grKfyRJfr7nk`) đã tắt (`enabled=false`). Không còn run nào đang chạy. Quota tài khoản 2 ~15,2h/30h đã dùng (còn ~14,8h; refresh 2026-11-14 00:00 UTC). Tài khoản 3 vẫn chưa có GPU.
- Chờ người dùng chọn: F1 seed 2022 + E4 seed 2022 (~9,7h) để xác nhận; hoặc hướng khác ngoài loss.


### TRẠNG THÁI HIỆN TẠI — cập nhật 2026-11-09 (kế hoạch H1 / H2 — RW-DIR) — chờ scheduler submit H1
Người dùng chọn (2026-11-09) kế hoạch **H1 / H2**: nhân ma trận chênh lệch Gram của BC với trọng số độ tin cậy W_ij = sqrt(r_i r_j)/mean (r = 1 nếu teacher đúng, r_wrong nếu sai),
rồi đưa vào Mahalanobis gốc của SKD. Nền **E4** (không dùng trọng số CE/KD của F1). H1: r_wrong = 0.5; H2: r_wrong = 0.25.
Chi tiết, kiểm tra, rủi ro, luật quyết định: [experiments/2026-11-09_h1_h2_reliability_weighted_gram.md](experiments/2026-11-09_h1_h2_reliability_weighted_gram.md).
- Code: `GRAM_WEIGHT_MODE` / `GRAM_R_WRONG` trong `kaggle/streamlined_kd_cifar100.py` (mặc định tắt, bit-exact với E4/F1; `r_wrong=1.0` bit-exact E4). Test CPU + smoke test đạt.
- Notebook: `kaggle/experiments/h1_rw_dir_05.ipynb` (slug `streamlined-kd-cifar100-h1-rw-dir-05`), `kaggle/experiments/h2_rw_dir_025.ipynb` (slug `streamlined-kd-cifar100-h2-rw-dir-025`). Không build lại.
- Scheduler v5 (`trig_01Sh2GBEHeb1grKfyRJfr7nk`) bật lại: cron `CRON_TZ=Asia/Saigon 0 1,7,13,19 * * *` (mỗi 6h, lần đầu 01:00 giờ Sài Gòn 10/11). Thứ tự: submit H1 → đọc H1, submit H2 → đọc H2, tổng kết.
- Quota tài khoản 2 ~15,2h/30h đã dùng (còn ~14,8h; refresh 2026-11-14 00:00 UTC). Tài khoản 3 vẫn chưa có GPU.
- Đối chứng: E4 71.70 / 71.73 / 72.10; F1 72.13 / 72.14 / 72.32 (epoch cuối / TB 10 cuối / best). Ngưỡng sàng lọc: epoch cuối ≳ 71.90 (E4 + 0.2).
- Rủi ro: BC đóng góp rất nhỏ (A3), nên H1/H2 có thể không phân biệt được với E4.
- **H1 ĐÃ SUBMIT (QUEUED):** https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-h1-rw-dir-05 — version 1, 2026-11-09 18:05 UTC, notebook `kaggle/experiments/h1_rw_dir_05.ipynb` (commit `48b47c1`);
  dataset, Docker image ghim, T4 đã xác minh. Quota tài khoản 2: 15,26h/30h trước H1. Dự kiến xong ~4,8h sau khi bắt đầu chạy.
- Ghi chú: người dùng được hỏi có muốn đổi thứ tự (multi-seed E4/F1 trước) hay chỉ chạy H1 không; chưa có trả lời nên tiếp tục đúng kế hoạch ban đầu (H1 → H2).
- 2026-11-10 00:10 UTC (07:10 giờ Sài Gòn): H1 vẫn **QUEUED** sau ~6h kể từ lúc submit; quota GPU tài khoản 2 không đổi (54.948s dùng, 0s reserved) nghĩa là chưa có phiên GPU nào bắt đầu.
  Không tạo run trùng, không submit H2. Nếu sang lần trigger sau (13:00 giờ Sài Gòn) vẫn QUEUED, đây là hàng đợi Kaggle bất thường: ghi blocker và nêu cho người dùng quyết định (không tự submit lại).
- 2026-11-10 01:28 UTC (người dùng cho phép hủy rồi push lại nếu vẫn kẹt): `kaggle kernels delete` lỗi phía Kaggle; không lấy được session ID để hủy riêng. Đã push lại cùng notebook → **version 2**, chuyển **RUNNING** ngay (quota GPU bắt đầu tính).
  Metadata xác minh lại (dataset, Docker ghim, T4). Lưu ý: version 1 có thể vẫn nằm hàng đợi; nếu thấy hai phiên chạy song song (quota tăng nhanh gấp đôi) thì hủy phiên v1 và ghi lại. H1 chạy từ ~01:28 UTC, dự kiến xong ~06:15 UTC.
- **H1 COMPLETE** (version 2, nền E4 + RW-DIR r_wrong 0.5): epoch cuối **71.97**, TB 10 cuối **71.95**, best **72.20** (ep 272), top-5 93.06, train top-1 67.34, runtime 19.334s (~5,37h).
  So E4 (71.70 / 71.73 / 72.10): **+0.27 / +0.22 / +0.10** → đạt ngưỡng sàng lọc ≥ 71.90 nhưng thấp hơn F1 (72.13). gradnorm_bc chỉ ~0.005 so với KD ~0.52 (BC ~1% gradient) nên nhiều khả năng một phần là nhiễu quỹ đạo.
  Teacher sai ở ~31% ảnh train, W ∈ [0.605, 1.21]. Chi tiết: [experiments/2026-11-09_h1_h2_reliability_weighted_gram.md](experiments/2026-11-09_h1_h2_reliability_weighted_gram.md).
- **H2 ĐÃ SUBMIT (RUNNING):** https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-h2-rw-dir-025 — version 1, 2026-11-10 ~07:40 UTC, notebook `kaggle/experiments/h2_rw_dir_025.ipynb` (commit `48b47c1`);
  dataset, Docker image ghim, T4 đã xác minh. Quota tài khoản 2: 20,65h/30h trước H2 (còn 9,35h; sau H2 còn ~4h < 5,5h nên seed 2022 phải chờ refresh 2026-11-14 00:00 UTC). Dự kiến xong ~13:00 UTC.
- Scheduler đang chạy mỗi giờ (cron `32 * * * *` UTC) theo yêu cầu người dùng.
- **H2 ĐÃ HỦY (2026-11-10 ~08:45 UTC, theo yêu cầu người dùng):** xóa kernel bằng `kaggle kernels delete`; quota ngừng tăng (tổng 22,27h/30h đã dùng, còn **7,73h**; refresh 2026-11-14 00:00 UTC). Không còn run nào chạy.
- Scheduler v6 đã **tắt** (`enabled=false`) để không tự submit lại H2. Chờ người dùng chọn: dùng ~7,7h còn lại cho **một** run (đủ một run, không đủ hai): F1 seed 2022 (kiểm chứng run tốt nhất) hoặc E4 seed 2022; run còn lại sau 14/11.
- Ghi chú: F1 (72.13) vẫn là run tốt nhất; H1 (71.97) đứng thứ hai trong các run seed 2021; cả hai chưa được xác nhận bằng seed khác.


### TRẠNG THÁI HIỆN TẠI — cập nhật 2026-11-11 — F1 SEED 2022 ĐÃ SUBMIT (QUEUED)
Người dùng yêu cầu chạy thêm 1 seed cho F1 (cấu hình tương tự). **F1 seed 2022** đã submit: https://www.kaggle.com/code/kitit5512trn/streamlined-kd-cifar100-f1-seed2022 — version 1,
notebook `kaggle/experiments/f1_teacher_correct_weight_seed2022.ipynb` (= notebook F1, chỉ đổi `SEED` mặc định 2022). Dataset, Docker ghim, T4 đã xác minh. Chi tiết: [experiments/2026-11-11_f1_seed2022.md](experiments/2026-11-11_f1_seed2022.md).
- Quy ước chỉ số hiện hành: **best top-1** (xem mục "QUY ƯỚC CHỈ SỐ — CẬP NHẬT 2026-11-11"). Đối chứng seed 2021: F1 best 72.32 (final 72.13); E4 best 72.10 (final 71.70).
- Quota tài khoản 2: 22,27h/30h trước run (còn 7,73h; refresh 2026-11-14 00:00 UTC). Sau run còn ~2,3h → E4 seed 2022 (để so cặp) phải chờ refresh.
- H2 đã hủy, H1 xong (best 72.20). Không còn kế hoạch H.
- **F1 seed 2022 COMPLETE (2026-11-11):** best **72.45** (ep 291), epoch cuối **72.34**, TB 10 cuối **72.27**, top-5 92.98, train top-1 67.52, runtime 16.670 s (~4,63h). Teacher 72.41/92.50.
  So F1 seed 2021 (72.32 / 72.13 / 72.14) và E4 seed 2021 (72.10 / 71.70 / 71.73): **F1 lặp lại và cao hơn ở seed 2022**; trung bình best hai seed = 72.39; **kém paper SKD (72.50) chỉ 0.05**. Chi tiết: [experiments/2026-11-11_f1_seed2022.md](experiments/2026-11-11_f1_seed2022.md).
- Giới hạn: chưa có E4 seed 2022 → chưa so cặp F1 vs E4 ở seed 2022. Quota tài khoản 2: 26,91h/30h đã dùng (còn ~3,1h) → **E4 seed 2022 chờ quota refresh 2026-11-14 00:00 UTC**.
- Scheduler v7 đã tắt (`enabled=false`). Không còn run nào đang chạy. Chờ người dùng quyết định bước tiếp theo.



### TRẠNG THÁI HIỆN TẠI — cập nhật 2026-11-11 (kế hoạch I1: bỏ `/C` trong Gram của BC) — **ĐÃ CHUYỂN SANG TÀI KHOẢN KAGGLE 1**
Người dùng chuyển sang **tài khoản 1** (`trananhkiet`, CLI: `export KAGGLE_API_TOKEN="$KAGGLE_KEY"`; quota 0/30h đã dùng, refresh 2026-11-14 00:00 UTC). Tài khoản 2 (`kitit5512trn`) chỉ còn ~3,1h: không dùng cho kế hoạch này.
Kế hoạch **I1** (người dùng muốn kết luận trong ~8h): sóng 1 = run thăm dò 10 epoch để chọn λ_BC; sóng 2 = hai run đầy đủ F1 + GRAM_DIV=1 + LAMBDA_BC=λ* (seed 2022 và 2021, song song).
Chi tiết, quy tắc chọn λ*, quy tắc kết luận: [experiments/2026-11-11_i1_gram_div.md](experiments/2026-11-11_i1_gram_div.md). Code: `GRAM_DIV` trong `kaggle/streamlined_kd_cifar100.py` (mặc định tắt, bit-exact);
`kaggle/make_variant_notebook.py` sinh notebook. Quy ước chỉ số: best top-1. Đối chứng: F1 seed 2021 = 72.32 best; F1 seed 2022 = 72.45 best.
- **Sóng 1 (probe 10 epoch) ĐÃ SUBMIT** 2026-11-11 ~17:09 UTC (tài khoản 1, `streamlined-kd-cifar100-i1-probe-10ep`, version 1, RUNNING; docker/dataset/T4 đã xác minh).
- **Sóng 2 (người dùng chọn: chỉ seed 2022 trước):** F1 + GRAM_DIV=1 + LAMBDA_BC=λ*, 300 epoch, seed 2022; submit khi probe đọc xong. Seed 2021 hoãn. Quy tắc kết luận: Δ = best(I1 s2022) − 72.45 (F1 s2022); ≥ +0.20 hứa hẹn, ≤ −0.20 có hại, còn lại không phân biệt được.
- **Probe COMPLETE:** ĝ_KD 0,500, ĝ_BC 0,0989 (epoch 3–10) → λ* = 2,73 → kẹp **1,0**. Bỏ `/C` làm gradient BC tăng ~20 lần (0,005 → ~0,10) nhưng vẫn nhỏ hơn KD ~5 lần.
- **Sóng 2 ĐÃ SUBMIT (RUNNING)** ~17:50 UTC 11/11: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-i1-lam1p0-seed2022 — version 1, F1 + GRAM_DIV=1 + LAMBDA_BC=1.0, seed 2022, notebook `kaggle/experiments/i1_lam1p0_seed2022.ipynb` (commit `ef0576c`);
  dataset, Docker ghim, T4 đã xác minh. Quota tài khoản 1: 0,19h/30h trước run. Dự kiến xong ~5,4h sau (~23:15 UTC).
- **I1 (run chính, λ_BC=1.0, seed 2022) COMPLETE (2026-11-11 22:37 UTC, ~4,81h):** best **72.06** (ep 287), epoch cuối **71.77**, TB 10 cuối **71.88**, top-5 92.97, train top-1 cuối **64.36** (F1 s2022: 67.52). Teacher 72.41/92.50.
  So F1 seed 2022 (72.45 / 72.34 / 72.27): **Δ best = −0.39 ≤ −0.20 → KẾT LUẬN: I1 có hại, dừng hướng này.** Gradient BC (khi bỏ `/C`) tăng 0.10 → 0.22, ~0.4× KD ở cuối; BC mạnh làm student khớp train kém hơn (−3.2 train top-1).
  Giới hạn: một seed, một λ (1.0, bị kẹp từ λ*=2.73); chưa loại trừ λ nhỏ hơn nhưng không có lý do mạnh để thử thêm. Chi tiết: [experiments/2026-11-11_i1_gram_div.md](experiments/2026-11-11_i1_gram_div.md).
- Scheduler v8 đã **tắt** (`enabled=false`). Không còn run nào đang chạy. Quota tài khoản 1: ~5,0h/30h đã dùng (còn ~25h; refresh 2026-11-14 00:00 UTC). Chờ người dùng quyết định bước tiếp theo.
- Bảng best top-1 (seed 2021 trừ khi ghi): F1 72.32 (s2022: 72.45) · F4 72.32 · F2 72.23 · H1 72.20 · E4 72.10 · G1 72.09 · I1 72.06 (s2022) · F3 71.97 · paper SKD 72.50.


### HƯỚNG NGHIÊN CỨU ĐANG MỞ — cập nhật 2026-11-12 (người dùng đề xuất; chưa có run nào chạy)
Tìm **cơ chế** đằng sau F1 (best 72.32 seed 2021 / 72.45 seed 2022) rồi đề xuất loss có tính mới (Error-Aware Decoupled KD: α_i·TCKD + β_i·NCKD theo loại lỗi của teacher).
Ba giả thuyết: A teacher reliability · B tăng KD chung (0.538/0.462) · C cân bằng CE/KD theo độ khó. Thí nghiệm theo ưu tiên: J1 (hằng 0.538/0.462) → J2 (xáo trộn trọng số đúng/sai) → J3 (chỉ weight KD, CE cố định). **E4 seed 2022 đã bị bỏ khỏi kế hoạch (người dùng, 2026-11-12)**; đối chứng là F1 cùng seed.
Chi tiết, sơ đồ, ghi chú rủi ro (tính mới so với TGeo-KD/DKD/RLD, nhiễu seed, chi phí quota): **[experiments/2026-11-12_research_direction_f1_mechanism.md](experiments/2026-11-12_research_direction_f1_mechanism.md)**.
Nền tảng liên quan: báo cáo deep research `reports/Loss KD CIFAR100 ResNet20.md` (+ `research_notes/Loss KD CIFAR100 ResNet20/`). Quota tài khoản 1: ~5,0h/30h đã dùng (refresh 2026-11-14 00:00 UTC). Scheduler đang tắt.


### TRẠNG THÁI HIỆN TẠI — cập nhật 2026-11-12 (kế hoạch J1 / J2, một seed 2022; tài khoản Kaggle 1)
Người dùng yêu cầu chạy **J1** (trọng số hằng 0.5384 cho mọi mẫu) và **J2** (xáo trộn trọng số 0.6/0.4 trong batch) trên **một seed (2022)**, rồi đưa kết luận ban đầu. Hai run chạy song song.
Plan, công thức, test, quy tắc kết luận: [experiments/2026-11-12_j1_j2_controls.md](experiments/2026-11-12_j1_j2_controls.md) (nằm trong hướng nghiên cứu [experiments/2026-11-12_research_direction_f1_mechanism.md](experiments/2026-11-12_research_direction_f1_mechanism.md); E4 seed 2022 đã bỏ).
Notebook: `kaggle/experiments/j1_const0p5384_seed2022.ipynb`, `kaggle/experiments/j2_shuffled_seed2022.ipynb`. Đối chứng: F1 seed 2022 (best 72.45 / epoch cuối 72.34 / TB 10 cuối 72.27). Quy ước: best top-1. Quota tài khoản 1: 5.02h/30h trước run.
- **J1 và J2 ĐÃ SUBMIT (cả hai RUNNING)** 2026-11-12 ~03:32 UTC, tài khoản 1, version 1, commit `d489cba`; dataset, Docker ghim, T4 đã xác minh:
  - J1: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-j1-const0p5384-seed2022
  - J2: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-j2-shuffled-seed2022
  Quota tài khoản 1 trước run: 5.02h/30h (còn 24.98h). Dự kiến xong ~8:30–9:00 UTC (4.6–5.4h). Quy tắc kết luận sơ bộ (best top-1, ngưỡng 0.20 so với F1 seed 2022 = 72.45): xem file plan.
- **J1 và J2 COMPLETE (2026-11-12 ~09:38 UTC, tài khoản 1, seed 2022):**
  | | best | epoch cuối | TB 10 cuối |
  |---|---:|---:|---:|
  | F1 (đối chứng) | 72.45 | 72.34 | 72.27 |
  | **J1** (hằng 0.5384) | **71.51** | 71.24 | 71.24 |
  | **J2** (xáo trộn đúng/sai) | **72.39** | 72.02 | 72.14 |
  Δ best so với F1: J1 **−0.94**, J2 **−0.06**. `sw_same` J2 = 58.0% (xác nhận xáo trộn). Runtime J1 4,80h, J2 4,95h; teacher 72.41 ở cả hai.
- **KẾT LUẬN SƠ BỘ (một seed):** giả thuyết B (tăng KD chung) **bị bác bỏ**; giả thuyết A (teacher reliability) **chưa có bằng chứng** (F1 ≈ J2); J2 hơn J1 +0.88 dù cùng trọng số trung bình → **giả thuyết D: lợi ích của F1 đến từ sự không đồng nhất ngẫu nhiên của trọng số CE/KD giữa các mẫu** (điều chuẩn), không phải từ đúng/sai của teacher. Hướng Error-Aware DKD cần cân nhắc lại.
  Giới hạn: một seed; best top-1 lạc quan; J1 thấp hơn E4 (seed khác) 0.59 và E4 seed 2022 đã bỏ nên không tách được nhiễu. Chi tiết, gợi ý bước sau (J2/J1 seed 2021, J5 trọng số iid ngẫu nhiên, quét phương sai): [experiments/2026-11-12_j1_j2_controls.md](experiments/2026-11-12_j1_j2_controls.md).
- Scheduler v9 đã **tắt** (`enabled=false`). Không còn run nào đang chạy. Quota tài khoản 1: ~14,8h/30h đã dùng (ước tính; còn ~15h; refresh 2026-11-14 00:00 UTC). Chờ người dùng quyết định bước tiếp theo.


### TRẠNG THÁI HIỆN TẠI — cập nhật 2026-11-12 (E4 seed 2022 + J2 seed 2021; sau đó DỪNG để đánh giá tổng thể) — tài khoản Kaggle 1
Người dùng yêu cầu chạy **E4 seed 2022** và **J2 seed 2021** (song song), rồi **dừng** để đánh giá tổng thể (đảo lại quyết định bỏ E4 seed 2022). Plan, bảng tổng hợp cần lập, giới hạn: [experiments/2026-11-12_e4s2022_j2s2021.md](experiments/2026-11-12_e4s2022_j2s2021.md).
Notebook: `kaggle/experiments/e4_mdteacher_300ep_seed2022.ipynb`, `kaggle/experiments/j2_shuffled_seed2021.ipynb`. Quy ước: best top-1. Quota tài khoản 1 trước run: 14.79h/30h (còn 15.21h).
- **E4 seed 2022 và J2 seed 2021 ĐÃ SUBMIT (cả hai RUNNING)** 2026-11-12 ~10:09 UTC, tài khoản 1, version 1, commit `c73443b`; dataset, Docker ghim, T4 đã xác minh:
  - E4 seed 2022: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-e4-seed2022
  - J2 seed 2021: https://www.kaggle.com/code/trananhkiet/streamlined-kd-cifar100-j2-shuffled-seed2021
  Dự kiến xong ~14:45–15:30 UTC (4.6–5.4h). Sau khi cả hai xong: ghi kết quả + bảng tổng hợp hai seed, **KHÔNG chạy thêm**, chờ người dùng đánh giá tổng thể.
- **E4 seed 2022 và J2 seed 2021 COMPLETE (2026-11-12 ~15:38 UTC):** E4 s2022 = **72.11** best / 71.84 cuối / 71.92 TB10 (train 67.12); J2 s2021 = **71.99** / 71.68 / 71.66 (train 66.50). Teacher 72.41; `sw_same` J2 58.0%.
  **Bảng hai seed (best / cuối / TB10):** E4 72.10/71.70/71.73 · 72.11/71.84/71.92 (TB 72.11/71.77/71.83); F1 72.32/72.13/72.14 · 72.45/72.34/72.27 (TB 72.39/72.24/72.21); J2 71.99/71.68/71.66 · 72.39/72.02/72.14 (TB 72.19/71.85/71.90); J1 (s2022) 71.51/71.24/71.24.
  **Nhận xét sơ bộ:** F1 − E4 = +0.28 best / +0.47 cuối (6/6 phép dương); J2 − F1 = −0.20 / −0.39 (6/6 âm); J2 ≈ E4 (+0.09); J1 − E4 = −0.60 (s2022). → B bị bác; **D không được ủng hộ** (kết luận một seed trước đó bị thay thế); **A được ủng hộ nhẹ** (chưa tách khỏi C = độ khó). Nhiễu giữa seed cùng cấu hình tới ~0.4 (J2) nên hiệu ứng 0.2–0.5 chỉ mức vừa trên nhiễu. Chi tiết: [experiments/2026-11-12_e4s2022_j2s2021.md](experiments/2026-11-12_e4s2022_j2s2021.md).
- Scheduler v10 đã **tắt** (`enabled=false`). **Không còn run nào đang chạy; DỪNG để người dùng đánh giá tổng thể.** Quota tài khoản 1: ~24,5h/30h đã dùng (ước tính; còn ~5,5h; refresh 2026-11-14 00:00 UTC).


### TRẠNG THÁI HIỆN TẠI — cập nhật 2026-11-12 (D2: bỏ TCKD trên mẫu teacher sai; một run seed 2022; tài khoản Kaggle 1)
Người dùng chọn hướng 2 (error-aware decoupled KD) và muốn kiểm tra kết quả sau ~6h. Plan, công thức, test, quy tắc kết luận, giới hạn: [experiments/2026-11-12_d2_nckd_on_wrong.md](experiments/2026-11-12_d2_nckd_on_wrong.md).
Notebook `kaggle/experiments/d2_nckd_on_wrong_seed2022.ipynb` (F1 + `WRONG_KD_MODE=nckd_only`). Đối chứng seed 2022: F1 72.45 best (cuối 72.34, TB10 72.27), E4 72.11. Quota tài khoản 1 trước run: 24.52h/30h (còn 5.48h; refresh 2026-11-14 00:00 UTC).
- **Chưa submit.**
