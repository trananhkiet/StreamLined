# Relational / correlation / feature-level KD losses to combine with logit KD, plus the SKD paper (CIFAR-100 ResNet56->ResNet20)

Research limits: only github.com / raw.githubusercontent.com and WebSearch worked; arxiv.org, openaccess.thecvf.com, ar5iv, p.aislab.kr, papernotes were DNS-blocked in this session. Numbers below come from the official SKD repo README/code and WebSearch summaries of other papers (secondary, flagged).

## 1. What does the SKD (Streamlined KD, CVPR 2026) paper report, with what hyperparameters and ablations?

### Takeaway
SKD = CE + masked KL(T=4) "instance" term + masked Mahalanobis-Gram "batch correlation" (bc) term; the official README reports 72.50% for ResNet56->ResNet20 (teacher 72.40, scratch student 65.79). The code uses weights CE 0.1 / KD 0.9 / BC 0.9, Tikhonov 0.1, and BOTH masks keep the LOW-confidence half of the batch (not the high-confidence half). Ablation tables and any "additional terms" discussion could not be read (PDF host blocked).

### Cited Findings
- Paper: "Streamlined Knowledge Distillation", Jeong, Lee, Choi, CVPR 2026 pp. 26646-26655; official code (refactored from mdistiller); also won a CVPR "Compute Transparency Champion Award" — [HyunJunSik/StreamLined README](https://github.com/trananhkiet/StreamLined) (mirror of the official repo) and [official repo](https://github.com/HyunJunSik/StreamLined)
- Abstract-level description: logit-only method with two knowledge types: instance-wise (KL) and direction-wise (Mahalanobis distance on the Gramian of normalized logits); claimed equivalent to L2 in covariance-whitened space; Cholesky used for the inverse — search summary of [CVPR 2026 poster page](https://cvpr.thecvf.com/virtual/2026/poster/39253) (page itself unreachable; secondary summary). One secondary summary says total loss L_SKD = L_INS + L_DIR equally weighted, CIFAR-100: 240 epochs, SGD lr 0.1 decayed at 150/180/210, 5 runs averaged (unverified against the PDF).
- README CIFAR-100 table (homogeneous), columns RN56->RN20 | RN110->RN32 | RN32x4->RN8x4 | WRN40-2->WRN16-2 | WRN40-2->WRN40-1 | VGG13->VGG13 — [StreamLined README](https://github.com/trananhkiet/StreamLined) (web-fetch table flattened; column alignment inferred, "Ours" row bold):
  - Teacher 72.40 | 74.31 | 79.34 | 75.59 | 75.59 | 74.64
  - Student 65.79 | 67.92 | 72.48 | 71.12 | 69.54 | 68.19
  - FitNet 69.64 | 72.35 | 75.26 | 74.46 | 73.28 | 70.09
  - RKD 70.86 | 73.81 | 74.93 | 74.73 | 74.30 | 71.61
  - CRD 67.28 | 71.97 | 74.16 | 74.22 | 72.44 | 72.74
  - OFD 68.77 | 71.40 | 73.88 | 72.31 | 73.26 | 74.21
  - ReviewKD 69.54 | 71.12 | 75.71 | 73.92 | 74.50 | 72.29
  - KD 71.74 | 74.01 | 74.75 | 76.04 | 74.52 | 74.08
  - CLKD 65.74 | 69.81 | 70.19 | 72.89 | 71.89 | 72.46
  - DKD 71.17 | 74.12 | 76.51 | 76.41 | 75.33 | 74.41
  - MLKD 72.21 | 74.24 | 75.59 | 76.83 | 74.78 | 74.25
  - SDD 69.42 | 72.78 | 74.40 | 75.01 | 72.53 | 72.29
  - **SKD 72.50 | 74.84 | 78.33 | 76.60 | 76.04 | 75.75**
  - So SKD beats this table's plain KD by only +0.76 on RN56->RN20 (72.50 vs 71.74); this KD baseline (71.74) is far above the standard mdistiller/RepDistiller KD (70.66) — i.e. the authors' own training recipe/baselines differ from the usual protocol (student 65.79 vs the usual 69.06).
- Code (Distiller/SKD.py) — [raw SKD.py](https://raw.githubusercontent.com/HyunJunSik/StreamLined/main/Distiller/SKD.py), via fetch summaries:
  - defaults: temp=4, tik=1e-1 (Tikhonov), ce=0.1, kd=0.9.
  - loss_ce = 0.1 * CE(student, y); loss_kd = 0.9 * KL(T=4, T^2 scaled, per-sample) * sample-mask; loss_bc = 0.9 * SKD_loss(...) * class-mask. Returned values already weighted; no warmup/ramp; epoch arg unused. So the "0.1 / 0.9 / 0.9" recollection is confirmed.
  - SKD_loss: L2-normalize logits (rows), per-batch similarity (Gram) matrix for student and teacher, difference, covariance of difference + Tikhonov*I (cov = torch.cov(diff.T) + 0.1*I), Mahalanobis distance via Cholesky, mean.
  - Sample mask: teacher max-softmax confidence compared with the batch median; keeps samples with confidence <= median (LOW-confidence half). Class mask: per-class teacher probability mass summed over batch, keeps classes with mass <= median (low-mass half).
  - Fetch summary reported "SKD_loss ... between normalized logits" — whether the Gram is over the batch (B x B) or classes (C x C) should be re-checked in the source; the user's term "batch correlation (BC)" and cov of `diff.T` with `diff.size(1)` suggest the covariance is over the batch-dim of a diff matrix; not verified.
- Extra experiments file only covers robustness to batch size and noisy labels for SKD(L_dir) vs L2; no tables extractable — [extra_experiments.md](https://raw.githubusercontent.com/HyunJunSik/StreamLined/main/extra_experiments.md)
- Repo contains Distiller implementations of AT, CLKD, CRD, DKD, FitNet, KD, MLKD, OFD, RKD, ReviewKD, SDD, SKD — [Distiller dir](https://github.com/HyunJunSik/StreamLined/tree/main/Distiller)

### Inferences
- Because SKD ships both masks, the "confidence mask" in the user's recipe is real but selects the LOW-confidence (hard) half of teacher predictions — it effectively focuses KD on hard samples and rarely-predicted classes. Worth testing the opposite mask as an ablation.
- Cost: bc term needs a B x B (or C x C) covariance and a Cholesky each step: negligible vs the forward passes; no extra parameters.

### Gaps
- Paper ablation tables (effect of each term, masks, Tikhonov, batch size) and any "future work / additional terms" text: PDF unreachable (openaccess.thecvf.com, p.aislab.kr blocked). Not found.
- The 5-run std and the exact schedule/batch size not verified from primary source.

## 2. Published ResNet56->ResNet20 numbers for CRD, ReviewKD, SimKD, ICKD, CCKD, RKD, PKT, VID, FitNet, NST, AT, OFD, FCFD, DiffKD, and newer methods; which are cheap and compatible with CE+KL+Gram-BC?

### Takeaway
Under the standard RepDistiller protocol (teacher ~72.3, scratch student 69.06) KD is 70.66, CRD 71.16, ReviewKD 71.89, DIST ~71.5-71.75, SimKD ~71.05; feature-only methods mostly sit at 69-70.6. SKD's table uses a different protocol (scratch 65.79), so its numbers are not comparable with these. Cheapest, parameter-free, logit-compatible add-ons: AT, PKT, RKD, NST, DIST (all use only existing features/logits); CRD needs a memory bank + embedding heads; ReviewKD/SimKD/VID/OFD/FitNet add connector parameters.

### Cited Findings
- RepDistiller protocol, RN56->RN20 (teacher 72.34, student 69.06): KD 70.66, FitNet 69.21, AT 70.55, SP 69.67, CC 69.63, VID 70.38, RKD 69.61, PKT 70.34, AB 69.47, FT 69.84, FSP 69.95, NST 69.60, CRD 71.16 — [HobbitLong/RepDistiller README](https://github.com/HobbitLong/RepDistiller)
- mdistiller README (RN56->RN20): KD 70.66, DKD 71.97 — [megvii-research/mdistiller](https://github.com/megvii-research/mdistiller)
- ReviewKD 71.89 (teacher/student 73.04?/69.06 in the reproduction; teacher accuracy differs by paper: 72.34, 72.37, 73.04) — summary of [Re: Knowledge Review](https://arxiv.org/pdf/2205.11246) via WebSearch (secondary)
- DIST 71.75 ± 0.30 (teacher 72.34) in one table; 71.55 (teacher 72.37) in another — WebSearch summary of [2410.06561](https://arxiv.org/pdf/2410.06561) (secondary)
- SimKD 71.05 (student 69.06) — WebSearch summary of [TopKD 2508.04539](https://arxiv.org/pdf/2508.04539); another paper gives 71.06 ([2502.05835](https://arxiv.org/pdf/2502.05835)); secondary, labels in source tables flagged ambiguous
- DKD 71.97 (mdistiller protocol) — [mdistiller](https://github.com/megvii-research/mdistiller); same value cited via [2502.05835](https://arxiv.org/pdf/2502.05835)
- Contradiction across protocols: SKD repo table gives KD 71.74, DKD 71.17, MLKD 72.21, CRD 67.28, ReviewKD 69.54, RKD 70.86 — [StreamLined README](https://github.com/trananhkiet/StreamLined) vs the RepDistiller/mdistiller numbers above (KD 70.66, DKD 71.97, CRD 71.16, ReviewKD 71.89). CRD/ReviewKD drop by 2-4 points in SKD's re-runs, so SKD's feature-method baselines look under-tuned relative to published numbers.
- Compatibility facts from code layout: SKD repo lists AT, CRD, FitNet, OFD, RKD, ReviewKD implementations, all in one `Distiller` interface — [Distiller dir](https://github.com/HyunJunSik/StreamLined/tree/main/Distiller)

### Inferences
- Cost classes (from method definitions, my own knowledge, not from fetched sources): parameter-free: AT (attention maps), RKD (distance/angle), PKT, NST, DIST (inter/intra Pearson on softmax), CC; extra params: FitNet/VID/OFD/ReviewKD/SimKD/ICKD connectors, CRD embeddings + memory bank (N=50k x 128 on CIFAR-100).
- For a CE + KL(T=4) + Gram-BC recipe, best low-risk additions: AT or RKD on the penultimate/stage features (no params). DIST is partly redundant with Gram-BC (both are batch-level relational losses on logits/probabilities).

### Gaps
- No verified RN56->RN20 numbers found for ICKD, CCKD (RepDistiller "CC" is 69.63), FCFD, DiffKD, and 2023-2026 feature/relational methods (VRM 2502.20760, BicKD 2602.01265, CMKD, MSDCRD 2502.05835, TopKD): arxiv blocked / only snippets.
- OFD under the RepDistiller protocol for RN56->RN20: not found.

## 3. Batch-level relation losses (DIST, CRD with memory bank, Mahalanobis/Bayes variants) and stacking

### Takeaway
No source found that explicitly stacks Gram-BC with DIST/CRD; relevant evidence is indirect. Treat stacking as untested and to be measured.

### Cited Findings
- DIST (inter-class + intra-class Pearson relations) RN56->RN20 ~71.5-71.75 — [2410.06561 summary](https://arxiv.org/pdf/2410.06561)
- CRD 71.16 standalone and its original paper combines it with KD (CRD+KD) as the best row in RepDistiller — [RepDistiller](https://github.com/HobbitLong/RepDistiller) (CRD+KD value for RN56->RN20 not in the fetched README extract)
- SKD's own pair of terms (instance KL + direction Gram) is itself a two-level logit KD; the SKD table shows rivals MLKD (multi-level logit alignment, batch/class/instance) 72.21 — [StreamLined README](https://github.com/trananhkiet/StreamLined)

### Inferences
- MLKD already includes batch-level (Gram) alignment; SKD's gain over MLKD on RN56->RN20 is +0.29, within run noise, suggesting diminishing returns when stacking more batch-level logit relations.

### Gaps
- No source on Bayes/Mahalanobis variants beyond SKD; no source on DIST+Gram stacking.

## 4. Typical gains from combining logit KD + one feature/relational term on small CIFAR-100 pairs

### Takeaway
Not verified from primary sources in this session. Search found no direct DKD+ReviewKD / MLKD+CRD / CTKD+X table for RN56->RN20.

### Cited Findings
- Logit-standardization paper has DKD/MLKD with LS plug-ins on CIFAR-100 (cells not legible via search) — [2403.01427](https://arxiv.org/pdf/2403.01427)
- TopKD compares with DKD+LS and WTTM, not feature combos — [2508.04539](https://arxiv.org/pdf/2508.04539)
- Single-method facts: KD->CRD+KD and KD->ReviewKD gains of ~0.5-1.2 points over KD on RN56->RN20 (KD 70.66 vs CRD 71.16, ReviewKD 71.89) — see section 2 sources.

### Inferences
- From memory (unsourced): typical CRD/feature + KD stacking gains on small CIFAR-100 pairs are ~0.3-1.0 points, with feature terms giving smaller returns when the student is already near the teacher (RN56->RN20 teacher-student gap only ~3 points).

### Gaps
- Need direct reading of the CMKD paper (2410.06561), LS paper tables and ReviewKD/DKD papers to get actual stacked numbers; arxiv blocked.
