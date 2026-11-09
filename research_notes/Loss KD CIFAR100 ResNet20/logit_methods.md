# Logit-based KD improvements, CIFAR-100 ResNet56 -> ResNet20

Access note: arxiv.org, ar5iv and related hosts did not resolve from this session (network allowlist). Primary papers were NOT read directly; numbers come from GitHub READMEs (fetched) and search-engine extracts of arXiv PDFs (search snippets, partly garbled). Items marked "unverified" need a check against the paper.

## Q1. Published resnet56->resnet20 numbers per method (mdistiller-style protocol)

### Takeaway
Under the standard mdistiller/DKD protocol (teacher 72.34, scratch student 69.06, KD 70.66) the best logit-only numbers are about 72.2-72.3 (MLKD 72.19, DKD+LSKD 72.32, MLKD+LSKD 72.33). Plain DKD is 71.97, DIST about 71.55-71.75, NKD about 71.62, TTM/WTTM about 71.83/71.92, CTKD 71.19, KD+LSKD 71.43. The student is capped by the teacher (72.34), so remaining headroom over about 72.3 is nil for logit-only methods.

### Cited Findings
- Baselines: teacher ResNet56 72.34, student ResNet20 from scratch 69.06, KD 70.66 (R56->R20), KD 70.67 (R110->R20) — [TTM/WTTM repo](https://github.com/zkxufo/TTM)
- DKD 71.97 vs KD 70.66 on R56->R20 — [mdistiller README](https://github.com/megvii-research/mdistiller)
- LSKD (Logit Standardization, Sun et al. CVPR 2024 Highlight) Table, R56->R20: KD 70.66 -> KD+Ours 71.43; CTKD 71.19 -> 71.34; DKD 71.97 -> 72.32; MLKD 72.19 -> 72.33. Same table R110->R20: KD 70.67/71.48, CTKD 70.99/71.39, DKD 71.06/71.85, MLKD 71.89/72.27 — [LSKD repo](https://github.com/sunshangquan/logit-standardization-KD); mirror [LEMMA](https://github.com/AIRLABkhu/LEMMA); paper [CVPR 2024 page](https://cvpr.thecvf.com/virtual/2024/poster/30792)
- LSKD hyperparameters for R56->R20 (from log filenames): KD+Ours base temp 2, kd weight 9; CTKD+Ours 2 / 9; DKD+Ours base temp 3, kd weight 18, beta 2.0; command flags `--logit-stand --base-temp 2 --kd-weight 9` — [LSKD repo](https://github.com/sunshangquan/logit-standardization-KD)
- TTM 71.83, WTTM 71.92, WTTM+CRD 72.20 on R56->R20 (R110->R20: 71.46/71.67/72.13) (Zheng & Yang, ICLR 2024). Repo example flags: `--ttm_l` (gamma, power-transform exponent) 0.1 for TTM, 0.2 for WTTM+CRD; `-b` KD weight 45 for TTM, 0.8 for WTTM+CRD; `-a` 4; T not stated in README — [TTM repo](https://github.com/zkxufo/TTM). The paper reports 5-run averages — [search extract of arXiv 2402.11148](https://arxiv.org/pdf/2402.11148)
- DIST (Huang 2022): 71.75 +/- 0.30 per WTTM paper table and the Wasserstein-KD paper; 71.55 in the CMKD paper's table; 71.78 in a "previous recipe" column of arXiv 2305.15781 (which reports 74.51 under a stronger recipe; DKD 73.10, KD 72.34 there); 71.27 in a T=4 table of the "Rethinking DKD" paper — [2402.11148](https://arxiv.org/pdf/2402.11148), [2412.08139](https://arxiv.org/pdf/2412.08139), [2305.15781](https://arxiv.org/pdf/2305.15781.pdf), [2512.04625](https://arxiv.org/pdf/2512.04625), [2410.06561](https://arxiv.org/pdf/2410.06561). All via search extracts; DIST numbers vary by paper/protocol.
- NKD (Yang et al., ICCV 2023, "From Knowledge Distillation to Self-KD"/arXiv 2208.10139): 71.62 on R56->R20 as reported in the CMKD paper table (search extract, column mapping somewhat uncertain) — [2410.06561](https://arxiv.org/pdf/2410.06561). NKD's own table not retrieved. NKD uses DKD's CIFAR training setup; ImageNet hyperparameters alpha=1.5, lambda=1 — [2208.10139](https://arxiv.org/pdf/2208.10139). Code: [cls_KD](https://github.com/yzd-v/cls_KD) (README only links nkd.md).
- MLKD (Jin, Wang, Lin, CVPR 2023; instance/batch/class-level alignment): 72.19 on R56->R20 (R110->R20 71.89) as quoted in the LSKD tables — [LSKD repo](https://github.com/sunshangquan/logit-standardization-KD). Another search extract claimed 70.78; that is the WRN40-2->MobileNetV2 column misread, discard.
- CTKD (curriculum learnable temperature): 71.19 (R56->R20) — [LSKD repo](https://github.com/sunshangquan/logit-standardization-KD)
- DTKD (Dynamic Temperature KD, arXiv 2404.12711): 72.05 vs KD 70.66 on R56->R20, 5-run avg (garbled extract, unverified) — [2404.12711](https://arxiv.org/pdf/2404.12711)
- GDKD (in "Rethinking DKD: predictive distribution perspective", Dec 2025): 71.65 on R56->R20, 5-run avg — [2512.04625](https://arxiv.org/pdf/2512.04625)
- DeepKD (2505.15133) table lists DKD 71.97, DKD+DOT 71.12, DKD+LSKD 72.32 for this pair; DeepKD's own number not retrieved — [2505.15133](https://arxiv.org/pdf/2505.15133)
- CRD (feature/contrastive) 71.16 on R56->R20 for reference — [TopKD extract](https://arxiv.org/pdf/2508.04539)
- Other 2024-2026 logit papers surfaced but R56->R20 number NOT retrieved: WKD-L (Wasserstein, NeurIPS 2024, 2412.08139), TopKD (2508.04539), DTS Dynamic Temperature Scheduler (2511.13767), CMKD (2410.06561), Dual-Head KD (2411.08937), Local Dense Logit (ICCV 2025), AEKT (2409.01679).

### Loss formulas / hyperparameters (from my background knowledge, not fetched; verify against code)
- KD: CE + w * T^2 * KL(p_t^T || p_s^T). mdistiller CIFAR default T=4, CE 0.1, KD 0.9.
- DKD: CE + alpha*TCKD + beta*NCKD, T=4, alpha=1, beta=8 default, 20-epoch linear warmup; per my recollection mdistiller's res56->res20 config uses beta=2 (consistent with LSKD's DKD beta=2.0 filename; unverified).
- DIST: inter-class (row) and intra-class (column) Pearson correlation between student and teacher softmax outputs over the batch; beta=gamma=2 on CIFAR in mdistiller (recollection, unverified).
- NKD: split target (normalized non-target distribution matched with KL, scaled by lambda-ish weight) and target term; teacher-free variant tf-NKD.
- LSKD: z-score logits per sample, z=(l-mean)/(std), then softmax at base temp tau (2 here), tau shared teacher/student; kd weight 9 (large because z-scored logits give smaller gradients).
- WTTM: drop temperature on student side, power transform on teacher probs (exponent gamma=1/T), KL weighted by sample-adaptive U(p_t) (teacher entropy-like), emphasizing low-confidence teacher samples.

### Inferences
- The ranking at R56->R20 is tight: MLKD ~ DKD+LSKD ~ MLKD+LSKD (72.2-72.3) > DTKD(?) 72.05 > DKD 71.97 > WTTM 71.92 > TTM 71.83 > DIST 71.6-71.8 > NKD 71.6 > KD+LSKD 71.43 > CTKD 71.19 > KD 70.66. Differences of 0.1-0.3 are within seed noise (std about 0.2-0.3).
- Teacher-bound ceiling: ~72.3 equals the teacher accuracy, so a loss at KD+BC is unlikely to exceed ~72.3 by logits alone.
- Different papers' DIST/NKD numbers conflict by up to 0.5; use one's own runs.

### Gaps
- Could not read arXiv PDFs; NKD own table, WKD-L, TopKD, DeepKD, DTS R56->R20 numbers not obtained.
- Exact CIFAR hyperparameters for DIST/NKD/WTTM not verified.
- No post-2025 papers with confirmed R56->R20 numbers beyond GDKD (71.65, Dec 2025); "Temperature-Adaptive Transformed Teacher Matching" (listed as arxiv 2608.29099 in a search result) could not be verified.

## Q2. What can be stacked on CE + KL(T=4, confidence mask) + Gram-matrix batch-correlation (BC) term; complementary vs redundant

### Takeaway
BC is essentially a DIST/MLKD-batch-level style relational term, so DIST and MLKD's batch-level term are most redundant with it. Best candidates are things acting on the per-sample KL: logit standardization (largest confirmed plug-in gain), WTTM/confidence weighting, and DKD-style decoupling; the confidence mask is conceptually overlapping with WTTM's adaptive weighting.

### Cited Findings
- LSKD is explicitly plug-and-play and improves KD, CTKD, DKD, MLKD (gain +0.77, +0.15, +0.35, +0.14 on R56->R20), with the gain shrinking as the base method gets stronger — [LSKD repo](https://github.com/sunshangquan/logit-standardization-KD)
- MLKD combines instance, batch (input correlation) and class (category correlation) alignment — [MLKD CVPR 2023](https://mlanthology.org/cvpr/2023/jin2023cvpr-multilevel); it already contains a batch-correlation term, and still gets 72.19.
- WTTM combines with feature method CRD (+0.28 over WTTM on R56->R20), showing logit and other-level terms stack — [TTM repo](https://github.com/zkxufo/TTM)
- DKD+DOT is worse than DKD (71.12 vs 71.97) on R56->R20, i.e. not every add-on stacks — [2505.15133](https://arxiv.org/pdf/2505.15133)

### Inferences
- Redundant with BC: DIST (inter/intra-class correlation of softmax outputs over batch), MLKD batch/class terms. Adding them double-counts the same relational signal; if kept, replace BC rather than stack.
- Likely complementary: LSKD (changes how the KL is computed; orthogonal to relational term; note T=4 would be replaced by z-score with base temp ~2 and kd weight ~9, which interacts with the confidence mask and loss weighting and needs retuning); DKD's NCKD/TCKD split (beta=2 for R56->R20), possibly with mask applied to the TCKD part.
- Partly overlapping: WTTM (sample-adaptive weighting by teacher confidence) vs. a confidence mask; they act on the same quantity in opposite directions in spirit (WTTM upweights low-confidence teacher samples, a mask drops them), so combining may conflict. Treat as alternatives.
- CTKD/DTKD dynamic temperature: cheap to add, but gains small for strong bases (LSKD+CTKD 71.34 vs CTKD 71.19); and LSKD subsumes the temperature problem.
- Implementation difficulty (my estimate): LSKD very easy (about 5 lines); DKD easy (about 20 lines, needs warmup); DIST/WTTM easy; NKD medium; MLKD medium (needs augmented views); CTKD medium (gradient reversal layer).
- Risks: z-score changes loss scale (kd weight 9 vs 0.9); DKD warmup and beta sensitive; single-seed differences <0.3 are noise.

### Gaps
- No paper tests combos with a Gram-matrix BC term specifically; no source for BC+LSKD or BC+DKD interactions. All complementarity statements are inference.

## Q3. Which help when the student is small and capacity-limited

### Takeaway
Evidence is indirect. The LSKD paper's motivation is exactly capacity gap (shared temperature forces student to match teacher logit range/variance), and DKD and TTM-type methods give their largest gains on weak students; but R56->R20 has a small teacher-student gap (72.34 vs ResNet20 69.06), so gains are smaller there than on R32x4->R8x4.

### Cited Findings
- LSKD argues a shared temperature forces teacher/student logits to match range and variance, limiting students given the capacity gap — [CVPR 2024 page](https://cvpr.thecvf.com/virtual/2024/poster/30792)
- Gains on R56->R20 over KD: DKD +1.31, LSKD +0.77, TTM +1.17, WTTM +1.26, MLKD +1.53 vs larger gains on R32x4->R8x4 (KD 73.33 -> DKD 76.32, KD+LSKD 76.62, MLKD 77.08) — [mdistiller](https://github.com/megvii-research/mdistiller), [LSKD repo](https://github.com/sunshangquan/logit-standardization-KD)
- Small ResNet20 student with weaker teacher: R110->R20 gains lower for DKD (71.06, only +0.39 over KD) while MLKD 71.89 and WTTM 71.67 stay higher — [LSKD repo](https://github.com/sunshangquan/logit-standardization-KD), [TTM repo](https://github.com/zkxufo/TTM)

### Inferences
- For ResNet20 (0.27M params), methods that soften or normalize the non-target part (DKD with beta=2 rather than 8, LSKD, TTM/WTTM) help; multi-level relational terms (MLKD) give the single biggest standalone gain, suggesting relational/batch signal (like BC) is valuable for small students.
- DKD's default beta=8 is tuned for larger gaps; R56->R20 uses lower beta (unverified recollection, consistent with LSKD DKD beta=2.0).

### Gaps
- No direct controlled study of capacity vs method on CIFAR ResNet20 was found; no ablation isolating small-student effects.
