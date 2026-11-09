# Sample-wise / instance-wise adaptive weighting and teacher-reliability handling in KD

Research constraint note: arxiv.org, ar5iv, openaccess.thecvf.com were all unreachable via WebFetch (DNS not in the session allowlist). All findings below come from WebSearch result summaries only; no paper table was read directly. Anything not directly supported by a search summary is flagged as "unverified" or placed in Gaps. Re-verify numbers against the PDFs before quoting.

## Q1. What published methods weight or correct the KD signal per sample (2020-2026)?

### Takeaway
There is a sizeable family: (a) correctness/geometry-based CE-vs-KD mixing (TGeo-KD, ADA-KD, WLS-KD), (b) logit correction for wrong teachers (Knowledge Adjustment/LA, RLD), (c) uncertainty/difficulty weights (WSL, ER-KD/EA-KD, CIST, AEKT, KD-AIF), (d) per-instance temperature (CTKD Instance-T, CIST), (e) optimizer-level CE/KD balancing (DOT, MoKD). Your 0.6/0.4-vs-0.4/0.6 correctness switch is essentially the hard-threshold version of TGeo-KD's stated principle.

### Cited Findings
- WSL (Zhou et al., ICLR 2021): distillation raises bias and lowers variance relative to direct training; "regularization samples" (high bias / low variance contribution) hurt KD; WSL assigns lower weights to them and higher weights to others, using student and teacher predictions. Hard-filtering these samples also hurts, so weighting is soft. CIFAR uses tau=4, alpha=2.25 (grid-searched). Code: github.com/bellymonster/Weighted-Soft-Label-Distillation. — [arXiv 2102.00650](https://arxiv.org/pdf/2102.00650); [summary](https://liner.com/review/rethinking-soft-labels-for-knowledge-distillation-a-biasvariance-tradeoff-perspective); [code](https://github.com/bellymonster/Weighted-Soft-Label-Distillation)
- TGeo-KD / "Less or More From Teacher" (ICLR 2024): learns a sample-wise fusion ratio alpha (small NN, bilevel objective) between teacher soft targets and ground truth, using student/teacher/GT trilateral geometry (intra- and inter-sample). Stated principle: alpha larger when the teacher is correct (and student-teacher agreement/"ST" high, more learning potential), smaller when teacher is wrong because it misleads. Contrasted with ADA-KD (per-class alpha from teacher class correctness) and WLS-KD (uses both teacher and student correctness). Claims consistently better than other re-weighting methods on CIFAR-100, ImageNet and non-vision tasks. Motivating CIFAR-100 experiment: ResNet-34 teacher -> ResNet-18 student. — [arXiv 2312.15112](https://arxiv.org/pdf/2312.15112); [ICLR slides](https://iclr.cc/media/iclr-2024/Slides/18738.pdf)
- Knowledge Adjustment (KA) / Dynamic Temperature Distillation (DTD), "Preparing Lessons" (Wen, Lai, Qian): for misjudged training samples the teacher prediction is fixed using the ground truth before loss computation (journal version calls it "Logits Adjustment", LA). CIFAR-100 Preact-ResNet50 -> Preact-ResNet18: KD 75.87 vs a checkmarked KD row 77.24 (snippet did not label the checkmark; treat as KD with DTD-KA, unverified). — [arXiv 1911.07471](https://arxiv.org/pdf/1911.07471)
- RLD, Knowledge Distillation with Refined Logits (ICCV 2025): argues prior correction strategies (modify teacher logits using labels) destroy class correlations; instead uses "sample confidence" (binary probabilities from logits) and "masked correlation" (mask teacher classes ranked equal/above the true class, keep correlation among the rest). Logit-only, defaults alpha=1, beta=4. — [arXiv 2408.07703](https://arxiv.org/html/2408.07703v2); [ICCV 2025 page](https://iccv.thecvf.com/virtual/2025/poster/2153)
- ER-KD "Knowledge From the Dark Side" -> EA-KD (ICCV 2025): reweights per-sample KD loss by entropy of teacher softened logits (difficulty proxy; more entropy = higher weight); EA-KD combines teacher and student entropy, only multiplies a sample-level weight in the loss. — [arXiv 2311.13621](https://arxiv.org/pdf/2311.13621); [EA-KD v2 html](https://arxiv.org/html/2311.13621v2)
- CIST, Consistently Informative Soft-Label Temperature (arXiv 2605.20357, May 2026): sample-wise adaptive temperatures for teacher and student plus reweighting of the distill loss by teacher confidence and student learning difficulty. Theory: teacher-label entropy is governed largely by max-logit/temperature ratio. — [arXiv 2605.20357](https://arxiv.org/html/2605.20357)
- AEKT (arXiv 2409.01679): adaptively controls target-class gradient according to teacher-vs-student confidence ratio on the target class. — [arXiv 2409.01679](https://arxiv.org/pdf/2409.01679v1)
- KD-AIF (arXiv 2501.02705): per-sample weights via influence functions from robust statistics. — [arXiv 2501.02705](https://arxiv.org/html/2501.02705v1)
- CTKD (AAAI 2023): global learnable temperature trained adversarially (gradient reversal) with curriculum on difficulty; Instance-T variant produces per-sample temperatures; plug-in to other KD methods, evaluated on eleven CIFAR-100 pairs. — [arXiv 2211.16231](https://arxiv.org/pdf/2211.16231); [project page](https://zhengli97.github.io/CTKD)
- Annealing-KD (Jafari et al., 2104.07163): feeds teacher soft targets incrementally via annealed temperature iterations (schedule, not per-sample). — [fugumt summary](https://fugumt.com/fugumt/paper_check/2104.07163v1_enmode)
- Tf-KD (Yuan et al., CVPR 2020): KD is a learned label-smoothing regularizer; a "virtual teacher" with LSR works; a follow-up (LSR-OS-TC) splits it into Output Smoothing and Teacher Correction and finds output smoothing key to stable hyperparameters. — [arXiv 1909.11723](https://arxiv.org/pdf/1909.11723); [CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.html)
- Dynamic Temperature KD (DTKD) lists ResNet56 teacher 72.34 / ResNet20 student 69.06 in CIFAR-100 table (baselines). — [arXiv 2404.12711](https://arxiv.org/pdf/2404.12711)

### Inferences
- Your hard correct/wrong switch matches the TGeo-KD design principle; TGeo-KD's extras beyond it are (i) learned (not fixed) ratio and (ii) use of student-teacher agreement. Those are the natural next steps.
- RLD's argument (label-based correction destroys class correlations) is consistent with your finding that swapping wrong teacher logits did not help.

### Gaps
- Exact WSL weight formula not retrieved (not in search summaries). From memory only (UNVERIFIED): weight is a softmax over the batch of the ratio of student-to-teacher CE (lower weight where student loss is large relative to teacher loss); check Sec. 3-4 of arXiv 2102.00650 or the repo loss file.
- Exact formulas for ER-KD, TGeo-KD alpha, CIST, AEKT not retrieved.
- Teacher-Assistant (TAKD), AID, "Knowledge distillation from a stronger teacher" (DIST), instance-temperature papers other than CTKD, and "correct-teacher-only" KD were not searched/found with detail. No source for a method literally named "AID" or "CCKD-L" was found; unverified.

## Q2. Which report gains on small students, with numbers (CIFAR-100, ResNet56->ResNet20 where available)?

### Takeaway
Reliable ResNet56->ResNet20 numbers I found: KD 70.66; DKD 71.97; RLD 72.00; LA 71.24; RC 71.63; LR 70.74; KD+DOT 71.07 (+0.41). Per-sample weighting papers mostly report on bigger pairs or other architectures; I found no verified ResNet56->ResNet20 number for WSL, ER-KD, TGeo-KD or CTKD. Typical gains over KD for this pair are 0.4 to 1.3 points, so your +0.43 (one seed) is on the order of DOT and within seed noise for this pair.

### Cited Findings
- ResNet56->ResNet20 CIFAR-100: KD 70.66, DKD 71.97, LA 71.24, RC 71.63, LR 70.74, RLD 72.00; ResNet110->ResNet20 RLD 71.67 vs DKD 71.06. (secondary source, not the paper table) — [papernotes RLD summary](https://en.papernotes.org/ICCV2025/model_compression/knowledge_distillation_with_refined_logits/)
- KD baseline 70.66 for ResNet56->ResNet20 also appears in the mdistiller/DKD repo table. DOT supplement: KD 70.66 -> KD+DOT 71.07 (+~0.4), CRD protocol. — [mdistiller](https://github.com/megvii-research/mdistiller); [DOT supplement](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Zhao_DOT_A_Distillation-Oriented_ICCV_2023_supplemental.pdf); [DOT arXiv 2307.08436](https://arxiv.org/pdf/2307.08436)
- DOT main-text claim: +1~2% on CIFAR-100, 3~6% Tiny-ImageNet, 1~2% ImageNet (other pairs). — [DOT arXiv](https://arxiv.org/pdf/2307.08436)
- WSL: WRN-40-2 -> WRN-40-1, KD 73.54 -> WSL 74.48 (+0.94) (from a review page; not verified against paper). WSL+RKD on resnet56->resnet20 listed as 70.73 in one snippet row (labelling uncertain; a 70.25 value also appeared, unlabeled). — [liner review](https://liner.com/review/rethinking-soft-labels-for-knowledge-distillation-a-biasvariance-tradeoff-perspective); [arXiv 2102.00650](https://arxiv.org/pdf/2102.00650)
- Entropy weighting preliminary experiment (EA-KD Table 1, pair not identified): KD 73.33; teacher-entropy weights 75.14 (+1.81); student-entropy weights 74.76 (+1.43); inverted teacher-entropy 72.73 (-0.60); inverted student-entropy 68.90 (-4.43). Shows direction matters a lot. — [EA-KD v2](https://arxiv.org/html/2311.13621v2). A third-party summary claims +1.48 avg over KD on 7 CIFAR-100 pairs (unverified) — [papernotes EA-KD](https://en.papernotes.org/ICCV2025/object_detection/ea-kd_entropy-based_adaptive_knowledge_distillation/)
- MoKD CIFAR-100 row (columns unlabeled in snippet): 72.29, 75.54, 76.34, 70.87, 77.92, 78.11 vs ReviewKD++ 72.05, 75.66, 76.07, 70.45, 77.68, 77.93. Margins under 0.5 typically. Paper v2 withdrawn. — [MoKD](https://arxiv.org/pdf/2505.08170)

### Inferences
- With a 70.66 KD baseline, +0.43 over (your) baseline is similar to DOT (+0.41) but well below DKD/RLD (+1.3). A logit-decomposition loss (DKD/RLD-style) may offer more headroom than any per-sample CE/KD mixing.
- Gains from pure weighting on ResNet20 are likely to be sub-1 point and need 3+ seeds (CIFAR-100 seed std for ResNet20 is commonly ~0.2-0.3; this is general knowledge, not sourced).

### Gaps
- No verified ResNet56->ResNet20 numbers for WSL, ER-KD/EA-KD, TGeo-KD, CTKD, AEKT, CIST, KD-AIF.
- Primary tables unreadable due to blocked hosts.

## Q3. Gradient conflict / loss balancing between CE and KD; why per-sample correctness weighting helps or saturates

### Takeaway
DOT and MoKD say that CE and KD act like two tasks whose gradients can conflict or one can dominate; static weights can't resolve this. Teacher-error literature says wrong-teacher samples inject misleading gradient, so downweighting KD there helps, but corrections that overwrite logits destroy class correlation. Saturation: once wrong-teacher samples are handled, the remaining signal is in how much to trust a correct teacher, which a binary flag cannot express.

### Cited Findings
- DOT: adding KD to task loss makes the student a multi-task learner with a trade-off; loss-weight tuning alone doesn't fix it; DOT handles task and distillation gradients separately and gives the distillation gradient larger momentum so it dominates optimization; argues optimizing KD loss well can lower both losses. — [DOT arXiv](https://arxiv.org/pdf/2307.08436); [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_DOT_A_Distillation-Oriented_Trainer_ICCV_2023_paper.pdf)
- MoKD: defines gradient conflict (negative inner product of KD and task gradients) and gradient dominance (magnitude mismatch); treats KD as multi-objective optimization toward Pareto optimality; appendix shows less conflict/dominance than baseline on detection. Withdrawn v2. — [MoKD](https://arxiv.org/pdf/2505.08170)
- WSL: bias-variance view; KD increases bias, decreases variance; regularization samples need downweighting but not removal. — [WSL](https://arxiv.org/pdf/2102.00650)
- TGeo-KD: wrong teacher -> misleading -> smaller alpha; high student-teacher agreement -> larger alpha. — [TGeo-KD](https://arxiv.org/pdf/2312.15112)
- RLD: label-based correction ruins class correlations; keep teacher logits except those ranked >= true class (masking). — [RLD](https://arxiv.org/html/2408.07703v2)
- Tf-KD: KD acts as learned label smoothing (regularization), so for samples where a teacher is wrong, KD's regularization role still holds but its "dark knowledge" role may not. — [Tf-KD](https://arxiv.org/pdf/1909.11723)

### Inferences
- Your results (0.7/0.3 no better; soft 0.4+0.2p no better; swap logits no better) fit: the binary flag already captures most of the benefit, and finer weights on p (teacher max prob) carry little additional info because correct samples are mostly high-confidence on train data with a strong teacher; ResNet56 reaches near-100% on train images without aug but not with aug, so p differences are small.
- The conflict story suggests checking cosine(grad_CE, grad_KD) per batch, split by teacher-correct/wrong; if wrong-teacher samples have strongly negative cosine, that explains why the reversal helps (inference, not sourced).

### Gaps
- No source found on PCGrad/GradNorm applied to CE-vs-KD specifically on CIFAR-100 small students.
- No source quantifying per-sample saturation.

## Q4. Concrete ideas beyond what was tried

### Takeaway
Candidates, ordered by evidence from the literature: (1) weight KD by teacher entropy (direction matters: higher teacher entropy = larger weight per EA-KD); (2) student-teacher agreement or WSL-style student/teacher CE ratio; (3) RLD-style masked correlation for wrong-teacher samples instead of swapping; (4) per-sample/instance temperature; (5) gradient-level balancing (DOT-style momentum or PCGrad projection); (6) learned ratio network (TGeo-KD).

### Cited Findings
- Entropy direction evidence: teacher-entropy weighting +1.81, inverted -0.60 (EA-KD Table 1) — [EA-KD](https://arxiv.org/html/2311.13621v2)
- Learned alpha net with bilevel objective using student/teacher/GT geometry — [TGeo-KD](https://arxiv.org/pdf/2312.15112)
- Sample-wise temperatures + teacher-confidence/student-difficulty weights — [CIST](https://arxiv.org/html/2605.20357)
- Teacher/student target-class confidence ratio controlling target gradient — [AEKT](https://arxiv.org/pdf/2409.01679v1)
- Masked correlation for wrong teacher — [RLD](https://arxiv.org/html/2408.07703v2)
- Larger-momentum KD gradient — [DOT](https://arxiv.org/pdf/2307.08436)

### Inferences (proposals, untested)
- Margin weight: m = z_t[y] - max_{j!=y} z_t[j] (teacher true-class logit margin on the augmented image). Use alpha_KD = sigmoid(k*m) in [0.3,0.7]; unifies correct/wrong (m<0 means wrong) and gives a continuous weight on margin rather than on p (p saturates near 1, margin does not).
- Disagreement: alpha_KD = base * (1 - JS(p_s,p_t)) or mix by KL(p_t||p_s) rank within batch; WSL-style variant w_i = softmax_batch(-CE_s,i / CE_t,i), scaled to mean 1 so total KD mass is preserved.
- Entropy: multiply KD loss by normalized teacher entropy H(p_t^T)/log K (mean-normalized to 1); combine with the correct/wrong switch.
- Wrong-teacher handling: instead of swapping logits, apply RLD masking (drop classes ranked >= true class from KD target) or lower temperature only on wrong samples.
- Per-sample temperature: T_i higher for low-entropy (over-confident) teacher outputs (CIST principle).
- Gradient-level: PCGrad-style projection of the KD gradient onto the CE-nonconflicting half-space only on wrong-teacher or low-agreement samples; or DOT-style separate momentum for CE and KD.
- Evaluation protocol: at least 3 seeds; +0.43 single-seed is not distinguishable from noise.

### Gaps
- None of these proposals were found tested on ResNet56->ResNet20; all untested.
- Learned/dynamic weight (TGeo-KD) cost and gain on small pairs unknown.
