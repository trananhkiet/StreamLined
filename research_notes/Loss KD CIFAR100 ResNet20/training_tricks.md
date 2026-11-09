# KD training-signal tricks (non-loss-term) for CIFAR-100 ResNet56 -> ResNet20

Method note: WebFetch could not reach arxiv.org, ar5iv, openaccess.thecvf.com, en.papernotes.org (DNS blocked). Most facts below come from WebSearch result summaries (not full-text reads). Treat exact numbers as "reported in search summary" unless stated. Anything not sourced is in Inferences/Gaps.

## 1. Teacher-side consistency, augmentation, mixup/CutMix, calibration/temperature, and interaction with per-sample weighting

### Takeaway
Beyer 2022 says the single most important design choices are: identical augmented view to teacher and student ("consistent"), strong augmentation (mixup), and very long schedules, with the student matching teacher outputs on the same views (function matching). Your setup (teacher sees same augmented image) already satisfies "consistent". Strong augmentation (CutMix/mixup) plus KD is reported to help across CIFAR-100 pairs including ResNet56->ResNet20 contexts, with the variance of the teacher's mean output as the proposed selection criterion. Label-smoothed (and likely over-regularized) teachers hurt distillation.

### Cited Findings
- Beyer et al. (CVPR 2022): key principles are consistent image views for teacher and student, aggressive augmentation, and very long training schedules; KD needs far more epochs than supervised training; no overfitting reported even at 1M epochs (their setting is ImageNet-scale/ big_vision, not CIFAR); ResNet-50 student reaches 82.8% top-1 ImageNet. Code: google-research/big_vision. — [arXiv 2106.05237 (search summary)](https://arxiv.org/pdf/2106.05237); [CVPR page](https://openaccess.thecvf.com/content/CVPR2022/html/Beyer_Knowledge_Distillation_A_Good_Teacher_Is_Patient_and_Consistent_CVPR_2022_paper.html)
- Wang et al. NeurIPS 2022 "What makes a good DA in KD": good DA reduces variance of the teacher's mean output probability / covariance of teacher-student CE; proxy = std of teacher's mean probability. KD+CutMix improved all evaluated pairs vs KD (>1 pt on resnet32x4/ShuffleV2); applied CutMix to 5 other top KD methods on CIFAR-100, all gained; proposes CutMixPick (entropy-based selection of mixed samples; student or teacher entropy, `--mix_n_run 2` candidates). Their ResNet56->ResNet20 CIFAR-100 plot averages 3 random runs. Exact per-pair numbers for 56->20 not retrieved. — [arXiv 2012.02909](https://arxiv.org/pdf/2012.02909); [Liner summary](https://liner.com/review/what-makes-a-good-data-augmentation-in-knowledge-distillation-a)
- Code recipe in README (vgg13->vgg8 example): `--distill kd -r 0.1 -a 0.9 -b 0 --t_output_as_target_for_input_mix --lw_mix [1,0,1] --mix_mode cutmix|mixup|cutmix_pick`. I.e., teacher is queried on the MIXED image (teacher output as target for mixed input), CE weight 0.1, KD weight 0.9. Script for all pairs: `scripts/S_test_loss_different_DA_cifar100.sh`; logs in release v0.2. — [GitHub MingSun-Tse/Good-DA-in-KD](https://github.com/MingSun-Tse/Good-DA-in-KD)
- "Understanding the role of mixup in KD": students from partially mixup-trained teachers outperform those w/o mixup on CIFAR-100 (details/numbers not retrieved). — [arXiv 2211.03946](https://arxiv.org/pdf/2211.03946)
- Stanton et al. (NeurIPS 2021): student-teacher fidelity (test agreement) for ResNet-56 teacher on CIFAR-100 stays <80% even with largest GAN-augmented distillation set; best augmentation policy reaches ~85% agreement; augmentations that help accuracy differ from those that help fidelity; with a small distillation set student can beat teacher in accuracy while having lower fidelity. — [arXiv 2106.05945](https://arxiv.org/pdf/2106.05945); [poster](https://izmailovpavel.github.io/files/gnosis/poster.pdf)
- Muller et al. 2019: teachers trained with label smoothing give worse students (penultimate clusters tight, class-similarity info lost) even though teacher accuracy/calibration improve. Implication: do not use LS-trained teacher; teacher calibration via LS is counterproductive. — [arXiv 1906.02629](https://arxiv.org/pdf/1906.02629)
- Cho & Hariharan (ICCV 2019): capacity mismatch means small students cannot mimic big teachers; sequential (teacher-assistant-style) chaining ineffective; remedy "ESKD": regularize teacher by stopping teacher training early, and stop KD near convergence so student fits training loss better. — [arXiv 1910.01348](https://arxiv.org/pdf/1910.01348); [ICCV page](https://openaccess.thecvf.com/content_ICCV_2019/html/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.html)

### Inferences
- Formulas / difficulty (my own, standard): 
  - Consistent KD (already done): L = T^2 * KL(softmax(z_t(x_aug)/T) || softmax(z_s(x_aug)/T)); difficulty trivial.
  - Mixup-KD: x = lam*xi + (1-lam)*xj; teacher forward on x (not mix of teacher outputs); CE target = lam*yi+(1-lam)*yj. Difficulty low (~15 lines). CutMix: lam = 1 - box_area/HW. With per-sample weights, the weight for a mixed image is ambiguous: use lam-weighted mixture of the two source weights, or recompute weight from the teacher's confidence on the mixed image (the latter is consistent with "teacher sees same input").
  - Function matching with long schedule: 240 ep -> 480-1000 ep; cost-only change. Beyer's regime (no CE, only KL, mixup, very long) is untested at ResNet20 scale; ResNet20 (0.27M params) may be capacity limited so extra epochs yield diminishing returns. Unverified.
  - Teacher-side temperature: fit a single scalar T_cal on held-out train-aug data (LBFGS on NLL) to measure teacher calibration; for strong aug (CutMix), teacher on mixed images is less confident, so the effective soft target is already smoother. Not literature-confirmed.
- Per-sample weighting + strong aug: weighting by teacher confidence/entropy on strong views will up-weight easy (weak-aug-like) samples and may cancel the regularizing effect of strong aug; Wang et al.'s CutMixPick (select mixes by entropy) is the closest published precedent for sample-level selection.

### Gaps
- Could not read the Beyer ablation tables (mixup vs none, epochs) or any CIFAR-scale results of function matching; no paper found that does Beyer-style protocol on ResNet56->ResNet20.
- Exact KD+CutMix 56->20 number from Wang et al. not retrieved (PDF fetch blocked).
- No source found on teacher temperature/calibration (post-hoc scaling) improving CIFAR-100 KD students.

## 2. Student-side: label smoothing, logit penalty, SAM, EMA, SWA, self-/born-again/online, two-stage, curriculum, temperature schedules

### Takeaway
Evidence retrieved is thin for EMA/SWA/logit-penalty with KD at this pair. Best-supported items: curriculum/dynamic temperature (CTKD, ~+0.5 on KD), logit standardization (+0.35 on DKD for 56->20; plug-in z-score of logits before softmax), swapped-logit/multi-level methods reaching ~72.3-72.7, SAM+self-distillation (SADT), and born-again self-distillation (gains on same-architecture students). LS and KD overlap as regularizers (Tf-KD).

### Cited Findings
- CTKD: learnable temperature via adversarial gradient-reversal with curriculum (difficulty increases); plug-in, negligible cost; example row KD 70.83 -> 71.32 with CTKD (pair not identified in snippet). Code: zhengli97/CTKD. — [arXiv 2211.16231](https://arxiv.org/pdf/2211.16231); [project](https://zhengli97.github.io/CTKD)
- Logit standardization (Sun et al. CVPR 2024): subtract per-sample logit mean and divide by std before softmax (for teacher and student, base temperature); 56->20 DKD 71.97 -> 72.32 (+0.35); KD baseline 70.66 for 56->20 (README column order inferred, not confirmed). MLKD 72.19 baseline. Code: sunshangquan/logit-standardization-KD. — [arXiv 2403.01427](https://arxiv.org/pdf/2403.01427); [GitHub](https://www.github.com/sunshangquan/logit-standardization-KD); [LEMMA README](https://github.com/AIRLABkhu/LEMMA)
- Dynamic Temperature KD (arXiv 2404.12711) and Adaptive temperature (ATKD) exist as per-sample temperature approaches; numbers not retrieved. — [2404.12711](https://arxiv.org/pdf/2404.12711)
- SLD (Swapped Logit Distillation): temperature set [1..6] median 3.5, single swap; student-swap loss warm-up 150 epochs (secondary source). — [arXiv 2504.20108](https://arxiv.org/pdf/2504.20108); [emergentmind](https://www.emergentmind.com/topics/logit-swap-self-distillation)
- Yuan et al. (CVPR 2020) KD as learned label smoothing; Tf-KD (self or hand-designed soft target) gives up to +0.65 ImageNet over baselines and beats plain LSR; soft-target regularization role "equally or more important" than class-similarity information. Follow-up notes Tf-KD hyperparameters hard to tune. — [arXiv 1909.11723](https://arxiv.org/pdf/1909.11723); [Applied Sci. follow-up](https://mdpi.com/2076-3417/11/10/4699)
- SADT: combines SAM with self-distillation (NeurIPS 2022), claims better convergence/generalization; CIFAR numbers not retrieved. — [arXiv 2211.00310](https://arxiv.org/pdf/2211.00310)
- Born-Again Networks: same-architecture student, loss = CE + match teacher; DenseNet-90-60 CIFAR-100 error 17.69 -> 16.00 (third-party summary); ensembling generations adds more. — [arXiv 1805.04770](https://arxiv.org/pdf/1805.04770.pdf)
- FFSD online distillation: ResNets on CIFAR-100, 300 epochs, batch 128 (no EMA/SWA comparison). — [arXiv 2103.14473](https://arxiv.org/pdf/2103.14473)

### Inferences
(All implementation estimates are mine, not cited.)
- Logit standardization: ~5 lines, no new params; formula z' = (z - mean(z)) / (std(z)+eps) / T; cheapest literature-backed target-changing trick. Low risk, expected ~+0.3 (reported on DKD).
- Temperature schedule/curriculum: linear ramp T from low to high or CTKD gradient-reversal layer; low-medium difficulty (CTKD code reusable). Expected gain 0.3-0.5 over KD baseline, may shrink on strong baselines.
- Student EMA (decay ~0.999, eval EMA weights; BN stats re-estimated or EMA buffers): ~10 lines, near-free; commonly reduces epoch-to-epoch noise (directly useful given your 0.2-0.3 noise) but no cited KD-specific gain here. SWA over last ~10-20% of epochs at constant LR: similar; but with cosine/step to near-zero LR benefit small. Needs BN re-calibration.
- SAM + KD: 2x compute, rho ~0.05 typical (my recollection, not verified here); SADT is the cited precedent; medium difficulty; ResNet20 is low-capacity so flatness gain uncertain.
- Logit penalty / norm regularization (e.g. lambda*mean(||z_s||^2) or student logit-norm): trivial to add, but it is a loss term, flagged as outside scope; no KD source retrieved.
- Born-again / two-stage: stage 1 KD student (ResNet20) becomes teacher/ensemble member for stage 2 same-size student; doubles training time; plausible +0.3-0.8 but no ResNet20 number found.
- Label smoothing on the student's CE branch overlaps with KD's own smoothing effect (Yuan et al.); expect little additive gain.

### Gaps
- No retrieved paper reporting EMA, SWA, or logit-penalty applied to KD on CIFAR-100 ResNet56->ResNet20.
- SAM+KD numbers for CIFAR-100 not retrieved.
- CTKD's own 56->20 Table 1 value not retrieved.

## 3. Overfitting signals: KD as regularizer, late shift towards CE (train +1.55, no test gain)

### Takeaway
The literature supports your reading: KD's benefit is substantially regularization/soft-target smoothing (Yuan 2020), small students have limited ability to fit teacher outputs (Cho 2019), and fidelity is low even when accuracy rises (Stanton 2021) - so train-accuracy gains from more CE weight are not expected to translate to test gains.

### Cited Findings
- KD as learned label smoothing / regularization; Tf-KD gains show non-teacher soft targets also help. — [arXiv 1909.11723](https://arxiv.org/pdf/1909.11723)
- Stanton 2021: students rarely match teacher (fidelity <80% at ResNet-56 CIFAR-100 even with big augmented sets); optimization difficulty, not generalization gap, is main bottleneck; better fidelity does not always mean better accuracy. — [arXiv 2106.05945](https://arxiv.org/pdf/2106.05945)
- Cho & Hariharan: early-stopped KD (stop near convergence, so student fits train loss better) and capacity mismatch. — [arXiv 1910.01348](https://arxiv.org/pdf/1910.01348)
- Beyer 2022: with consistent views and strong aug, no overfitting even at very long schedules -> regularization from teacher targets on varied views, not from short training. — [arXiv 2106.05237](https://arxiv.org/pdf/2106.05237)

### Inferences
- Pure train-accuracy increase without test gain at low LR = student fitting hard labels of (weakly) augmented samples; a stronger/more diverse view stream (CutMix/mixup, more epochs) is the literature's answer, not a CE shift. Use the train-vs-test gap and teacher-student agreement (fidelity) on the test set as diagnostics, not train accuracy.

### Gaps
- No paper found reporting specifically a late-stage CE re-weighting schedule on KD.

## 4. Seed variance and significance claims

### Takeaway
Standard practice is weak: DKD reports averages over 5 trials; Wang et al. average 3 runs with shaded std; many papers (SLD, MLKD comparisons) report single numbers with 0.1-0.3 pt margins. No paper found with formal significance tests. Your 0.2-0.3 epoch noise equals the gaps being claimed in the literature (e.g., +0.35 for logit standardization on DKD).

### Cited Findings
- DKD 56->20 71.97 is an average over 5 trials (per search summary of the DKD protocol). — [DKD arXiv 2203.08679](https://arxiv.org/pdf/2203.08679); [mdistiller](https://github.com/megvii-research/mdistiller)
- Wang et al. 2022: 3 random runs, shaded stddev for 56->20 plot. — [arXiv 2012.02909](https://arxiv.org/pdf/2012.02909)
- Online-distillation CutMix paper: 3 runs, shaded std. — [arXiv 2206.12370](https://arxiv.org/pdf/2206.12370)
- NormKD 71.40 for 56->20 was reproduced by SLD authors rather than copied (vs reported); SLD paper's epoch count inconsistent (240 vs 480 stated). — [arXiv 2504.20108](https://arxiv.org/pdf/2504.20108)

### Inferences
- With std per run of roughly 0.2-0.3 (consistent with your noise), 3 seeds give SE ~0.15; need >=5 seeds and averaging last-N epochs (or EMA weights) to resolve 0.3 gains. Paired seeds (same data order) reduce variance. These stat claims are my own reasoning.

### Gaps
- Exact seed std for ResNet20 KD on CIFAR-100 not retrieved from any source.

## 5. Claims above 72.5 on ResNet56 -> ResNet20 (CIFAR-100)

### Takeaway
Reported numbers cluster at 71.2-72.7. >72.5 claims found: SLD 72.67 (swapped-logit, from search summary) and SKD "72.50" (CVPR 2026 Streamlined KD; column label unconfirmed). Both are method changes plus longer/extra-loss recipes, not pure schedule tricks.

### Cited Findings
- KD baseline 70.66; DKD 71.97; DKD+logit-std 72.32; MLKD 72.19; LS-MLKD 72.33; NormKD (reproduced) 71.40; GDKD 71.65. — [2403.01427](https://arxiv.org/pdf/2403.01427); [SLD paper](https://arxiv.org/pdf/2504.20108); search summaries
- SLD 72.67 on this pair (stated in search summary alongside LS-MLKD; not confirmed in table). — [arXiv 2504.20108](https://arxiv.org/pdf/2504.20108)
- Streamlined KD (CVPR 2026) table row for ResNet56->ResNet20: 71.74 | 71.17 | 72.21 | 72.00 | 72.50, labels unconfirmed; 72.50 likely its own method. — [papernotes summary via search](https://en.papernotes.org/CVPR2026/model_compression/streamlined_knowledge_distillation/) (page unreachable by fetch)
- LS-MLKD/MLKD protocol: batch 64, 240 epochs (some say 480 for SLD). — [arXiv 2504.20108](https://arxiv.org/pdf/2504.20108)
- DeepKD (2505.15133) reports gains over MLKD with top-k; not matched to this pair. — [arXiv 2505.15133](https://arxiv.org/pdf/2505.15133)

### Inferences
- >72.5 appears reachable only with multi-level/swap logit methods, multi-temperature, plus possibly doubled epochs; pure data-side tricks (CutMix) reported as gains of ~0.5-1 but unverified for 56->20.

### Gaps
- Could not verify SLD's 72.67 or SKD's 72.50 in primary tables; no >72.5 claim from EMA/SWA/SAM/two-stage found. Teacher accuracy (typically 72.34 for ResNet56 in mdistiller) not verified here.
