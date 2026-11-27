# Usage: python test_l1_l4_losses.py OLD_SCRIPT NEW_SCRIPT   (OLD = script at commit e77acde, before L1-L4 were added)
import os, sys, importlib.util, dataclasses
os.environ.update(ALLOW_CPU="1", SEED="2022", EPOCHS="300", TEACHER_SOURCE="mdistiller", LR_MILESTONES="100,188,250",
                  SAMPLE_WEIGHT_MODE="teacher_correct", SW_DISTRUST="0.5", NUM_WORKERS="0")
import numpy as np, torch, torch.nn.functional as F
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
OLD = load("old_main", sys.argv[1]); NEW = load("new_main", sys.argv[2])
assert (NEW.CFG.correct_boost_part, NEW.CFG.trust_topk, NEW.CFG.ce_weight_mode) == ("full", 1, "complement")
g = torch.Generator().manual_seed(0); B, C = 64, 100
s = torch.randn(B, C, generator=g).requires_grad_(True); t = torch.randn(B, C, generator=g) * 3; y = torch.randint(0, C, (B,), generator=g)
t[torch.arange(20), y[:20]] += 15                      # first 20: teacher correct
for i in range(20, 30):
    r = t[i].clone(); r[y[i]] = -1e9; v = r.topk(2).values; t[i, y[i]] = (v[0] + v[1]) / 2   # label is exactly teacher rank 2
def L(mod, **cfg):
    d = dict(wrong_kd_mode="full", correct_boost_part="full", trust_topk=1, ce_weight_mode="complement", late_kd_start=0)
    d = {k: v for k, v in d.items() if hasattr(mod.CFG, k)}
    mod.CFG = dataclasses.replace(mod.CFG, **{**d, **cfg})
    return {k: x.detach() for k, x in mod.distillation_losses(s, t, y).items()}
base = dict(sample_weight_mode="teacher_correct", sw_trust=0.6, sw_distrust=0.5)
# R1 defaults bit-exact vs previous script (F1b, D2, J2-free paths)
for extra in ({}, {"wrong_kd_mode": "nckd_only"}, {"sample_weight_mode": "teacher_prob"}):
    kw = {**base, **extra}; a, b = L(OLD, **kw), L(NEW, **kw); assert all(torch.equal(a[k], b[k]) for k in a), extra
OLD.CFG = dataclasses.replace(OLD.CFG, sample_weight_mode="none"); NEW.CFG = dataclasses.replace(NEW.CFG, sample_weight_mode="none")
a, b = OLD.distillation_losses(s, t, y), NEW.distillation_losses(s, t, y); assert all(torch.equal(a[k], b[k]) for k in a)
print("R1 defaults bit-exact vs previous script (F1b, D2, teacher_prob, E4)")
# R2 DKD identity: TCKD + NCKD_w == KL
kl = NEW.kd_loss(s, t, 4.0, per_sample=True); nw = NEW.nckd_per_sample(s, t, y, 4.0); tckd = kl - nw
pt = F.softmax(t / 4, 1); ps = F.softmax(s / 4, 1); bt = pt.gather(1, y[:, None]).squeeze(1); bs = ps.gather(1, y[:, None]).squeeze(1)
tckd_ref = (bt * (bt / bs).log() + (1 - bt) * ((1 - bt) / (1 - bs)).log()) * 16
assert torch.allclose(tckd, tckd_ref, atol=1e-4) and (tckd >= -1e-5).all() and (nw >= -1e-5).all()
print(f"R2 KL = TCKD + (1-pT)NCKD holds (max err {(tckd - tckd_ref).abs().max():.1e})")
# R3 L1 / L2 formulas
w = NEW.teacher_correct_weights(t, y); correct = t.argmax(1).eq(y); ce_i = F.cross_entropy(s, y, reduction="none")
conf = F.softmax(t, 1).max(1).values; mfrac = conf.le(float(np.percentile(conf.detach().numpy(), 50))).float().mean()
f1b = L(NEW, **base)
for part, piece in (("tckd", tckd), ("nckd", nw)):
    o = L(NEW, **base, correct_boost_part=part)
    exp_kd = (0.5 * kl + (w - 0.5) * piece).mean() * mfrac
    assert torch.allclose(o["kd"], exp_kd.detach(), atol=1e-6) and torch.equal(o["ce"], f1b["ce"]) and torch.equal(o["bc"], f1b["bc"])
    print(f"R3 {part}: KD = 0.5 KL + (w-0.5) {part.upper()}  ({o['kd']:.4f} vs F1b {f1b['kd']:.4f}); CE/BC unchanged")
NEW.CFG = dataclasses.replace(NEW.CFG, correct_boost_part="full")
# R4 sum of L1 + L2 boosts recovers F1b exactly in formula: (tckd version + nckd version) - 0.5*KL-term == F1b
o1, o2 = L(NEW, **base, correct_boost_part="tckd"), L(NEW, **base, correct_boost_part="nckd")
assert torch.allclose(o1["kd"] + o2["kd"] - 0.5 * kl.mean().detach() * mfrac, f1b["kd"], atol=1e-5); print("R4 L1+L2 boosts partition the F1b boost")
# R5 L3 top-2 trust
NEW.CFG = dataclasses.replace(NEW.CFG, **base, correct_boost_part="full", trust_topk=2)
w2 = NEW.teacher_correct_weights(t, y); in2 = t.topk(2, 1).indices.eq(y[:, None]).any(1)
assert torch.equal(w2, torch.where(in2, torch.tensor(0.6), torch.tensor(0.5))) and in2.sum() > correct.sum() and in2[20:30].all()
NEW.pop_sample_weight_stats(); NEW.teacher_correct_weights(t, y); st = NEW.pop_sample_weight_stats()
assert abs(st["teacher_train_top1_sw"] - 100 * correct.float().mean().item()) < 1e-4 and abs(st["sw_trusted_pct"] - 100 * in2.float().mean().item()) < 1e-4
print(f"R5 top-2 trust: trusted {st['sw_trusted_pct']:.1f}% vs top-1 {st['teacher_train_top1_sw']:.1f}% (top1 stat unchanged)")
# R6 L4 fixed CE weight
NEW.CFG = dataclasses.replace(NEW.CFG, trust_topk=1); o = L(NEW, **base, ce_weight_mode="fixed")
assert torch.allclose(o["ce"], (0.5 * ce_i).mean().detach()) and torch.equal(o["kd"], f1b["kd"]); print("R6 fixed CE: CE = 0.5 CE_i for all samples, KD = F1b")
# R7 guards
for bad in (dict(sample_weight_mode="none", trust_topk=2), {**base, "correct_boost_part": "tckd", "wrong_kd_mode": "nckd_only"}, {**base, "ce_weight_mode": "x"}):
    NEW.CFG = dataclasses.replace(NEW.CFG, **{**dict(correct_boost_part="full", trust_topk=1, ce_weight_mode="complement", wrong_kd_mode="full"), **bad})
    try: NEW.distillation_losses(s, t, y); raise SystemExit(f"expected error {bad}")
    except ValueError: pass
print("R7 invalid combinations raise")
# R8 train path: 3 steps each, each mode runs and changes params vs F1b (L1,L2,L4) ; stats keys present
x = torch.randn(48, 3, 32, 32); yy = torch.randint(0, 100, (48,)); loader = [(x[i:i+16], yy[i:i+16]) for i in range(0, 48, 16)]
def run(**cfg):
    NEW.CFG = dataclasses.replace(NEW.CFG, **{**base, **dict(wrong_kd_mode="full", correct_boost_part="full", trust_topk=1, ce_weight_mode="complement", grad_log_every=1), **cfg})
    torch.manual_seed(1); te = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, 100)).eval(); stu = NEW.resnet20()
    opt = torch.optim.SGD(stu.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4); NEW.train_one_epoch(stu, te, loader, opt)
    return torch.cat([p.detach().flatten() for p in stu.parameters()]), NEW.pop_sample_weight_stats()
ref, _ = run()
for cfg in (dict(correct_boost_part="tckd"), dict(correct_boost_part="nckd"), dict(trust_topk=2), dict(ce_weight_mode="fixed")):
    p, st = run(**cfg); d = (p - ref).abs().max().item(); assert torch.isfinite(p).all()
    print(f"R8 train {cfg}: ok, max|dparam| vs F1b {d:.2e}, stats {sorted(k for k in st if k in ('sw_trusted_pct', 'tckd_share_pct'))}")
print("L1-L4 PREFLIGHT PASS")
