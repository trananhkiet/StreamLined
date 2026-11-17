# Usage: python test_vw1_view_fixed.py OLD_SCRIPT NEW_SCRIPT   (OLD = script at commit a7dfc76, before VW1 was added; it already has VS1)
import os, sys, importlib.util, dataclasses
os.environ.update(ALLOW_CPU="1", SEED="2022", EPOCHS="300", TEACHER_SOURCE="mdistiller", LR_MILESTONES="100,188,250",
                  SAMPLE_WEIGHT_MODE="teacher_correct", SW_DISTRUST="0.5", NUM_WORKERS="0")
import numpy as np, torch, torch.nn.functional as F
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
OLD = load("old_main", sys.argv[1]); NEW = load("new_main", sys.argv[2])
assert NEW.CFG.sw_view_weak == 0.598 and NEW.CFG.sw_view_strong == 0.540
g = torch.Generator().manual_seed(0)
B, C = 64, 100
s = torch.randn(B, C, generator=g).requires_grad_(True); t = torch.randn(B, C, generator=g) * 3; y = torch.randint(0, C, (B,), generator=g)
v = (torch.rand(B, generator=g) < 0.5).long(); t[:20, :] += 0; t[torch.arange(20), y[:20]] += 15
def L(mod, mode, **kw):
    mod.CFG = dataclasses.replace(mod.CFG, sample_weight_mode=mode)
    if hasattr(mod, "VIEW_SHUFFLE_GENERATOR"): mod.VIEW_SHUFFLE_GENERATOR.clear()
    return {k: x.detach() for k, x in mod.distillation_losses(s, t, y, **kw).items()}
# R1 F1b and VS1 unchanged vs previous script (bit-exact)
for mode, kw in (("teacher_correct", {}), ("teacher_correct_view_shuffled", {"view_ids": v})):
    a, b = L(OLD, mode, **kw), L(NEW, mode, **kw); assert all(torch.equal(a[k], b[k]) for k in a), mode
print("R1 F1b and VS1 losses bit-exact vs previous script")
# R2 VW1 weights and loss formula
NEW.CFG = dataclasses.replace(NEW.CFG, sample_weight_mode="view_fixed")
w = NEW.teacher_correct_weights(t, y, v)
assert torch.equal(w, torch.where(v == 0, torch.tensor(0.598), torch.tensor(0.540)))
o = L(NEW, "view_fixed", view_ids=v)
ce_i = F.cross_entropy(s, y, reduction="none"); kl_i = NEW.kd_loss(s, t, 4.0, per_sample=True)
conf = F.softmax(t, 1).max(1).values; mfrac = conf.le(float(np.percentile(conf.numpy(), 50))).float().mean()
assert torch.allclose(o["ce"], ((1 - w) * ce_i).mean()) and torch.allclose(o["kd"], (w * kl_i).mean() * mfrac)
f = L(NEW, "teacher_correct"); assert torch.equal(o["bc"], f["bc"])
print("R2 VW1: weak 0.598 / strong 0.540 for every sample regardless of teacher correctness; CE/KD formula OK; BC identical")
# R3 VW1 ignores teacher correctness: flipping teacher argmax does not change weights
t2 = t.clone(); t2[torch.arange(B), y] += 50
NEW.CFG = dataclasses.replace(NEW.CFG, sample_weight_mode="view_fixed")
assert torch.equal(NEW.teacher_correct_weights(t2, y, v), w); print("R3 VW1 weights independent of teacher correctness")
# R4 missing view ids -> error
try: NEW.teacher_correct_weights(t, y, None); raise SystemExit("expected error")
except ValueError: print("R4 VW1 without view ids raises")
# R5 train path: VW1 with weak=strong=w equals F1b with SW_TRUST=SW_DISTRUST=w (bit-exact); VW1 0.598/0.540 differs
x = torch.randn(48, 3, 32, 32); yy = torch.randint(0, 100, (48,)); vv = torch.randint(0, 2, (48,))
two = [(x[i:i+16], yy[i:i+16]) for i in range(0, 48, 16)]; three = [(x[i:i+16], yy[i:i+16], vv[i:i+16]) for i in range(0, 48, 16)]
def run(mode, loader, **cfg):
    NEW.CFG = dataclasses.replace(NEW.CFG, sample_weight_mode=mode, grad_log_every=1, **cfg)
    torch.manual_seed(1); te = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, 100)).eval(); st = NEW.resnet20()
    opt = torch.optim.SGD(st.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4); NEW.train_one_epoch(st, te, loader, opt)
    return torch.cat([p.detach().flatten() for p in st.parameters()])
a = run("teacher_correct", two, sw_trust=0.57, sw_distrust=0.57); b = run("view_fixed", three, sw_view_weak=0.57, sw_view_strong=0.57)
assert torch.equal(a, b); print("R5 VW1 with equal view weights == F1b with equal weights (3 train steps, bit-exact)")
NEW.pop_sample_weight_stats(); c = run("view_fixed", three, sw_view_weak=0.598, sw_view_strong=0.540); d = (a - c).abs().max().item(); assert d > 0
st = NEW.pop_sample_weight_stats(); print(f"R6 VW1 0.598/0.540 changes the update (max|dparam| {d:.2e}); epoch stats keys:", sorted(k for k in st if k.startswith("vs_")))
assert abs(st["vs_w_weak"] - 0.598) < 1e-6 and abs(st["vs_w_strong"] - 0.540) < 1e-6
print("VW1 PREFLIGHT PASS")
