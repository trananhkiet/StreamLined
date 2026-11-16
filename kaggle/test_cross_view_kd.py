# Usage (from a scratch dir): git show HEAD~N:kaggle/streamlined_kd_cifar100.py > old_main.py; python test_cross_view_kd.py old_main.py <repo>/kaggle/streamlined_kd_cifar100.py
# old_main.py = the script BEFORE CROSS_VIEW_KD was added (commit 40d00d2).
import os, sys, importlib.util, dataclasses, copy
os.environ.update(ALLOW_CPU="1", SEED="2022", EPOCHS="300", TEACHER_SOURCE="mdistiller", LR_MILESTONES="100,188,250", NUM_WORKERS="0")
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
OLD = load("old_main", sys.argv[1]); NEW = load("new_main", sys.argv[2])
assert NEW.CFG.cross_view_kd is False
torch.manual_seed(0)
B, C = 64, 100
s = torch.randn(B, C, requires_grad=True); t = torch.randn(B, C) * 3; tw = torch.randn(B, C) * 3; y = torch.randint(0, C, (B,))
def losses(mod, **kw):
    out = mod.distillation_losses(s, t, y, **kw)
    return {k: v.detach().clone() for k, v in out.items()}
# T1 mode off == HEAD
a, b = losses(OLD), losses(NEW)
assert all(torch.equal(a[k], b[k]) for k in a), (a, b); print("T1 mode-off loss bit-exact vs HEAD:", {k: float(v) for k, v in b.items()})
# T2 kd target = original teacher logits -> identical
c = losses(NEW, kd_teacher_logits=t.clone()); assert all(torch.equal(a[k], c[k]) for k in a); print("T2 cross path with unchanged target bit-exact")
# T3 changed target: CE/BC identical; KD = kd_loss(s, tw) * mask(from t)
d = losses(NEW, kd_teacher_logits=tw)
assert torch.equal(d["ce"], a["ce"]) and torch.equal(d["bc"], a["bc"]) and not torch.equal(d["kd"], a["kd"])
conf = F.softmax(t, 1).max(1).values; mask = conf.le(float(np.percentile(conf.numpy(), 50)))
expect = NEW.CFG.kd_weight * (NEW.kd_loss(s, tw, NEW.CFG.temperature) * mask).mean()
assert torch.allclose(d["kd"], expect.detach(), atol=0, rtol=0), (d["kd"], expect); print("T3 only KD changes; mask/BC/CE from original teacher logits")
# T4 train_one_epoch: OLD vs NEW(off) vs NEW(on with aux == x) -> identical params after 3 steps
x = torch.randn(3 * 16, 3, 32, 32); yy = torch.randint(0, C, (48,))
two = [(x[i:i+16], yy[i:i+16]) for i in range(0, 48, 16)]
five = [(x[i:i+16], yy[i:i+16], x[i:i+16].clone(), torch.arange(i, i+16), torch.tensor([0, 1] * 8)) for i in range(0, 48, 16)]
def run(mod, loader, cv):
    mod.CFG = dataclasses.replace(mod.CFG, grad_log_every=1, **({"cross_view_kd": cv} if hasattr(mod.CFG, "cross_view_kd") else {}))
    torch.manual_seed(1); st = mod.resnet20(); te = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, 100)).eval(); te.requires_grad_(False)  # sane-scale stand-in teacher (untrained ResNet56 saturates softmax)
    opt = torch.optim.SGD(st.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    mod.train_one_epoch(st, te, loader, opt)
    return torch.cat([p.detach().flatten() for p in st.parameters()])
p_old, p_off, p_on = run(OLD, two, False), run(NEW, two, False), run(NEW, five, True)
assert torch.equal(p_old, p_off), "mode-off training diverged from HEAD"
d_on = (p_off - p_on).abs().max().item()
assert d_on < 1e-4, f"cross path with aux==x changed training: {d_on}"
print(f"T4 3 training steps: HEAD == NEW(off) bit-exact; NEW(on, aux==x) max|dparam|={d_on:.2e} (float rounding of subset teacher forward)")
st = NEW.pop_cv_stats(); assert abs(st["cv_strong_pct"] - 50.0) < 1e-9; print("   cv stats:", st)
# T4b aux != x must change training (KD target of strong rows really is replaced)
five_b = [(a_, b_, torch.randn_like(c_), d_, e_) for a_, b_, c_, d_, e_ in five]
d_b = (p_on - run(NEW, five_b, True)).abs().max().item()
assert d_b > 100 * max(d_on, 1e-7), d_b; print(f"T4b different weak view changes the update: max|dparam|={d_b:.2e} (target really replaced)")
# T5 pairing of CrossViewCIFAR
class Base: pass
base = Base(); rng = np.random.default_rng(0)
base.data = rng.integers(0, 256, size=(20, 32, 32, 3), dtype=np.uint8); base.targets = list(rng.integers(0, 100, 20))
import torchvision.transforms as T
weak = T.Compose([T.RandomHorizontalFlip(), T.ToTensor()]); strong = T.Compose([T.RandomHorizontalFlip(), T.ColorJitter(0.5), T.ToTensor()])
ws, ss = NEW.CrossViewCIFAR(base, weak, None, 0), NEW.CrossViewCIFAR(base, strong, weak, 1)
for i in range(20):
    x1, y1, xt1, i1, f1 = ws[i]; x2, y2, xt2, i2, f2 = ss[i]
    orig = T.ToTensor()(Image.fromarray(base.data[i])); flip = orig.flip(-1)
    assert xt1 is x1 and f1 == 0 and f2 == 1 and i1 == i2 == i and y1 == y2 == base.targets[i]
    assert torch.equal(xt2, orig) or torch.equal(xt2, flip), "aux weak view is not from the same image"
    others = [T.ToTensor()(Image.fromarray(base.data[j])) for j in range(20) if j != i]
    assert not any(torch.equal(xt2, o) or torch.equal(xt2, o.flip(-1)) for o in others)
print("T5 pairing: aux weak view always comes from the same original image/label/index (20/20)")
loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([ws, ss]), batch_size=8, shuffle=True)
xb, yb, xtb, ib, sb = next(iter(loader)); assert xb.shape == xtb.shape and sb.dtype == torch.int64
print("T5b ConcatDataset + DataLoader collates the 5-tuple:", tuple(xb.shape), sb.tolist())
print("ALL TESTS PASSED")
