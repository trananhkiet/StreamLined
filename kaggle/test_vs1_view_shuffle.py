# Usage: python test_vs1.py OLD_SCRIPT NEW_SCRIPT   (OLD = streamlined_kd_cifar100.py before VS1 was added)
import os, sys, importlib.util, dataclasses
os.environ.update(ALLOW_CPU="1", SEED="2022", EPOCHS="300", TEACHER_SOURCE="mdistiller", LR_MILESTONES="100,188,250",
                  SAMPLE_WEIGHT_MODE="teacher_correct", SW_DISTRUST="0.5", NUM_WORKERS="0")
import numpy as np, torch, torch.nn.functional as F
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
OLD = load("old_main", sys.argv[1]); NEW = load("new_main", sys.argv[2])
assert NEW.CFG.sample_weight_mode == "teacher_correct" and NEW.CFG.sw_trust == 0.6 and NEW.CFG.sw_distrust == 0.5
VS = lambda: setattr(NEW, "CFG", dataclasses.replace(NEW.CFG, sample_weight_mode="teacher_correct_view_shuffled"))
F1B = lambda: setattr(NEW, "CFG", dataclasses.replace(NEW.CFG, sample_weight_mode="teacher_correct"))
R = {}
def realistic_batch(B=64, C=100, seed=0, p_weak=0.978, p_strong=0.40):
    g = torch.Generator().manual_seed(seed)
    view = (torch.rand(B, generator=g) < 0.5).long()                     # ConcatDataset+shuffle: random mix of views
    y = torch.randint(0, C, (B,), generator=g)
    t = torch.randn(B, C, generator=g) * 2
    right = torch.where(view == 0, torch.rand(B, generator=g) < p_weak, torch.rand(B, generator=g) < p_strong)
    wrong_cls = (y + 1 + torch.randint(0, C - 1, (B,), generator=g)) % C
    t[torch.arange(B), torch.where(right, y, wrong_cls)] += 14
    s = (0.5 * t + torch.randn(B, C, generator=g) * 1.5).requires_grad_(True)
    return s, t, y, view

# P1 regression: VS1 off == HEAD F1b (loss) bit-exact
s, t, y, v = realistic_batch(seed=1)
a = {k: x.detach() for k, x in OLD.distillation_losses(s, t, y).items()}
b = {k: x.detach() for k, x in NEW.distillation_losses(s, t, y).items()}
assert all(torch.equal(a[k], b[k]) for k in a); print("P1 F1b with VS1 off: loss bit-exact vs HEAD", {k: round(float(x), 5) for k, x in b.items()})

# P2 F1b weighting is per-sample BEFORE reduction (gradient of CE term w.r.t. logits row i == (1-w_i)(p_i - e_y)/B)
F1B(); ce = NEW.distillation_losses(s, t, y)["ce"]
(gce,) = torch.autograd.grad(ce, s)
w = torch.where(t.argmax(1).eq(y), 0.6, 0.5)
expect = (1 - w)[:, None] * (F.softmax(s.detach(), 1) - F.one_hot(y, 100)) / len(y)
assert torch.allclose(gce, expect, atol=1e-7); print("P2 F1b CE weight is applied per sample before the mean (grad check OK)")

# P3 VS1 shuffle invariants on many realistic batches
VS(); NEW.VIEW_SHUFFLE_GENERATOR.clear()
tot = {"f1b_align": 0, "vs_align": 0, "n": 0, "tc": {0: [0, 0], 1: [0, 0]}}
for k in range(400):
    s, t, y, v = realistic_batch(seed=100 + k)
    base = torch.where(t.argmax(1).eq(y), 0.6, 0.5)
    rng_before = torch.get_rng_state()
    sh = NEW.view_stratified_shuffle(base, v)
    assert torch.equal(rng_before, torch.get_rng_state()), "global RNG touched"
    correct = t.argmax(1).eq(y)
    for view in (0, 1):
        m = v == view
        assert torch.equal(torch.sort(sh[m]).values, torch.sort(base[m]).values), "multiset changed"
        assert abs(sh[m].sum().item() - base[m].sum().item()) < 1e-5  # float32 summation order only
        assert (sh[m] == 0.6).sum() == (base[m] == 0.6).sum() and (sh[m] == 0.5).sum() == (base[m] == 0.5).sum()
        tot["tc"][view][0] += ((sh == 0.6) & m & correct).sum().item(); tot["tc"][view][1] += ((sh == 0.6) & m).sum().item()
        tot.setdefault("acc", {0: [0, 0], 1: [0, 0]})[view][0] += (m & correct).sum().item(); tot["acc"][view][1] += m.sum().item()
    tot["f1b_align"] += ((base == 0.6) == correct).sum().item(); tot["vs_align"] += ((sh == 0.6) == correct).sum().item(); tot["n"] += len(y)
pw = tot["tc"][0][0] / tot["tc"][0][1]; ps = tot["tc"][1][0] / tot["tc"][1][1]
aw = tot["acc"][0][0] / tot["acc"][0][1]; as_ = tot["acc"][1][0] / tot["acc"][1][1]; R["teacher_acc_weak"], R["teacher_acc_strong"] = aw, as_
R["trust_on_correct_weak"], R["trust_on_correct_strong"] = pw, ps
R["align_f1b"], R["align_vs1"] = tot["f1b_align"] / tot["n"], tot["vs_align"] / tot["n"]
print(f"P3 per-view multiset/sum/#0.5/#0.6 preserved in 400/400 batches; global RNG untouched")
print(f"   synthetic teacher accuracy: weak {aw:.3f}, strong {as_:.3f}")
print(f"   P(teacher correct | weight 0.6): weak {pw:.3f} (random expectation = {aw:.3f}), strong {ps:.3f} (expectation = {as_:.3f}); F1b = 1.000")
print(f"   weight<->correctness agreement: F1b {R['align_f1b']:.3f}, VS1 {R['align_vs1']:.3f} (random within view: weak {aw**2+(1-aw)**2:.3f}, strong {as_**2+(1-as_)**2:.3f})")
assert abs(pw - aw) < 0.02 and abs(ps - as_) < 0.03 and aw > 0.95 and 0.3 < as_ < 0.5

# P4 view with < 2 samples untouched; non-contiguous views
base = torch.tensor([0.6, 0.5, 0.6, 0.5, 0.6]); vv = torch.tensor([1, 0, 0, 0, 0])
out = NEW.view_stratified_shuffle(base, vv); assert out[0] == base[0] and torch.equal(torch.sort(out[1:]).values, torch.sort(base[1:]).values)
print("P4 view with a single sample keeps its weight; non-contiguous view ids handled")

# P5 CE/KD/BC with VS1: BC identical to F1b; CE/KD equal the formula with the shuffled weights
s, t, y, v = realistic_batch(seed=7)
F1B(); f = NEW.distillation_losses(s, t, y)
VS(); NEW.VIEW_SHUFFLE_GENERATOR.clear(); g_ = torch.Generator().manual_seed(NEW.CFG.seed + 54321)
o = NEW.distillation_losses(s, t, y, view_ids=v)
base = torch.where(t.argmax(1).eq(y), 0.6, 0.5); sh = base.clone()
for view in (0, 1):
    idx = torch.nonzero(v == view).flatten(); sh[idx] = base[idx[torch.randperm(idx.numel(), generator=g_)]]
ce_i = F.cross_entropy(s, y, reduction="none"); kl_i = NEW.kd_loss(s, t, NEW.CFG.temperature, per_sample=True)
conf = F.softmax(t, 1).max(1).values; mfrac = conf.le(float(np.percentile(conf.numpy(), 50))).float().mean()
assert torch.equal(o["bc"], f["bc"]); assert torch.allclose(o["ce"], ((1 - sh) * ce_i).mean()); assert torch.allclose(o["kd"], (sh * kl_i).mean() * mfrac)
print("P5 VS1: BC identical to F1b; CE/KD equal (1-w~)CE_i and w~ KL_i * mask with the shuffled weights")

# P6 aggregate loss and gradient norm, F1b vs VS1 on the same 400 realistic batches
acc = {m: {"ce": [], "kd": [], "g": []} for m in ("f1b", "vs1")}
NEW.VIEW_SHUFFLE_GENERATOR.clear()
for k in range(400):
    s, t, y, v = realistic_batch(seed=1000 + k)
    for mode in ("f1b", "vs1"):
        (F1B if mode == "f1b" else VS)()
        out = NEW.distillation_losses(s, t, y, view_ids=v if mode == "vs1" else None)
        (gr,) = torch.autograd.grad(out["ce"] + out["kd"] + out["bc"], s)
        acc[mode]["ce"].append(out["ce"].item()); acc[mode]["kd"].append(out["kd"].item()); acc[mode]["g"].append(gr.norm().item())
for key in ("ce", "kd", "g"):
    a_, b_ = np.mean(acc["f1b"][key]), np.mean(acc["vs1"][key]); R[f"ratio_{key}"] = b_ / a_
    print(f"P6 mean {key:2s}: F1b {a_:.5f}  VS1 {b_:.5f}  ratio {b_/a_:.4f}")
assert all(0.9 < R[f"ratio_{k}"] < 1.1 for k in ("ce", "kd", "g")), "abnormal scaling"

# P7 ViewTagged wrapper keeps the underlying sample and RNG draws identical; collate gives a 3-tuple
import torchvision.transforms as T
from PIL import Image
class Fake(torch.utils.data.Dataset):
    def __init__(s_, tf): s_.data = np.random.default_rng(0).integers(0, 256, (10, 32, 32, 3), dtype=np.uint8); s_.targets = list(range(10)); s_.tf = tf
    def __len__(s_): return 10
    def __getitem__(s_, i): return s_.tf(Image.fromarray(s_.data[i])), s_.targets[i]
tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
torch.manual_seed(3); plain = [Fake(tf)[i] for i in range(10)]
torch.manual_seed(3); tagged = [NEW.ViewTagged(Fake(tf), 1)[i] for i in range(10)]
assert all(torch.equal(p[0], q[0]) and p[1] == q[1] and q[2] == 1 for p, q in zip(plain, tagged))
cat = torch.utils.data.ConcatDataset([NEW.ViewTagged(Fake(tf), 0), NEW.ViewTagged(Fake(tf), 1)])
assert [cat[i][2] for i in range(20)] == [0] * 10 + [1] * 10
xb, yb, vb = next(iter(torch.utils.data.DataLoader(cat, batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(0))))
assert vb.dtype == torch.int64 and xb.shape == (8, 3, 32, 32)
print("P7 ViewTagged: identical images/labels under the same RNG; ConcatDataset view ids 0 (weak) / 1 (strong); collate OK:", vb.tolist())

# P8 train_one_epoch: HEAD F1b == NEW F1b (2-tuple); NEW VS1 with SW_TRUST==SW_DISTRUST (shuffle no-op) == F1b with same weights
x = torch.randn(48, 3, 32, 32); yy = torch.randint(0, 100, (48,)); vv = torch.randint(0, 2, (48,))
torch.manual_seed(1); _te = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, 100))   # same init as run() below
with torch.no_grad(): _pred = _te(x).argmax(1)
yy = torch.where(torch.arange(48) % 2 == 0, _pred, yy)   # stand-in teacher correct on ~half -> mixed 0.6/0.5 weights
two = [(x[i:i+16], yy[i:i+16]) for i in range(0, 48, 16)]; three = [(x[i:i+16], yy[i:i+16], vv[i:i+16]) for i in range(0, 48, 16)]
def run(mod, loader, mode, trust=0.6):
    mod.CFG = dataclasses.replace(mod.CFG, sample_weight_mode=mode, sw_trust=trust, grad_log_every=1)
    torch.manual_seed(1); te = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3072, 100)).eval(); st = mod.resnet20(); te.requires_grad_(False)
    opt = torch.optim.SGD(st.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4); mod.train_one_epoch(st, te, loader, opt)
    return torch.cat([p.detach().flatten() for p in st.parameters()])
assert torch.equal(run(OLD, two, "teacher_correct"), run(NEW, two, "teacher_correct")); print("P8a 3 train steps: HEAD F1b == NEW F1b (bit-exact)")
assert torch.equal(run(NEW, two, "teacher_correct", 0.5), run(NEW, three, "teacher_correct_view_shuffled", 0.5)); print("P8b VS1 path with equal weights == F1b path (bit-exact): only the weight placement differs")
d = (run(NEW, two, "teacher_correct") - run(NEW, three, "teacher_correct_view_shuffled")).abs().max().item(); print(f"P8c VS1 vs F1b with 0.6/0.5 weights: max|dparam| {d:.2e} (>0: shuffle really changes the update)")
assert d > 0
import json; json.dump(R, open("preflight_vs1.json", "w"), indent=2)
print("PREFLIGHT PASS")
