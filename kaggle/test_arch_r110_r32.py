# Usage: python test_arch_r110_r32.py OLD_SCRIPT NEW_SCRIPT R110_CKPT R56_CKPT   (OLD = script at be913f5, before TEACHER_ARCH/STUDENT_ARCH; ckpts = mdistiller cifar_teachers resnet110/resnet56 ckpt_epoch_240.pth)
import os, sys, importlib.util, dataclasses, shutil
os.environ.update(ALLOW_CPU="1", SEED="2022", EPOCHS="300", TEACHER_SOURCE="mdistiller", LR_MILESTONES="100,188,250",
                  SAMPLE_WEIGHT_MODE="teacher_correct", SW_DISTRUST="0.5", NUM_WORKERS="0")
import torch
from pathlib import Path
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
OLD = load("old_main", sys.argv[1]); NEW = load("new_main", sys.argv[2]); R110, R56 = Path(sys.argv[3]), Path(sys.argv[4])
assert NEW.CFG.teacher_arch == "resnet56" and NEW.CFG.student_arch == "resnet20"
# A1 defaults unchanged: student/teacher init bit-exact, F1b loss bit-exact
torch.manual_seed(5); a = OLD.resnet20(); torch.manual_seed(5); b = NEW.MODEL_FACTORY[NEW.CFG.student_arch]()
assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
s = torch.randn(64, 100, requires_grad=True); t = torch.randn(64, 100) * 3; y = torch.randint(0, 100, (64,))
la = OLD.distillation_losses(s, t, y); lb = NEW.distillation_losses(s, t, y)
assert all(torch.equal(la[k], lb[k]) for k in la); print("A1 defaults: ResNet20 init and F1b loss bit-exact vs previous script")
# A2 real mdistiller checkpoints strict-load
def strict_load(arch, path):
    m = NEW.MODEL_FACTORY[arch](); st = NEW.load_state_dict(path)
    for k, v in m.state_dict().items():
        if k.endswith("num_batches_tracked") and k not in st: st[k] = torch.zeros_like(v)
    m.load_state_dict(st, strict=True); return m
m110 = strict_load("resnet110", R110); m56 = strict_load("resnet56", R56)
n = lambda m: sum(p.numel() for p in m.parameters())
print(f"A2 strict load OK: ResNet110 {n(m110)/1e6:.3f}M params, ResNet56 {n(m56)/1e6:.3f}M, ResNet32 {n(NEW.resnet32())/1e6:.3f}M, ResNet20 {n(NEW.resnet20())/1e6:.3f}M")
raw = NEW._torch_load(R110); print("   checkpoint keys:", [k for k in raw if k != "model"][:6], "stored acc:", float(raw.get("best_acc", raw.get("accuracy", float("nan")))) if isinstance(raw, dict) else None)
# A3 hash check through the real code path (cached file in WORK_DIR)
for arch, src, fname in (("resnet110", R110, "mdistiller_resnet110_vanilla_ckpt_epoch_240.pth"), ("resnet56", R56, "mdistiller_resnet56_vanilla_ckpt_epoch_240.pth")):
    shutil.copy(src, NEW.WORK_DIR / fname); NEW.CFG = dataclasses.replace(NEW.CFG, teacher_arch=arch)
    assert NEW.obtain_mdistiller_teacher() == NEW.WORK_DIR / fname
print("A3 obtain_mdistiller_teacher: SHA-256 verified for resnet110 and resnet56 (resnet56 file name unchanged)")
# A4 invalid combos
NEW.CFG = dataclasses.replace(NEW.CFG, teacher_source="skd", teacher_arch="resnet110")
try: NEW.obtain_teacher_checkpoint(); raise SystemExit("expected error")
except ValueError: print("A4 TEACHER_SOURCE=skd with ResNet110 raises (only ResNet56 is released by SKD)")
# A5 three F1b training steps with ResNet110 teacher -> ResNet32 student
NEW.CFG = dataclasses.replace(NEW.CFG, teacher_source="mdistiller", teacher_arch="resnet110", student_arch="resnet32", grad_log_every=1)
m110.eval().requires_grad_(False); torch.manual_seed(1); st = NEW.MODEL_FACTORY["resnet32"]()
x = torch.randn(48, 3, 32, 32); yy = torch.randint(0, 100, (48,)); loader = [(x[i:i+16], yy[i:i+16]) for i in range(0, 48, 16)]
opt = torch.optim.SGD(st.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4); before = [p.detach().clone() for p in st.parameters()]
m = NEW.train_one_epoch(st, m110, loader, opt); assert any(not torch.equal(p, q) for p, q in zip(before, st.parameters()))
print("A5 ResNet110 -> ResNet32 F1b: 3 train steps OK, metrics", {k: round(v, 3) for k, v in m.items()})
print("ARCH PREFLIGHT PASS")
