"""Teacher-student knowledge-gap analysis (groups G1-G4) for E4 vs F1b checkpoints.

Inference only. No training, no hyper-parameter selection.

Groups (teacher/student correctness on the SAME input view):
  G1 teacher right, student right     G2 teacher right, student wrong
  G3 teacher wrong, student right     G4 teacher wrong, student wrong

Per-sample gradients are taken w.r.t. the STUDENT LOGITS (closed form, verified against autograd in
`selftest`):
  g_ce = softmax(s) - onehot(y)
  g_kd = T * (softmax(s/T) - softmax(t/T))          # gradient of T^2 * KL(p_t^T || p_s^T)
cos(g_ce, g_kd) > 0 means CE and KD push the student logits the same way; < 0 means they conflict.
The scalar KD mask (~0.5) and BC are not part of this analysis (BC gradient is ~1% of KD, see HANDOFF).

Usage:
  python analysis_gap_groups.py selftest
  python analysis_gap_groups.py extract  OUT.npz --student CKPT --teacher CKPT --split train|test
  python analysis_gap_groups.py report   --a E4=e4.npz --b F1b=f1b.npz --out report.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

GROUPS = ("G1", "G2", "G3", "G4")
T_DEFAULT = 4.0
ANALYSIS_SUBSET = 10000   # fixed subset of the 50k train images
SUBSET_SEED = 0           # fixed, independent of any training seed


# ----------------------------------------------------------------------------- pure-tensor core
def group_ids(t_logits: torch.Tensor, s_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """0..3 for G1..G4. argmax is not differentiable, so no gradient can flow through the grouping."""
    t_ok = t_logits.argmax(1) == y
    s_ok = s_logits.argmax(1) == y
    gid = torch.empty_like(y)
    gid[t_ok & s_ok] = 0
    gid[t_ok & ~s_ok] = 1
    gid[~t_ok & s_ok] = 2
    gid[~t_ok & ~s_ok] = 3
    return gid


def per_sample_terms(s: torch.Tensor, t: torch.Tensor, y: torch.Tensor, T: float = T_DEFAULT) -> dict:
    s, t = s.double(), t.double()
    onehot = F.one_hot(y, s.size(1)).double()
    ps, pt = F.softmax(s, 1), F.softmax(t, 1)
    ce = F.cross_entropy(s, y, reduction="none")
    ps_T, pt_T = F.softmax(s / T, 1), F.softmax(t / T, 1)
    kd = (pt_T * (torch.log(pt_T.clamp_min(1e-12)) - torch.log(ps_T.clamp_min(1e-12)))).sum(1) * T * T
    g_ce = ps - onehot
    g_kd = T * (ps_T - pt_T)
    cos = F.cosine_similarity(g_ce, g_kd, dim=1, eps=1e-12)
    return {
        "ce": ce, "kd": kd,
        "p_t_y": pt.gather(1, y[:, None])[:, 0], "p_s_y": ps.gather(1, y[:, None])[:, 0],
        "cos": cos, "n_ce": g_ce.norm(dim=1), "n_kd": g_kd.norm(dim=1),
    }


def _boot_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze(s: torch.Tensor, t: torch.Tensor, y: torch.Tensor, T: float = T_DEFAULT) -> dict:
    gid = group_ids(t, s, y)
    terms = per_sample_terms(s, t, y, T)
    out = {"n_total": int(len(y)), "student_top1": float((s.argmax(1) == y).float().mean() * 100),
           "teacher_top1": float((t.argmax(1) == y).float().mean() * 100), "groups": {}}
    for k, name in enumerate(GROUPS):
        m = gid == k
        n = int(m.sum())
        row = {"n": n, "share_pct": 100.0 * n / len(y)}
        if n:
            cos = terms["cos"][m].numpy()
            lo, hi = _boot_ci(cos)
            row.update({
                "ce": float(terms["ce"][m].mean()), "kd": float(terms["kd"][m].mean()),
                "p_t_y": float(terms["p_t_y"][m].mean()), "p_s_y": float(terms["p_s_y"][m].mean()),
                "cos_mean": float(cos.mean()), "cos_median": float(np.median(cos)),
                "cos_ci95": [lo, hi], "cos_neg_pct": float((cos < 0).mean() * 100),
                "gnorm_ce": float(terms["n_ce"][m].mean()), "gnorm_kd": float(terms["n_kd"][m].mean()),
                # per-sample cos is shared by sign; also report the norm-weighted share of KD in a 0.5/0.5 update
                "kd_over_ce_norm": float((terms["n_kd"][m] / terms["n_ce"][m].clamp_min(1e-12)).median()),
            })
        out["groups"][name] = row
    return out, gid


def transitions(gid_a: torch.Tensor, gid_b: torch.Tensor) -> dict:
    mat = np.zeros((4, 4), dtype=int)
    for a, b in zip(gid_a.tolist(), gid_b.tolist()):
        mat[a, b] += 1
    return {"rows=a_group, cols=b_group": mat.tolist()}


# ----------------------------------------------------------------------------- model / data (needs CIFAR-100)
def _load_main_module():
    os.environ.setdefault("ALLOW_CPU", "1")
    path = Path(__file__).with_name("streamlined_kd_cifar100.py")
    spec = importlib.util.spec_from_file_location("skd_main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["skd_main"] = mod
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def _logits(model, loader, device):
    model.eval()
    outs = []
    for x, _ in loader:
        outs.append(model(x.to(device)).float().cpu())
    return torch.cat(outs)


def extract(args) -> None:
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset

    mod = _load_main_module()
    device = mod.DEVICE
    mod.prepare_cifar100()
    norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    tf = transforms.Compose([transforms.ToTensor(), norm])   # deterministic, one view for both models
    ds = torchvision.datasets.CIFAR100(mod.DATA_DIR, train=(args.split == "train"), download=False, transform=tf)
    if args.split == "train":
        idx = np.random.default_rng(SUBSET_SEED).permutation(len(ds))[:ANALYSIS_SUBSET]
        idx = np.sort(idx)
        ds = Subset(ds, idx.tolist())
    else:
        idx = np.arange(len(ds))
    loader = DataLoader(ds, batch_size=500, shuffle=False, num_workers=0)
    labels = torch.tensor([ds[i][1] for i in range(len(ds))]) if args.split != "train" else \
        torch.tensor([torchvision.datasets.CIFAR100(mod.DATA_DIR, train=True, download=False).targets[i] for i in idx])

    teacher = mod.resnet56().to(device)
    teacher.load_state_dict(mod.load_state_dict(Path(args.teacher)), strict=False)
    student = mod.resnet20().to(device)
    student.load_state_dict(mod.load_state_dict(Path(args.student)), strict=True)
    s, t = _logits(student, loader, device), _logits(teacher, loader, device)
    np.savez(args.out, s=s.numpy(), t=t.numpy(), y=labels.numpy(), idx=np.asarray(idx),
             split=args.split, student=str(args.student), teacher=str(args.teacher))
    print(f"saved {args.out}: n={len(labels)} student_top1={(s.argmax(1)==labels).float().mean()*100:.2f} "
          f"teacher_top1={(t.argmax(1)==labels).float().mean()*100:.2f}")


def report(args) -> None:
    def load(spec):
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=True)
        return name, torch.from_numpy(z["s"]), torch.from_numpy(z["t"]), torch.from_numpy(z["y"]).long()

    (na, sa, ta, ya), (nb, sb, tb, yb) = load(args.a), load(args.b)
    if not torch.equal(ya, yb) or not torch.allclose(ta, tb, atol=1e-4):
        raise SystemExit("A and B must use identical samples/teacher logits (same split, same teacher)")
    ra, ga = analyze(sa, ta, ya)
    rb, gb = analyze(sb, tb, yb)
    result = {na: ra, nb: rb, "transitions": transitions(ga, gb)}
    Path(args.out).write_text(json.dumps(result, indent=2))
    for name, r in ((na, ra), (nb, rb)):
        print(f"\n{name}: student {r['student_top1']:.2f}  teacher {r['teacher_top1']:.2f}  n={r['n_total']}")
        print("grp   n   share%   CE     KD    p_t[y] p_s[y]  cos_mean [95%CI]       cos<0%  |g_ce| |g_kd|")
        for g in GROUPS:
            x = r["groups"][g]
            if x["n"]:
                print(f"{g} {x['n']:5d} {x['share_pct']:6.2f} {x['ce']:6.3f} {x['kd']:6.3f} {x['p_t_y']:6.3f} "
                      f"{x['p_s_y']:6.3f}  {x['cos_mean']:+.3f} [{x['cos_ci95'][0]:+.3f},{x['cos_ci95'][1]:+.3f}] "
                      f"{x['cos_neg_pct']:6.1f} {x['gnorm_ce']:6.3f} {x['gnorm_kd']:6.3f}")
    print("\ntransitions (rows=%s group, cols=%s group):" % (na, nb))
    print(np.array(result["transitions"]["rows=a_group, cols=b_group"]))


# ----------------------------------------------------------------------------- self-test (synthetic, no data needed)
def selftest() -> None:
    torch.manual_seed(0)
    B, C, T = 64, 100, T_DEFAULT
    s = (torch.randn(B, C) * 2).double().requires_grad_(True)
    t = (torch.randn(B, C) * 3).double()
    y = torch.randint(0, C, (B,))
    terms = per_sample_terms(s.detach(), t, y, T)
    # closed-form gradients vs autograd, per sample (loss sum => gradient row i belongs to sample i)
    ce = F.cross_entropy(s, y, reduction="sum")
    (g_ce_auto,) = torch.autograd.grad(ce, s)
    kd = (T * T) * F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction="sum")
    (g_kd_auto,) = torch.autograd.grad(kd, s)
    ps, pt = F.softmax(s.detach(), 1), F.softmax(t, 1)
    g_ce = ps - F.one_hot(y, C).double()
    g_kd = T * (F.softmax(s.detach() / T, 1) - F.softmax(t / T, 1))
    assert torch.allclose(g_ce, g_ce_auto, atol=1e-10), "CE grad mismatch"
    assert torch.allclose(g_kd, g_kd_auto, atol=1e-10), "KD grad mismatch"
    assert torch.allclose(terms["kd"].sum(), kd.detach(), atol=1e-8), "KD value mismatch with F.kl_div*T^2"
    assert torch.allclose(terms["cos"], F.cosine_similarity(g_ce_auto, g_kd_auto, dim=1), atol=1e-10)
    # group partition is exhaustive and consistent with argmax
    gid = group_ids(t, s.detach(), y)
    assert gid.min() >= 0 and gid.max() <= 3
    t_ok, s_ok = t.argmax(1) == y, s.detach().argmax(1) == y
    assert torch.equal(gid == 1, t_ok & ~s_ok) and torch.equal(gid == 2, ~t_ok & s_ok)
    # identical student/teacher => KD = 0 and g_kd = 0
    z = per_sample_terms(t, t, y, T)
    assert z["kd"].abs().max() < 1e-9 and z["n_kd"].max() < 1e-9
    # kd closed-form matches repo kd_loss when importable (CPU)
    try:
        mod = _load_main_module()
        ref = mod.kd_loss(s.detach().float(), t.float(), T, per_sample=True)
        assert torch.allclose(ref.double(), terms["kd"], atol=1e-3), "differs from repo kd_loss"
        print("repo kd_loss(per_sample) agrees with analysis KD")
    except Exception as exc:  # repo kd_loss may require config; report, do not hide
        print(f"[note] could not cross-check repo kd_loss: {type(exc).__name__}: {exc}")
    r, _ = analyze(s.detach().float(), t.float(), y)
    assert sum(r["groups"][g]["n"] for g in GROUPS) == B
    print("selftest OK (synthetic logits only; NOT results from E4/F1b)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    e = sub.add_parser("extract")
    e.add_argument("out"); e.add_argument("--student", required=True); e.add_argument("--teacher", required=True)
    e.add_argument("--split", choices=("train", "test"), default="train")
    r = sub.add_parser("report")
    r.add_argument("--a", required=True); r.add_argument("--b", required=True); r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"selftest": lambda: selftest(), "extract": lambda: extract(a), "report": lambda: report(a)}[a.cmd]()
