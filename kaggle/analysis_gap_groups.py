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
    all_targets = np.asarray(ds.targets)                      # read once; never re-instantiate the dataset per sample
    if args.split == "train":
        idx = np.sort(np.random.default_rng(SUBSET_SEED).permutation(len(ds))[:ANALYSIS_SUBSET])
        ds = Subset(ds, idx.tolist())
    else:
        idx = np.arange(len(ds))
    labels = torch.from_numpy(all_targets[idx]).long()
    loader = DataLoader(ds, batch_size=500, shuffle=False, num_workers=0)

    teacher = mod.resnet56().to(device)
    t_state = mod.load_state_dict(Path(args.teacher))
    for key, value in teacher.state_dict().items():   # same handling as run_experiment: BN step counters only
        if key.endswith("num_batches_tracked") and key not in t_state:
            t_state[key] = torch.zeros_like(value)
    teacher.load_state_dict(t_state, strict=True)
    student = mod.resnet20().to(device)
    raw = mod._torch_load(Path(args.student))
    if isinstance(raw, dict) and "student" in raw:     # last_checkpoint.pt layout
        raw = raw["student"]
    student.load_state_dict(raw, strict=True)
    s, t = _logits(student, loader, device), _logits(teacher, loader, device)
    np.savez(args.out, s=s.numpy(), t=t.numpy(), y=labels.numpy(), idx=np.asarray(idx),
             split=args.split, student=str(args.student), teacher=str(args.teacher))
    print(f"saved {args.out}: n={len(labels)} student_top1={(s.argmax(1)==labels).float().mean()*100:.2f} "
          f"teacher_top1={(t.argmax(1)==labels).float().mean()*100:.2f}")


def extract_aug(args) -> None:
    """Training-distribution analysis: weak view (crop+flip) and strong view (crop+flip+RandAugment) of the SAME fixed
    10k train subset. Augmented tensors are generated ONCE (fixed seed) and every model sees exactly the same tensors."""
    import random
    import torchvision
    import torchvision.transforms as transforms

    mod = _load_main_module()
    device = mod.DEVICE
    mod.prepare_cifar100()
    norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    weak = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), norm])
    strong = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                 mod.PaperRandAugment(2), transforms.ToTensor(), norm])
    base = torchvision.datasets.CIFAR100(mod.DATA_DIR, train=True, download=False)       # PIL images, no transform
    idx = np.sort(np.random.default_rng(SUBSET_SEED).permutation(len(base))[:ANALYSIS_SUBSET])
    y = torch.from_numpy(np.asarray(base.targets)[idx]).long()
    random.seed(args.aug_seed); np.random.seed(args.aug_seed); torch.manual_seed(args.aug_seed)
    xs_weak = torch.stack([weak(base[i][0]) for i in idx])
    xs_strong = torch.stack([strong(base[i][0]) for i in idx])
    x_all = torch.cat([xs_weak, xs_strong])
    view = torch.cat([torch.zeros(len(idx)), torch.ones(len(idx))]).long()
    y_all = torch.cat([y, y])

    @torch.no_grad()
    def run(model):
        model.eval()
        return torch.cat([model(x_all[i:i + 500].to(device)).float().cpu() for i in range(0, len(x_all), 500)])

    teacher = mod.resnet56().to(device)
    t_state = mod.load_state_dict(Path(args.teacher))
    for key, value in teacher.state_dict().items():
        if key.endswith("num_batches_tracked") and key not in t_state:
            t_state[key] = torch.zeros_like(value)
    teacher.load_state_dict(t_state, strict=True)
    t = run(teacher)
    for v, name in ((0, "weak"), (1, "strong")):
        print(f"teacher top1 on {name} view: {(t[view == v].argmax(1) == y_all[view == v]).float().mean() * 100:.2f}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for spec in args.student:
        name, path = spec.split("=", 1)
        student = mod.resnet20().to(device)
        raw = mod._torch_load(Path(path))
        if isinstance(raw, dict) and "student" in raw:
            raw = raw["student"]
        student.load_state_dict(raw, strict=True)
        sl = run(student)
        out = Path(args.out_dir) / f"{name}.npz"
        np.savez(out, s=sl.numpy(), t=t.numpy(), y=y_all.numpy(), view=view.numpy(), idx=np.asarray(idx))
        print(f"saved {out}: " + "  ".join(
            f"{vn} student_top1={(sl[view == v].argmax(1) == y_all[view == v]).float().mean() * 100:.2f}"
            for v, vn in ((0, "weak"), (1, "strong"))))


def d0_cross_view(s_strong, t_strong, t_weak, y, T: float = T_DEFAULT) -> dict:
    """Same student strong logits; KD target = teacher on the strong view (original) vs teacher on the paired weak view."""
    orig = per_sample_terms(s_strong, t_strong, y, T)
    cross = per_sample_terms(s_strong, t_weak, y, T)
    pt_s, pt_w = F.softmax(t_strong.double(), 1), F.softmax(t_weak.double(), 1)
    ps_s = F.softmax(s_strong.double(), 1)
    tw_ok, ts_ok = t_weak.argmax(1) == y, t_strong.argmax(1) == y
    m = 0.5 * (pt_s + pt_w)
    js = 0.5 * ((pt_s * (pt_s.clamp_min(1e-12) / m).log()).sum(1) + (pt_w * (pt_w.clamp_min(1e-12) / m).log()).sum(1))
    def blk(t):
        c = t["cos"].numpy()
        return {"conflict_pct": float((c < 0).mean() * 100), "cos_mean": float(c.mean()), "cos_ci95": list(_boot_ci(c)),
                "kd_mean": float(t["kd"].mean()), "gnorm_kd_mean": float(t["n_kd"].mean()),
                "gnorm_kd_p99": float(np.percentile(t["n_kd"].numpy(), 99)), "gnorm_ce_mean": float(t["n_ce"].mean()),
                "finite": bool(torch.isfinite(t["kd"]).all() and torch.isfinite(t["n_kd"]).all())}
    out = {"n": int(len(y)), "original": blk(orig), "cross": blk(cross),
           "teacher_weak_acc": float(tw_ok.float().mean() * 100), "teacher_strong_acc": float(ts_ok.float().mean() * 100),
           "teacher_weak_conf_y": float(pt_w.gather(1, y[:, None]).mean()), "teacher_strong_conf_y": float(pt_s.gather(1, y[:, None]).mean()),
           "teacher_weak_maxprob": float(pt_w.max(1).values.mean()), "teacher_strong_maxprob": float(pt_s.max(1).values.mean()),
           "tweak_right_tstrong_wrong_pct": float((tw_ok & ~ts_ok).float().mean() * 100),
           "tweak_wrong_tstrong_right_pct": float((~tw_ok & ts_ok).float().mean() * 100),
           "teacher_argmax_agree_pct": float((t_weak.argmax(1) == t_strong.argmax(1)).float().mean() * 100),
           "js_teacher_weak_strong_mean": float(js.mean()),
           "student_strong_acc": float((s_strong.argmax(1) == y).float().mean() * 100),
           "student_strong_agree_tweak_pct": float((s_strong.argmax(1) == t_weak.argmax(1)).float().mean() * 100),
           "semantic_lost_pct": float(((pt_s.gather(1, y[:, None])[:, 0] < 0.01) & (ps_s.gather(1, y[:, None])[:, 0] < 0.01)).float().mean() * 100),
           "by_teacher_case": {}}
    for name, m_ in (("Tw_right&Ts_right", tw_ok & ts_ok), ("Tw_right&Ts_wrong", tw_ok & ~ts_ok),
                     ("Tw_wrong&Ts_right", ~tw_ok & ts_ok), ("Tw_wrong&Ts_wrong", ~tw_ok & ~ts_ok)):
        n = int(m_.sum())
        row = {"n": n, "pct": 100.0 * n / len(y)}
        if n:
            for tag, t in (("orig", orig), ("cross", cross)):
                c = t["cos"][m_].numpy()
                row[f"{tag}_conflict_pct"] = float((c < 0).mean() * 100)
                row[f"{tag}_cos_mean"] = float(c.mean())
                row[f"{tag}_gnorm_kd"] = float(t["n_kd"][m_].mean())
        out["by_teacher_case"][name] = row
    o, c = out["original"], out["cross"]
    r_mean, r_p99 = c["gnorm_kd_mean"] / o["gnorm_kd_mean"], c["gnorm_kd_p99"] / o["gnorm_kd_p99"]
    crit = {
        "C1_conflict": c["conflict_pct"] <= 0.75 * o["conflict_pct"] and (o["conflict_pct"] - c["conflict_pct"]) >= 10.0,
        "C2_gnorm": 0.67 <= r_mean <= 1.5 and 0.67 <= r_p99 <= 1.5 and o["finite"] and c["finite"],
        "C3_pairing": out["teacher_weak_acc"] >= 95.0,   # label/idx equality is asserted by the caller
        "C4_semantic": out["semantic_lost_pct"] <= 25.0,
    }
    out["gnorm_kd_ratio_mean"], out["gnorm_kd_ratio_p99"] = r_mean, r_p99
    out["criteria"] = crit
    out["decision"] = "PASS" if all(crit.values()) else "FAIL"
    return out


def d0(args) -> None:
    results = {}
    for spec in args.npz:
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=True)
        view = z["view"]
        w, st = np.where(view == 0)[0], np.where(view == 1)[0]
        assert len(w) == len(st), "unequal weak/strong counts"
        # pairing: row k of the weak block and row k of the strong block come from the same original train image
        idx = z["idx"]
        assert len(idx) == len(w), "idx length must equal the number of original images"
        assert np.array_equal(z["y"][w], z["y"][st]), "PAIRING FAILURE: labels differ between paired weak/strong rows"
        y = torch.from_numpy(z["y"][st]).long()
        r = d0_cross_view(torch.from_numpy(z["s"][st]), torch.from_numpy(z["t"][st]), torch.from_numpy(z["t"][w]), y)
        r["pairing_labels_identical"] = True
        results[name] = r
        o, c = r["original"], r["cross"]
        print(f"\n===== D0 [{name}]  n_strong={r['n']}  decision={r['decision']}")
        print(f"teacher acc weak {r['teacher_weak_acc']:.2f} / strong {r['teacher_strong_acc']:.2f};  p_T[y] weak {r['teacher_weak_conf_y']:.3f} / strong {r['teacher_strong_conf_y']:.3f};"
              f"  maxprob weak {r['teacher_weak_maxprob']:.3f} / strong {r['teacher_strong_maxprob']:.3f}")
        print(f"Tweak right & Tstrong wrong {r['tweak_right_tstrong_wrong_pct']:.2f}%;  Tweak wrong & Tstrong right {r['tweak_wrong_tstrong_right_pct']:.2f}%;"
              f"  teacher argmax agree {r['teacher_argmax_agree_pct']:.2f}%;  JS(Tw,Ts) {r['js_teacher_weak_strong_mean']:.3f}")
        print(f"student strong acc {r['student_strong_acc']:.2f};  S_strong argmax == T_weak argmax {r['student_strong_agree_tweak_pct']:.2f}%;  semantic-lost {r['semantic_lost_pct']:.2f}%")
        print("            conflict%   cos_mean [CI95]            KD      |g_KD| mean  |g_KD| p99  |g_CE| mean")
        for tag, b in (("original", o), ("cross", c)):
            print(f"{tag:10s}  {b['conflict_pct']:7.2f}   {b['cos_mean']:+.3f} [{b['cos_ci95'][0]:+.3f},{b['cos_ci95'][1]:+.3f}]  {b['kd_mean']:6.3f}   {b['gnorm_kd_mean']:.3f}      {b['gnorm_kd_p99']:.3f}      {b['gnorm_ce_mean']:.3f}")
        print(f"|g_KD| ratio cross/original: mean {r['gnorm_kd_ratio_mean']:.3f}, p99 {r['gnorm_kd_ratio_p99']:.3f}")
        for k, v in r["by_teacher_case"].items():
            extra = "" if not v["n"] else (f"  conflict orig {v['orig_conflict_pct']:.1f}% -> cross {v['cross_conflict_pct']:.1f}%"
                                         f"  cos {v['orig_cos_mean']:+.3f} -> {v['cross_cos_mean']:+.3f}  |g_KD| {v['orig_gnorm_kd']:.3f} -> {v['cross_gnorm_kd']:.3f}")
            print(f"  {k:18s} n={v['n']:5d} ({v['pct']:5.2f}%){extra}")
        print("criteria:", r["criteria"])
    Path(args.out).write_text(json.dumps(results, indent=2))


def report(args) -> None:
    def load(spec):
        name, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=True)
        keep = np.ones(len(z["y"]), dtype=bool)
        if getattr(args, "view", "all") != "all":
            keep = z["view"] == {"weak": 0, "strong": 1}[args.view]
        return (name, torch.from_numpy(z["s"][keep]), torch.from_numpy(z["t"][keep]),
                torch.from_numpy(z["y"][keep]).long())

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
    g = sub.add_parser("extract_aug")
    g.add_argument("out_dir"); g.add_argument("--teacher", required=True)
    g.add_argument("--student", action="append", required=True, help="NAME=PATH (repeatable)")
    g.add_argument("--aug_seed", type=int, default=12345)
    dd = sub.add_parser("d0")
    dd.add_argument("--npz", action="append", required=True, help="NAME=PATH of an extract_aug npz (repeatable)")
    dd.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--a", required=True); r.add_argument("--b", required=True); r.add_argument("--out", required=True)
    r.add_argument("--view", choices=("all", "weak", "strong"), default="all")
    a = ap.parse_args()
    {"selftest": lambda: selftest(), "extract": lambda: extract(a), "extract_aug": lambda: extract_aug(a),
     "report": lambda: report(a), "d0": lambda: d0(a)}[a.cmd]()
