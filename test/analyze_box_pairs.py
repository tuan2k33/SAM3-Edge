"""Offline analysis of the (GT box, predicted box) pairs written by collect_box_pairs.py (CPU only).

Errors are prediction - GT. In PATCHES: coordinate error (normalised to the image) / (14 / content), where `content`
is the resolution of the image content (the network input for plain sizes, e.g. 420; 420 for "420in1008").
Only pairs with IoU >= --min-iou are used for the error statistics (default 0.5, to keep gross mismatches out).

    python test/analyze_box_pairs.py [--dir test/results_pairs] [--min-iou 0.5]
"""
import argparse, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(HERE, "results_pairs"))
ap.add_argument("--min-iou", type=float, default=0.5)
ap.add_argument("--set", choices=["first", "held"], default="first",
                help="first = the first 64 images (exploration); held = images 65-128 (held-out validation, *_s64.json)")
args = ap.parse_args()

runs = {}
for f in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
    held = f.endswith("_s64.json")
    if held != (args.set == "held"):
        continue
    r = json.load(open(f)); runs[(r["size"], r["variant"])] = r


def key(k):
    s = k[0]; c, _, cv = s.partition("in")
    return (int(cv or c), int(c), k[1] != "orig", k[1])


def se(a):
    a = np.asarray(a); return a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")


def ols(x, y):
    """y = a + b x ; returns a, b, se_a, se_b, r2"""
    x = np.asarray(x); y = np.asarray(y); X = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ beta
    s2 = (res ** 2).sum() / max(len(y) - 2, 1); cov = s2 * np.linalg.inv(X.T @ X)
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta[0], beta[1], np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]), r2


rows = []
print(f"## Errors in patches (prediction - GT), pairs with IoU >= {args.min_iou}\n")
print("| size | variant | pairs | mean IoU | recall@.5 | x1 | y1 | x2 | y2 | width | height | width % | height % |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(runs, key=key):
    r = runs[k]; P = np.array(r["pairs"]); P = P[P[:, 8] >= args.min_iou]
    if not len(P):
        print(f"| {k[0]} | {k[1]} | 0 | | | | | | | | | | |"); continue
    pn = 14.0 / r["content"]
    g, p = P[:, :4], P[:, 4:8]
    e = (p - g) / pn                                              # patches
    ew = ((p[:, 2] - p[:, 0]) - (g[:, 2] - g[:, 0])) / pn
    eh = ((p[:, 3] - p[:, 1]) - (g[:, 3] - g[:, 1])) / pn
    rw = ((p[:, 2] - p[:, 0]) / (g[:, 2] - g[:, 0]) - 1) * 100
    rh = ((p[:, 3] - p[:, 1]) / (g[:, 3] - g[:, 1]) - 1) * 100
    f = lambda a: f"{np.mean(a):+.2f}±{se(a):.2f}"
    print(f"| {k[0]} | {k[1]} | {len(P)} | {P[:, 8].mean():.3f} | {r['tp50'] / r['gt']:.3f} | "
          f"{f(e[:, 0])} | {f(e[:, 1])} | {f(e[:, 2])} | {f(e[:, 3])} | {f(ew)} | {f(eh)} | {f(rw)} | {f(rh)} |")

print("\n## Where does the error come from? OLS of the error (patches) on a box property\n")
print("x2 error ~ a + b * (GT x2), x1 error ~ a + b * (GT x1), width error ~ a + b * (GT width)  [coordinates and width normalised to 0..1]\n")
print("| size | variant | x1: a | x1: b | R² | x2: a | x2: b | R² | width: a | width: b | R² |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for k in sorted(runs, key=key):
    r = runs[k]; P = np.array(r["pairs"]); P = P[P[:, 8] >= args.min_iou]
    if len(P) < 10:
        continue
    pn = 14.0 / r["content"]; g, p = P[:, :4], P[:, 4:8]
    out = []
    for xg, y in ((g[:, 0], (p[:, 0] - g[:, 0]) / pn), (g[:, 2], (p[:, 2] - g[:, 2]) / pn),
                  (g[:, 2] - g[:, 0], ((p[:, 2] - p[:, 0]) - (g[:, 2] - g[:, 0])) / pn)):
        a, b, sa, sb, r2 = ols(xg, y); out += [f"{a:+.2f}±{sa:.2f}", f"{b:+.2f}±{sb:.2f}", f"{r2:.2f}"]
    print(f"| {k[0]} | {k[1]} | " + " | ".join(out) + " |")


print("\n## Relative width / height error vs imgsz (the primary criterion: is there a trend, and how large is it?)\n")
print(f"Mean of (pred size / GT size - 1) in %, pairs with IoU >= {args.min_iou}. 'slope' = change per +10 cells of H (OLS over the plain sizes of that variant);")
print("a flat variant has slope ~ 0 and values within a band around 0 (about +-15% is acceptable); IoU is only a secondary reference.\n")
by_var = {}
for (size, var), r in runs.items():
    if "in" in size:
        continue
    P = np.array(r["pairs"]); P = P[P[:, 8] >= args.min_iou]
    if len(P) < 10:
        continue
    g, p = P[:, :4], P[:, 4:8]
    rw = ((p[:, 2] - p[:, 0]) / (g[:, 2] - g[:, 0]) - 1) * 100
    rh = ((p[:, 3] - p[:, 1]) / (g[:, 3] - g[:, 1]) - 1) * 100
    by_var.setdefault(var, []).append((int(size) // 14, int(size), rw.mean(), se(rw), rh.mean(), se(rh), P[:, 8].mean()))
for var, rows_ in by_var.items():
    rows_.sort()
    if len(rows_) < 3:
        continue
    H = np.array([x[0] for x in rows_]); W = np.array([x[2] for x in rows_]); Hh = np.array([x[4] for x in rows_])
    sw = np.polyfit(H, W, 1)[0] * 10; sh = np.polyfit(H, Hh, 1)[0] * 10
    print(f"**{var}**  slope width {sw:+.2f}%/10 cells, height {sh:+.2f}%/10 cells; width range [{W.min():+.1f}, {W.max():+.1f}]%, height range [{Hh.min():+.1f}, {Hh.max():+.1f}]%\n")
    print("| imgsz | H | width % | height % | mean IoU (ref.) |"); print("|---|---|---|---|---|")
    for h, sz, w_, sw_, h_, sh_, iu in rows_:
        print(f"| {sz} | {h} | {w_:+.1f}±{sw_:.1f} | {h_:+.1f}±{sh_:.1f} | {iu:.3f} |")
    print()


print(f"\n## Summary per variant over the plain sizes (set: {args.set}) -- the selection table\n")
print("size error = |mean width %| and |mean height %| averaged over sizes (0 = unbiased); centre error = |mean (x1+x2)/2 error| and |mean (y1+y2)/2 error| in patches, averaged over sizes;")
print("slope = change of the mean width % per +10 cells of H (flatness). IoU is only a reference.\n")
print("| variant | sizes | mean abs width % | mean abs height % | width slope | mean abs centre x (patch) | mean abs centre y (patch) | mean IoU |")
print("|---|---|---|---|---|---|---|---|")
summ = {}
for (size, var), r in runs.items():
    if "in" in size:
        continue
    P = np.array(r["pairs"]); P = P[P[:, 8] >= args.min_iou]
    if len(P) < 10:
        continue
    g, p = P[:, :4], P[:, 4:8]; pn = 14.0 / r["content"]
    rw = ((p[:, 2] - p[:, 0]) / (g[:, 2] - g[:, 0]) - 1) * 100; rh = ((p[:, 3] - p[:, 1]) / (g[:, 3] - g[:, 1]) - 1) * 100
    cx = (((p[:, 0] + p[:, 2]) - (g[:, 0] + g[:, 2])) / 2 / pn).mean(); cy = (((p[:, 1] + p[:, 3]) - (g[:, 1] + g[:, 3])) / 2 / pn).mean()
    summ.setdefault(var, []).append((int(size) // 14, rw.mean(), rh.mean(), cx, cy, P[:, 8].mean()))
for var, rows_ in sorted(summ.items(), key=lambda kv: kv[0]):
    rows_.sort()
    H = np.array([x[0] for x in rows_]); W = np.array([x[1] for x in rows_]); Hh = np.array([x[2] for x in rows_])
    CX = np.array([x[3] for x in rows_]); CY = np.array([x[4] for x in rows_]); I = np.array([x[5] for x in rows_])
    slope = np.polyfit(H, W, 1)[0] * 10 if len(H) >= 3 else float("nan")
    print(f"| {var} | {len(H)} | {np.abs(W).mean():.2f} | {np.abs(Hh).mean():.2f} | {slope:+.2f} | {np.abs(CX).mean():.2f} | {np.abs(CY).mean():.2f} | {I.mean():.3f} |")
