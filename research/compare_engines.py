"""What does a changed engine (pruned, another precision, ...) do to the detections of a reference engine? Class `person`, COCO val2017, both
engines on the same images. Read docs/NOTES.md section 8 for how to read the numbers.

    python research/compare_engines.py 420 weight/<reference>/sam3.engine weight/<changed>/sam3.engine [more engines] [--n 300] [--thr 0.5]

Against the ground truth (crowd regions ignored): recall and precision at --thr, and AP@0.5, which does not depend on a threshold.
Against the reference engine, box by box (same object = IoU >= 0.5):
  same object   how the score of an object changes: mean and median change, the 10% worst, how many fall under --thr, how many are gone
  new boxes     boxes over --thr that the reference did not have over --thr: a real object (matches the ground truth), a box the reference
                also had with a lower score, or an invented one (no ground truth, no reference box at all)"""
import argparse, json, os, sys, zipfile
from collections import defaultdict

import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "test"))
import golden

ap = argparse.ArgumentParser()
ap.add_argument("imgsz", type=int); ap.add_argument("engines", nargs="+")
ap.add_argument("--n", type=int, default=300); ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--coco", default=os.environ.get("COCO_DIR", "/mnt/c/Dev/projects/ModelEdge/Detector/data/coco"))
a = ap.parse_args()
S, T = a.imgsz, a.thr

ann = json.load(open(f"{a.coco}/instances_val2017.json"))
by_img = defaultdict(list)
for x in ann["annotations"]:
    if x["category_id"] == 1: by_img[x["image_id"]].append(x)
imgs = sorted(ann["images"], key=lambda i: i["id"])[:a.n]
z = zipfile.ZipFile(f"{a.coco}/val2017.zip")
loaders = {im["file_name"]: (lambda n=im["file_name"]: cv2.imdecode(np.frombuffer(z.read("val2017/" + n), np.uint8), cv2.IMREAD_COLOR)) for im in imgs}
GT, CROWD = {}, {}
for im in imgs:
    box = lambda x: np.array([x["bbox"][0], x["bbox"][1], x["bbox"][0] + x["bbox"][2], x["bbox"][1] + x["bbox"][3]]) * np.array([S / im["width"], S / im["height"]] * 2)
    GT[im["file_name"]] = [box(x) for x in by_img[im["id"]] if not x["iscrowd"]]
    CROWD[im["file_name"]] = [box(x) for x in by_img[im["id"]] if x["iscrowd"]]


def nms(d, low=0.05):
    d = d[d[:, 4] >= low]; d = d[np.argsort(-d[:, 4])]; keep = []
    for r in d:
        if all(golden.iou(r, k) <= 0.6 for k in keep): keep.append(r)
    return keep


def in_crowd(r, crowds):
    for c in crowds:
        w, h = min(r[2], c[2]) - max(r[0], c[0]), min(r[3], c[3]) - max(r[1], c[1])
        if max(w, 0) * max(h, 0) >= 0.5 * (r[2] - r[0]) * (r[3] - r[1]): return True
    return False


def against_gt(dets):
    """every box as (score, is true positive), ignoring boxes in crowd regions -> recall, precision at T, AP@0.5; also the set of true-positive boxes"""
    rows, ngt, tp_ids = [], 0, set()
    for name, keep in dets.items():
        used = set(); ngt += len(GT[name])
        for k, r in enumerate(keep):
            c = [(golden.iou(r, g), j) for j, g in enumerate(GT[name]) if j not in used]
            i, j = max(c, default=(0.0, -1))
            if i >= 0.5: used.add(j); rows.append((r[4], 1)); tp_ids.add((name, k))
            elif not in_crowd(r, CROWD[name]): rows.append((r[4], 0))
    rows.sort(reverse=True); tp = np.cumsum([t for _, t in rows]); n = np.arange(1, len(rows) + 1)
    prec, rec = tp / n, tp / max(ngt, 1)
    mono = np.maximum.accumulate(prec[::-1])[::-1]
    apv = float(np.sum(np.diff(np.concatenate([[0], rec])) * mono))
    at = sum(1 for s, _ in rows if s >= T)
    return (tp[at - 1] / ngt if at else 0.0), (tp[at - 1] / at if at else 0.0), apv, ngt, int(tp[at - 1]) if at else 0, at, tp_ids


res = {e: {n: nms(d) for n, d in golden.run(e, S, loaders).items()} for e in a.engines}
ref = res[a.engines[0]]
print(f"{len(imgs)} images, {sum(map(len, GT.values()))} labelled people, imgsz {S}, score threshold {T}\n")
for e in a.engines:
    r, p, apv, ngt, tp, at, tp_ids = against_gt(res[e])
    print(f"{os.path.basename(os.path.dirname(e))}")
    print(f"  ground truth   recall {r:.3f} ({tp}/{ngt})   precision {p:.3f} ({at - tp} wrong of {at})   AP@0.5 {apv:.3f}")
    if e == a.engines[0]: continue
    d_old, d_new, gone = [], [], 0
    for name in ref:
        for row in (x for x in ref[name] if x[4] >= T):
            c = [(golden.iou(row, q), q[4]) for q in res[e][name]]
            i, s = max(c, default=(0.0, 0.0))
            if i >= 0.5: d_old.append(row[4]); d_new.append(s)
            else: gone += 1
    d_old, d_new = np.array(d_old), np.array(d_new); ch = d_new - d_old; n_ref = len(ch) + gone
    print(f"  same object    {n_ref} reference boxes: score {d_old.mean():.3f} -> {d_new.mean():.3f} (mean change {ch.mean():+.3f}, median {np.median(ch):+.3f}, "
          f"worst 10% {np.percentile(ch, 10):+.3f}, worst {ch.min():+.3f}); under {T}: {int((d_new < T).sum())} ({(d_new < T).mean():.1%}); gone: {gone}")
    real = promoted = invented = 0
    for name in ref:
        for k, row in enumerate(res[e][name]):
            if row[4] < T: continue
            best = max(((golden.iou(row, q), q[4]) for q in ref[name]), default=(0.0, 0.0))
            if best[0] >= 0.5 and best[1] >= T: continue                      # the reference had it too
            if (name, k) in tp_ids: real += 1
            elif best[0] >= 0.5: promoted += 1
            elif not in_crowd(row, CROWD[name]): invented += 1
    print(f"  new boxes      {real + promoted + invented}: {real} real objects, {promoted} the reference had with a lower score (no label), {invented} invented")
