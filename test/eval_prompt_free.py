"""Recall of a prompt-free engine on COCO val2017: for every ground-truth instance, the best mask IoU over the engine's masks, before and after the host post-process
(infer_prompt_free.postprocess, defaults). Class-less: no label is checked, and recall counts a miss but never a spurious mask, so read it with the masks per image.
    python test/eval_prompt_free.py weight/sam3_420_prompt-free/sam3.engine 420 [N images = 200]"""
import json, os, sys, zipfile
from collections import defaultdict

import cv2, numpy as np, torch, torch.nn.functional as F

from infer_prompt_free import Engine, postprocess

COCO = os.environ.get("COCO_DIR", "/mnt/c/Dev/projects/ModelEdge/Detector/data/coco")
engine, S = sys.argv[1], int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
run = Engine(engine, S)


def polygons(a, im):
    m = np.zeros((S, S), np.uint8)
    for poly in a["segmentation"]:  # non-crowd instances are polygon lists, in image pixels
        cv2.fillPoly(m, [(np.array(poly).reshape(-1, 2) * [S / im["width"], S / im["height"]]).round().astype(np.int32)], 1)
    return m


ann = json.load(open(f"{COCO}/instances_val2017.json"))
names = {c["id"]: c["name"] for c in ann["categories"]}
imgs = sorted(ann["images"], key=lambda i: i["id"])[:N]
by_img = defaultdict(list)
for a in ann["annotations"]: by_img[a["image_id"]].append(a)
z = zipfile.ZipFile(f"{COCO}/val2017.zip")
rows, counts = [], []  # (category, area fraction, best IoU all points / after post-process)
for im in imgs:
    gts = [a for a in by_img[im["id"]] if not a["iscrowd"] and a["area"] > 0]
    if not gts: continue
    det, mk = run(cv2.imdecode(np.frombuffer(z.read("val2017/" + im["file_name"]), np.uint8), cv2.IMREAD_COLOR))
    # pixel counts pass 65504 (a mask over 37% of a 420 frame): fp16 sums overflowed and large objects counted as missed
    P = torch.cat([(F.interpolate(c[:, None], size=(S, S), mode="bilinear")[:, 0] > 0.5).flatten(1) for c in mk.split(64)])   # bool [N, S*S]
    g = (torch.from_numpy(np.stack([polygons(a, im) for a in gts])).cuda() > 0).flatten(1).float()
    inter = torch.cat([g @ c.float().T for c in P.split(64)], 1)
    iou = inter / (g.sum(1, keepdim=True) + P.sum(1)[None] - inter + 1e-6)
    keep = postprocess(det, mk)
    counts.append(len(keep))
    b_all = iou.max(1).values.float().cpu().numpy(); b_post = iou[:, keep].max(1).values.float().cpu().numpy() if len(keep) else np.zeros(len(gts))
    for k, a in enumerate(gts): rows.append((names[a["category_id"]], a["area"] / (im["width"] * im["height"]), b_all[k], b_post[k]))

r = np.array([x[2:] for x in rows]); cat = np.array([x[0] for x in rows]); frac = np.array([x[1] for x in rows])
def line(label, m):
    if m.sum() == 0: return
    print(f"{label:22s} n={int(m.sum()):5d}  recall@0.5 / @0.7 (mask IoU):  all points {np.mean(r[m, 0] >= .5):.2f}/{np.mean(r[m, 0] >= .7):.2f}   after post-process {np.mean(r[m, 1] >= .5):.2f}/{np.mean(r[m, 1] >= .7):.2f}")
print(f"{len(imgs)} images, {len(rows)} instances, {np.mean(counts):.0f} masks per image after post-process")
line("all", np.ones(len(rows), bool))
for lo, hi, n in ((0.1, 1, "large (>10% of image)"), (0.01, 0.1, "medium (1-10%)"), (0, 0.01, "small (<1%)")): line(n, (frac >= lo) & (frac < hi))
for c in ("person", "bus", "car", "chair", "dog", "bottle"): line(c, cat == c)
