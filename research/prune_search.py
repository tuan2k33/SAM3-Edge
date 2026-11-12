"""Which sub-blocks can go without losing detections? Eager fp16, set-classes wrapper. Every one of the 106 sub-blocks (ViT 32 x attention, mlp;
DETR encoder 6 x self_attn, cross_attn, mlp; decoder 6 x self_attn, text_cross_attn, vision_cross_attn, mlp) is switched off alone (output 0, the
residual passes) and ranked by what it does to the unpruned model's detections (score >= 0.5). Then the ViT blocks are chosen greedily (at every step the one that
costs the least score given those already cut), and encoder/decoder blocks are added on top of the first 10 of them.

    HF_TOKEN=offline HF_HUB_OFFLINE=1 python research/prune_search.py [--imgsz 420] [--classes person car chair] [--rank-images 24] [--images 40] [--pool 26] [--max-vit 18]
Then build the candidates as real engines: research/prune_probe.py. Results: docs/NOTES.md section 8."""
import argparse, os, sys

import cv2, numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "test"), os.path.dirname(os.path.abspath(__file__))]
import export as E, golden
from prune_probe import Zero
from transformers.models.sam3 import Sam3Model, Sam3Processor

ap = argparse.ArgumentParser()
ap.add_argument("--imgsz", type=int, default=420); ap.add_argument("--classes", nargs="+", default=["person", "car", "chair"])
ap.add_argument("--rank-images", type=int, default=24); ap.add_argument("--images", type=int, default=40)
ap.add_argument("--sets", type=int, nargs="+", default=[2, 4, 6, 8, 12])
ap.add_argument("--pool", type=int, default=26); ap.add_argument("--max-vit", type=int, default=18)
a = ap.parse_args()
S = a.imgsz

model = Sam3Model.from_pretrained("facebook/sam3", token=os.environ.get("HF_TOKEN")).half().to("cuda").eval()
te, am = E.encode_concepts(model, Sam3Processor.from_pretrained("facebook/sam3", token=os.environ.get("HF_TOKEN")), a.classes)
w = E.Sam3Wrapper(model, te, am, S).cuda().eval().half(); w.core.scale = w.core.scale.float()
STAGES = {"vit": (model.vision_encoder.backbone.layers, ("attention", "mlp")),
          "enc": (model.detr_encoder.layers, ("self_attn", "cross_attn", "mlp")),
          "dec": (model.detr_decoder.layers, ("self_attn", "text_cross_attn", "vision_cross_attn", "mlp"))}
loaders = list(golden.images().values())
X = [torch.from_numpy((cv2.cvtColor(cv2.resize(l(), (S, S)), cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5).permute(2, 0, 1)[None].cuda().half()
     for l in loaders[::max(1, len(loaders) // a.images)][:a.images]]


def run(images):
    with torch.no_grad():
        return [w(x)[0][0].float().cpu().numpy() for x in images]


def compare(ref, new):
    """(share of the reference boxes found at IoU >= 0.5, their mean score before, after, share still scoring >= 0.5, median IoU)"""
    old, sc, ious = [], [], []
    for r, n in zip(ref, new):
        for row in r[r[:, 4] >= 0.5]:
            c = [(golden.iou(row, q), q[4]) for q in n[(n[:, 5] == row[5]) & (n[:, 4] >= 0.05)]]
            i, s = max(c, default=(0.0, 0.0))
            old.append(row[4]); ious.append(i); sc.append(s if i >= 0.5 else 0.0)
    old, sc, ious = map(np.array, (old, sc, ious))
    return (ious >= 0.5).mean(), old.mean(), sc.mean(), (sc >= 0.5).mean(), np.median(ious)


def cut(blocks):
    """blocks: (stage, attribute, layer)"""
    saved = [(STAGES[st][0][k], attr, getattr(STAGES[st][0][k], attr)) for st, attr, k in blocks]
    for st, attr, k in blocks:
        setattr(STAGES[st][0][k], attr, Zero(pair=attr != "mlp"))
    return lambda: [setattr(l, attr, m) for l, attr, m in saved]


def cumulative(title, base, order):
    print(f"{title}\n   n  found  mean score      still >= 0.5  median IoU  added blocks")
    for n in a.sets:
        if n > len(order): break
        undo = cut(base + order[:n]); f, o, s, keep, iou = compare(ref, run(X)); undo()
        print(f"  {n:2d}  {f:.3f}  {o:.3f} -> {s:.3f}  {keep:.3f}         {iou:.3f}       " + " ".join(f"{st}.{attr}:{k}" for st, attr, k in sorted(order[:n])), flush=True)


ref = run(X)
print(f"imgsz {S}, classes {a.classes}, {len(X)} images, {sum(int((r[:, 4] >= 0.5).sum()) for r in ref)} reference boxes", flush=True)
rank = []
for st, (layers, attrs) in STAGES.items():
    for k in range(len(layers)):
        for attr in attrs:
            undo = cut([(st, attr, k)]); f, o, s, keep, iou = compare(ref[:a.rank_images], run(X[:a.rank_images])); undo()
            rank.append(((1 - keep) + (1 - f) + (1 - iou), st, attr, k, f, s - o, keep, iou))
rank.sort()
print("single cuts, least harmful first (found, score change, still >= 0.5, median IoU):")
for d, st, attr, k, f, ds, keep, iou in rank:
    print(f"  {st}.{attr}:{k:<2d}  {f:.3f}  {ds:+.3f}  {keep:.3f}  {iou:.3f}")
order = [(st, attr, k) for _, st, attr, k, *_ in rank]

# ViT: greedy. Single-cut ranks do not add up (the damage is cumulative), so at every step the block that hurts the scores least, given the
# blocks already cut, is added. Candidates: the --pool best single cuts.
pool, chosen = [b for b in order if b[0] == "vit"][:a.pool], []
print(f"ViT, greedy over the {len(pool)} best single cuts ({a.rank_images} images to choose, all {len(X)} to report):\n   n  found  mean score      still >= 0.5  median IoU  added")
while pool and len(chosen) < a.max_vit:
    best = None
    for b in pool:
        undo = cut(chosen + [b]); f, o, s, keep, iou = compare(ref[:a.rank_images], run(X[:a.rank_images])); undo()
        if best is None or (f, s) > best[0]: best = ((f, s), b)
    chosen.append(best[1]); pool.remove(best[1])
    undo = cut(chosen); f, o, s, keep, iou = compare(ref, run(X)); undo()
    print(f"  {len(chosen):2d}  {f:.3f}  {o:.3f} -> {s:.3f}  {keep:.3f}         {iou:.3f}       {chosen[-1][1]}:{chosen[-1][2]}", flush=True)
print("greedy order: " + ",".join(f"{attr}:{k}" for _, attr, k in chosen))
cumulative(f"the first 10 of the greedy ViT order + encoder/decoder blocks:", chosen[:10], [b for b in order if b[0] != "vit"])
