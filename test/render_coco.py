"""coco128 images with ground truth (blue), before = original decoder (red) and the shipped 71/72 grid (green).

Same setup as collect_box_pairs.py (oracle prompts: the <= 4 COCO classes with a GT box >= 2% of the image, score >= 0.5,
NMS 0.6, first --nimg usable images, eager fp16), but every box is drawn instead of paired. The raw pairs in
results_pairs/ carry no image id, so the forward passes are rerun (about a minute per size).

    HF_TOKEN=... python test/render_coco.py [--sizes 420 700] [--nimg 64] [--skip 0]   # -> test/coco_vis/<size>/<image>.jpg
"""
import argparse, collections, glob, importlib.util, os, sys
import cv2, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import export as E
OE = E   # export.py holds the set-classes and the text-as-input wrappers
from transformers.models.sam3 import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3DetrDecoder

ap = argparse.ArgumentParser()
ap.add_argument("--coco", default=os.path.join(HERE, "data", "coco128"))
ap.add_argument("--sizes", nargs="+", type=int, default=[420, 700])
ap.add_argument("--nimg", type=int, default=64)
ap.add_argument("--skip", type=int, default=0)
ap.add_argument("--out", default=os.path.join(HERE, "coco_vis"))
args = ap.parse_args()
SHIPPED = Sam3DetrDecoder._get_coords
NAMES = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter "
         "bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
         "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass "
         "cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
         "potted_plant bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink "
         "refrigerator book clock vase scissors teddy_bear hair_drier toothbrush").split()
NAMES = [n.replace("_", " ") for n in NAMES]


def orig_grid(self, h, w, dtype, device):
    f = lambda n: (torch.arange(0, n, device=device, dtype=torch.float32) / n).to(dtype)
    return f(h), f(w)


def iou(a, b):
    ix1, iy1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1]); ix2, iy2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / np.maximum((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter, 1e-12)


def nms(d, thr=0.6):
    out = []
    for r in d[np.argsort(-d[:, 4])]:
        if not out or iou(r, np.array(out)).max() <= thr:
            out.append(r)
    return np.array(out) if out else np.zeros((0, 6))


items = []
for lp in sorted(glob.glob(args.coco + "/labels/train2017/*.txt")):
    ip = lp.replace("/labels/", "/images/").replace(".txt", ".jpg"); img = cv2.imread(ip)
    if img is None:
        continue
    gt = collections.defaultdict(list)
    for ln in open(lp):
        c, cx, cy, w, h = ln.split(); c = int(c); cx, cy, w, h = map(float, (cx, cy, w, h))
        if w * h >= 0.02:
            gt[c].append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    if gt:
        top = [c for c, _ in sorted(gt.items(), key=lambda kv: -len(kv[1]))[:4]]
        items.append((os.path.basename(ip)[:-4], img, top, {c: np.array(gt[c]) for c in top}))
    if len(items) >= args.nimg + args.skip:
        break
items = items[args.skip:]

tok = os.environ["HF_TOKEN"]
model = Sam3Model.from_pretrained("facebook/sam3", token=tok).half().to("cuda").eval()
proc = Sam3Processor.from_pretrained("facebook/sam3", token=tok)
te_all, am_all = E.encode_concepts(model, proc, NAMES); TE, AM = te_all.cuda(), (am_all > 0).cuda()
COL = {"gt": (255, 0, 0), "before": (0, 0, 255), "shipped": (0, 160, 0)}

for S in args.sizes:
    w = OE.DynamicVisionDecoderWrapper(model, "cuda", S).cuda().eval().half(); w.scale = w.scale.float()
    os.makedirs(os.path.join(args.out, str(S)), exist_ok=True)
    for name, img, ids, gtd in items:
        h0, w0 = img.shape[:2]
        x = (cv2.cvtColor(cv2.resize(img, (S, S)), cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
        x = torch.from_numpy(x).permute(2, 0, 1)[None].cuda().half()
        preds = {}
        for tag, grid in (("before", orig_grid), ("shipped", SHIPPED)):
            Sam3DetrDecoder._get_coords = grid
            with torch.no_grad():
                d = w(x, TE[ids].unsqueeze(0), AM[ids].unsqueeze(0))[0][0].float().cpu().numpy()
            d[:, [0, 2]] *= w0 / S; d[:, [1, 3]] *= h0 / S
            preds[tag] = [nms(d[(d[:, 5] == i) & (d[:, 4] >= 0.5) & (d[:, 2] > d[:, 0]) & (d[:, 3] > d[:, 1])]) for i in range(len(ids))]
        vis = img.copy(); t = max(2, round(max(h0, w0) / 400))
        for i, cid in enumerate(ids):
            for g in gtd[cid]:
                cv2.rectangle(vis, (int(g[0] * w0), int(g[1] * h0)), (int(g[2] * w0), int(g[3] * h0)), COL["gt"], t + 2)
            for tag in ("before", "shipped"):
                for b in preds[tag][i]:
                    cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), COL[tag], t)
        cv2.rectangle(vis, (0, 0), (max(250, int(w0 * 0.45)), 78), (255, 255, 255), -1)
        for k, (txt, col) in enumerate([(f"{name}  imgsz={S}  {', '.join(NAMES[c] for c in ids)}", (0, 0, 0)), ("ground truth", COL["gt"]),
                                        ("before (original grid)", COL["before"]), ("shipped (71/72 grid)", COL["shipped"])]):
            cv2.putText(vis, txt, (6, 18 + 19 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1 if k == 0 else 2)
        cv2.imwrite(os.path.join(args.out, str(S), f"{name}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("done", S, len(items), "images ->", os.path.join(args.out, str(S)), flush=True)
    del w; torch.cuda.empty_cache()
