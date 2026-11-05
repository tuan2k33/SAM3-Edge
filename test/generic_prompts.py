"""Class-agnostic prompts on coco128: does a generic prompt ("object", "main object", ...) make SAM3 find the GT objects?

Same images, fp16 model and decoding as collect_box_pairs.py (first 64 usable images, imgsz 1008, NMS 0.6).
Per prompt: recall@IoU0.5 of the GT boxes >= 2% area (any class), precision of the predictions against every GT box
of any size, predictions per image. "oracle" = the COCO class names present in the image, predictions merged over classes.

    HF_TOKEN=... python test/generic_prompts.py [--imgsz 1008] [--thr 0.5 0.3]
"""
import argparse, collections, glob, importlib.util, os, sys
import cv2, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "SAM3Fixed"))
import export as E
_sp = importlib.util.spec_from_file_location("open_export", os.path.join(HERE, "..", "SAM3Open", "export.py"))
OE = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(OE)
from transformers.models.sam3 import Sam3Model, Sam3Processor

ap = argparse.ArgumentParser()
ap.add_argument("--coco", default=os.path.join(HERE, "data", "coco128"))
ap.add_argument("--imgsz", type=int, default=1008)
ap.add_argument("--nimg", type=int, default=64)
ap.add_argument("--skip", type=int, default=0)
ap.add_argument("--thr", type=float, nargs="+", default=[0.5, 0.3])
args = ap.parse_args()

PROMPTS = ["object", "objects", "thing", "main object", "salient object", "foreground object", "prominent object", "item"]
NAMES = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter "
         "bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
         "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass "
         "cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
         "potted_plant bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink "
         "refrigerator book clock vase scissors teddy_bear hair_drier toothbrush").split()
NAMES = [n.replace("_", " ") for n in NAMES]


def iou(a, b):
    ix1, iy1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1]); ix2, iy2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / np.maximum((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter, 1e-12)


def nms(d, thr=0.6):
    d = d[np.argsort(-d[:, 4])]; out = []
    for r in d:
        if not out or iou(r, np.array(out)).max() <= thr:
            out.append(r)
    return np.array(out) if out else np.zeros((0, 6))


items = []
for lp in sorted(glob.glob(args.coco + "/labels/train2017/*.txt")):
    img = cv2.imread(lp.replace("/labels/", "/images/").replace(".txt", ".jpg"))
    if img is None:
        continue
    rows = [list(map(float, ln.split())) for ln in open(lp)]
    big = [r for r in rows if r[3] * r[4] >= 0.02]
    if big:
        xyxy = lambda rs: np.array([[r[1] - r[3] / 2, r[2] - r[4] / 2, r[1] + r[3] / 2, r[2] + r[4] / 2] for r in rs])
        items.append((img, xyxy(big), xyxy(rows), sorted({int(r[0]) for r in big})[:4]))
    if len(items) >= args.nimg + args.skip:
        break
items = items[args.skip:]
print("images", len(items), "GT>=2%", sum(len(g) for _, g, _, _ in items), "GT all", sum(len(a) for _, _, a, _ in items), flush=True)

tok = os.environ["HF_TOKEN"]
model = Sam3Model.from_pretrained("facebook/sam3", token=tok).half().to("cuda").eval()
proc = Sam3Processor.from_pretrained("facebook/sam3", token=tok)
te, am = E.encode_concepts(model, proc, PROMPTS + NAMES)
TE, AM = te.cuda(), (am > 0).cuda()
w = OE.DynamicVisionDecoderWrapper(model, "cuda", args.imgsz).cuda().eval().half()
w.scale = w.scale.float()
S = args.imgsz

stat = {(n, t): dict(tp=0, gt=0, npred=0, ppos=0) for n in PROMPTS + ["oracle"] for t in args.thr}
for img, gt, allgt, cls in items:
    x = cv2.cvtColor(cv2.resize(img, (S, S)), cv2.COLOR_BGR2RGB).astype(np.float32)
    x = torch.from_numpy((x - 127.5) / 127.5).permute(2, 0, 1)[None].cuda().half()
    runs = {n: [PROMPTS.index(n)] for n in PROMPTS}
    runs["oracle"] = [len(PROMPTS) + c for c in cls]
    for name, ids in runs.items():
        with torch.no_grad():
            d = w(x, TE[ids].unsqueeze(0), AM[ids].unsqueeze(0))[0].float().cpu().numpy()
        d[:, :4] = np.clip(d[:, :4] / S, 0, 1)
        for t in args.thr:
            p = nms(d[(d[:, 4] >= t) & (d[:, 2] > d[:, 0]) & (d[:, 3] > d[:, 1])])
            s = stat[(name, t)]
            s["gt"] += len(gt); s["npred"] += len(p)
            s["tp"] += sum(bool(len(p)) and iou(g, p[:, :4]).max() >= 0.5 for g in gt)
            s["ppos"] += sum(iou(q[:4], allgt).max() >= 0.5 for q in p)

n = len(items)
print(f"\nimgsz {S}, {n} images")
print(f"{'prompt':20s} {'thr':>4s} {'recall@.5':>10s} {'precision':>10s} {'pred/img':>9s}")
for (name, t), s in stat.items():
    print(f"{name:20s} {t:4.1f} {s['tp'] / max(s['gt'], 1):10.3f} {s['ppos'] / max(s['npred'], 1):10.3f} {s['npred'] / n:9.2f}")
