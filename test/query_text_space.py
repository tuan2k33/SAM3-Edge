"""Do SAM3's decoder object queries live in the text embedding space? (HF transformers, local clone, fp16)

Sam3DotProductScoring scores a query as  query_proj(q) . text_proj(pool(text_mlp(text)))  / 16.  So a query's projected vector
can be compared with the projected vector of ANY text, not only the prompt it was decoded with. For each GT box (>= 2% area,
coco128, first 64 usable images) take the query whose box has the best IoU (>= 0.5) with it, and classify that query among the
80 COCO names by the same dot product. The decoding prompt is varied (generic words and the oracle class names), chance = 1/80.

    HF_HUB_OFFLINE=1 python test/query_text_space.py
"""
import argparse, collections, glob, os, sys
import cv2, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/Dev/projects/transformers/src"))
from transformers import Sam3Model, Sam3Processor

ap = argparse.ArgumentParser()
ap.add_argument("--coco", default=os.path.join(HERE, "data", "coco128"))
ap.add_argument("--imgsz", type=int, default=1008)
ap.add_argument("--nimg", type=int, default=64)
args = ap.parse_args()

NAMES = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter "
         "bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
         "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass "
         "cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
         "potted_plant bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink "
         "refrigerator book clock vase scissors teddy_bear hair_drier toothbrush").split()
NAMES = [n.replace("_", " ") for n in NAMES]
GENERIC = ["object", "foreground object", "prominent object", "item"]

items = []
for lp in sorted(glob.glob(args.coco + "/labels/train2017/*.txt")):
    img = cv2.imread(lp.replace("/labels/", "/images/").replace(".txt", ".jpg"))
    if img is None:
        continue
    rows = [list(map(float, ln.split())) for ln in open(lp)]
    big = [(int(r[0]), r[1] - r[3] / 2, r[2] - r[4] / 2, r[1] + r[3] / 2, r[2] + r[4] / 2) for r in rows if r[3] * r[4] >= 0.02]
    if big:
        items.append((img, big))
    if len(items) >= args.nimg:
        break

snap = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/")))[0]
proc = Sam3Processor.from_pretrained(snap)
model = Sam3Model.from_pretrained(snap, image_size=args.imgsz).half().to("cuda").eval()
ds = model.dot_product_scoring
tokenize = lambda s: proc.tokenizer(s, return_tensors="pt", padding="max_length", max_length=32).to("cuda")


@torch.no_grad()
def text_proj(s):
    t = tokenize(s)
    feats = model.get_text_features(input_ids=t["input_ids"], attention_mask=t["attention_mask"], return_dict=True).pooler_output
    f = ds.text_mlp_out_norm(ds.text_mlp(feats) + feats)
    return ds.text_proj(ds._pool_text_features(f, t["attention_mask"].bool())), t


P = torch.cat([text_proj(n)[0] for n in NAMES])             # [80, 256]
P = P.float()
cap = {}
ds.register_forward_hook(lambda m, a, k, o: cap.__setitem__("h", k["decoder_hidden_states"][-1] if "decoder_hidden_states" in k else a[0][-1]), with_kwargs=True)


def iou(a, b):
    ix1, iy1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1]); ix2, iy2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / np.maximum((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter, 1e-12)


stat = collections.defaultdict(lambda: dict(n=0, loc=0, top1=0, top5=0))
rec = collections.defaultdict(list)   # tag -> (class, query vector projected, own-prompt logit)
S = args.imgsz
for img, gts in items:
    x = cv2.cvtColor(cv2.resize(img, (S, S)), cv2.COLOR_BGR2RGB).astype(np.float32)
    x = torch.from_numpy((x - 127.5) / 127.5).permute(2, 0, 1)[None].cuda().half()
    oracle = sorted({g[0] for g in gts})[:4]
    for tag, prompts in [(g, [g]) for g in GENERIC] + [("oracle", [NAMES[c] for c in oracle])]:
        for p in prompts:
            _, t = text_proj(p)
            with torch.no_grad():
                out = model(pixel_values=x, input_ids=t["input_ids"], attention_mask=t["attention_mask"])
                q = ds.query_proj(cap["h"])[0].float()                                  # [200, 256]
            boxes = out.pred_boxes[0].float().cpu().numpy()
            sc = (q @ P.T) * ds.scale                                                    # [200, 80] logits against every name
            for c, *g in gts:
                if tag == "oracle" and NAMES[c] != p:
                    continue
                s = stat[tag]; s["n"] += 1
                i = iou(np.array(g), boxes); j = int(i.argmax())
                if i[j] < 0.5:
                    continue
                s["loc"] += 1; rec[tag].append((c, q[j].cpu()))


print(f"\nimgsz {S}, {len(items)} images, chance top1 {1/80:.3f} top5 {5/80:.3f}; |text| min/max {P.norm(dim=1).min():.1f}/{P.norm(dim=1).max():.1f}")
Pc = P - P.mean(0)
print(f"{'decoded with':18s} {'n':>4s} | {'raw dot':>13s} | {'cosine':>13s} | {'dot, mean-removed':>17s}   (top1/top5)")
for tag, r in rec.items():
    Q = torch.stack([v for _, v in r]).cuda(); y = torch.tensor([c for c, _ in r]).cuda()
    res = []
    for sc in (Q @ P.T, torch.nn.functional.normalize(Q, dim=1) @ torch.nn.functional.normalize(P, dim=1).T, (Q - Q.mean(0)) @ Pc.T):
        top = sc.topk(5, dim=1).indices
        res.append(f"{(top[:, 0] == y).float().mean():.3f}/{(top == y[:, None]).any(1).float().mean():.3f}")
    print(f"{tag:18s} {len(r):4d} | {res[0]:>13s} | {res[1]:>13s} | {res[2]:>17s}")
    if tag == "oracle":
        print("  most frequent raw argmax:", collections.Counter(NAMES[i] for i in (Q @ P.T).argmax(1).tolist()).most_common(4))
