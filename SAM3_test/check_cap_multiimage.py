"""Re-check of the _get_coords cap on many boxes (see GETCOORDS_INVESTIGATION.md, "Re-check").

Metric: mean |coordinate deviation| (px) of matched boxes vs the ORIGINAL (buggy) formula at imgsz 1008
-- a proxy, not ground truth. Data: bus.jpg (bus+person) + 12 frames of a video (person).
Run from the repo root:  HF_TOKEN=... VIDEO=/path/to/pedestrians.mp4 python SAM3_test/check_cap_multiimage.py
Note: _VisionEncoder mutates the shared model, so wrappers are rebuilt whenever imgsz changes.
"""
import sys, os, numpy as np, torch, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "SAM3Fixed"))
import export as E
from transformers.models.sam3 import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3DetrDecoder
tok = os.environ["HF_TOKEN"]
model = Sam3Model.from_pretrained("facebook/sam3", token=tok).to("cuda").eval()
proc = Sam3Processor.from_pretrained("facebook/sam3", token=tok)

def orig(self, height, width, dtype, device):          # the original (buggy) grid: max (H-1)/H
    return (torch.arange(0, height, device=device, dtype=dtype) / height,
            torch.arange(0, width, device=device, dtype=dtype) / width)
def set_cap(c):
    if c == "orig": Sam3DetrDecoder._get_coords = orig
    else:
        def capped(self, h, w, dtype, device):
            return (torch.arange(0, h, device=device, dtype=dtype) / (h - 1) * c,
                    torch.arange(0, w, device=device, dtype=dtype) / (w - 1) * c)
        Sam3DetrDecoder._get_coords = capped

CAPS = [("orig", "orig"), ("0.98", 0.98), ("71/72", 71/72), ("0.987654321", 0.987654321), ("0.99", 0.99), ("1.0", 1.0)]
SIZES = [420, 504, 700, 1008]

# data: bus.jpg (bus+person) and 12 frames of pedestrians.mp4 (person)
data = [(cv2.imread(os.path.join(HERE, "bus.jpg")), ["bus", "person"], "bus.jpg")]
cap = cv2.VideoCapture(os.environ.get("VIDEO", "/mnt/c/Dev/pedestrians.mp4")); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in np.linspace(10, n - 10, 12).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read(); data.append((f, ["person"], f"ped{i}"))

enc = {}
def emb(cl):
    k = tuple(cl)
    if k not in enc: enc[k] = E.encode_concepts(model, proc, cl)
    return enc[k]
wr = {}
def det(img, cl, imgsz):
    k = (tuple(cl), imgsz)
    if wr and next(iter(wr))[1] != imgsz: wr.clear()   # _VisionEncoder mutates the shared model: one imgsz at a time
    if k not in wr:
        te, am = emb(cl); wr[k] = E.Sam3Wrapper(model, te, am, imgsz, with_mask=False).cuda().eval()
    H0, W0 = img.shape[:2]
    x = cv2.cvtColor(cv2.resize(img, (imgsz, imgsz)), cv2.COLOR_BGR2RGB).astype(np.float32)
    x = torch.from_numpy((x - 127.5) / 127.5).permute(2, 0, 1)[None].cuda()
    with torch.no_grad(): d = wr[k](x)[0].float().cpu().numpy()
    d[:, [0, 2]] *= W0 / imgsz; d[:, [1, 3]] *= H0 / imgsz
    return d, len(cl)

def iou(a, b):
    ix1, iy1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1]); ix2, iy2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(ix2-ix1, 0, None) * np.clip(iy2-iy1, 0, None)
    return inter / np.maximum((a[2]-a[0])*(a[3]-a[1]) + (b[:,2]-b[:,0])*(b[:,3]-b[:,1]) - inter, 1e-9)
def keep(d, thr):
    d = d[(d[:, 4] >= thr) & (d[:, 2] > d[:, 0]) & (d[:, 3] > d[:, 1])]; d = d[np.argsort(-d[:, 4])]; out = []
    for r in d:
        if all(iou(r, np.array([o]))[0] <= 0.6 or o[5] != r[5] for o in out): out.append(r)
    return np.array(out) if out else np.zeros((0, 6))

# reference = original (buggy) formula at 1008, as in the md
set_cap("orig"); ref = {name: keep(det(img, cl, 1008)[0], 0.5) for img, cl, name in data}
res = {}
for imgsz in SIZES:
  for cname, c in CAPS:
    set_cap(c)
    if True:
        ds = []; match = tot = 0
        for img, cl, name in data:
            t = keep(det(img, cl, imgsz)[0], 0.3); r = ref[name]; tot += len(r)
            for b in r:
                m = t[t[:, 5] == b[5]] if len(t) else t
                if len(m):
                    v = iou(b, m); j = int(np.argmax(v))
                    if v[j] >= 0.5: match += 1; ds.append(np.abs(m[j, :4] - b[:4]).mean())
        res[(cname, imgsz)] = (np.mean(ds) if ds else float("nan"), match, tot)
print(f"\nmean|d| (px) of box coordinates vs the ORIGINAL formula at 1008, over bus.jpg + 12 pedestrian frames  (lower = closer to the 1008 box)")
print(f"{'cap':>12} " + " ".join(f"{s:>16}" for s in SIZES) + "      mean over sizes<1008")
for cname, _ in CAPS:
    vals = [res[(cname, s)] for s in SIZES]
    print(f"{cname:>12} " + " ".join(f"{v[0]:7.2f} ({v[1]:>2}/{v[2]:>2})" for v in vals) + f"      {np.mean([v[0] for v in vals[:3]]):.2f}")
