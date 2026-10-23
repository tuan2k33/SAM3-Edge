"""bus.jpg, 'person' + 'bus': before (original decoder) / after (fixes as shipped) / reference, one image per imgsz.

bus.jpg has no ground-truth labels: the reference (blue) is the shipped model at 1008, the resolution it was trained at
(on COCO it matches GT to ~1% in box size, see BOX_SHRINK_INVESTIGATION.md). Eager fp16, one wrapper per imgsz.

    HF_TOKEN=... python test/compare_bus.py [--sizes 420 700]
"""
import argparse, importlib.util, os, sys
import cv2, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "SAM3Fixed"))
import export as E
_sp = importlib.util.spec_from_file_location("open_export", os.path.join(HERE, "..", "SAM3Open", "export.py"))
OE = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(OE)
from transformers.models.sam3 import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3DetrDecoder

ap = argparse.ArgumentParser(); ap.add_argument("--sizes", nargs="+", type=int, default=[420, 700])
args = ap.parse_args()
SHIPPED = Sam3DetrDecoder._get_coords


def orig_grid(self, h, w, dtype, device):
    f = lambda n: (torch.arange(0, n, device=device, dtype=torch.float32) / n).to(dtype)
    return f(h), f(w)


tok = os.environ["HF_TOKEN"]
model = Sam3Model.from_pretrained("facebook/sam3", token=tok).half().to("cuda").eval()
proc = Sam3Processor.from_pretrained("facebook/sam3", token=tok)
CLASSES = ["person", "bus"]
te, am = E.encode_concepts(model, proc, CLASSES); te, am = te.cuda(), (am > 0).cuda()
img = cv2.imread(os.path.join(HERE, "bus.jpg")); h0, w0 = img.shape[:2]


def run(S, shipped, grid_only=False):
    E._APPLY_QUERY_BOX_SCALE = OE._APPLY_QUERY_BOX_SCALE = shipped and not grid_only
    model.detr_decoder.position_encoding.box_scale = 1.0
    Sam3DetrDecoder._get_coords = SHIPPED if shipped else orig_grid
    w = OE.DynamicVisionDecoderWrapper(model, "cuda", S).cuda().eval().half(); w.scale = w.scale.float()
    x = (cv2.cvtColor(cv2.resize(img, (S, S)), cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
    with torch.no_grad():
        d = w(torch.from_numpy(x).permute(2, 0, 1)[None].cuda().half(), te.unsqueeze(0), am.unsqueeze(0))[0].float().cpu().numpy()
    d = d[(d[:, 4] >= 0.5) & (d[:, 2] > d[:, 0]) & (d[:, 3] > d[:, 1])]
    d[:, [0, 2]] *= w0 / S; d[:, [1, 3]] *= h0 / S
    return d


def draw(vis, d, col, thick):
    for x1, y1, x2, y2, sc, c in d:
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, thick)


def nms(d, thr=0.6):
    keep = []
    for r in d[np.argsort(-d[:, 4])]:
        same = [k for k in keep if k[5] == r[5]]
        ok = True
        for k in same:
            iw = max(0, min(r[2], k[2]) - max(r[0], k[0])); ih = max(0, min(r[3], k[3]) - max(r[1], k[1]))
            inter = iw * ih
            if inter / ((r[2] - r[0]) * (r[3] - r[1]) + (k[2] - k[0]) * (k[3] - k[1]) - inter) > thr:
                ok = False
        if ok:
            keep.append(r)
    return np.array(keep)


ref = nms(run(1008, True))
for S in args.sizes:
    before, grid, after = nms(run(S, False)), nms(run(S, True, grid_only=True)), nms(run(S, True))
    vis = img.copy()
    cv2.rectangle(vis, (0, 0), (330, 138), (255, 255, 255), -1)
    for i, (t, col) in enumerate([(f"SAM3 imgsz={S} bus+person", (0, 0, 0)), ("before (original decoder)", (0, 0, 255)),
                                  ("only 71/72 grid", (0, 140, 255)), ("after (71/72 + sineq)", (0, 170, 0)), ("reference (1008)", (255, 0, 0))]):
        cv2.putText(vis, t, (8, 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    draw(vis, ref, (255, 0, 0), 4); draw(vis, before, (0, 0, 255), 2); draw(vis, grid, (0, 140, 255), 2); draw(vis, after, (0, 170, 0), 2)
    cv2.imwrite(os.path.join(HERE, f"sam3_bus_before_after_ref_{S}.jpg"), vis)
    print(S, "before", len(before), "grid", len(grid), "after", len(after), "ref", len(ref))
    for name, d in (("before", before), ("grid", grid), ("after", after)):
        o = ref[np.argsort(ref[:, 0])]; d = d[np.argsort(d[:, 0])]
        if len(d) == len(o):
            print(f"  {name}: mean width ratio to ref {np.mean((d[:, 2] - d[:, 0]) / (o[:, 2] - o[:, 0])):.3f}, height {np.mean((d[:, 3] - d[:, 1]) / (o[:, 3] - o[:, 1])):.3f}")
