"""Is a class's box independent of which other classes are baked into the wrapper?

Builds Sam3Wrapper for ['bus'], ['bus','person'], ['person','bus'], ['person'] on bus.jpg and compares the
top-scoring box per class (810x1080 px). Expected: identical (differences only from fp16 rounding).
Run from the repo root:  HF_TOKEN=... python SAM3_test/check_class_independence.py
"""
import os, sys
import cv2, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "SAM3Fixed"))
import export as E
from transformers.models.sam3 import Sam3Model, Sam3Processor

tok = os.environ["HF_TOKEN"]
model = Sam3Model.from_pretrained("facebook/sam3", token=tok).to("cuda").eval()
proc = Sam3Processor.from_pretrained("facebook/sam3", token=tok)
img = cv2.imread(os.path.join(HERE, "bus.jpg")); H0, W0 = img.shape[:2]


def top_boxes(classes, imgsz):
    te, am = E.encode_concepts(model, proc, classes)
    w = E.Sam3Wrapper(model, te, am, imgsz, with_mask=False).cuda().eval()
    x = cv2.cvtColor(cv2.resize(img, (imgsz, imgsz)), cv2.COLOR_BGR2RGB).astype(np.float32)
    x = torch.from_numpy((x - 127.5) / 127.5).permute(2, 0, 1)[None].cuda()
    with torch.no_grad():
        d = w(x)[0].float().cpu().numpy()
    Q = d.shape[0] // len(classes)
    out = {}
    for i, c in enumerate(classes):
        r = d[i * Q:(i + 1) * Q]; j = int(np.argmax(r[:, 4]))
        out[c] = (r[j, :4] * np.array([W0, H0, W0, H0]) / imgsz, float(r[j, 4]))
    del w; torch.cuda.empty_cache()
    return out


for imgsz in (1008, 420):
    runs = {tuple(c): top_boxes(c, imgsz) for c in (["bus"], ["bus", "person"], ["person", "bus"], ["person"])}
    print(f"imgsz={imgsz}")
    for cls in ("bus", "person"):
        ref = next(r[cls][0] for k, r in runs.items() if k == (cls,))
        for k, r in runs.items():
            if cls in r:
                print(f"  {cls:6s} in {str(list(k)):20s} box={np.round(r[cls][0], 1)} score={r[cls][1]:.3f} max|d| vs alone: {np.abs(r[cls][0] - ref).max():.2f}px")
