"""Which lever lifts prompt-free recall: all 3 masks per point, mask resolution, grid density, imgsz? Eager PyTorch fp16 (no engine build), the same
COCO val2017 recall as test/eval_prompt_free.py (its Engine is swapped for the eager wrapper).

    HF_HUB_OFFLINE=1 python research/prompt_free_levers.py 420 [--stride 2] [--all-masks] [--mask-size 36] [--min-area 0.003] [--n 200]
Results: docs/NOTES.md section 6."""
import argparse, os, runpy, sys

import cv2, numpy as np, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "test")]
import export, infer_prompt_free
from transformers.models.sam3_tracker import Sam3TrackerModel

ap = argparse.ArgumentParser()
ap.add_argument("imgsz", type=int); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--all-masks", action="store_true")
ap.add_argument("--mask-size", type=int, default=export.MASK_SIZE); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--min-area", type=float, default=0.003, help="post-process: smallest mask kept, a fraction of the frame")
a = ap.parse_args()

tracker = Sam3TrackerModel.from_pretrained("facebook/sam3", token=os.environ.get("HF_TOKEN")).float().eval()
wrapper = export.PromptFreeWrapper(tracker, "cpu", a.imgsz, a.stride, a.all_masks, a.mask_size).half().cuda().eval()  # prompt embeddings are computed in fp32
wrapper.scale = wrapper.scale.float()
del tracker


class Eager:
    def __init__(self, path, S): self.S = S
    def __call__(self, bgr):
        S = self.S
        x = torch.from_numpy((cv2.cvtColor(cv2.resize(bgr, (S, S)), cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5).permute(2, 0, 1)[None].cuda().half()
        with torch.no_grad():
            det, mk = wrapper(x)
        return det[0].float(), mk[0].float()


infer_prompt_free.Engine = Eager
_post = infer_prompt_free.postprocess
infer_prompt_free.postprocess = lambda det, mk: _post(det, mk, min_area=a.min_area)
print(f"imgsz {a.imgsz}, stride {a.stride} ({(a.imgsz // 14 // a.stride) ** 2} points), {'all 3 masks' if a.all_masks else 'best mask'}, mask {a.mask_size}, min area {a.min_area}", flush=True)
sys.argv = ["eval_prompt_free.py", "eager", str(a.imgsz), str(a.n)]
runpy.run_path(os.path.join(ROOT, "test", "eval_prompt_free.py"), run_name="__main__")
