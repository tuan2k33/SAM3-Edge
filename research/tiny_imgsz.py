"""Does SAM3 (HF transformers) run at tiny imgsz (H = imgsz/14 = 1, 2, 3 cells)?

For each size the model is loaded with image_size=<size> (config override), fp16 on GPU, one random image, prompt "person".
Where it fails (neck, decoder grid, ...) is printed. _get_coords variants: stock (i/n), pr (i/(n-1)*71/72), clamp (shipped fix).

    python research/tiny_imgsz.py [--sizes 14 28 42 420] [--variants stock pr clamp]
"""
import argparse, glob, os, sys, traceback
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
from transformers import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3DetrDecoder

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[14, 28, 42, 420])
ap.add_argument("--variants", nargs="+", default=["stock", "pr", "clamp"])
args = ap.parse_args()

snap = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/")))[0]


def stock(self, height, width, dtype, device):
    return torch.arange(0, height, device=device, dtype=dtype) / height, torch.arange(0, width, device=device, dtype=dtype) / width


def pr(self, height, width, dtype, device):
    f = lambda n: torch.arange(0, n, device=device, dtype=dtype) / (n - 1) * 71 / 72
    return f(height), f(width)


def clamp(self, height, width, dtype, device):
    f = lambda n: torch.arange(0, n, device=device, dtype=dtype) / torch.clamp(n - 1, min=1) * 71 / 72
    return f(height), f(width)


VARIANTS = dict(stock=stock, pr=pr, clamp=clamp)
proc = Sam3Processor.from_pretrained(snap)
tok = proc.tokenizer("person", return_tensors="pt", padding="max_length", max_length=32)

for size in args.sizes:
    model = Sam3Model.from_pretrained(snap, image_size=size).half().to("cuda").eval()
    x = torch.randn(1, 3, size, size, device="cuda").half()
    for v in args.variants:
        Sam3DetrDecoder._get_coords = VARIANTS[v]
        try:
            with torch.no_grad():
                out = model(pixel_values=x, input_ids=tok["input_ids"].cuda(), attention_mask=tok["attention_mask"].cuda())
            b = out.pred_boxes
            print(f"imgsz {size:4d} {v:5s}: ok, boxes {tuple(b.shape)}, nan={bool(torch.isnan(b).any())}", flush=True)
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            print(f"imgsz {size:4d} {v:5s}: FAIL {type(e).__name__}: {str(e)[:110]}  @ {os.path.basename(tb.filename)}:{tb.lineno} {tb.name}", flush=True)
    del model
    torch.cuda.empty_cache()
