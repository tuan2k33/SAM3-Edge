"""image_size not a multiple of 14 in HF SAM3: is 430 the same as cropping the top-left 420x420 of the 430 input?

    python test/floor_imgsz.py [--size 430]
"""
import argparse, glob, os, sys
import cv2, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
from transformers import Sam3Model, Sam3Processor

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, default=430)
args = ap.parse_args()
fl = args.size // 14 * 14

snap = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/")))[0]
proc = Sam3Processor.from_pretrained(snap)
tok = proc.tokenizer("person", return_tensors="pt", padding="max_length", max_length=32)
img = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(HERE, "bus.jpg")), (args.size, args.size)), cv2.COLOR_BGR2RGB)
x = torch.from_numpy((img.astype(np.float32) - 127.5) / 127.5).permute(2, 0, 1)[None].cuda().half()


def run(image_size, inp):
    m = Sam3Model.from_pretrained(snap, image_size=image_size).half().to("cuda").eval()
    with torch.no_grad():
        o = m(pixel_values=inp, input_ids=tok["input_ids"].cuda(), attention_mask=tok["attention_mask"].cuda())
    return o.pred_boxes.float().cpu(), o.pred_logits.float().cpu()


b_full, l_full = run(args.size, x)
b_crop, l_crop = run(fl, x[:, :, :fl, :fl].contiguous())
print(f"input {args.size}x{args.size}, floor to {fl}")
print("max |boxes diff|:", (b_full - b_crop).abs().max().item(), " max |logits diff|:", (l_full - l_crop).abs().max().item())
top = l_full[0].argmax()
print("top query box (normalised to the feature grid):", [round(v, 4) for v in b_full[0, top].tolist()])
