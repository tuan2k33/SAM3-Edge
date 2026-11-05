"""How many cells does each SAM3 stage actually receive when image_size changes? (HF transformers, local clone, fp16)

Hooks print the shapes at: ViT output, FPN neck levels, DETR encoder input, DETR decoder input + spatial_shapes, and the RPB mask.

    python test/grid_shapes.py [--sizes 420 700 1008 430]
"""
import argparse, glob, os, sys
import torch

sys.path.insert(0, os.path.expanduser("~/Dev/projects/transformers/src"))
from transformers import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3DetrDecoder

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[420, 700, 1008, 430])
args = ap.parse_args()

snap = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/")))[0]
tok = Sam3Processor.from_pretrained(snap).tokenizer("person", return_tensors="pt", padding="max_length", max_length=32)


def shp(t):
    if isinstance(t, (list, tuple)):
        return [shp(i) for i in t]
    return tuple(t.shape) if hasattr(t, "shape") else t


for size in args.sizes:
    model = Sam3Model.from_pretrained(snap, image_size=size).half().to("cuda").eval()
    log = []
    hs = []
    hs.append(model.vision_encoder.backbone.register_forward_hook(
        lambda m, a, o: log.append(("ViT backbone output", shp(getattr(o, "last_hidden_state", o))))))
    hs.append(model.vision_encoder.neck.register_forward_hook(
        lambda m, a, o: log.append(("FPN neck levels (feature maps)", shp(o[0] if isinstance(o, tuple) else o)))))
    hs.append(model.detr_encoder.register_forward_pre_hook(
        lambda m, a, k: log.append(("DETR encoder vision_features in", shp(k["vision_features"]))), with_kwargs=True))
    hs.append(model.detr_decoder.register_forward_pre_hook(
        lambda m, a, k: log.append(("DETR decoder vision_features in", shp(k["vision_features"]))), with_kwargs=True))
    hs.append(model.detr_decoder.register_forward_pre_hook(
        lambda m, a, k: log.append(("decoder spatial_shapes (H, W) -> _get_coords", k["spatial_shapes"].tolist())), with_kwargs=True))
    orig = Sam3DetrDecoder._get_rpb_matrix

    def rpb(self, ref, shape, _o=orig):
        r = _o(self, ref, shape)
        if not any(n.startswith("RPB") for n, _ in log):
            log.append(("RPB mask [B, heads, Q, H*W]", tuple(r.shape)))
        return r
    Sam3DetrDecoder._get_rpb_matrix = rpb

    x = torch.randn(1, 3, size, size, device="cuda").half()
    with torch.no_grad():
        model(pixel_values=x, input_ids=tok["input_ids"].cuda(), attention_mask=tok["attention_mask"].cuda())
    Sam3DetrDecoder._get_rpb_matrix = orig
    print(f"\n=== image_size = {size}  (size // 14 = {size // 14}, cells = {(size // 14) ** 2}) ===", flush=True)
    for name, s in log:
        print(f"  {name:48s} {s}", flush=True)
    for h in hs:
        h.remove()
    del model
    torch.cuda.empty_cache()
