"""
SAM3 Specialize — ONNX + TRT (bbox-only or instance segmentation)
------------------------------------------------------------------
Decomposes SAM3 into ONNX-traceable sub-modules for dynamic batch.

--mode det:
    output:  detections [B, N_cls*200, 6]          x1 y1 x2 y2 score cls_id (pixel space)
    DS blobs: detections
    parser:  NvDsInferParseSAM3Det

--mode seg (default):
    output:  detections [B, N_cls*200, 6]          x1 y1 x2 y2 score cls_id (pixel space)
             masks      [B, N_cls*200, Hm, Wm]     probability masks [0, 1]
    DS blobs: detections;masks
    parser:  NvDsInferParseSAM3Full

Output dir: ../weight/sam3_{imgsz}_{mode}_{classes}/  (e.g. weight/sam3_1008_seg_adult_child_phone/)

Usage:
    CUDA_VISIBLE_DEVICES=1 python export.py --classes adult child phone --mode det --device cuda
    CUDA_VISIBLE_DEVICES=1 python export.py --classes adult child phone --mode seg --device cuda
"""

import argparse
import gc
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.sam3 import Sam3Model, Sam3Processor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# the shared modules + the imgsz fix live in sam3_common.py at the repo root (applied at import)
from sam3_common import _TRAIN_GRID, _get_coords_denom_fix, _sine_pos_enc, _VisionEncoder, _VisionEncoderFull, _Decoder, _DecoderFull  # noqa: F401


# ── Unified wrapper ───────────────────────────────────────────────────────────

class Sam3Wrapper(nn.Module):
    """
    Single-input wrapper. Text features for each class are pre-baked as ONNX constants.

    with_mask=False (default):
        pixel_values [B,3,H,W]  →  detections [B, N_cls*Q, 6]
        columns: x1 y1 x2 y2 score cls_id  (pixel space, score=sigmoid(logit))

    with_mask=True:
        pixel_values [B,3,H,W]  →  detections [B, N_cls*Q, 6]
                                    masks      [B, N_cls*Q, Hm, Wm]
        cls_id for query i: i // Q  (Q=200)
    """

    def __init__(self, sam3: Sam3Model, text_embeds_all: torch.Tensor,
                 attn_masks_all: torch.Tensor, imgsz: int = 1008, with_mask: bool = False):
        super().__init__()
        device = str(next(sam3.parameters()).device)
        self.with_mask   = with_mask
        self.num_classes = text_embeds_all.shape[0]

        if with_mask:
            self.vision_enc    = _VisionEncoderFull(sam3, device, imgsz)
            self.decoder       = _DecoderFull(sam3)
            self.mask_embedder = sam3.mask_decoder.mask_embedder
            self.pixel_decoder = sam3.mask_decoder.pixel_decoder
            self.instance_proj = sam3.mask_decoder.instance_projection
            ps      = sam3.vision_encoder.backbone.config.patch_size
            self.Hp = imgsz // ps
            self.Wp = imgsz // ps
        else:
            self.vision_enc = _VisionEncoder(sam3, device, imgsz)
            self.decoder    = _Decoder(sam3)

        self.register_buffer("text_feat_all", text_embeds_all)
        self.register_buffer("text_mask_all", attn_masks_all > 0)

        self.register_buffer(
            "scale",
            torch.tensor([imgsz, imgsz, imgsz, imgsz], dtype=torch.float32),
        )

    def _prototypes(self, fpn0, fpn1, enc_hidden):
        B = fpn0.shape[0]
        enc_vis = (
            enc_hidden[:, :self.Hp * self.Wp, :]
            .transpose(1, 2)
            .reshape(B, -1, self.Hp, self.Wp)
        )
        pixel_embed = self.pixel_decoder([fpn0, fpn1, enc_vis])
        return self.instance_proj(pixel_embed)  # [B, K, 288, 288]

    def forward(self, pixel_values: torch.Tensor):
        B = pixel_values.shape[0]

        if self.with_mask:
            fpn2, fpn_pos, fpn0, fpn1 = self.vision_enc(pixel_values)
        else:
            fpn2, fpn_pos = self.vision_enc(pixel_values)

        N = self.num_classes

        # classes ride the batch axis: row b*N+i = image b x class i
        fpn2    = fpn2.repeat_interleave(N, dim=0)
        fpn_pos = fpn_pos.repeat_interleave(N, dim=0)
        text_feat = self.text_feat_all.repeat(B, 1, 1)
        text_mask = self.text_mask_all.repeat(B, 1)

        if self.with_mask:
            fpn0 = fpn0.repeat_interleave(N, dim=0)
            fpn1 = fpn1.repeat_interleave(N, dim=0)
            pred_boxes, pred_logits, dec_queries, enc_hidden, presence = self.decoder(
                fpn2, fpn_pos, text_feat, text_mask
            )
            protos = self._prototypes(fpn0, fpn1, enc_hidden)
            coeffs = self.mask_embedder(dec_queries)
            masks  = torch.einsum("bqk,bkhw->bqhw", coeffs, protos).sigmoid()
            masks  = F.interpolate(masks, (36, 36), mode="bilinear", align_corners=False)
            Q = masks.shape[1]
            masks = masks.reshape(B, N * Q, *masks.shape[2:])  # class-major: query i -> class i // Q
        else:
            pred_boxes, pred_logits, presence = self.decoder(fpn2, fpn_pos, text_feat, text_mask)

        # the grid fix can push edge boxes slightly past the frame
        pred_boxes = pred_boxes.clamp(0.0, 1.0)

        boxes_px = pred_boxes * self.scale
        # final score = query score gated by the scene-level presence token; without it absent concepts leak as false positives
        scores = (pred_logits.sigmoid() * presence.sigmoid()).unsqueeze(-1)
        Q = pred_boxes.shape[1]
        cls_id = (
            torch.arange(N, dtype=pred_boxes.dtype, device=pred_boxes.device)
            .repeat(B).view(B * N, 1, 1).expand(-1, Q, 1)
        )
        detections = torch.cat([boxes_px, scores, cls_id], dim=-1)
        detections = detections.reshape(B, N * Q, 6)  # [B, N_cls*Q, 6]

        if self.with_mask:
            return detections, masks  # [B, N_cls*Q, Hm, Wm] fp16
        return detections


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_imgsz(model, imgsz, device="cpu"):
    """Check the actual condition Sam3Wrapper/_VisionEncoder needs (does NOT
    call the original HuggingFace model.get_vision_features() -- that always
    fails for imgsz != 1008 because it has no positional-embedding resize
    logic, whereas Sam3Wrapper below implements its OWN resize logic
    (tile+crop) and therefore supports imgsz values other than 1008, as long
    as two conditions hold: it must be a multiple of patch_size, and the
    image must be square (H=W, since the wrapper only takes one imgsz value)."""
    print(f"[Validate] imgsz={imgsz} ...")
    ps = model.vision_encoder.backbone.config.patch_size
    if imgsz % ps != 0:
        print(f"[Validate] FAILED: imgsz={imgsz} is not divisible by patch_size={ps}")
        return False
    print(f"[Validate] OK — imgsz={imgsz} is a multiple of patch_size={ps} "
          f"(Hp=Wp={imgsz // ps}), square image")
    return True


def normalize_class(cls):
    """Only used for the engine folder name (paths can't contain spaces):
    'blue bus' and 'blue-bus' both map to 'blue-bus'. The prompt sent to the
    model and labels.txt keep the original text as typed (raw arg, may
    contain spaces)."""
    return "-".join(cls.strip().split())


def encode_concepts(model, processor, classes):
    print(f"[Encode] {len(classes)} class(es): {classes}")
    embeds, masks = [], []
    dev = next(model.parameters()).device
    for cls in classes:
        inp = processor(text=cls, return_tensors="pt")
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.no_grad():
            out = model.get_text_features(
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
            )
        embeds.append(out.pooler_output)
        masks.append(inp["attention_mask"])
        print(f"  [+] '{cls}'  {tuple(out.pooler_output.shape)}")
    return torch.cat(embeds, dim=0), torch.cat(masks, dim=0)


def onnx_to_fp16(fp32_path, fp16_path):
    """TensorRT 11 is strongly typed: the engine precision is the ONNX dtype, so convert the graph, input and output included
    (fp16 pixel_values in, fp16 detections out). The converter runs without shape inference (it needs ~7 GB here)
    and then leaves Cast(to=fp32) nodes inside the graph, patched below."""
    import onnx
    from onnx import TensorProto
    from onnxconverter_common import float16
    t0 = time.perf_counter()
    m = onnx.load(fp32_path)
    m = float16.convert_float_to_float16(m, keep_io_types=False, disable_shape_infer=True)
    for nd in m.graph.node:
        if nd.op_type == "Cast":
            for a in nd.attribute:
                if a.name == "to" and a.i == TensorProto.FLOAT:
                    a.i = TensorProto.FLOAT16
    onnx.save(m, fp16_path)
    print(f"[FP16] {fp16_path}  ({os.path.getsize(fp16_path)/1e6:.0f} MB, {time.perf_counter()-t0:.0f}s)")


def run_trtexec(onnx_path, engine_path, imgsz, min_batch, opt_batch, max_batch):
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes=pixel_values:{min_batch}x3x{imgsz}x{imgsz}",
        f"--optShapes=pixel_values:{opt_batch}x3x{imgsz}x{imgsz}",
        f"--maxShapes=pixel_values:{max_batch}x3x{imgsz}x{imgsz}",
    ]
    print(f"\n[TRT] {' '.join(cmd)}\n")
    t0 = time.perf_counter()
    r  = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"trtexec failed (exit {r.returncode})")
    print(f"\n[TRT] Saved: {engine_path}  ({(time.perf_counter()-t0)/60:.1f} min)")


def generate_ds_config(classes, engine_path, Q=200, with_mask=False, Hm=288, Wm=288):
    config_dir = os.path.dirname(os.path.abspath(engine_path))
    os.makedirs(config_dir, exist_ok=True)

    nc = len(classes)
    N  = nc * Q

    labels_path = os.path.join(config_dir, "labels.txt")
    with open(labels_path, "w") as f:
        f.write("\n".join(classes) + "\n")

    # the custom nvinfer plugin is shared by both architectures: deepstream/ at the repo root (build it, see CLAUDE.md)
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "deepstream", "libnvdsinfer_sam3.so")
    class_attrs = "\n".join(
        f"[class-attrs-{i}]\npre-cluster-threshold=0.3" for i in range(nc)
    )

    if with_mask:
        output_blob_names = "detections;masks"
        blob_comment      = f"# detections [{N},6] fp16  masks [{N},{Hm},{Wm}] fp16"
        parser_line       = f"parse-bbox-instance-mask-func-name=NvDsInferParseSAM3Full"
        network_type      = 3
    else:
        output_blob_names = "detections"
        blob_comment      = f"# detections [{N},6] fp16  x1 y1 x2 y2 score cls_id"
        parser_line       = f"parse-bbox-func-name=NvDsInferParseSAM3Det"
        network_type      = 0

    config_path = os.path.join(config_dir, "config_infer.txt")
    with open(config_path, "w") as f:
        f.write(f"""\
[property]
gpu-id=0

model-engine-file={os.path.abspath(engine_path)}
labelfile-path={labels_path}
num-detected-classes={nc}
batch-size=4

net-scale-factor=0.00784313725490196
offsets=127.5;127.5;127.5
model-color-format=0

network-mode=1
process-mode=1
network-type={network_type}
{"segmentation-threshold=0.5" if with_mask else ""}
cluster-mode=4
gie-unique-id=3

{blob_comment}
output-blob-names={output_blob_names}

{parser_line}
custom-lib-path={lib_path}

[class-attrs-all]
topk={N}

{class_attrs}
""")
    print(f"[Config] {config_dir}/config_infer.txt  ({nc} classes, N={N}, mask={with_mask})")


# ── Main ──────────────────────────────────────────────────────────────────────

def specialize_and_export(
    checkpoint, classes, output_path,
    imgsz=1008, opset=17, device="cpu",
    min_batch=1, opt_batch=4, max_batch=16,
    skip_trt=False, with_mask=False,
):
    token = os.environ.get("HF_TOKEN")
    print(f"[Load] {checkpoint} ...")
    t0        = time.perf_counter()
    model     = Sam3Model.from_pretrained(checkpoint, token=token).to(device).eval()
    processor = Sam3Processor.from_pretrained(checkpoint, token=token)
    print(f"[Load] {time.perf_counter()-t0:.1f}s")

    if not validate_imgsz(model, imgsz, device):
        raise ValueError(f"imgsz={imgsz} not compatible (must be a multiple of patch_size)")

    text_embeds, attn_masks = encode_concepts(model, processor, classes)

    print(f"[Build] Sam3Wrapper (with_mask={with_mask}) ...")
    wrapper = Sam3Wrapper(
        model, text_embeds, attn_masks, imgsz, with_mask=with_mask,
    ).to(device).eval()

    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    print("[Check] Sanity forward pass ...")
    with torch.no_grad():
        out = wrapper(dummy)

    nc = len(classes)
    if with_mask:
        detections, masks = out
        Q       = detections.shape[1] // nc
        Hm, Wm  = masks.shape[2], masks.shape[3]
        print(f"[Check] detections {tuple(detections.shape)}  expect [1,{nc*Q},6]")
        print(f"[Check] masks      {tuple(masks.shape)}  expect [1,{nc*Q},{Hm},{Wm}]")
        print(f"[Check] boxes  [{detections[0,:,0].min():.1f}, {detections[0,:,2].max():.1f}]")
        print(f"[Check] scores [{detections[0,:,4].min():.3f}, {detections[0,:,4].max():.3f}]")
        print(f"[Check] masks  [{masks.min():.3f}, {masks.max():.3f}]")
        output_names = ["detections", "masks"]
        dynamic_axes = {
            "pixel_values": {0: "batch"},
            "detections":   {0: "batch"},
            "masks":        {0: "batch"},
        }
    else:
        detections = out
        Q       = detections.shape[1] // nc
        Hm, Wm  = 0, 0
        print(f"[Check] detections {tuple(detections.shape)}  expect [1,{nc*Q},6]")
        print(f"[Check] boxes  [{detections[0,:,0].min():.1f}, {detections[0,:,2].max():.1f}]")
        print(f"[Check] scores [{detections[0,:,4].min():.3f}, {detections[0,:,4].max():.3f}]")
        output_names = ["detections"]
        dynamic_axes = {
            "pixel_values": {0: "batch"},
            "detections":   {0: "batch"},
        }

    print(f"[Export] {output_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, output_path,
            input_names  = ["pixel_values"],
            output_names = output_names,
            dynamic_axes = dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"[Export] Saved: {output_path}  ({size_mb:.0f} MB)")
    del wrapper, model, out
    gc.collect()
    torch.cuda.empty_cache()
    onnx_to_fp16(output_path, output_path)

    engine_path = output_path.replace(".onnx", ".engine")
    if not skip_trt:
        run_trtexec(output_path, engine_path, imgsz, min_batch, opt_batch, max_batch)

    generate_ds_config(classes, engine_path, Q=Q, with_mask=with_mask, Hm=Hm, Wm=Wm)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="facebook/sam3")
    p.add_argument("--classes",    nargs="+", required=True,
                    help="class names (kept as-is for the prompt); only the "
                         "engine folder name replaces spaces with '-' to avoid path issues")
    p.add_argument("--imgsz", type=int, default=1008,
                    help="must be a multiple of patch_size (14); square image (H=W)")
    p.add_argument("--opset",      type=int, default=17)
    p.add_argument("--device",     default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--min-batch",  type=int, default=1)
    p.add_argument("--opt-batch",  type=int, default=4)
    p.add_argument("--max-batch",  type=int, default=16)
    p.add_argument("--skip-trt",   action="store_true")
    p.add_argument("--mode",       default="seg", choices=["det", "seg"],
                   help="det=bbox only  seg=det+masks[B,N,36,36] fp16 (default)")
    return p.parse_args()


def main():
    args        = parse_args()
    suffix      = "_".join(normalize_class(c) for c in args.classes)
    weight_dir  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weight")
    config_dir  = os.path.join(weight_dir, f"sam3_{args.imgsz}_{args.mode}_{suffix}")
    os.makedirs(config_dir, exist_ok=True)
    output_path = os.path.join(config_dir, "sam3.onnx")
    specialize_and_export(
        checkpoint=args.checkpoint,
        classes=args.classes,
        output_path=output_path,
        imgsz=args.imgsz,
        opset=args.opset,
        device=args.device,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        skip_trt=args.skip_trt,
        with_mask=(args.mode == "seg"),
    )


if __name__ == "__main__":
    main()
