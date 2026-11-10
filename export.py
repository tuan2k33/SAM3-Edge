#!/usr/bin/env python3
"""
SAM3 (facebook/sam3) -> ONNX -> TensorRT engine, one script for every mode. The imgsz != 1008 decoder-grid fix (below) is applied at import,
so it holds for all of them, and every engine outputs detections AND masks.

  --mode set-classes   classes baked into the graph (--classes a b c): pixel_values -> detections + masks
                         -> weight/sam3_{imgsz}_set-classes_{classes}/sam3.{onnx,engine} (+ config_infer.txt, labels.txt for nvinfer)
  --mode open-text     the prompt is a runtime input: text_encoder + vision_decoder engines, built once, no re-export to change classes
                         (K = --k-max class slots per image; unused slots have an all-False text_mask and score 0)
                         -> weight/sam3_{imgsz}_open-text/{text_encoder,vision_decoder}.{onnx,engine}
  --mode prompt-free   not implemented yet (planned: grid points through the SAM3 tracker, see CLAUDE.md)

Outputs (fp16, pixels of the S x S input): detections [B,K*200,6] = x1 y1 x2 y2 score cls_id (cls_id = class slot), masks [B,K*200,36,36] probabilities.
The row of query q of class slot k is k*200 + q.

Usage:
    python3 export.py --mode set-classes --classes truck car --imgsz 420
    python3 export.py --mode open-text --imgsz 420 --k-max 5
"""
import argparse
import ctypes
import gc
import math
import os
import shlex
import subprocess
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.sam3 import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3DetrDecoder, Sam3ViTRotaryEmbedding


ROOT = os.path.dirname(os.path.abspath(__file__))
Q = 200          # object queries per class slot
CTX_LEN = 32     # SAM3 text tokenizer context length
MASK_SIZE = 36   # masks are interpolated down to MASK_SIZE x MASK_SIZE


# ── Fixes for running at imgsz != 1008 ─────────────────────────────────────
# Applied at import, no-ops at H = 72. Evidence: test/BOX_SHRINK_INVESTIGATION.md
# Decoder grid ends at the trained 71/72 for any H. (A query-box scale was tried and is removed for now.)
_TRAIN_GRID = 72                 # H at the checkpoint's training size (imgsz 1008 / patch 14)


def _get_coords_denom_fix(self, height, width, dtype, device):
    coords_h = (torch.arange(0, height, device=device, dtype=torch.float32) / (height - 1) * (_TRAIN_GRID - 1) / _TRAIN_GRID).to(dtype)
    coords_w = (torch.arange(0, width, device=device, dtype=torch.float32) / (width - 1) * (_TRAIN_GRID - 1) / _TRAIN_GRID).to(dtype)
    return coords_h, coords_w


Sam3DetrDecoder._get_coords = _get_coords_denom_fix


# ── Sinusoidal pos enc ────────────────────────────────────────────────────────

def _sine_pos_enc(B, H, W, device, dtype, num_pos_feats=128, temp=10000):
    scale = 2 * math.pi
    y = torch.arange(1, H+1, dtype=dtype, device=device).view(1,H,1).expand(B,H,W)
    x = torch.arange(1, W+1, dtype=dtype, device=device).view(1,1,W).expand(B,H,W)
    y = y / (H + 1e-6) * scale
    x = x / (W + 1e-6) * scale
    dim_t = torch.arange(num_pos_feats, dtype=dtype, device=device)
    dim_t = temp ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = x[..., None] / dim_t
    pos_y = y[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


# ── Vision encoders ───────────────────────────────────────────────────────────

class _VisionEncoder(nn.Module):
    """ViT backbone + FPN neck.

    forward → (fpn_feat_2, fpn_pos_2)          bbox-only path
    Subclass _VisionEncoderFull additionally returns fpn0/fpn1 for the mask decoder.
    """

    def __init__(self, sam3: Sam3Model, device: str, imgsz: int):
        super().__init__()
        bb = sam3.vision_encoder.backbone
        self.patch_embed = bb.embeddings.patch_embeddings
        self.dropout     = bb.embeddings.dropout
        self.layer_norm  = bb.layer_norm
        self.layers      = bb.layers
        self.neck        = sam3.vision_encoder.neck

        ps = bb.config.patch_size
        Hp = Wp = imgsz // ps
        self.Hp, self.Wp = Hp, Wp
        D = bb.config.hidden_size

        for layer in self.layers:
            if getattr(layer, "window_size", 0) != 0:
                continue
            cfg = getattr(layer, "config", bb.config)
            scale = cfg.window_size / Hp
            if hasattr(layer, "position_ids"):
                hh, ww = torch.meshgrid(torch.arange(Hp), torch.arange(Wp), indexing="ij")
                pos = torch.stack([ww.flatten(), hh.flatten()], dim=-1) * scale
                layer.position_ids = pos.to(device)
            else:
                layer.rotary_emb = Sam3ViTRotaryEmbedding(
                    cfg, end_x=Hp, end_y=Wp, scale=scale
                ).to(device)

        orig = bb.embeddings.position_embeddings.data
        n0   = int(orig.shape[1] ** 0.5)
        pe   = orig.reshape(1, n0, n0, D).permute(0, 3, 1, 2)
        rh   = Hp // n0 + 1
        rw   = Wp // n0 + 1
        pe   = pe.tile([1, 1, rh, rw])[:, :, :Hp, :Wp]
        pe   = pe.permute(0, 2, 3, 1).reshape(1, Hp * Wp, D)
        self.register_buffer("vit_pos_embed", pe.to(device))

        npf  = sam3.vision_encoder.neck.config.fpn_hidden_size // 2
        pos2 = _sine_pos_enc(1, Hp, Wp, torch.device(device), torch.float32, npf)
        self.register_buffer("fpn_pos_2", pos2)

    def _encode(self, images: torch.Tensor):
        B   = images.shape[0]
        emb = self.patch_embed(images) + self.vit_pos_embed
        emb = self.dropout(emb)
        hs  = self.layer_norm(emb.view(B, self.Hp, self.Wp, -1))
        for layer in self.layers:
            hs = layer(hs)
        spatial = (hs.view(B, self.Hp * self.Wp, -1)
                     .view(B, self.Hp, self.Wp, -1)
                     .permute(0, 3, 1, 2))
        fpn, _ = self.neck(spatial)
        return fpn, B

    def forward(self, images: torch.Tensor):
        fpn, B = self._encode(images)
        return fpn[2], self.fpn_pos_2.expand(B, -1, -1, -1)


class _VisionEncoderFull(_VisionEncoder):
    """Also returns fpn0 (finest 288×288) and fpn1 (mid 144×144) for the mask decoder."""

    def forward(self, images: torch.Tensor):
        fpn, B = self._encode(images)
        # fpn[0]=finest 288×288, fpn[1]=mid 144×144, fpn[2]=coarse 72×72
        return fpn[2], self.fpn_pos_2.expand(B, -1, -1, -1), fpn[0], fpn[1]


# ── Decoders ──────────────────────────────────────────────────────────────────

class _Decoder(nn.Module):
    """DETR encoder + decoder + scoring.

    forward → (boxes, logits)                   bbox-only path
    Subclass _DecoderFull additionally returns dec_queries/enc_hidden for the mask head.
    """

    def __init__(self, sam3: Sam3Model):
        super().__init__()
        self.detr_encoder        = sam3.detr_encoder
        self.detr_decoder        = sam3.detr_decoder
        self.dot_product_scoring = sam3.dot_product_scoring
        self.box_head            = sam3.detr_decoder.box_head

    @staticmethod
    def _inv_sigmoid(x, eps=1e-3):
        x = x.clamp(min=0, max=1)
        return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))

    @staticmethod
    def _cxcywh_to_xyxy(x):
        cx, cy, w, h = x.unbind(-1)
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

    def _decode(self, fpn_feat_2, fpn_pos_2, text_features, text_mask):
        enc = self.detr_encoder(
            vision_features   = [fpn_feat_2],
            text_features     = text_features,
            vision_pos_embeds = [fpn_pos_2],
            text_mask         = text_mask,
        )
        dec = self.detr_decoder(
            vision_features     = enc.last_hidden_state,
            text_features       = enc.text_features,
            vision_pos_encoding = enc.pos_embeds_flattened,
            text_mask           = text_mask,
            spatial_shapes      = enc.spatial_shapes,
        )
        offsets    = self.box_head(dec.intermediate_hidden_states)
        ref_inv    = self._inv_sigmoid(dec.reference_boxes)
        all_boxes  = self._cxcywh_to_xyxy((ref_inv + offsets).sigmoid())
        all_logits = self.dot_product_scoring(
            decoder_hidden_states = dec.intermediate_hidden_states,
            text_features         = enc.text_features,
            text_mask             = text_mask,
        ).squeeze(-1)
        return (
            all_boxes[-1],                          # [B, Q, 4]  xyxy normalized [0,1]
            all_logits[-1],                         # [B, Q]     logits (pre-sigmoid)
            dec.intermediate_hidden_states[-1],     # [B, Q, D]  for mask coefficients
            enc.last_hidden_state,                  # [B, Hp*Wp+text, D] for prototypes
            dec.presence_logits[-1],                # [B, 1]  scene-level presence (last layer)
        )

    def forward(self, fpn_feat_2, fpn_pos_2, text_features, text_mask):
        boxes, logits, _, _, presence = self._decode(fpn_feat_2, fpn_pos_2, text_features, text_mask)
        return boxes, logits, presence


class _DecoderFull(_Decoder):
    """Also returns dec_queries and enc_hidden for the mask head."""

    def forward(self, fpn_feat_2, fpn_pos_2, text_features, text_mask):
        return self._decode(fpn_feat_2, fpn_pos_2, text_features, text_mask)


# ── Model ─────────────────────────────────────────────────────────────────────

class DynamicVisionDecoderWrapper(nn.Module):
    """pixel_values [B,3,S,S] + text_features [B,K,32,256] + text_mask [B,K,32]
    -> detections [B,K*200,6] (x1 y1 x2 y2 score cls_id = slot k, pixels) + masks [B,K*200,36,36].
    K = the number of class slots, a constant of the exported graph (a K read from the input shape becomes a dynamic Range node that TensorRT 11 cannot
    import); None reads it from the input, for eager use. The ViT runs once per image; its outputs are repeated K times so the K class slots go through the DETR
    decoder as K batch rows (row b*K+k)."""

    def __init__(self, sam3, device, imgsz, K=None):
        super().__init__()
        self.K = K
        self.vision_enc = _VisionEncoderFull(sam3, device, imgsz)
        self.decoder = _DecoderFull(sam3)
        self.mask_embedder = sam3.mask_decoder.mask_embedder
        self.pixel_decoder = sam3.mask_decoder.pixel_decoder
        self.instance_proj = sam3.mask_decoder.instance_projection
        self.Hp = self.Wp = imgsz // sam3.vision_encoder.backbone.config.patch_size
        self.register_buffer("scale", torch.tensor([imgsz, imgsz, imgsz, imgsz], dtype=torch.float32))
        self.register_buffer("first_token", torch.arange(CTX_LEN) == 0)

    def _prototypes(self, fpn0, fpn1, enc_hidden):
        B = fpn0.shape[0]
        enc_vis = enc_hidden[:, :self.Hp * self.Wp, :].transpose(1, 2).reshape(B, -1, self.Hp, self.Wp)
        return self.instance_proj(self.pixel_decoder([fpn0, fpn1, enc_vis]))  # [B, 32, 288, 288]

    def forward(self, pixel_values, text_features, text_mask):
        K = self.K or text_features.shape[1]
        rep = lambda t: t.unsqueeze(1).expand(-1, K, *t.shape[1:]).flatten(0, 1)
        fpn2, fpn_pos, fpn0, fpn1 = self.vision_enc(pixel_values)

        # An all-masked slot would softmax over nothing (NaN): give it one dummy visible token and zero its score afterwards.
        valid = text_mask.any(dim=-1)                                   # [B,K]
        safe_mask = text_mask | (self.first_token & ~valid.unsqueeze(-1))

        boxes, logits, dec_queries, enc_hidden, presence = self.decoder(
            rep(fpn2), rep(fpn_pos), text_features.flatten(0, 1), safe_mask.flatten(0, 1))
        protos = self._prototypes(rep(fpn0), rep(fpn1), enc_hidden)
        masks = torch.einsum("bqk,bkhw->bqhw", self.mask_embedder(dec_queries), protos).sigmoid()
        masks = F.interpolate(masks, (MASK_SIZE, MASK_SIZE), mode="bilinear", align_corners=False)
        masks = masks.unflatten(0, (-1, K)).flatten(1, 2)               # [B,K*Q,36,36], class-major

        # the grid fix can push edge boxes slightly past the frame
        boxes_px = (boxes.clamp(0.0, 1.0) * self.scale).unflatten(0, (-1, K))   # [B,K,Q,4]
        # final score = query score gated by the scene-level presence token; without it absent concepts leak as false positives
        scores = (logits.sigmoid() * presence.sigmoid()).unflatten(0, (-1, K)) * valid.unsqueeze(-1).float()
        cls_id = torch.arange(K, device=scores.device, dtype=scores.dtype).view(1, K, 1).expand_as(scores)
        det = torch.cat([boxes_px, scores.unsqueeze(-1), cls_id.unsqueeze(-1)], dim=-1)
        return det.flatten(1, 2), masks                                 # [B,K*Q,6], [B,K*Q,36,36]


class Sam3Wrapper(nn.Module):
    """set-classes: the text features of the N classes are ONNX constants, so the engine takes only the image: pixel_values -> detections, masks."""

    def __init__(self, sam3, text_embeds_all, attn_masks_all, imgsz):
        super().__init__()
        self.core = DynamicVisionDecoderWrapper(sam3, str(next(sam3.parameters()).device), imgsz, K=text_embeds_all.shape[0])
        self.register_buffer("text_feat_all", text_embeds_all)          # [N,32,256]
        self.register_buffer("text_mask_all", attn_masks_all > 0)       # [N,32]

    def forward(self, pixel_values):
        B = pixel_values.shape[0]
        return self.core(pixel_values, self.text_feat_all.unsqueeze(0).expand(B, -1, -1, -1), self.text_mask_all.unsqueeze(0).expand(B, -1, -1))


class TextEncoderWrapper(nn.Module):
    """Only holds text_encoder + text_projection (what get_text_features() actually calls) -- NOT a reference to the full sam3 model, otherwise
    ONNX export serializes the whole vision backbone as unused initializers (bloated the first attempt's text_encoder.onnx to 1.4GB)."""

    def __init__(self, sam3):
        super().__init__()
        self.text_encoder = sam3.text_encoder
        self.text_projection = sam3.text_projection

    def forward(self, input_ids, attention_mask):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return self.text_projection(text_outputs.last_hidden_state), attention_mask > 0


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


def generate_ds_config(classes, engine_path, Hm, Wm):
    config_dir = os.path.dirname(os.path.abspath(engine_path))
    os.makedirs(config_dir, exist_ok=True)

    nc = len(classes)
    N  = nc * Q

    labels_path = os.path.join(config_dir, "labels.txt")
    with open(labels_path, "w") as f:
        f.write("\n".join(classes) + "\n")

    lib_path = os.path.join(ROOT, "deepstream", "libnvdsinfer_sam3.so")  # build it, see CLAUDE.md
    class_attrs = "\n".join(
        f"[class-attrs-{i}]\npre-cluster-threshold=0.3" for i in range(nc)
    )

    output_blob_names = "detections;masks"
    blob_comment      = f"# detections [{N},6] fp16  masks [{N},{Hm},{Wm}] fp16"
    parser_line       = "parse-bbox-instance-mask-func-name=NvDsInferParseSAM3Full"

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
network-type=3
segmentation-threshold=0.5
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
    print(f"[Config] {config_dir}/config_infer.txt  ({nc} classes, N={N})")


def run_trtexec(onnx_path, engine_path, shapes, extra_args=()):
    """shapes: {input_name: (min, opt, max)}, each an 'AxBxC' string. TensorRT 11 is strongly typed: the precision is the ONNX dtype (see onnx_to_fp16)."""
    cmd = ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    for flag, idx in (("--minShapes", 0), ("--optShapes", 1), ("--maxShapes", 2)):
        cmd.append(f"{flag}=" + ",".join(f"{n}:{v[idx]}" for n, v in shapes.items()))
    cmd += extra_args
    print(f"\n[TRT] {' '.join(cmd)}\n", flush=True)
    t0 = time.perf_counter()
    if subprocess.run(cmd).returncode != 0:
        raise RuntimeError("trtexec failed")
    print(f"[TRT] {os.path.basename(engine_path)} built in {time.perf_counter() - t0:.0f}s", flush=True)


def export_onnx(module, dummy, path, input_names, output_names, dynamic_axes, opset):
    print(f"[Export] {path} ...", flush=True)
    with torch.no_grad():
        torch.onnx.export(module, dummy, path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes,
                          opset_version=opset, dynamo=False)
    print(f"[Export] Saved: {path}  ({os.path.getsize(path) / 1e6:.0f} MB)")


def free_memory():
    """Give the freed model back to the OS before trtexec: the 7 GB dev machine otherwise kills this process while it builds."""
    gc.collect()
    torch.cuda.empty_cache()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def load(args):
    token = os.environ.get("HF_TOKEN")
    t0 = time.perf_counter()
    print(f"[Load] {args.checkpoint} ...")
    model = Sam3Model.from_pretrained(args.checkpoint, token=token).to(args.device).eval()
    processor = Sam3Processor.from_pretrained(args.checkpoint, token=token)
    print(f"[Load] {time.perf_counter() - t0:.1f}s")
    if not validate_imgsz(model, args.imgsz, args.device):
        raise ValueError(f"imgsz={args.imgsz} not compatible (must be a multiple of patch_size)")
    return model, processor


# ── Modes ─────────────────────────────────────────────────────────────────────

def export_set_classes(args):
    assert args.classes, "--mode set-classes needs --classes"
    out_dir = args.output_dir or os.path.join(ROOT, "weight", f"sam3_{args.imgsz}_set-classes_" + "_".join(normalize_class(c) for c in args.classes))
    os.makedirs(out_dir, exist_ok=True)
    onnx_path, engine_path = os.path.join(out_dir, "sam3.onnx"), os.path.join(out_dir, "sam3.engine")
    model, processor = load(args)
    text_embeds, attn_masks = encode_concepts(model, processor, args.classes)

    wrapper = Sam3Wrapper(model, text_embeds, attn_masks, args.imgsz).to(args.device).eval()
    dummy = torch.randn(1, 3, args.imgsz, args.imgsz, device=args.device)
    with torch.no_grad():
        detections, masks = wrapper(dummy)
    nc, (Hm, Wm) = len(args.classes), masks.shape[2:]
    print(f"[Check] detections {tuple(detections.shape)} expect [1,{nc * Q},6]; masks {tuple(masks.shape)}; "
          f"scores [{detections[0, :, 4].min():.3f}, {detections[0, :, 4].max():.3f}]; masks [{masks.min():.3f}, {masks.max():.3f}]")
    export_onnx(wrapper, dummy, onnx_path, ["pixel_values"], ["detections", "masks"],
                {"pixel_values": {0: "batch"}, "detections": {0: "batch"}, "masks": {0: "batch"}}, args.opset)
    del wrapper, model, detections, masks
    free_memory()
    onnx_to_fp16(onnx_path, onnx_path)
    free_memory()

    generate_ds_config(args.classes, engine_path, Hm, Wm)  # before the build: this process can be OOM-killed while trtexec runs (the child still finishes)
    if not args.skip_trt:
        S, mn, op, mx = args.imgsz, args.min_batch, args.opt_batch, args.max_batch or 16
        run_trtexec(onnx_path, engine_path, {"pixel_values": (f"{mn}x3x{S}x{S}", f"{op}x3x{S}x{S}", f"{mx}x3x{S}x{S}")}, shlex.split(args.trt_args))


def export_open_text(args):
    out_dir = args.output_dir or os.path.join(ROOT, "weight", f"sam3_{args.imgsz}_open-text")
    os.makedirs(out_dir, exist_ok=True)
    model, processor = load(args)
    S, K, B = args.imgsz, args.k_max, args.max_batch or 4

    # text encoder
    text_wrapper = TextEncoderWrapper(model).to(args.device).eval()
    dummy_prompt = processor(text="dog", return_tensors="pt")
    ids, amask = dummy_prompt["input_ids"].to(args.device), dummy_prompt["attention_mask"].to(args.device)
    with torch.no_grad():
        text_features, text_mask = text_wrapper(ids, amask)
    print(f"[Check] text_features {tuple(text_features.shape)} text_mask {tuple(text_mask.shape)} (CTX_LEN={CTX_LEN})")
    text_onnx = os.path.join(out_dir, "text_encoder.onnx")
    export_onnx(text_wrapper, (ids, amask), text_onnx, ["input_ids", "attention_mask"], ["text_features", "text_mask"],
                {n: {0: "batch"} for n in ("input_ids", "attention_mask", "text_features", "text_mask")}, args.opset)

    # vision + decoder, paired by batch row
    vision_wrapper = DynamicVisionDecoderWrapper(model, args.device, S, K=K).to(args.device).eval()
    dummy_pixels = torch.randn(1, 3, S, S, device=args.device)
    text_features = text_features.unsqueeze(1).expand(-1, K, -1, -1).contiguous()
    text_mask = text_mask.unsqueeze(1).expand(-1, K, -1).contiguous()
    text_mask[:, K // 2:] = False  # exercise the unused-slot path
    with torch.no_grad():
        detections, masks = vision_wrapper(dummy_pixels, text_features, text_mask)
    assert torch.isfinite(detections).all() and torch.isfinite(masks).all(), "non-finite output (unused-slot handling broken)"
    print(f"[Check] detections {tuple(detections.shape)} masks {tuple(masks.shape)}")
    vision_onnx = os.path.join(out_dir, "vision_decoder.onnx")
    export_onnx(vision_wrapper, (dummy_pixels, text_features, text_mask), vision_onnx, ["pixel_values", "text_features", "text_mask"],
                ["detections", "masks"],
                {"pixel_values": {0: "batch"}, "text_features": {0: "batch", 1: "slots"}, "text_mask": {0: "batch", 1: "slots"},
                 "detections": {0: "batch", 1: "queries"}, "masks": {0: "batch", 1: "queries"}}, args.opset)
    del vision_wrapper, text_wrapper, model
    free_memory()
    for p in (text_onnx, vision_onnx):
        onnx_to_fp16(p, p)
        free_memory()

    if not args.skip_trt:
        extra = shlex.split(args.trt_args)
        run_trtexec(text_onnx, text_onnx.replace(".onnx", ".engine"),
                    {"input_ids": ("1x32", f"{B}x32", f"{B}x32"), "attention_mask": ("1x32", f"{B}x32", f"{B}x32")}, extra)
        run_trtexec(vision_onnx, vision_onnx.replace(".onnx", ".engine"),
                    {"pixel_values": (f"1x3x{S}x{S}", f"{B}x3x{S}x{S}", f"{B}x3x{S}x{S}"),
                     "text_features": (f"1x{K}x32x256", f"{B}x{K}x32x256", f"{B}x{K}x32x256"),
                     "text_mask": (f"1x{K}x32", f"{B}x{K}x32", f"{B}x{K}x32")}, extra)


def export_prompt_free(args):
    raise NotImplementedError("--mode prompt-free is planned (grid points through the SAM3 tracker, see CLAUDE.md); use set-classes or open-text")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["set-classes", "open-text", "prompt-free"])
    p.add_argument("--classes", nargs="+", help="set-classes: class names (kept as typed for the prompt; the engine folder name replaces spaces with '-')")
    p.add_argument("--checkpoint", default="facebook/sam3")
    p.add_argument("--imgsz", type=int, default=420, help="square, a multiple of patch_size (14)")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cpu: the FP32 model (3.4 GB) does not fit a 4 GB GPU")
    p.add_argument("--min-batch", type=int, default=1, help="set-classes")
    p.add_argument("--opt-batch", type=int, default=4, help="set-classes")
    p.add_argument("--max-batch", type=int, help="max batch = max cameras (default: 16 for set-classes, 4 for open-text)")
    p.add_argument("--k-max", type=int, default=5, help="open-text: class slots per camera baked into vision_decoder.engine; must match deepstream/sam3open_text.K_MAX")
    p.add_argument("--skip-trt", action="store_true", help="only write the ONNX files")
    p.add_argument("--trt-args", default="", help='extra trtexec flags, e.g. "--memPoolSize=workspace:1536 --builderOptimizationLevel=1" on low-RAM machines')
    p.add_argument("--output-dir", help="default: weight/sam3_{imgsz}_{mode}[_{classes}]/")
    return p.parse_args()


def main():
    args = parse_args()
    {"set-classes": export_set_classes, "open-text": export_open_text, "prompt-free": export_prompt_free}[args.mode](args)


if __name__ == "__main__":
    main()

