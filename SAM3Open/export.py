#!/usr/bin/env python3
"""
Dynamic-prompt ONNX export for the ORIGINAL SAM3 (facebook/sam3, 840M params),
mirroring efficientsam3's export_onnx_open_vocab_paired.py: prompts are
encoded and passed in at RUNTIME (not baked as buffers like Sam3Wrapper in
SAM3Fixed/export.py), so no re-export is needed to change classes.

Two graphs:
  1. text_encoder.onnx  : input_ids [B,32] + attention_mask [B,32]
                              -> text_features [B,32,256], text_mask [B,32]
  2. vision_decoder.onnx: pixel_values [B,3,H,W] + text_features [B,K,32,256]
                              + text_mask [B,K,32]  -> detections [B,K*200,6]
                              (K = --k-max class slots per image; ViT runs once
                              per image, the K slots ride the batch axis through
                              the DETR decoder. Unused slots: text_mask all False
                              -> score forced to 0. cls_id = slot index.)

Unlike efficientsam3, SAM3's detr_encoder/detr_decoder consume text_features
per BATCH ROW directly (no FindStage.text_ids indexing) -- so image[i] slot k is
automatically paired with text_features[i,k].

Usage:
    python3 export.py --imgsz 1008   # -> ../weight/sam3_1008_open/{text,vision}.{onnx,engine}
    python3 export.py --k-max 4      # engine with 4 class slots per camera (default 5)

The vision engine is built with K fixed (min=opt=max=--k-max): nvinfer always
allocates the K of the profile's opt shape, so a dynamic K axis does not make it
follow the number of classes actually in use. Decoder cost is paid for all K
slots (measured ~+20 ms per slot on RTX 3050 Ti, ViT/encoder ~260 ms per image).
"""
import os
import shlex
import subprocess
import sys
import time
import argparse

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SAM3_DEPLOY_ROOT = os.path.dirname(os.path.abspath(__file__))

from transformers.models.sam3 import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3DetrDecoder, Sam3SinePositionEmbedding, Sam3ViTRotaryEmbedding


# ── Self-contained: the fixes and the two modules below are a copy of SAM3Fixed/export.py (keep in sync) ──

# ── Fixes for running at imgsz != 1008 ─────────────────────────────────────
#
# The checkpoint was trained at one size only (imgsz 1008 = 72x72 cells of 14 px). At any other
# size the boxes come out smaller (imgsz < 1008) or slightly larger (imgsz > 1008) than the
# ground truth, by an amount that varies smoothly with H = imgsz/14: the model's own decoder is
# where that happens (SAM3_test/BOX_SHRINK_INVESTIGATION.md has the full evidence). Two patches,
# both no-ops at H = 72 and both applied when this module is imported (before any model is built):
#
# 1) Grid (centre of the boxes). Sam3DetrDecoder._get_coords builds arange(0,H)/H, which ends at
#    (H-1)/H; the box relative-position bias (every decoder layer) was trained on a grid ending
#    at 71/72. We stretch it to always end there: arange/(H-1) * 71/72. This re-centres the boxes
#    (centre error ~0.2 -> ~0.05 patch) but barely changes their size.
#
# 2) Query box scale (size of the boxes). The decoder turns each query's box (cx, cy, w, h) into a
#    sine embedding (Sam3SinePositionEmbedding.encode_boxes -> ref_point_head -> query_pos).
#    Multiplying the box by (H/72)**a before that encoding grows the predicted boxes about their
#    centre and flattens the width/height error across imgsz (width slope per 10 cells of H:
#    +2.0% -> ~0, mean |error| 5.8% -> 1.9% over imgsz 336-1260). a = 1 means "express the box in
#    the 72-grid frame, where one cell has the trained size"; there is no theory behind a > 1 and
#    the gain is small, so a = 1.
#
# Sam3Wrapper.forward() still clamps boxes to [0,1] as a safety net.
_TRAIN_GRID = 72                 # H at the checkpoint's training size (imgsz 1008 / patch 14)
_QUERY_BOX_SCALE_EXP = 1.0       # a in (H/_TRAIN_GRID)**a
_APPLY_QUERY_BOX_SCALE = True    # experiments switch it off to start from the unpatched decoder


def _get_coords_denom_fix(self, height, width, dtype, device):
    # computed in fp32, cast once: same formula, no fp16 rounding noise in the grid
    coords_h = (torch.arange(0, height, device=device, dtype=torch.float32) / (height - 1) * (_TRAIN_GRID - 1) / _TRAIN_GRID).to(dtype)
    coords_w = (torch.arange(0, width, device=device, dtype=torch.float32) / (width - 1) * (_TRAIN_GRID - 1) / _TRAIN_GRID).to(dtype)
    return coords_h, coords_w


Sam3DetrDecoder._get_coords = _get_coords_denom_fix

# Always wrap the ORIGINAL encode_boxes (kept on the class), so importing both SAM3Fixed/export.py
# and SAM3Open/export.py in one process does not apply the scale twice.
if not hasattr(Sam3SinePositionEmbedding, "_orig_encode_boxes"):
    Sam3SinePositionEmbedding._orig_encode_boxes = Sam3SinePositionEmbedding.encode_boxes


def _encode_boxes_scaled(self, boxes):
    return Sam3SinePositionEmbedding._orig_encode_boxes(self, boxes * getattr(self, "box_scale", 1.0))


Sam3SinePositionEmbedding.encode_boxes = _encode_boxes_scaled


def _query_box_scale(H):
    return (H / _TRAIN_GRID) ** _QUERY_BOX_SCALE_EXP


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
        if _APPLY_QUERY_BOX_SCALE:
            self.detr_decoder.position_encoding.box_scale = _query_box_scale(int(fpn_feat_2.shape[-1]))
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



IMGSZ_DEFAULT = 1008
K_MAX_DEFAULT = 5  # class slots per camera; sam3open_text.K_MAX is the upper bound of the snapshot
CTX_LEN = 32  # SAM3 text tokenizer context length (see encode_concepts() output shape)


class TextEncoderWrapper(nn.Module):
    """Only holds text_encoder + text_projection (what get_text_features()
    actually calls) -- NOT a reference to the full sam3 model, otherwise
    ONNX export serializes the whole vision backbone as unused initializers
    (bloated the first attempt's text_encoder.onnx to 1.4GB)."""

    def __init__(self, sam3):
        super().__init__()
        self.text_encoder = sam3.text_encoder
        self.text_projection = sam3.text_projection

    def forward(self, input_ids, attention_mask):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooler_output = self.text_projection(text_outputs.last_hidden_state)
        text_mask = attention_mask > 0
        return pooler_output, text_mask


class DynamicVisionDecoderWrapper(nn.Module):
    """pixel_values [B,3,H,W] + text_features [B,K,32,256] + text_mask [B,K,32]
    -> detections [B,K*200,6] (x1 y1 x2 y2 score cls_id, cls_id = slot k).
    K is read from the input shape, so it can be a fixed or a dynamic axis of
    the ONNX/TRT profile. The ViT runs once per image; its output is repeated K
    times so the K class slots go through the DETR decoder as K batch rows
    (row b*K+k)."""

    def __init__(self, sam3, device, imgsz=IMGSZ_DEFAULT):
        super().__init__()
        self.vision_enc = _VisionEncoder(sam3, device, imgsz)
        self.decoder = _Decoder(sam3)
        self.register_buffer('scale', torch.tensor([imgsz, imgsz, imgsz, imgsz], dtype=torch.float32))
        self.register_buffer('first_token', torch.arange(CTX_LEN) == 0)

    def forward(self, pixel_values, text_features, text_mask):
        K = text_features.shape[1]
        fpn_feat_2, fpn_pos_2 = self.vision_enc(pixel_values)
        fpn_feat_2 = fpn_feat_2.unsqueeze(1).expand(-1, K, -1, -1, -1).flatten(0, 1)
        fpn_pos_2 = fpn_pos_2.unsqueeze(1).expand(-1, K, -1, -1, -1).flatten(0, 1)

        # An all-masked slot would softmax over nothing (NaN) -- give it one
        # dummy visible token and zero its score afterwards.
        valid = text_mask.any(dim=-1)                                   # [B,K]
        safe_mask = text_mask | (self.first_token & ~valid.unsqueeze(-1))

        boxes, logits, presence = self.decoder(
            fpn_feat_2, fpn_pos_2,
            text_features.flatten(0, 1),
            safe_mask.flatten(0, 1),
        )
        Q = boxes.shape[1]
        boxes_px = (boxes * self.scale).unflatten(0, (-1, K))           # [B,K,Q,4]
        scores = (logits.sigmoid() * presence.sigmoid()).unflatten(0, (-1, K))
        scores = scores * valid.unsqueeze(-1).float()
        cls_id = torch.arange(K, device=scores.device, dtype=scores.dtype).view(1, K, 1).expand_as(scores)
        det = torch.cat([boxes_px, scores.unsqueeze(-1), cls_id.unsqueeze(-1)], dim=-1)
        return det.flatten(1, 2)                                        # [B,K*Q,6]


def run_trtexec(onnx_path, engine_path, shapes, extra_args):
    """shapes: {input_name: (min, opt, max)} where each is an 'AxBxC' string."""
    cmd = ['trtexec', f'--onnx={onnx_path}', f'--saveEngine={engine_path}', '--fp16']
    for flag, idx in (('--minShapes', 0), ('--optShapes', 1), ('--maxShapes', 2)):
        cmd.append(f"{flag}=" + ','.join(f'{n}:{v[idx]}' for n, v in shapes.items()))
    cmd += extra_args
    print(f'\n[TRT] {" ".join(cmd)}\n', flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f'trtexec failed (exit {r.returncode})')
    print(f'[TRT] {os.path.basename(engine_path)} built in {time.perf_counter() - t0:.0f}s')


def build_engines(args):
    extra = shlex.split(args.trt_args)
    d, B, K, S = args.output_dir, args.max_batch, args.k_max, args.imgsz
    run_trtexec(
        os.path.join(d, 'text_encoder.onnx'), os.path.join(d, 'text_encoder.engine'),
        {'input_ids': ('1x32', f'{B}x32', f'{B}x32'),
         'attention_mask': ('1x32', f'{B}x32', f'{B}x32')}, extra)
    run_trtexec(
        os.path.join(d, 'vision_decoder.onnx'), os.path.join(d, 'vision_decoder.engine'),
        {'pixel_values': (f'1x3x{S}x{S}', f'{B}x3x{S}x{S}', f'{B}x3x{S}x{S}'),
         'text_features': (f'1x{K}x32x256', f'{B}x{K}x32x256', f'{B}x{K}x32x256'),
         'text_mask': (f'1x{K}x32', f'{B}x{K}x32', f'{B}x{K}x32')}, extra)


def export(args):
    device = args.device
    t0 = time.perf_counter()
    print('[Load] facebook/sam3 ...')
    model = Sam3Model.from_pretrained('facebook/sam3').to(device).eval()
    processor = Sam3Processor.from_pretrained('facebook/sam3')
    print(f'[Load] {time.perf_counter() - t0:.1f}s')
    os.makedirs(args.output_dir, exist_ok=True)

    # --- text encoder ---
    text_wrapper = TextEncoderWrapper(model).to(device).eval()
    dummy_prompt = processor(text="dog", return_tensors="pt")
    dummy_ids = dummy_prompt["input_ids"].to(device)
    dummy_mask = dummy_prompt["attention_mask"].to(device)
    print(f'[Info] tokenized shape: {tuple(dummy_ids.shape)} (CTX_LEN={CTX_LEN})')

    with torch.no_grad():
        text_features, text_mask = text_wrapper(dummy_ids, dummy_mask)
    print(f'[Check] text_features {tuple(text_features.shape)} text_mask {tuple(text_mask.shape)}')

    text_onnx_path = os.path.join(args.output_dir, 'text_encoder.onnx')
    print(f'[Export] {text_onnx_path} ...')
    with torch.no_grad():
        torch.onnx.export(
            text_wrapper, (dummy_ids, dummy_mask), text_onnx_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['text_features', 'text_mask'],
            dynamic_axes={'input_ids': {0: 'batch'}, 'attention_mask': {0: 'batch'},
                           'text_features': {0: 'batch'}, 'text_mask': {0: 'batch'}},
            opset_version=args.opset, dynamo=False,
        )
    print(f'[Export] Saved: {text_onnx_path}')

    # --- vision + decoder (paired by batch row) ---
    vision_wrapper = DynamicVisionDecoderWrapper(model, device, imgsz=args.imgsz).to(device).eval()
    dummy_pixels = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

    K = args.k_max
    text_features = text_features.unsqueeze(1).expand(-1, K, -1, -1).contiguous()
    text_mask = text_mask.unsqueeze(1).expand(-1, K, -1).contiguous()
    text_mask[:, K // 2:] = False  # exercise the unused-slot path

    print('[Check] Vision+decoder sanity forward pass ...')
    with torch.no_grad():
        detections = vision_wrapper(dummy_pixels, text_features, text_mask)
    assert torch.isfinite(detections).all(), 'non-finite detections (unused-slot handling broken)'
    print(f'[Check] detections {tuple(detections.shape)}')

    vision_onnx_path = os.path.join(args.output_dir, 'vision_decoder.onnx')
    print(f'[Export] {vision_onnx_path} ...')
    with torch.no_grad():
        torch.onnx.export(
            vision_wrapper, (dummy_pixels, text_features, text_mask), vision_onnx_path,
            input_names=['pixel_values', 'text_features', 'text_mask'],
            output_names=['detections'],
            dynamic_axes={'pixel_values': {0: 'batch'},
                           'text_features': {0: 'batch', 1: 'slots'},
                           'text_mask': {0: 'batch', 1: 'slots'},
                           'detections': {0: 'batch', 1: 'queries'}},
            opset_version=args.opset, dynamo=False,
        )
    print(f'[Export] Saved: {vision_onnx_path}')

    if not args.skip_trt:
        del vision_wrapper, text_wrapper, model
        torch.cuda.empty_cache()
        build_engines(args)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--imgsz', type=int, default=IMGSZ_DEFAULT)
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--k-max', type=int, default=K_MAX_DEFAULT,
                    help='class slots per camera baked into vision_decoder.engine; must be <= and ideally equal to sam3open_text.K_MAX (default %(default)s)')
    p.add_argument('--max-batch', type=int, default=4,
                    help='engine max batch = max cameras (sam3open_text.MAX_SOURCES, default %(default)s)')
    p.add_argument('--skip-trt', action='store_true', help='only write the ONNX files')
    p.add_argument('--trt-args', default='',
                    help='extra trtexec flags, e.g. "--memPoolSize=workspace:1536 --builderOptimizationLevel=1" on low-RAM machines')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output-dir', default=None,
                    help='default: ../weight/sam3_{imgsz}_open/ (repo-root weight/ folder, shared with SAM3Fixed)')
    args = p.parse_args()
    if args.output_dir is None:
        weight_dir = os.path.join(os.path.dirname(SAM3_DEPLOY_ROOT), 'weight')
        args.output_dir = os.path.join(weight_dir, f'sam3_{args.imgsz}_open')
    return args


if __name__ == '__main__':
    export(parse_args())
