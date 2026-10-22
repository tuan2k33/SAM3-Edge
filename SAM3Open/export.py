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

import torch
import torch.nn as nn

SAM3_DEPLOY_ROOT = os.path.dirname(os.path.abspath(__file__))
# _VisionEncoder/_Decoder live in the sibling SAM3Fixed/export.py, not here
# -- both architectures share the same ONNX-traceable submodules.
sys.path.insert(0, os.path.join(os.path.dirname(SAM3_DEPLOY_ROOT), "SAM3Fixed"))

from transformers.models.sam3 import Sam3Model, Sam3Processor
from export import _VisionEncoder, _Decoder

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
