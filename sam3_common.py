"""The SAM3 modules both export scripts trace (SAM3Fixed/export.py, SAM3Open/export.py): the imgsz != 1008 decoder-grid fix (applied at import),
the ViT + FPN vision encoder and the DETR encoder/decoder, cut so they export to ONNX and run at any square imgsz that is a multiple of 14."""
import math

import torch
import torch.nn as nn

from transformers.models.sam3 import Sam3Model
from transformers.models.sam3.modeling_sam3 import Sam3DetrDecoder, Sam3ViTRotaryEmbedding


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
