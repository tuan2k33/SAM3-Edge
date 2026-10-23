"""Collect (ground-truth box, predicted box) pairs on coco128 for a given imgsz and decoder-grid variant.

Raw pairs are saved (normalised to the original image) so every analysis can be redone offline on the CPU
(see analyze_box_pairs.py). One JSON per (size, variant, nimg), written as soon as it is finished; existing files
are skipped, so a run can be stopped and resumed.

Setup: prompts are the COCO class names that have a GT box with area >= 2% of the image (at most 4 per image,
"oracle" prompts: this measures box geometry, not false positives). Prediction: score >= 0.5, NMS 0.6, boxes
clipped to the image, a true fp16 model. A GT box is paired with the best-IoU unused prediction of its class if IoU >= 0.3.

--sizes  items are "420" (network input 420x420) or "420in1008" (the image is resized to 420 and pasted into the
         top-left of a grey 1008 canvas, then run at 1008: content resolution 420, grid = the trained 72x72).
--variants
  orig        original grid arange(0,H)/H
  <number>    grid arange(0,H)/(H-1) * number   (a fraction like 71/72 is accepted; 1.0 = grid ending at 1.0)
  cell        original grid, deltas scaled by H/72 before the 8x + log2 encoding (distances in units of cells)
  prod        the fixes as shipped in SAM3Fixed/export.py / SAM3Open/export.py (grid 71/72 + query box scale (H/72)**a)
  <enc>+<grid>  a decoder patch plus a grid variant, e.g. sineq1.2+71/72 (enc part: sineq<a>, grid part: orig or a cap)
  ropecell    global-layer RoPE positions scaled by Hp/72 (1 cell = 1/3 RoPE unit at every size, as in training)
  peinterp    absolute position embedding = the trained 72x72 (tiled) map resampled bilinearly to Hp x Hp
  sinekey     decoder key sine positions in the trained 72-frame: (i+1)/72 * 2pi for the first H cells (instead of (i+1)/H)
  sineq       box coordinates fed to the decoder's box sine encoding scaled by H/72 (the 72-frame);
              sineq<a> (e.g. sineq0.5) scales by (H/72)**a instead
  frame72     sinekey + sineq + the RPB in the 72-frame (= cell): the whole decoder in the trained cell geometry
  encpad      bisection, only for "<c>in<canvas>" sizes: the vision encoder runs on the padded canvas (H=72), its
              feature map is cropped to the content (c/14 cells) and the decoder runs with the geometry of that
              smaller grid (grid, sine position, RPB at H=c/14), boxes normalised to the content

Run from the repo root:  HF_TOKEN=... python SAM3_test/collect_box_pairs.py --sizes 420 --variants orig,71/72,cell
Needs coco128 in SAM3_test/data/coco128 (git-ignored); see check_cap_coco128.py for the download line.
"""
import argparse, collections, glob, importlib.util, json, math, os, sys
import cv2, numpy as np, torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "SAM3Fixed"))
import export as E
_sp = importlib.util.spec_from_file_location("open_export", os.path.join(HERE, "..", "SAM3Open", "export.py"))
OE = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(OE)   # text-as-input wrapper: built once per imgsz
from transformers.models.sam3 import Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3DetrDecoder, box_cxcywh_to_xyxy

SHIPPED_COORDS = Sam3DetrDecoder._get_coords               # the grid shipped in SAM3Fixed/export.py and SAM3Open/export.py
ap = argparse.ArgumentParser()
ap.add_argument("--coco", default=os.path.join(HERE, "data", "coco128"))
ap.add_argument("--sizes", default="420")
ap.add_argument("--variants", default="orig")
ap.add_argument("--nimg", type=int, default=64)
ap.add_argument("--skip", type=int, default=0, help="skip the first K usable images (e.g. --skip 64 --nimg 64 = the held-out second half)")
ap.add_argument("--out", default=os.path.join(HERE, "results_pairs"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

NAMES = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter "
         "bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
         "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass "
         "cup fork knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
         "potted_plant bed dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink "
         "refrigerator book clock vase scissors teddy_bear hair_drier toothbrush").split()
NAMES = [n.replace("_", " ") for n in NAMES]

# ---- decoder variants -------------------------------------------------------------------------------------------
ORIG_RPB = Sam3DetrDecoder._get_rpb_matrix


def orig_grid(self, h, w, dtype, device):
    f = lambda n: (torch.arange(0, n, device=device, dtype=torch.float32) / n).to(dtype)
    return f(h), f(w)


def _rpb_cell_units(self, reference_boxes, spatial_shape):
    """_get_rpb_matrix with the grid-to-box-edge deltas scaled by H/72 before the 8x + log2 encoding."""
    height, width = spatial_shape
    boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes)
    batch_size, num_queries, _ = boxes_xyxy.shape
    coords_h, coords_w = self._get_coords(height, width, dtype=reference_boxes.dtype, device=reference_boxes.device)
    deltas_y = (coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]).view(batch_size, num_queries, -1, 2)
    deltas_x = (coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]).view(batch_size, num_queries, -1, 2)
    deltas_y = deltas_y * (height / 72)
    deltas_x = deltas_x * (width / 72)
    enc = lambda d: torch.sign(d * 8) * torch.log2(torch.abs(d * 8) + 1.0) / math.log2(8)
    deltas_x = self.box_rpb_embed_x(enc(deltas_x))
    deltas_y = self.box_rpb_embed_y(enc(deltas_y))
    rpb = (deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)).flatten(2, 3)
    return rpb.permute(0, 3, 1, 2).contiguous()


ENC_VARIANTS = ("ropecell", "peinterp", "encpad", "sinekey", "sineq", "frame72")


def is_enc(v):
    return v in ENC_VARIANTS or v.startswith("sineq")


def _pe_resampled(vit, Hp):
    """Trained 72x72 absolute position map (the 24x24 embedding tiled 3x3) resampled to Hp x Hp."""
    orig = vit.embeddings.position_embeddings.data.float()         # [1, n0*n0, D]
    n0 = int(orig.shape[1] ** 0.5); D = orig.shape[-1]
    pe = orig.reshape(1, n0, n0, D).permute(0, 3, 1, 2).tile(1, 1, 72 // n0, 72 // n0)
    pe = F.interpolate(pe, size=(Hp, Hp), mode="bilinear", align_corners=False)
    return pe.permute(0, 2, 3, 1).reshape(1, Hp * Hp, D)


def apply_enc_variant(w, v, Hp):
    """Returns an undo() callable."""
    ve = w.vision_enc
    saved_pos = {id(l): l.position_ids for l in ve.layers}
    saved_pe = ve.vit_pos_embed
    saved_fpn_pos = ve.fpn_pos_2
    penc = model.detr_decoder.position_encoding
    had_inst_attr = "encode_boxes" in penc.__dict__
    saved_enc_boxes = penc.__dict__.get("encode_boxes")
    if v == "ropecell":
        for l in ve.layers:
            if getattr(l, "window_size", 0) == 0:                    # global-attention layers
                l.position_ids = l.position_ids * (Hp / 72)
    elif v == "peinterp":
        ve.vit_pos_embed = _pe_resampled(model.vision_encoder.backbone, Hp).to(saved_pe.dtype).to(saved_pe.device)

    if v in ("sinekey", "frame72"):                                   # key sine positions of the trained 72x72 grid
        npf = ve.neck.config.fpn_hidden_size // 2
        full = E._sine_pos_enc(1, 72, 72, saved_fpn_pos.device, torch.float32, npf)
        ve.fpn_pos_2 = full[:, :, :Hp, :Hp].contiguous().to(saved_fpn_pos.dtype)
    if v.startswith("sineq") or v == "frame72":                       # decoder box sine encoding in the 72-frame
        base = penc.encode_boxes
        alpha = float(v[5:]) if v.startswith("sineq") and len(v) > 5 else 1.0
        factor = (Hp / 72) ** alpha
        penc.encode_boxes = lambda boxes: base(boxes * factor)

    def undo():
        for l in ve.layers:
            l.position_ids = saved_pos[id(l)]
        ve.vit_pos_embed = saved_pe
        ve.fpn_pos_2 = saved_fpn_pos
        if had_inst_attr:
            penc.encode_boxes = saved_enc_boxes
        else:
            penc.__dict__.pop("encode_boxes", None)
    return undo


def forward_encpad(w, x, tf, tm, content):
    """DynamicVisionDecoderWrapper.forward with the feature map cropped to the content region before the decoder."""
    K = tf.shape[1]; hc = content // 14
    fpn, _ = w.vision_enc(x)                                       # [1, 256, canvas/14, canvas/14]
    fpn = fpn[:, :, :hc, :hc].contiguous()
    npf = w.vision_enc.neck.config.fpn_hidden_size // 2
    pos = E._sine_pos_enc(1, hc, hc, fpn.device, torch.float32, npf).to(fpn.dtype)
    fpn = fpn.unsqueeze(1).expand(-1, K, -1, -1, -1).flatten(0, 1)
    pos = pos.unsqueeze(1).expand(-1, K, -1, -1, -1).flatten(0, 1)
    valid = tm.any(dim=-1)
    safe_mask = tm | (w.first_token & ~valid.unsqueeze(-1))
    boxes, logits, presence = w.decoder(fpn, pos, tf.flatten(0, 1), safe_mask.flatten(0, 1))
    Q = boxes.shape[1]
    boxes_px = (boxes * content).unflatten(0, (-1, K))                 # normalised to the content -> content px
    scores = (logits.sigmoid() * presence.sigmoid()).unflatten(0, (-1, K)) * valid.unsqueeze(-1).to(logits.dtype)
    cls_id = torch.arange(K, device=scores.device, dtype=scores.dtype).view(1, K, 1).expand_as(scores)
    det = torch.cat([boxes_px.float(), scores.unsqueeze(-1).float(), cls_id.unsqueeze(-1).float()], dim=-1)
    return det.flatten(1, 2)


def set_variant(v):
    # the shipped query-box scale is only on for "prod"; every other variant starts from the unpatched decoder
    E._APPLY_QUERY_BOX_SCALE = OE._APPLY_QUERY_BOX_SCALE = (v == "prod")
    model.detr_decoder.position_encoding.box_scale = 1.0
    Sam3DetrDecoder._get_rpb_matrix = ORIG_RPB
    Sam3DetrDecoder._get_coords = orig_grid
    if v == "prod":                                           # exactly what the export scripts ship
        Sam3DetrDecoder._get_coords = SHIPPED_COORDS
        return
    if v == "frame72":
        Sam3DetrDecoder._get_rpb_matrix = _rpb_cell_units
        return
    if v == "orig" or is_enc(v):
        return
    if v == "cell":
        Sam3DetrDecoder._get_rpb_matrix = _rpb_cell_units
    else:
        c = float(v.split("/")[0]) / float(v.split("/")[1]) if "/" in v else float(v)

        def capped(self, h, w, dtype, device):
            f = lambda n: (torch.arange(0, n, device=device, dtype=torch.float32) / (n - 1) * c).to(dtype)
            return f(h), f(w)
        Sam3DetrDecoder._get_coords = capped


def parse_size(s):
    if "in" in s:
        content, canvas = s.split("in")
        return int(content), int(canvas)
    return int(s), int(s)


def out_path(size, variant):
    tag = f"_n{args.nimg}" + (f"_s{args.skip}" if args.skip else "")
    return os.path.join(args.out, f"{size}_{variant.replace('/', '-')}{tag}.json")


# ---- data -------------------------------------------------------------------------------------------------------
items = []
for lp in sorted(glob.glob(args.coco + "/labels/train2017/*.txt")):
    ip = lp.replace("/labels/", "/images/").replace(".txt", ".jpg")
    img = cv2.imread(ip)
    if img is None:
        continue
    H, W = img.shape[:2]
    gt = collections.defaultdict(list)
    for ln in open(lp):
        c, cx, cy, w, h = ln.split(); c = int(c); cx, cy, w, h = map(float, (cx, cy, w, h))
        if w * h >= 0.02:
            gt[c].append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])      # normalised xyxy
    if gt:
        top = [c for c, _ in sorted(gt.items(), key=lambda kv: -len(kv[1]))[:4]]
        items.append((img, top, {c: np.array(gt[c]) for c in top}))
    if len(items) >= args.nimg + args.skip:
        break
items = items[args.skip:]
print("images", len(items), "GT boxes", sum(len(v) for _, _, g in items for v in g.values()), flush=True)


def iou(a, b):
    ix1, iy1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    ix2, iy2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / np.maximum((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter, 1e-12)


def nms(d, thr=0.6):
    d = d[np.argsort(-d[:, 4])]; out = []
    for r in d:
        if not out or iou(r, np.array(out)).max() <= thr:
            out.append(r)
    return np.array(out) if out else np.zeros((0, 6))


tok = os.environ["HF_TOKEN"]
model = Sam3Model.from_pretrained("facebook/sam3", token=tok).half().to("cuda").eval()   # fp16 weights (fp32 + autocast spills out of 4 GB VRAM: 8 s/forward)
proc = Sam3Processor.from_pretrained("facebook/sam3", token=tok)
te_all, am_all = E.encode_concepts(model, proc, NAMES)            # all 80 names once (fp32)
TE, AM = te_all.cuda(), (am_all > 0).cuda()

for size in args.sizes.split(","):
    content, canvas = parse_size(size)
    todo = [v for v in args.variants.split(",") if not os.path.exists(out_path(size, v))]
    for v in args.variants.split(","):
        if v not in todo:
            print(f"skip {size} {v}", flush=True)
    if not todo:
        continue
    w = OE.DynamicVisionDecoderWrapper(model, "cuda", canvas).cuda().eval().half()   # one per canvas size (shared model is mutated)
    w.scale = w.scale.float()                                                       # keep pixel scaling in fp32
    X = []
    for img, _, _ in items:
        x = cv2.cvtColor(cv2.resize(img, (content, content)), cv2.COLOR_BGR2RGB).astype(np.float32)
        x = (x - 127.5) / 127.5
        if canvas != content:
            full = np.zeros((canvas, canvas, 3), np.float32); full[:content, :content] = x; x = full
        X.append(torch.from_numpy(x).permute(2, 0, 1)[None].cuda().half())
    for v in todo:
        enc_part, _, grid_part = v.partition("+")
        set_variant(grid_part if "+" in v else v)
        undo = apply_enc_variant(w, enc_part, canvas // 14)
        res = dict(size=size, variant=v, nimg=len(items), content=content, canvas=canvas, pairs=[], gt=0, npred=0, tp50=0)
        for (img, ids, gtd), x in zip(items, X):
            with torch.no_grad():
                if v == "encpad":
                    d = forward_encpad(w, x, TE[ids].unsqueeze(0), AM[ids].unsqueeze(0), content)[0].float().cpu().numpy()
                else:
                    d = w(x, TE[ids].unsqueeze(0), AM[ids].unsqueeze(0))[0].float().cpu().numpy()
            d[:, :4] = np.clip(d[:, :4] / content, 0, 1)             # canvas px -> normalised image coordinates
            for i, cid in enumerate(ids):
                p = d[d[:, 5] == i]
                p = nms(p[(p[:, 4] >= 0.5) & (p[:, 2] > p[:, 0]) & (p[:, 3] > p[:, 1])])
                g = gtd[cid]; res["gt"] += len(g); res["npred"] += len(p); used = set()
                for gb in g:
                    if not len(p):
                        continue
                    s = iou(gb, p[:, :4]); order = [j for j in np.argsort(-s) if j not in used]
                    if order and s[order[0]] >= 0.3:
                        j = order[0]; used.add(j); res["tp50"] += int(s[j] >= 0.5)
                        res["pairs"].append([float(t) for t in gb] + [float(t) for t in p[j, :4]] + [float(s[j]), float(p[j, 4])])
        undo()
        json.dump(res, open(out_path(size, v), "w"))
        print(f"done {size} {v}: {len(res['pairs'])} pairs of {res['gt']} GT -> {out_path(size, v)}", flush=True)
    del w, X
    torch.cuda.empty_cache()
