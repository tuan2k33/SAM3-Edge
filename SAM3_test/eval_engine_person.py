"""Box size/centre error of a SAM3Fixed TensorRT engine (baked class 'person') against COCO ground truth.

Runs the engine itself (TensorRT, FP16) on every coco128 image that has a 'person' GT box covering >= 2% of the
image, and reports the same quantities as analyze_box_pairs.py (mean width/height error in %, centre error in
patches, recall@0.5, mean IoU). Use it to check that the fixes in SAM3Fixed/export.py really end up in the engine.

    python SAM3_test/eval_engine_person.py weight/sam3_420_det_person/sam3_new.engine 420 [--set first|held|all]
"""
import argparse, glob, os
import cv2, numpy as np, tensorrt as trt, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("engine"); ap.add_argument("imgsz", type=int)
ap.add_argument("--coco", default=os.path.join(HERE, "data", "coco128"))
ap.add_argument("--set", choices=["first", "held", "all"], default="all")
args = ap.parse_args()

logger = trt.Logger(trt.Logger.ERROR)
eng = trt.Runtime(logger).deserialize_cuda_engine(open(args.engine, "rb").read())
ctx = eng.create_execution_context()
S = args.imgsz
ctx.set_input_shape("pixel_values", (1, 3, S, S))
names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
out_name = [n for n in names if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT][0]
dt = {trt.float32: torch.float32, trt.float16: torch.float16}
inp = torch.zeros((1, 3, S, S), dtype=dt[eng.get_tensor_dtype("pixel_values")], device="cuda")
out = torch.zeros(tuple(ctx.get_tensor_shape(out_name)), dtype=dt[eng.get_tensor_dtype(out_name)], device="cuda")
ctx.set_tensor_address("pixel_values", inp.data_ptr()); ctx.set_tensor_address(out_name, out.data_ptr())
stream = torch.cuda.Stream()


def iou(a, b):
    ix1, iy1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1]); ix2, iy2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    return inter / np.maximum((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter, 1e-12)


def nms(d, thr=0.6):
    d = d[np.argsort(-d[:, 4])]; keep = []
    for r in d:
        if not keep or iou(r, np.array(keep)).max() <= thr:
            keep.append(r)
    return np.array(keep) if keep else np.zeros((0, 6))


# same image selection as collect_box_pairs.py: usable images (>= 1 GT box >= 2% area), first 64 / next 64
usable = []
for lp in sorted(glob.glob(args.coco + "/labels/train2017/*.txt")):
    if cv2.imread(lp.replace("/labels/", "/images/").replace(".txt", ".jpg")) is None:
        continue
    rows = [ln.split() for ln in open(lp)]
    big = [r for r in rows if float(r[3]) * float(r[4]) >= 0.02]
    if big:
        usable.append((lp, big))
sel = {"first": usable[:64], "held": usable[64:128], "all": usable[:128]}[args.set]

pairs, ngt, tp50 = [], 0, 0
for lp, rows in sel:
    gt = np.array([[float(r[1]) - float(r[3]) / 2, float(r[2]) - float(r[4]) / 2, float(r[1]) + float(r[3]) / 2, float(r[2]) + float(r[4]) / 2]
                   for r in rows if int(r[0]) == 0 and float(r[3]) * float(r[4]) >= 0.02])
    if not len(gt):
        continue
    im = cv2.imread(lp.replace("/labels/", "/images/").replace(".txt", ".jpg"))
    x = cv2.cvtColor(cv2.resize(im, (S, S)), cv2.COLOR_BGR2RGB).astype(np.float32)
    inp.copy_(torch.from_numpy((x - 127.5) / 127.5).permute(2, 0, 1)[None])
    ctx.execute_async_v3(stream_handle=stream.cuda_stream); stream.synchronize()
    d = out[0].float().cpu().numpy()                                    # [200, 6] x1 y1 x2 y2 score cls (px of the input)
    d = d[(d[:, 4] >= 0.5) & (d[:, 2] > d[:, 0]) & (d[:, 3] > d[:, 1])]
    d[:, :4] = np.clip(d[:, :4] / S, 0, 1)
    p = nms(d); used = set(); ngt += len(gt)
    for g in gt:
        if not len(p):
            continue
        v = iou(g, p[:, :4]); order = [j for j in np.argsort(-v) if j not in used]
        if order and v[order[0]] >= 0.3:
            j = order[0]; used.add(j); tp50 += int(v[j] >= 0.5); pairs.append(list(g) + list(p[j, :4]) + [v[j]])
P = np.array(pairs); P = P[P[:, 8] >= 0.5]; g, p = P[:, :4], P[:, 4:8]; pn = 14.0 / S
rw = ((p[:, 2] - p[:, 0]) / (g[:, 2] - g[:, 0]) - 1) * 100; rh = ((p[:, 3] - p[:, 1]) / (g[:, 3] - g[:, 1]) - 1) * 100
cx = (((p[:, 0] + p[:, 2]) - (g[:, 0] + g[:, 2])) / 2 / pn).mean(); cy = (((p[:, 1] + p[:, 3]) - (g[:, 1] + g[:, 3])) / 2 / pn).mean()
se = lambda a: a.std(ddof=1) / np.sqrt(len(a))
print(f"{os.path.basename(args.engine)} @ {S}, set={args.set}: {len(P)} person boxes (IoU>=0.5) of {ngt} GT | recall@.5 {tp50 / ngt:.3f} | "
      f"width {rw.mean():+.1f}±{se(rw):.1f}%  height {rh.mean():+.1f}±{se(rh):.1f}% | centre ({cx:+.2f}, {cy:+.2f}) patch | mean IoU {P[:, 8].mean():.3f}")
