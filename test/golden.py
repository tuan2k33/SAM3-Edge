"""Golden detections of a set-classes engine, to check refactors against. Images: test/bus.jpg, the first 8 coco128 images (test/data/coco128) and the first 60 of COCO val2017 (the images.cocodataset.org zip, $VAL2017_ZIP).
Input contract: RGB, resized straight to imgsz x imgsz, (x - 127.5) / 127.5, fp16 [1,3,S,S]. Output contract: `detections` fp16 [N,6] = x1 y1 x2 y2 score cls_id in input pixels.

    python test/golden.py save  weight/sam3_420_det_person/sam3.engine 420 test/golden/person_420.npz
    python test/golden.py check weight/<new>/sam3.engine 420 test/golden/person_420.npz   # exit 1 if a golden box (score >= 0.5) is lost, or the median IoU / the p95 score drift get worse than a plain rebuild
"""
import glob, os, sys, zipfile

import cv2, numpy as np, tensorrt as trt, torch

HERE = os.path.dirname(os.path.abspath(__file__))
# Rebuilding the same ONNX shuffles fp16 tactics: golden boxes (score >= 0.5) then keep IoU >= 0.82, drift <= 0.17, median IoU 0.996, p95 drift 0.019.
# A box is lost below IOU_MIN / above SCORE_TOL; the aggregates catch a broad drift.
THR_KEEP, THR_MUST, IOU_MIN, SCORE_TOL, IOU_MEDIAN, DRIFT_P95 = 0.3, 0.5, 0.7, 0.25, 0.98, 0.05
VAL2017 = os.environ.get("VAL2017_ZIP", "/mnt/c/Dev/projects/ModelEdge/Detector/data/coco/val2017.zip")
N_VAL = 60


def images():
    """name -> loader returning the BGR image."""
    out = {"bus.jpg": lambda: cv2.imread(os.path.join(HERE, "bus.jpg"))}
    for p in sorted(glob.glob(os.path.join(HERE, "data", "coco128", "images", "train2017", "*.jpg")))[:8]:
        out[os.path.basename(p)] = lambda p=p: cv2.imread(p)
    if os.path.exists(VAL2017):
        z = zipfile.ZipFile(VAL2017)
        for n in sorted(n for n in z.namelist() if n.endswith(".jpg"))[:N_VAL]:
            out["val2017_" + os.path.basename(n)] = lambda n=n: cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
    return out


def run(engine, S, loaders):
    eng = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(open(engine, "rb").read())
    ctx = eng.create_execution_context()
    ctx.set_input_shape("pixel_values", (1, 3, S, S))
    outs = [n for n in (eng.get_tensor_name(i) for i in range(eng.num_io_tensors)) if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]  # detections (+ masks)
    res = {}
    for name, load in loaders.items():
        x = (cv2.cvtColor(cv2.resize(load(), (S, S)), cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
        inp = torch.from_numpy(x).permute(2, 0, 1)[None].half().contiguous().cuda()
        bufs = {n: torch.zeros(tuple(ctx.get_tensor_shape(n)), dtype=torch.float16, device="cuda") for n in outs}
        ctx.set_tensor_address("pixel_values", inp.data_ptr())
        for n, t in bufs.items():
            ctx.set_tensor_address(n, t.data_ptr())
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
        res[name] = bufs["detections"][0].float().cpu().numpy()
    return res


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    i = max(w, 0) * max(h, 0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


if __name__ == "__main__":
    mode, engine, S, npz = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    if mode == "save":
        loaders = images()
        os.makedirs(os.path.dirname(npz), exist_ok=True)
        np.savez(npz, imgsz=S, **run(engine, S, loaders))
        print(f"saved {len(loaders)} images -> {npz}")
        sys.exit(0)
    gold = dict(np.load(npz)); assert int(gold.pop("imgsz")) == S, "golden was saved at another imgsz"
    loaders = {n: l for n, l in images().items() if n in gold}
    assert len(loaders) == len(gold), f"missing images (is $VAL2017_ZIP set?): {sorted(set(gold) - set(loaders))[:3]} ..."
    new, bad, ious, drifts = run(engine, S, loaders), 0, [], []
    for n in gold:
        g, d = gold[n][gold[n][:, 4] >= THR_MUST], new[n][new[n][:, 4] >= THR_KEEP]
        for row in g:
            c = [(iou(row, r), abs(r[4] - row[4])) for r in d if r[5] == row[5]]
            best, drift = max(c, default=(0.0, 1.0))
            ious.append(best); drifts.append(drift)
            if best < IOU_MIN or drift > SCORE_TOL:
                bad += 1; print(f"  {n}: golden box {row[:5].round(2)} lost (best IoU {best:.2f}, score drift {drift:.2f})")
    med, p95 = float(np.median(ious)), float(np.percentile(drifts, 95))
    broad = med < IOU_MEDIAN or p95 > DRIFT_P95
    print(f"{len(gold)} images, {len(ious)} golden boxes with score >= {THR_MUST}: median IoU {med:.3f}, p95 score drift {p95:.3f}, {bad} lost:", "FAIL" if bad or broad else "OK")
    sys.exit(1 if bad or broad else 0)
