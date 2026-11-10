"""Golden detections of a SAM3Fixed engine, to check refactors against. Images: test/bus.jpg + the first 8 coco128 images (test/data/coco128, if present).
Input contract: RGB, resized straight to imgsz x imgsz, (x - 127.5) / 127.5, fp16 [1,3,S,S]. Output contract: `detections` fp16 [N,6] = x1 y1 x2 y2 score cls_id in input pixels.

    python test/golden.py save  weight/sam3_420_det_person/sam3.engine 420 test/golden/person_420.npz
    python test/golden.py check weight/<new>/sam3.engine 420 test/golden/person_420.npz   # exit 1 if a golden box (score >= 0.5) is lost or drifts
"""
import glob, os, sys

import cv2, numpy as np, tensorrt as trt, torch

HERE = os.path.dirname(os.path.abspath(__file__))
THR_KEEP, THR_MUST, IOU_MIN, SCORE_TOL = 0.3, 0.5, 0.9, 0.05


def images():
    paths = [os.path.join(HERE, "bus.jpg")] + sorted(glob.glob(os.path.join(HERE, "data", "coco128", "images", "train2017", "*.jpg")))[:8]
    return {os.path.basename(p): p for p in paths}


def run(engine, S, paths):
    eng = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(open(engine, "rb").read())
    ctx = eng.create_execution_context()
    ctx.set_input_shape("pixel_values", (1, 3, S, S))
    out_name = next(n for n in (eng.get_tensor_name(i) for i in range(eng.num_io_tensors)) if eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT and n == "detections")
    res = {}
    for name, p in paths.items():
        x = (cv2.cvtColor(cv2.resize(cv2.imread(p), (S, S)), cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
        inp = torch.from_numpy(x).permute(2, 0, 1)[None].half().contiguous().cuda()
        out = torch.zeros(tuple(ctx.get_tensor_shape(out_name)), dtype=torch.float16, device="cuda")
        ctx.set_tensor_address("pixel_values", inp.data_ptr()); ctx.set_tensor_address(out_name, out.data_ptr())
        ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
        res[name] = out[0].float().cpu().numpy()
    return res


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    i = max(w, 0) * max(h, 0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


if __name__ == "__main__":
    mode, engine, S, npz = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    if mode == "save":
        paths = images()
        os.makedirs(os.path.dirname(npz), exist_ok=True)
        np.savez(npz, imgsz=S, **run(engine, S, paths))
        print(f"saved {len(paths)} images -> {npz}")
        sys.exit(0)
    gold = dict(np.load(npz)); assert int(gold.pop("imgsz")) == S, "golden was saved at another imgsz"
    paths = {n: p for n, p in images().items() if n in gold}
    assert len(paths) == len(gold), f"missing images: {sorted(set(gold) - set(paths))}"
    new, bad = run(engine, S, paths), 0
    for n in gold:
        g, d = gold[n][gold[n][:, 4] >= THR_KEEP], new[n][new[n][:, 4] >= THR_KEEP]
        for row in g:
            best = max((iou(row, r) for r in d if r[5] == row[5]), default=0.0)
            m = next((r for r in d if r[5] == row[5] and iou(row, r) == best), None)
            lost = best < IOU_MIN or abs(m[4] - row[4]) > SCORE_TOL
            if lost and row[4] >= THR_MUST:
                bad += 1; print(f"  {n}: golden box {row[:5].round(2)} lost or drifted (best IoU {best:.2f})")
        print(f"{n:20s} golden {len(g):3d} boxes >= {THR_KEEP} | new {len(d):3d}")
    print("OK" if not bad else f"FAIL: {bad} golden boxes lost or drifted")
    sys.exit(1 if bad else 0)
