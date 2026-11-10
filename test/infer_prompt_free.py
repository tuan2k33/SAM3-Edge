"""Run a prompt-free engine on one image, post-process on the host, draw the result.
    python test/infer_prompt_free.py weight/sam3_420_prompt-free/sam3.engine 420 test/bus.jpg out.jpg [--min-area 0.003] [--max-area 1.0]

The engine returns one mask per grid point. `postprocess` is what a consumer should do with them (the same job as SAM2's automatic mask generator, other filters):
drop masks under --min-area (fraction of the frame), NMS by mask IoU with the largest mask first, drop masks that lie >= 90% inside a kept larger one, then drop masks over
--max-area (off by default; big masks first gather their parts, then leave). No score filter: the tracker's predicted IoU ranks parts above whole objects."""
import argparse

import cv2, numpy as np, tensorrt as trt, torch, torch.nn.functional as F


def postprocess(det, mk, min_area=0.003, max_area=1.0, nms_iou=0.5, contain=0.9):
    """det [P,6] and mk [P,h,w] (probabilities) of one image, torch tensors -> row indices to keep, largest mask first."""
    area = mk.flatten(1).sum(1) / mk[0].numel()                    # soft area, a fraction of the frame
    idx = (area >= min_area).nonzero()[:, 0]
    idx = idx[area[idx].argsort(descending=True)]
    if len(idx) == 0:
        return idx.cpu().numpy()
    b = (mk[idx] > 0.5).flatten(1).float(); inter = (b @ b.T).cpu().numpy(); a = b.sum(1).cpu().numpy()
    iou = inter / np.maximum(a[:, None] + a[None] - inter, 1)
    inside = inter / np.maximum(a[None, :], 1)                      # share of mask j lying inside mask i
    sup, keep = np.zeros(len(idx), bool), []
    for i in range(len(idx)):                                        # NMS, largest first
        if not sup[i]:
            keep.append(i); sup |= iou[i] >= nms_iou
    kept = []
    for i in keep:                                                   # parts of an already kept mask go
        if not kept or not (inside[kept, i] >= contain).any():
            kept.append(i)
    return idx.cpu().numpy()[[i for i in kept if float(area[idx[i]]) <= max_area]]


class Engine:
    def __init__(self, path, S):
        self.S = S
        eng = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = eng.create_execution_context()
        self.inp = torch.zeros(1, 3, S, S, dtype=torch.float16, device="cuda")
        self.out = {n: torch.zeros(tuple(self.ctx.get_tensor_shape(n)), dtype=torch.float16, device="cuda") for n in ("detections", "masks")}
        self.ctx.set_tensor_address("pixel_values", self.inp.data_ptr())
        for n, t in self.out.items():
            self.ctx.set_tensor_address(n, t.data_ptr())

    def __call__(self, bgr):
        S = self.S
        self.inp.copy_(torch.from_numpy((cv2.cvtColor(cv2.resize(bgr, (S, S)), cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5).permute(2, 0, 1)[None])
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
        return self.out["detections"][0].float(), self.out["masks"][0].float()


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("engine"); p.add_argument("imgsz", type=int); p.add_argument("image"); p.add_argument("out")
    p.add_argument("--min-area", type=float, default=0.003); p.add_argument("--max-area", type=float, default=1.0)
    a = p.parse_args()
    bgr = cv2.imread(a.image); H, W = bgr.shape[:2]
    det, mk = Engine(a.engine, a.imgsz)(bgr)
    keep = postprocess(det, mk, a.min_area, a.max_area)
    print(f"{len(det)} points -> {len(keep)} masks")
    vis, rng = bgr.copy(), np.random.RandomState(0)
    for i in keep:
        c = tuple(int(v) for v in rng.randint(60, 255, 3))
        m = F.interpolate(mk[i][None, None], size=(H, W), mode="bilinear")[0, 0].cpu().numpy() > 0.5
        vis[m] = (0.5 * vis[m] + 0.5 * np.array(c)).astype(np.uint8)
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(vis, cs, -1, c, 2)
    cv2.imwrite(a.out, vis); print("wrote", a.out)
