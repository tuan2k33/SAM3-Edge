"""Run a prompt-free engine on one image: score filter + mask NMS on the host, draw the result.
    python test/infer_prompt_free.py weight/sam3_420_prompt-free/sam3.engine 420 test/bus.jpg out.jpg"""
import sys

import cv2, numpy as np, tensorrt as trt, torch, torch.nn.functional as F

engine, S, img, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
THR_SCORE, THR_NMS = 0.7, 0.5
eng = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(open(engine, "rb").read())
ctx = eng.create_execution_context()
bgr = cv2.imread(img); H, W = bgr.shape[:2]
x = (cv2.cvtColor(cv2.resize(bgr, (S, S)), cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
inp = torch.from_numpy(x).permute(2, 0, 1)[None].half().contiguous().cuda()
bufs = {n: torch.zeros(tuple(ctx.get_tensor_shape(n)), dtype=torch.float16, device="cuda") for n in ("detections", "masks")}
ctx.set_tensor_address("pixel_values", inp.data_ptr())
for n, t in bufs.items(): ctx.set_tensor_address(n, t.data_ptr())
ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
det, mk = bufs["detections"][0].float(), bufs["masks"][0].float()
order = det[:, 4].argsort(descending=True)
order = order[det[order, 4] >= THR_SCORE]
b = (mk[order] > 0.5).flatten(1).float(); sel = []
for i in range(len(order)):
    if all(float((b[i] * b[j]).sum() / ((b[i] + b[j]).clamp(max=1).sum() + 1e-6)) < THR_NMS for j in sel): sel.append(i)
print(f"{len(det)} points, {len(order)} with score >= {THR_SCORE}, {len(sel)} after mask NMS")
vis, rng = bgr.copy(), np.random.RandomState(0)
for i in sel:
    d = det[order[i]].cpu().numpy(); c = tuple(int(v) for v in rng.randint(60, 255, 3))
    m = F.interpolate(mk[order[i]][None, None], size=(H, W), mode="bilinear")[0, 0].cpu().numpy() > 0.5
    vis[m] = (0.5 * vis[m] + 0.5 * np.array(c)).astype(np.uint8)
    cv2.rectangle(vis, (int(d[0] * W / S), int(d[1] * H / S)), (int(d[2] * W / S), int(d[3] * H / S)), c, 2)
cv2.imwrite(out, vis); print("wrote", out)
