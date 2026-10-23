"""Run SAM3Fixed TensorRT engines (FP16, baked class 'person') on bus.jpg and draw all of them on one image (one colour per engine).

    python test/infer_engine_bus.py weight/sam3_420_det_person/sam3.engine:420 weight/sam3_1008_det_person/sam3.engine:1008
"""
import os, sys
import cv2, numpy as np, tensorrt as trt, torch

THR = float(os.environ.get("THR", 0.5))
HERE = os.path.dirname(os.path.abspath(__file__))
img = cv2.imread(os.path.join(HERE, "bus.jpg")); h0, w0 = img.shape[:2]
logger = trt.Logger(trt.Logger.ERROR)
vis = img.copy(); colors = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 200, 255)]
for n, spec in enumerate(sys.argv[1:]):
    path, S = spec.rsplit(":", 1); S = int(S)
    eng = trt.Runtime(logger).deserialize_cuda_engine(open(path, "rb").read()); ctx = eng.create_execution_context()
    ctx.set_input_shape("pixel_values", (1, 3, S, S))
    out_name = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors) if eng.get_tensor_mode(eng.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT][0]
    inp = torch.from_numpy((cv2.cvtColor(cv2.resize(img, (S, S)), cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5).permute(2, 0, 1)[None].contiguous().cuda()
    out = torch.zeros(tuple(ctx.get_tensor_shape(out_name)), dtype=torch.float16, device="cuda")
    ctx.set_tensor_address("pixel_values", inp.data_ptr()); ctx.set_tensor_address(out_name, out.data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream); torch.cuda.synchronize()
    d = out[0].float().cpu().numpy(); d = d[d[:, 4] >= THR]
    col = colors[n % len(colors)]
    cv2.putText(vis, f"{S}", (10, 30 + 30 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    print(f"{path} @ {S}: {len(d)} boxes")
    for x1, y1, x2, y2, sc, _ in d:
        x1, x2, y1, y2 = x1 / S * w0, x2 / S * w0, y1 / S * h0, y2 / S * h0
        print(f"  ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) score {sc:.2f}")
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
cv2.imwrite(os.path.join(HERE, "sam3_bus_person_fixed_engines.jpg"), vis)
