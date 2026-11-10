# Open-text (`ds9_open_text.py`)

Open-text mode (`export.py --mode open-text`): the prompt is encoded into a tensor at runtime instead of being baked into the graph. Never requires re-exporting to change the prompt. Each camera gets exactly **one** free-text prompt (not a class list — see the main [README](../README.md#how-it-works) for why). For multiple classes on the same camera, use [set-classes](GUIDE_set_classes.md) instead.

---

## Quick start — live RTSP cameras

**1. Write `config_open_text.txt`**

```ini
[sources]
urls =
    rtsp://user:pass@192.168.1.10:554
    rtsp://user:pass@192.168.1.11:554

[cam0]
prompt = forklift

[cam1]
prompt = spilled liquid on the floor

[model]
conf = 0.3

[pipeline]
width = 1280
height = 720
infer_interval = 5
run_seconds = 60
```

Each camera gets a **comma-separated class list** (`prompt = car, truck`) of up to `K_MAX=5` classes (`export.py --k-max`, must match `sam3open_text.K_MAX`) — changing prompts never requires re-exporting. Up to `MAX_SOURCES=4` cameras (`sam3open_text.py`, matching `vision_decoder.engine`'s max batch; rebuild the engine with a bigger `--max-batch` to raise this). Unused class slots still cost decoder time.

> `config_open_text.txt` can contain RTSP credentials — it is in `.gitignore` and never committed.

**2. Run**

```bash
CUDA_VISIBLE_DEVICES=1 python3 ds9_open_text.py
```

**Editing `config_open_text.txt` while running hot-swaps prompts live** — the changed prompt is re-encoded and `nvinfer`'s context is reloaded; there is no engine rebuild at all (the engine itself never changes, only the injected text tensor does).

---

## Manual steps

**Export the engines only** (built once, ever — prompts are supplied at
runtime, never baked in, so there's normally no need to re-export):

```bash
python3 export.py --mode open-text --imgsz 420        # from the repo root
# → weight/sam3_420_open-text/{text_encoder,vision_decoder}.{onnx,engine}  (fp16 in and out, detections + masks)
```

**Build the custom DeepStream plugin** (one for every mode, source in this folder; see the root `CLAUDE.md`):

```bash
g++ -shared -fPIC -O2 -std=c++20 -o libnvdsinfer_sam3.so nvdsparsebbox_sam3.cpp \
    -I/opt/nvidia/deepstream/deepstream/includes \
    -I/usr/local/cuda-13.2/targets/x86_64-linux/include
```

---

## Note on ONNX/engine resolution

Both the ONNX graphs and the TRT engines have `imgsz` (H and W) hardcoded at
export time (default 1008x1008; `export.py --imgsz 420` gives a much lighter
model, output in `weight/sam3_420_open-text/`) — only `batch` is a dynamic axis.
`ds9_open_text.py` reads the matching size from `config_open_text.txt` `[model] imgsz`. This isn't just a build-time
engine-profile choice: `_VisionEncoder.__init__` (in
`export.py`) rebuilds each ViT layer's `rotary_emb` and re-tiles the
absolute position embedding for a specific `Hp,Wp` grid *at trace time* (not
inside `forward()`), so a different resolution needs a full re-export
(`export.py --mode open-text --imgsz <N>`), not just a new TensorRT optimization profile on the
existing ONNX.
