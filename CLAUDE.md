# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Real-time open-vocabulary detection and instance segmentation on multi-camera RTSP streams using [SAM3](https://github.com/facebookresearch/sam3) and NVIDIA DeepStream 9. Two independent, self-contained architectures for getting a prompt into SAM3 live side by side as top-level folders — `SAM3Fixed/` (classes baked at export time) and `SAM3Open/` (open-vocabulary, prompt supplied at runtime). Each folder holds everything for that architecture: the export script, the custom `nvinfer` plugin, and the DeepStream pipeline script(s) + configs. Generated engines from both go into a shared `weight/` folder at the repo root.

## Setup

No pinned `requirements.txt` — always install latest torch/transformers/tensorrt matching your CUDA/TensorRT install:
```bash
pip install torch torchvision numpy opencv-python pillow transformers einops ftfy regex iopath onnx tensorrt
```
`Sam3Model`/`Sam3Processor` ship inside `transformers` itself — no separate `facebookresearch/sam3` install needed.

Also requires (not pip-installable): DeepStream 9 (`pyservicemaker`), TensorRT 10.x (`trtexec`, `libnvinfer`), CUDA 13.2, `ffmpeg`. All `gpu-id` properties in DeepStream configs must stay `0`; select the physical GPU via `CUDA_VISIBLE_DEVICES=N` instead.

## Precision: FP16 only

Always run and test in FP16 — TensorRT engines (`--fp16`, `network-mode=1`) and PyTorch tests/benchmarks alike (`model.half()`, pixel scaling kept in fp32). Never use FP32, and never `torch.autocast` over FP32 weights: FP32 is only marginally more accurate and much slower, and on the 4 GB dev GPU autocast keeps both weight copies and spills out of VRAM (8 s/forward vs 0.2 s for a real fp16 model). Do not spend time on FP32 comparisons.

## Common commands

Export an engine (dispatches to the right architecture based on `--prompt`):
```bash
python3 export.py --prompt truck car --mode det --imgsz 420   # baked -> weight/sam3_420_det_car_truck/
python3 export.py --imgsz 1008                                 # open-vocab -> weight/sam3_1008_open/ (once, ever)
```

Run the SAM3Fixed pipeline (baked classes, multi-class per camera, engine auto-built on first run):
```bash
cd SAM3Fixed && python3 ds9_rtsp.py    # reads config.txt; edit it live to hot-swap prompts
```

Run the SAM3Open pipeline (open-vocab, comma-separated class list per camera, up to 4 cameras × K_MAX=5 classes):
```bash
cd SAM3Open && python3 ds9_rtsp.py   # reads config.txt; edit it live to hot-swap prompts
```

Recompile a custom DeepStream parser after editing its `.cpp` (each folder is self-contained):
```bash
cd SAM3Fixed && g++ -shared -fPIC -O2 -o libnvdsinfer_sam3.so nvdsparsebbox_sam3.cpp \
    -I/opt/nvidia/deepstream/deepstream/includes \
    -I/usr/local/cuda-13.2/targets/x86_64-linux/include -std=c++14

cd SAM3Open && g++ -shared -fPIC -O2 -o libnvdsinfer_sam3.so nvdsparsebbox_sam3.cpp \
    -I/opt/nvidia/deepstream/deepstream/includes \
    -I/usr/local/cuda-13.2/targets/x86_64-linux/include -std=c++14
```

There is no automated test suite — the two production pipeline scripts above are the verification path (run them against real or fake-RTSP sources and check the detections/output video).

Per-stage throughput (decode fps, text encoder, vision decoder, baked engine), from the repo root:
```bash
python3 bench_stages.py --imgsz 420 --batch 1 2 4
```

## Architecture

### Two self-contained folders, one shared weight/ folder

`SAM3Fixed/` and `SAM3Open/` each hold their own export script, custom `nvinfer` plugin (`nvdsparsebbox_sam3.cpp`/`libnvdsinfer_sam3.so` — same filenames in both folders, no collision since they're in separate directories), and DeepStream pipeline script(s) + `config.txt`. Nothing about one architecture reaches into the other: `SAM3Open/export.py` carries its own copy of `_VisionEncoder`/`_Decoder` and the imgsz fixes (keep in sync with `SAM3Fixed/export.py`). Both write their generated engines into the shared `weight/` folder at the repo root (`weight/sam3_{imgsz}_{mode}_{classes}/` for Fixed, `weight/sam3_{imgsz}_open/` for Open) rather than next to themselves.

### Two ways to give SAM3 a prompt

**`SAM3Fixed/`** — classes are baked into the ONNX graph at export time as constants (`export.py`, formerly `specialize.py`). The resulting engine takes only an image as input and outputs detections directly, running natively in `nvinfer` exactly like a YOLO model. Changing classes requires re-exporting (or hot-swapping to a different already-built engine, see Multi-camera union-prompt pattern below). Multiple classes per camera are supported (cost scales with class count — see `Sam3Wrapper` below).

**`SAM3Open/`** — open-vocabulary: the prompt is encoded into a tensor at runtime instead of being baked in (`export.py`, formerly `export_onnx_dynamic.py`, produces a separate `text_encoder.engine` + `vision_decoder.engine`, built once, never re-exported). `ds9_rtsp.py` is a real, working camera pipeline: each camera gets a comma-separated class list (`prompt = car, truck`, up to `K_MAX=5` classes — baked into the engine by `export.py --k-max`, must match `sam3open_text.K_MAX`), and up to `MAX_SOURCES=4` cameras (matching the engine's max profile batch — `sam3open_text.py`) are scored in the SAME `nvinfer` forward pass, each against its own classes. Each class gets its own text embedding (slot k of the camera's row); the ViT runs once per image, the K_MAX slots ride the batch axis through the DETR decoder (decoder cost is paid for all K_MAX slots of every camera, used or not; unused slots have an all-False `text_mask` and score 0).

`export.py` at the repo root is a thin dispatcher over both: `--prompt <classes>` calls `SAM3Fixed/export.py`; no `--prompt` calls `SAM3Open/export.py`.

### `export.py`'s export pattern (SAM3Fixed)

SAM3 is decomposed into ONNX-traceable submodules (`_VisionEncoder`/`_VisionEncoderFull`, `_Decoder`/`_DecoderFull`) rather than exporting the HuggingFace model directly, because the stock model hard-fails at any resolution other than 1008 (missing dynamic RoPE/position-embedding resize logic). `_VisionEncoder.__init__` rebuilds each ViT layer's `rotary_emb` for the target `Hp,Wp` and re-tiles the absolute position embedding before tracing — this is what makes `--imgsz` other than 1008 possible at all (must still be a multiple of `patch_size`, square). `SAM3Open/export.py` has the same modules (a copy) and takes `--imgsz` too (default 1008; 420 verified end-to-end on 2 cameras). `SAM3Open/config.txt` `[model] imgsz` must match the engines' export size.

`Sam3Wrapper` bakes N classes as ONNX constants (`text_feat_all`/`text_mask_all`) and folds them into the batch axis: the ViT runs once, its outputs are repeated to `B*N` rows (row `b*N+i` = image b × class i), and the DETR decoder runs once over all rows, then reshapes back to `[B, N*Q, ...]` (class-major, `cls_id = i // Q`). FLOPs of the decoder still scale with N (the DETR encoder fuses text per prompt) but there is no per-class graph unroll, and activation memory scales with `B*N`.

### Object queries (200 per class/slot) are not worth cutting

The 200 learned object queries (`Sam3DetrDecoder.query_embed`, paper default `Q=200`) cannot be shrunk cheaply: truncating to the first N (tried 100/50/25 on 24 frames of `pedestrians.mp4`, imgsz 420, Q=200 as reference, conf 0.3 + NMS 0.6) kept ~83-90% / 68-82% / ~59% of the reference detections with growing score drift, and the PyTorch fp32 forward time did not change measurably (~420 ms; backbone + DETR encoder dominate). Only 85 of 200 queries ever fired on that clip but which ones depends on the image (query 87 alone took 22/236 detections), so there is no fixed "best N". Each query is conditioned on one prompt, so a shared pool across classes is not possible either; cost scales with the number of classes/slots. There is deliberately no `--num-queries` option.

### Multi-camera union-prompt pattern (SAM3Fixed)

`SAM3Fixed/spec_cache.py` lets cameras with different prompts share one engine: it computes the sorted union of every camera's requested classes, builds (or reuses) one engine in `weight/` baked with that union via `export.py`, and `PerCameraFilter` (in `ds9_rtsp.py`) filters detections back down per-camera by `cls_id` afterward. Engine folders are named `sam3_{imgsz}_{mode}_{sorted-union-classes}/` so the same class set always resolves to the same folder regardless of which camera declared it first.

Because `nvinfer`'s `config-file-path` property is live-swappable while the pipeline runs (as long as input resolution is unchanged — verified experimentally), this same union-prompt engine can be hot-swapped at runtime when a camera's prompt changes, without restarting GStreamer. `ds9_rtsp.py`'s watcher thread implements this: it polls `config.txt` for changes, recomputes the union, builds a new engine if that union hasn't been seen before, and calls `nvinfer.set({"config-file-path": ...})`.

### Per-camera prompt injection (SAM3Open)

`vision_decoder.engine` takes `pixel_values`/`text_features`/`text_mask` as inputs (batch dim = camera count, matching `nvstreammux`). `nvinfer` handles `pixel_values` (the image) automatically like any detector; `text_features`/`text_mask` are non-image input layers, injected by `SAM3Open/nvdsparsebbox_sam3.cpp`'s `NvDsInferInitializeInputLayers` — a fixed symbol name `nvinfer` looks up via `dlsym` (a custom name is silently never called; verified empirically). It reads two flat snapshot files written by `SAM3Open/sam3open_text.py`:
```
/dev/shm/sam3open_text_features.bin   MAX_SOURCES x K_MAX x 32 x 256  float32
/dev/shm/sam3open_text_mask.bin       MAX_SOURCES x K_MAX x 32        uint8   (TRT reports BOOL; DeepStream maps it to UINT8)
```
row *b* slot *k* = camera *b*'s k-th class encoding (from `text_encoder.engine`, run directly via TensorRT in Python — not through `nvinfer`). This function is called once per `nvinfer` context (re)load, not per frame — so a prompt change needs a snapshot rewrite followed by `nvinfer.set({"config-file-path": <SAME path>})`, which reloads the context and re-triggers it (verified: reloading the *same* config path re-reads the snapshot files). No TensorRT engine rebuild is ever needed for a prompt change. `SAM3Open/ds9_rtsp.py`'s watcher thread implements this the same way `SAM3Fixed/ds9_rtsp.py`'s does for the baked engine (see Multi-camera union-prompt pattern above).

`SAM3Open/nvdsparsebbox_sam3.cpp`'s output parser reads `detections [K_MAX*200,6]` float32 (`x1 y1 x2 y2 score cls_id`, pixel units); `cls_id` is the camera's class slot (NMS only suppresses within the same slot), and `SAM3Open/ds9_rtsp.py`'s `PromptLabeler` maps `(camera, cls_id)` to the display label from the camera's class list instead of a fixed `labels.txt`.

### The imgsz != 1008 box-accuracy fix

The checkpoint was only trained at imgsz 1008 (H = 72 cells of 14 px). At other `imgsz` the boxes shrink toward the top-left corner (width ~ -10% at 420), smoothly in H. Both `SAM3Fixed/export.py` and `SAM3Open/export.py` (self-contained, keep the two blocks in sync) apply, at **module import time** and baked into every exported engine (no flag):
- `Sam3DetrDecoder._get_coords` -> `arange(0,H)/(H-1) * 71/72` (`_TRAIN_GRID = 72`): fixes the box *centre*; the original grid `arange/H` is only a convention.
- `Sam3SinePositionEmbedding.encode_boxes` multiplies the box by `box_scale = (H/72)^1.0` (set per forward in `_Decoder._decode` from the feature map size): fixes the box *size* (width -10.4% -> -1.8% at 420 on held-out images, TensorRT FP16 engine). Root cause is in the decoder's box position encoding, not the grid; the exponent is empirical (a=1 = "72-grid frame"), no derivation.

At H = 72 both are the identity. Rejected: letting the grid end at 1.0, content resolution, window padding, encoder RoPE/abs-pos. Full evidence: `test/BOX_SHRINK_INVESTIGATION.md`.

### test/ (investigation artifacts)

`BOX_SHRINK_INVESTIGATION.md` (final fix, ground-truth evidence; scripts `collect_box_pairs.py`, `analyze_box_pairs.py`, `eval_engine_person.py` for engines; raw pairs in `results_pairs/`) plus `compare_bus.py` / `infer_engine_bus.py` (before/after/reference images on `bus.jpg`, the Ultralytics sample, 810×1080; run from the repo root with `HF_TOKEN`). Read the md before touching that patch.

### DeepStream pipeline gotchas (measured on this hardware — RTX 50-series, driver 610.x, DS 9.1)

- **Video encoder**: use `nvv4l2h264enc`, and force `video/x-raw(memory:NVMM),format=NV12` via a `capsfilter` immediately before it — without the explicit format, GStreamer negotiates I420 (listed first in the sink template) and the encoder misreads chroma, corrupting the image (bitstream stays valid, so `ffmpeg -v error` reports zero errors — only visual inspection catches it). Also set `idrinterval` explicitly; `iframeinterval` alone gets ignored and produces too few real seek points. `nvh264enc` and `nvcudah264enc` do not work at all with a DeepStream hardware-decode source on this GPU/driver combination.
- **Tracker**: `NvSORT` (cascaded/greedy matching, no visual tracker) is cheaper than `NvDCF` and tolerates a higher `interval`, but its stock `probationAge`/`minTrackerConfidence` assume near-every-frame detection — at `interval=5` they need lowering (`probationAge=1`, `minTrackerConfidence≈0.1`) or tracks never "mature" and zero boxes get emitted (see `config_tracker_NvSORT_i5.yml`, present in both `SAM3Fixed/` and `SAM3Open/`).
- Both custom parsers read detection count and layer shapes from the actual output tensor at runtime rather than assuming a fixed layout, so they stay correct across different baked class counts (SAM3Fixed) or camera counts (SAM3Open) without recompiling.
