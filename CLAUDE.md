# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Real-time open-vocabulary detection and instance segmentation on multi-camera RTSP streams using [SAM3](https://github.com/facebookresearch/sam3) and NVIDIA DeepStream 9. One export script (`export.py`) with three modes for getting a prompt into SAM3 -- `set-classes` (classes baked at export time), `open-text` (prompt supplied at runtime) and `prompt-free` (class-less: the SAM3 tracker segments everything from a grid of points) -- and a `deepstream/` folder with everything that runs the engines (pipelines, configs, the custom `nvinfer` plugin). Every engine outputs detections and masks. Generated engines go into `weight/` at the repo root.

Docs, all English: `README.md` (users), `deepstream/GUIDE.md` (running the pipelines), `docs/NOTES.md` (what was tried and measured, and why; read the matching section before changing precision, the DeepStream setup, the query count, prompt-free or the box fix), `research/BOX_SHRINK.md` (evidence for the imgsz fix). Do not add more Markdown files: extend these.

## Layout

- `export.py`: every engine (`--mode set-classes | open-text | prompt-free`) and the SAM3 modules it traces.
- `deepstream/`: pipelines, configs, the `nvinfer` plugin source, `trt11/` (DeepStream 9.2 for TensorRT 11 + Python 3.14), `test/` (parser unit test).
- `test/`: checks of the product: `golden.py` + `golden/`, `infer_prompt_free.py`, `eval_prompt_free.py`, `bench_stages.py`, `rtsp_serve.py`; datasets in `test/data/` (git-ignored).
- `research/`: investigation scripts, logs and results (box shrink, probes, `bf16_probe.py`, `prune_probe.py`, `prune_search.py`, `compare_engines.py`, `prompt_free_levers.py`). They import `export.py` and read `test/data/` and `test/bus.jpg`; run them from the repo root.
- `weight/` (engines, `timing.cache`), `output/` and `deepstream/output/` (videos): git-ignored.

## Setup

No pinned `requirements.txt` — install latest torch/transformers/tensorrt matching your CUDA/TensorRT install, into the project venv (`.venv` is a symlink to `~/.dev/venvs/SAM3-Edge`, created by `/mnt/c/Dev/env/mkproj`; never `--break-system-packages`):
```bash
.venv/bin/pip install torch torchvision numpy opencv-python pillow transformers einops ftfy regex iopath onnx onnxconverter-common tensorrt
```
`Sam3Model`/`Sam3Processor` ship inside `transformers` itself — no separate `facebookresearch/sam3` install needed.

Also requires (not pip-installable): TensorRT (`trtexec`, `libnvinfer`; 11.4 on the dev machine), CUDA 13, `ffmpeg`, and for the pipelines DeepStream 9 (`pyservicemaker`). Stock DeepStream 9.2 is built for TensorRT 10.16 + Python 3.12; on the dev machine it is rebuilt for TensorRT 11 + the project's `.venv` (below). An engine only loads in the TensorRT major version that built it: the ONNX is what travels. All `gpu-id` properties in DeepStream configs must stay `0`; select the physical GPU via `CUDA_VISIBLE_DEVICES=N` instead.

## Rules

- **One environment:** TensorRT 11 and the project's `.venv` (Python 3.14) for everything, DeepStream included. Never add a second TensorRT or Python for DeepStream: the user rejected that.
- **FP16 only**, engines and PyTorch tests alike (`model.half()`, pixel scaling in fp32). Never FP32, never `torch.autocast` over FP32 weights (spills out of the 4 GB GPU). BF16 was tried and dropped. Details: `docs/NOTES.md` section 2.
- Research scripts that require `HF_TOKEN` run from the local cache with `HF_TOKEN=offline HF_HUB_OFFLINE=1`; never read `.env`.
- Set `HF_HUB_OFFLINE=1` when exporting from the local HF cache (`[Load]` 20-36 min -> ~15 s). The machine has 7 GB RAM: the export process can be OOM-killed while `trtexec` runs (the child keeps going, check `sam3.engine`).
- `run_trtexec` reuses `weight/timing.cache`; delete it after a TensorRT or driver change.
- After a prompt swap, tracks from before it keep their old class number for a few seconds: accepted by the user, do not add code for it.
- There is deliberately no `--num-queries` option (`docs/NOTES.md` section 5).
- `--prune` (off by default; the user tried it as the default and went back) leaves 16 ViT sub-blocks out (`export.PRUNE_VIT`) and prints a warning: 20% faster; at 420 AP@0.5 0.653 -> 0.623 and an object loses about 0.07 of score, at 700 AP 0.832 -> 0.823. Engine folder suffix `_pruned`. `test/golden/person_420.npz` is the unpruned engine. Never cut DETR encoder or decoder blocks (scores collapse, nothing gained): `docs/NOTES.md` section 8. Compare engines with `research/compare_engines.py`. Timings taken right after a series of builds are wrong on this laptop (the GPU throttles above ~75 C): let it cool first.
- `--imgsz` is trimmed down to a multiple of 14 with a warning (the pipelines do the same with the `imgsz` of their config).

## Common commands

Export an engine (every mode writes detections and masks; the imgsz 71/72 grid fix applies to all of them):
```bash
python3 export.py --mode set-classes --classes truck car --imgsz 420   # classes baked -> weight/sam3_420_set-classes_truck_car/
python3 export.py --mode open-text --imgsz 420 --k-max 5               # prompt at runtime -> weight/sam3_420_open-text/ (once, ever)
python3 export.py --mode prompt-free --imgsz 420                       # class-less: tracker masks from a grid of points, batch 1
```

DeepStream, on the dev machine (`docs/NOTES.md` section 3 explains every step of the setup):
```bash
bash deepstream/trt11/setup.sh                                  # once, no sudo: DeepStream 9.2 in ~/.dev/SAM3-Edge-deepstream, rebuilt for TensorRT 11 + .venv, plus the SAM3 plugin
source deepstream/env_local.sh
cd deepstream && ../.venv/bin/python ds9_set_classes.py         # reads config_set_classes.txt ($DS_CONFIG overrides); edit the config live to hot-swap prompts
cd deepstream && ../.venv/bin/python ds9_open_text.py           # reads config_open_text.txt
python3 test/rtsp_serve.py test/data/pedestrians_2min.mp4 2     # fake cameras rtsp://127.0.0.1:8554/cam0..1 (system python: uses libgstrtspserver); use rtspt:// URLs
```
On a stock DeepStream install (`/opt/nvidia/deepstream/deepstream`, TensorRT 10) skip the first two lines, build the plugin with the command in `deepstream/GUIDE.md` and build the engines there: an engine only loads in the TensorRT major version that built it.

Checks (there is no automated suite):
```bash
python3 test/golden.py check weight/<folder>/sam3.engine 420 test/golden/person_420.npz    # a rebuilt set-classes `person` engine against saved detections
g++ -std=c++20 -I deepstream/test deepstream/test/test_parser.cpp -o deepstream/test/test_parser && deepstream/test/test_parser   # parser logic, no DeepStream needed; delete the binary after
python3 test/infer_prompt_free.py weight/sam3_420_prompt-free/sam3.engine 420 test/bus.jpg out.jpg [--min-area 0.003] [--max-area 1.0]
cd test && python3 eval_prompt_free.py ../weight/sam3_420_prompt-free/sam3.engine 420 200   # COCO recall, needs data/coco
python3 test/bench_stages.py --imgsz 420 --batch 1 2 4                                      # per-stage throughput
```
Refactors of `export.py` are checked by comparing the PyTorch forward of the old and new code on the same weights (bit-identical), then `golden.py check` on a rebuilt engine. The DeepStream side is checked by running the pipelines (file or fake-RTSP sources) and looking at the detections and the output video.

## Architecture

### Three ways to give SAM3 a prompt

**`--mode set-classes`** — classes are baked into the ONNX graph at export time as constants. The resulting engine takes only an image as input and outputs detections and masks, running natively in `nvinfer` exactly like a YOLO model. Changing classes requires re-exporting (or hot-swapping to a different already-built engine, see Multi-camera union-prompt pattern below). Multiple classes per camera are supported (cost scales with class count — see `Sam3Wrapper` below).

**`--mode open-text`** — open-vocabulary: the prompt is encoded into a tensor at runtime instead of being baked in (a separate `text_encoder.engine` + `vision_decoder.engine`, built once, never re-exported). `deepstream/ds9_open_text.py` is a real, working camera pipeline: each camera gets a comma-separated class list (`prompt = car, truck`; up to `K_MAX=5` different classes over all cameras — baked into the engine by `export.py --k-max`, must match `sam3open_text.K_MAX`), and up to `MAX_SOURCES=4` cameras (matching the engine's max profile batch — `sam3open_text.py`) are scored in the SAME `nvinfer` forward pass against the union of their classes, then filtered per camera. Each class gets its own text embedding (slot k of every row); the ViT runs once per image, the K_MAX slots ride the batch axis through the DETR decoder (decoder cost is paid for all K_MAX slots of every camera, used or not; unused slots have an all-False `text_mask` and score 0).

**`--mode prompt-free`** — no text and no class: a grid of points (`--grid-stride`, 2 -> one per 2x2 patches, 225 points at 420) goes through the SAM3 tracker (`tracker_model` / `tracker_neck` in the `facebook/sam3` checkpoint, which `Sam3Model` drops; `Sam3TrackerModel.from_pretrained` loads them with the renames built in, ViT included). Like SAM2's automatic mask generator it returns every region it can segment, objects and stuff (road, wall, grass) alike, and nothing says which is which. One engine `sam3.engine`, batch 1 (static); the ViT is the detector's, the neck/prompt encoder/mask decoder are the tracker's (~12M). The prompt embeddings of the fixed points are constants; all 3 masks of every point are returned at the decoder's own resolution (N = 675 masks of 120 x 120 at 420; keeping only the best one at 36 x 36 gave recall 0.35 instead of 0.62 at the same latency); box = the extent of its mask, score = the tracker's predicted IoU, cls_id 0. 122 ms at batch 1 at imgsz 420 on the 3050 Ti (set-classes: 84 ms). Not wired to DeepStream; no golden.

The prompt-free output is raw and must be post-processed on the host (`test/infer_prompt_free.py: postprocess`); do not filter on score. Measurements: `docs/NOTES.md` section 6.

### Engine contract (what a refactor must not change)

Every mode (set-classes, open-text and prompt-free) keeps the same I/O so the DeepStream side and the Python runtimes do not care which one built the engine:

- **Input** `pixel_values` fp32 `[B,3,S,S]` (fp16 inside; engines exported before 2026-11-12 took fp16 and give no detections in DeepStream), RGB, resized straight to S x S (no letterbox), `(x - 127.5) / 127.5` (nvinfer: `net-scale-factor=1/127.5`, `offsets=127.5`). S is square and a multiple of 14 (`export.py` trims `--imgsz` down to one).
- **Output** `detections` fp16 `[B,N,6]` = `x1 y1 x2 y2 score cls_id`, in pixels of the S x S input (the parser rescales). N = classes x 200 (set-classes) or K_MAX x 200 (open-text); `cls_id = i // 200` is the class slot. Boxes are clamped to the frame.
- **Output** `masks` fp16 `[B,N,Hm,Wm]`, probabilities in [0,1], row i belongs to `detections` row i (always present; 36 x 36 for set-classes and open-text, `S / 14 * 4` for prompt-free). `open-text` also takes fp16 `text_features [B,K,32,256]` and a bool `text_mask [B,K,32]`.
- **Check** a rebuilt engine against the saved one: `python test/golden.py check weight/<new>/sam3.engine 420 test/golden/person_420.npz` (bus.jpg, 8 coco128 images and 60 val2017 images, class `person`, 190 golden boxes with score >= 0.5; exit 1 if one is lost (IoU < 0.7 or score drift > 0.25) or the median IoU falls under 0.98 or the p95 score drift passes 0.05; a plain rebuild of the same ONNX gives min IoU 0.82, median 0.996, p95 drift 0.019, so rebuilt engines pass but a moved box does not). `save` writes a new golden from an engine you trust.

### `export.py`'s export pattern

SAM3 is decomposed into ONNX-traceable submodules (`_VisionEncoder`/`_VisionEncoderFull`, `_Decoder`/`_DecoderFull`) rather than exporting the HuggingFace model directly, because the stock model hard-fails at any resolution other than 1008 (missing dynamic RoPE/position-embedding resize logic). `_VisionEncoder.__init__` rebuilds each ViT layer's `rotary_emb` for the target `Hp,Wp` and re-tiles the absolute position embedding before tracing — this is what makes `--imgsz` other than 1008 possible at all (must still be a multiple of `patch_size`, square). Both the `set-classes` and the `open-text` wrapper use them (`--imgsz` default 420; 420 verified end-to-end on 2 cameras). `deepstream/config_open_text.txt` `[model] imgsz` must match the engines' export size.

`Sam3Wrapper` (set-classes) bakes N classes as ONNX constants and calls `DynamicVisionDecoderWrapper` (the open-text wrapper: text features as inputs) with them; (`text_feat_all`/`text_mask_all` buffers); the class slots fold into the batch axis: the ViT runs once, its outputs are repeated to `B*N` rows (row `b*N+i` = image b × class i), and the DETR decoder runs once over all rows, then reshapes back to `[B, N*Q, ...]` (class-major, `cls_id = i // Q`). FLOPs of the decoder still scale with N (the DETR encoder fuses text per prompt) but there is no per-class graph unroll, and activation memory scales with `B*N`.

Precision: TensorRT 11 is strongly typed, so the FP32 graph is exported and converted with `onnx_to_fp16` (outputs and text inputs become fp16, `pixel_values` stays fp32 with a `Cast` behind it because `nvinfer` only writes float32 images). `docs/NOTES.md` section 1.

### Multi-camera union-prompt pattern (set-classes)

`deepstream/spec_cache.py` lets cameras with different prompts share one engine: it computes the sorted union of every camera's requested classes, builds (or reuses) one engine in `weight/` baked with that union via `export.py`, and `PerCameraFilter` (in `ds9_set_classes.py`) filters detections back down per-camera by `cls_id` afterward. Engine folders are named `sam3_{imgsz}_set-classes_{sorted-union-classes}/` so the same class set always resolves to the same folder regardless of which camera declared it first.

Because `nvinfer`'s `config-file-path` property is live-swappable while the pipeline runs (as long as input resolution is unchanged — verified experimentally), this same union-prompt engine can be hot-swapped at runtime when a camera's prompt changes, without restarting GStreamer. `ds9_set_classes.py`'s watcher thread implements this: it polls `config_set_classes.txt` for changes, recomputes the union, builds a new engine if that union hasn't been seen before, and calls `nvinfer.set({"config-file-path": ...})`.

### Prompt injection (open-text)

`vision_decoder.engine` takes `pixel_values`/`text_features`/`text_mask` as inputs (batch dim = camera count, matching `nvstreammux`). `nvinfer` handles `pixel_values` (the image) automatically like any detector; `text_features`/`text_mask` are non-image input layers, injected by `deepstream/nvdsparsebbox_sam3.cpp`'s `NvDsInferInitializeInputLayers` — a fixed symbol name `nvinfer` looks up via `dlsym` (a custom name is silently never called; verified empirically). It reads two flat snapshot files written by `deepstream/sam3open_text.py`:
```
/dev/shm/sam3open_text_features.bin   MAX_SOURCES x K_MAX x 32 x 256  float16 (the engine's dtype: the plugin copies the bytes as they are)
/dev/shm/sam3open_text_mask.bin       MAX_SOURCES x K_MAX x 32        uint8   (TRT reports BOOL; DeepStream maps it to UINT8)
```
every row holds the same slots, slot *k* = the k-th class of the union of all cameras' classes (every batch row gets the same class slots, the union of all cameras' classes (at most `K_MAX` different classes over all cameras), and the pipeline keeps each camera's own classes by `cls_id`: `nvstreammux` does not keep camera b in batch row b, so per-row prompts scored cameras against each other's classes (fixed 2026-11-12)); the encodings come from `text_encoder.engine` (run directly via TensorRT in Python — not through `nvinfer`). This function is called once per `nvinfer` context (re)load, not per frame — so a prompt change needs a snapshot rewrite followed by `nvinfer.set({"config-file-path": <SAME path>})`, which reloads the context and re-triggers it (verified: reloading the *same* config path re-reads the snapshot files). No TensorRT engine rebuild is ever needed for a prompt change. `ds9_open_text.py`'s watcher thread implements this the same way `ds9_set_classes.py`'s does for the baked engine (see Multi-camera union-prompt pattern above).

`deepstream/nvdsparsebbox_sam3.cpp`'s `NvDsInferParseSam3OpenDet` reads `detections [K_MAX*200,6]` (float32 or float16, by the layer's dtype) (`x1 y1 x2 y2 score cls_id`, pixel units); `cls_id` is the slot in that union (NMS only suppresses within the same slot), and `ds9_open_text.py`'s `PromptLabeler` labels a detection with the union's class name if the camera asked for that class and hides it otherwise (no fixed `labels.txt`).

### The two imgsz != 1008 box fixes

The checkpoint was trained at imgsz 1008 only: a 72 x 72 grid of feature cells (14 px each). Run at another grid H, the decoder is outside the one geometry it knows and the boxes come out shifted toward the top-left and too small, smoothly in H (width -10% at 420, -13% at 336, +1.5% at 1260). `export.py` puts the decoder back into the trained geometry with two patches, applied at **module import time**, baked into every exported engine, both the identity at H = 72:

1. **Grid (centre).** `Sam3DetrDecoder._get_coords` returns `arange(0,H)/(H-1) * 71/72`: the cell coordinates end at the trained 71/72 for any H. This fixes the box *centre*.
2. **Size.** The decoder learned that a box of normalized width w covers w x 72 cells. At grid H it covers w x H cells, so the size the query is told (the w, h part of the sine embedding of its reference box, `Sam3SinePositionEmbedding.encode_boxes`) is given in cells / 72: `w x H/72`, `h x H/72`. The centre is left alone. The factor is the ratio of the two grids, nothing is tuned. Switch: `export.BOX_SIZE_FIX` (research scripts turn it off to test their own variants).

Evidence, measurements (AP@0.5 0.49 -> 0.65 at 420 with patch 2) and what was rejected: `docs/NOTES.md` section 7 and `research/BOX_SHRINK.md`. Read them before touching either patch. A change of either needs a new `test/golden/person_420.npz`.
