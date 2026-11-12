# Running the DeepStream pipelines

Two pipelines, one per prompt mode. Both read RTSP (or `file://`) sources, run one shared `nvinfer` + NvSORT tracker, and write an annotated H264 mp4 to `output/`. Both reload prompts while running when you edit their config file.

| | `ds9_set_classes.py` | `ds9_open_text.py` |
|---|---|---|
| Engine | `export.py --mode set-classes`: classes baked in, one engine per class set | `export.py --mode open-text`: built once, prompts are runtime inputs |
| Config | `config_set_classes.txt` | `config_open_text.txt` |
| Changing a prompt | another engine (built on demand, minutes) or, if the union of classes is unchanged, only the per-camera filter | text features re-encoded and reloaded (well under a second) |
| Limits | cost grows with the number of classes in the union | up to `K_MAX` = 5 different classes over all cameras, up to `MAX_SOURCES` = 4 cameras |

## 1. Environment

**This repo's setup: DeepStream 9.2 with TensorRT 11.4 and Python 3.14, no sudo.** DeepStream 9.2 is released for TensorRT 10.16 and Python 3.12; `trt11/setup.sh` unpacks it into `~/.dev/SAM3-Edge-deepstream`, rebuilds the two libraries and the Python binding that depend on those versions, and builds the SAM3 plugin. Why each step is needed: [docs/NOTES.md](../docs/NOTES.md), section 3.

```bash
bash deepstream/trt11/setup.sh        # once, from the repo root; about a minute after the 1.4 GB download
source deepstream/env_local.sh        # in every shell that runs a pipeline
```

**A stock DeepStream install** (`/opt/nvidia/deepstream/deepstream`, its own TensorRT 10 and Python 3.12) also works: build the engines on that machine (an engine only loads in the TensorRT major version that built it) and build the plugin yourself:

```bash
cd deepstream && g++ -shared -fPIC -O2 -std=c++20 -o libnvdsinfer_sam3.so nvdsparsebbox_sam3.cpp \
    -I/opt/nvidia/deepstream/deepstream/includes -I/usr/local/cuda/include
```

All `gpu-id` properties must stay `0`; pick the GPU with `CUDA_VISIBLE_DEVICES=N`. On WSL2 set `DS_NVBUF_MEMTYPE=1` (`env_local.sh` does). `DS_ROOT` overrides the DeepStream path, `DS_CONFIG` the config file.

## 2. set-classes

```bash
python3 export.py --mode set-classes --classes adult child phone --imgsz 420      # from the repo root
# -> weight/sam3_420_set-classes_adult_child_phone/ : sam3.onnx, sam3.engine, config_infer.txt, labels.txt
```

`config_set_classes.txt`:

```ini
[sources]
urls =
    rtsp://user:pass@192.168.1.10:554
    rtsp://user:pass@192.168.1.11:554

[model]
imgsz = 420
classes = adult, child, phone   # for cameras with no [camN] section
conf = 0.4

[cam0]
classes = adult, phone

[cam1]
classes = child

[pipeline]
width = 1920
height = 1080
infer_interval = 4
run_seconds = 60
```

```bash
cd deepstream && ../.venv/bin/python ds9_set_classes.py
```

All cameras share one engine baked with the union of their classes (sorted, so the same set always maps to the same `weight/` folder); the pipeline then keeps each camera's own classes. If the engine for a union does not exist it is built by calling `export.py` (minutes; export it beforehand on a small GPU).

## 3. open-text

```bash
python3 export.py --mode open-text --imgsz 420        # once; -> weight/sam3_420_open-text/{text_encoder,vision_decoder}.{onnx,engine}
```

`config_open_text.txt`:

```ini
[sources]
urls =
    rtsp://user:pass@192.168.1.10:554
    rtsp://user:pass@192.168.1.11:554

[cam0]
prompt = forklift, person

[cam1]
prompt = spilled liquid on the floor

[model]
conf = 0.3
imgsz = 420          # must match the exported engines

[pipeline]
width = 1280
height = 720
infer_interval = 5
run_seconds = 60
```

```bash
cd deepstream && ../.venv/bin/python ds9_open_text.py
```

Each camera gets a comma-separated class list. Every frame is scored against the union of all cameras' classes (at most `K_MAX` = 5 different classes, `export.py --k-max`, must match `sam3open_text.K_MAX`) and each camera keeps its own. Unused class slots still cost decoder time. More than `MAX_SOURCES` = 4 cameras needs an engine exported with a bigger `--max-batch`. Editing the config while running re-encodes the new prompts and reloads `nvinfer`'s context; the engine never changes.

## 4. Config fields

| Field | Meaning |
|---|---|
| `urls` | one source per line: `rtsp://`, `rtspt://` (RTSP over TCP, use it if the picture breaks up) or `file:///abs/path.mp4` |
| `imgsz` | SAM3 input size of the engine: square, trimmed down to a multiple of 14 (420 is the light size, 1008 the trained one) |
| `conf` | score threshold |
| `classes` / `prompt` | comma-separated class names; under `[model]` the fallback (set-classes), under `[camN]` that camera's |
| `width` / `height` | pipeline resolution, every source is scaled to it |
| `infer_interval` | infer every N+1 frames, the tracker fills the rest |
| `run_seconds` | stop after N seconds; 0 = until Ctrl+C |
| `output_file` | default: a timestamped file in `output/` |

Do not commit a config that holds camera credentials.

## 5. Good to know

- **The resolution is fixed at export.** Only the batch is dynamic: the exporter rebuilds the ViT's rotary embeddings and re-tiles the position embedding for one grid size at trace time, so another `imgsz` means another export, not another TensorRT profile.
- **After a prompt change**, tracks from before it keep their old class number for a few seconds, so labels can be wrong during the transition.
- **Testing without cameras:** `python3 test/rtsp_serve.py test/data/video.mp4 2` serves `rtspt://127.0.0.1:8554/cam0` and `cam1` from an H264 mp4 (system Python, it uses `libgstrtspserver`).
- **Parser unit test** (no DeepStream needed): `g++ -std=c++20 -I deepstream/test deepstream/test/test_parser.cpp -o deepstream/test/test_parser && deepstream/test/test_parser`.
- Encoder, tracker and other pipeline pitfalls that cost time once: [docs/NOTES.md](../docs/NOTES.md), section 9.
