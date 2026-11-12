# SAM3-Edge

Real-time open-vocabulary detection and instance segmentation on multi-camera RTSP streams with [SAM3](https://github.com/facebookresearch/sam3), TensorRT and NVIDIA DeepStream.

`export.py` turns SAM3 into TensorRT engines (FP16, any square input size; it is trimmed to a multiple of 14); `deepstream/` runs them on cameras and writes annotated video.

**Tested on:** DeepStream 9.2 + pyservicemaker, TensorRT 11.4, Python 3.14, CUDA 13, in one environment (RTX 3050 Ti 4 GB, WSL2). Both pipelines run on this stack, on file and RTSP sources, with prompts changed while running. DeepStream 9.2 is released for TensorRT 10.16 + Python 3.12: `deepstream/trt11/setup.sh` rebuilds the parts of it that depend on those versions (no sudo).

## Which mode do I want?

| | `set-classes` | `open-text` | `prompt-free` |
|---|---|---|---|
| Prompt | baked into the engine at export | free text, supplied at runtime | none: the tracker segments everything from a grid of points |
| Classes | any number (cost grows with it) | up to `K_MAX` (5) different classes over all cameras | class-less masks |
| Changing the prompt | another engine (re-export, or hot-swap to one already built) | edit the config, no re-export ever | n/a |
| DeepStream pipeline | `ds9_set_classes.py` | `ds9_open_text.py` | none yet (batch 1, post-process on the host) |

## Quick start

```bash
pip install torch torchvision numpy opencv-python pillow transformers einops ftfy regex iopath onnx onnxconverter-common tensorrt   # in a venv; SAM3 ships inside transformers
python3 export.py --mode set-classes --classes truck car --imgsz 420   # -> weight/sam3_420_set-classes_truck_car/
python3 export.py --mode open-text --imgsz 420                         # -> weight/sam3_420_open-text/   (once, ever)
python3 export.py --mode prompt-free --imgsz 420                       # -> weight/sam3_420_prompt-free/

bash deepstream/trt11/setup.sh && source deepstream/env_local.sh       # DeepStream 9.2 for TensorRT 11 + this Python, once
cd deepstream && ../.venv/bin/python ds9_set_classes.py                # or ds9_open_text.py; edit the config file first
```

Also needed: TensorRT (`trtexec`), CUDA, `ffmpeg`. SAM3.1 (`sam3.1_multiplex.pt`) has a different architecture and is not supported. Configs, a stock DeepStream install, hot-swapping prompts and testing without cameras: **[deepstream/GUIDE.md](deepstream/GUIDE.md)**.

Every engine has the same interface: input `pixel_values` fp32 `[B,3,S,S]` (RGB, plain resize, `(x - 127.5) / 127.5`); outputs `detections` fp16 `[B,N,6]` = `x1 y1 x2 y2 score cls_id` in input pixels and `masks` fp16 `[B,N,h,w]` probabilities (36x36; 120x120 for prompt-free at 420).

## How it works

**set-classes.** The text embeddings of the classes are ONNX constants, so the engine takes only an image and runs in `nvinfer` like any detector. Cameras with different classes share one engine baked with the union of all of them (`spec_cache.py`); the pipeline keeps each camera's own classes. Because `nvinfer`'s `config-file-path` can be swapped while running, a prompt change moves the pipeline onto another engine without a restart.

```
export.py --mode set-classes   (once per class set)
  -> weight/sam3_{imgsz}_set-classes_{classes}/sam3.engine

ds9_set_classes.py
  N x nvurisrcbin -> nvstreammux -> nvinfer (interval) -> nvtracker
    -> PerCameraFilter (each camera's own classes) -> tiler -> nvdsosd -> nvv4l2h264enc -> mp4
```

**open-text.** The prompt is a runtime input: a text-encoder engine (run from Python when a prompt changes) and a vision-decoder engine (run by `nvinfer`). The plugin injects the text features as non-image input layers; every frame is scored against the union of all cameras' classes and each camera keeps its own. A prompt change rewrites the features and reloads `nvinfer`'s context; no engine is ever rebuilt.

```
export.py --mode open-text   (once)
  -> text_encoder.engine      input_ids + attention_mask -> text_features + text_mask
  -> vision_decoder.engine    pixel_values + text_features + text_mask -> detections + masks

ds9_open_text.py
  sam3open_text.py encodes the classes, writes them to /dev/shm
  N x nvurisrcbin -> nvstreammux -> nvinfer (text features injected by nvdsparsebbox_sam3.cpp) -> nvtracker
    -> PromptLabeler (keeps and labels each camera's own classes) -> tiler -> nvdsosd -> nvv4l2h264enc -> mp4
```

**prompt-free.** One point prompt per grid cell (`--grid-stride`, 225 points at imgsz 420) goes through the SAM3 tracker's mask decoder; its 3 masks per point (part, object, whole) come back at 120x120 with a box and a predicted IoU. Nothing says which masks are objects and which are background, so post-process on the host (`test/infer_prompt_free.py`). On COCO val2017 (200 images) the engine at imgsz 420 has a mask of IoU >= 0.5 for 62% of the ground-truth instances before the post-process and 48-57% after it, at 122 ms per frame; an eager run at imgsz 700 reached 74% (not built as an engine).

**Speed knobs.** SAM3 was trained at 1008 only. The exporter rebuilds the position embeddings for the chosen `--imgsz` and puts the decoder back into the trained 72 x 72 geometry (cell grid and box size), so boxes stay centred and the right size. Smaller is much faster (1008 -> 420 is about 2.9x) and loses small objects; below 280 is not recommended. `--prune` leaves 16 ViT sub-blocks out for another 20% (AP@0.5 0.653 -> 0.623 at 420, 0.832 -> 0.823 at 700: a pruned 700 engine is far more accurate than a full 420 one).

Measurements, dead ends and the reasons behind these choices: [docs/NOTES.md](docs/NOTES.md).

## Layout

```
SAM3-Edge/
├── export.py        # SAM3 -> ONNX -> TensorRT, --mode set-classes | open-text | prompt-free
├── deepstream/      # pipelines, configs, the nvinfer plugin source, GUIDE.md, trt11/ (DeepStream 9.2 for TensorRT 11 + Python 3.14)
├── test/            # golden check of an engine, prompt-free inference and COCO recall, per-stage benchmark, fake RTSP cameras
├── research/        # investigation scripts, logs and results (box shrink at other input sizes, probes)
├── docs/NOTES.md    # what was tried and measured, and why
└── weight/, output/ # engines and videos (not in git)
```

## Updates

**2026-11**
- One `export.py` for everything (`--mode set-classes | open-text | prompt-free`); every engine outputs detections and masks. `SAM3Fixed/` and `SAM3Open/` are gone.
- New mode `prompt-free`: class-less masks from a grid of points through the SAM3 tracker (COCO val2017 recall 0.62 at imgsz 420).
- Boxes are the right size at any imgsz: the decoder is put back into its trained 72 x 72 geometry (grid and box size). `person` @420 on COCO val2017: AP@0.5 0.49 -> 0.65. Re-export to get it.
- `--prune`: 16 ViT sub-blocks left out, 20% faster, slightly lower precision and recall (off by default).
- Runs on DeepStream 9.2 + TensorRT 11.4 + Python 3.14 (`deepstream/trt11/setup.sh`, no sudo). Engines take an fp32 image again (fp16 input gave no detections in `nvinfer`).
- Fixed: open-text labels with different prompts per camera (`K_MAX` is now shared by all cameras), the set-classes prompt hot-swap.
- `--imgsz` is trimmed down to a multiple of 14. Docs merged into `deepstream/GUIDE.md` and `docs/NOTES.md`, research moved to `research/`, MIT license.

**2026-10**
- Box-centre fix at imgsz != 1008 (decoder grid ends at the trained 71/72).
- open-text: a class list per camera (`--k-max`). Configurable `imgsz` in the pipelines. Renamed to SAM3-Edge.

**2026-08**
- Input sizes other than 1008 (square, multiple of 14; not recommended below 280).
- Engines in one shared `weight/` folder. Prompts can be changed while the pipeline runs.

**2026-06 / 07**
- First release, dynamic batch, score gated by the presence token.

## AI Disclosure

Parts of the code, the tests and the documentation in this repository were written with the help of an AI coding assistant (Claude Code), under the author's direction and review. The measurements quoted here come from runs on the author's machine.

## License

The code in this repository is released under the [MIT License](LICENSE). It contains no model weights: SAM3 and its checkpoints are Meta's and come under the [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE); DeepStream, TensorRT and CUDA are NVIDIA's and come under their own licenses (`deepstream/trt11/` patches your local copy of the DeepStream sources and redistributes none of them).

This is a personal project, not affiliated with or endorsed by any company.
