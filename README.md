# SAM3-Edge

Real-time open-vocabulary detection and instance segmentation on multi-camera RTSP streams using [SAM3](https://github.com/facebookresearch/sam3) and NVIDIA DeepStream 9.

**Workflow:** `python3 export.py --mode set-classes|open-text|prompt-free ...` → write `deepstream/config_set_classes.txt` or `deepstream/config_open_text.txt` → run `deepstream/ds9_set_classes.py` or `deepstream/ds9_open_text.py` → saves annotated mp4. Setup steps live in [deepstream/GUIDE_set_classes.md](deepstream/GUIDE_set_classes.md) and [deepstream/GUIDE_open_text.md](deepstream/GUIDE_open_text.md).

---

## Tested on

- NVIDIA GPU
- DeepStream 9 + pyservicemaker, CUDA 13, TensorRT 10

---

## Updates

<details open>
<summary><strong>2026-11-10</strong></summary>

- **One folder less, one script less.** `SAM3Fixed/` and `SAM3Open/` are gone: a single `export.py` takes `--mode set-classes` (classes baked, the old Fixed),
  `--mode open-text` (prompt supplied at runtime, the old Open) or `--mode prompt-free` (no text, no class; raw masks, post-process on the host, see CLAUDE.md) instead of the old `--mode det/seg`.
  Every engine now outputs detections **and** masks (36x36 per object); the bbox-only `det` mode is removed. Engine folders: `weight/sam3_{imgsz}_set-classes_{classes}/`
  and `weight/sam3_{imgsz}_open-text/`.
- The encoder/decoder modules and the imgsz 71/72 grid fix live once in `export.py` (applied at import, so it holds for every mode). One DeepStream plugin source
  (`deepstream/nvdsparsebbox_sam3.cpp`, C++20). All DeepStream files (pipelines, configs, tracker yml, `spec_cache.py`, `sam3open_text.py`, guides) are in `deepstream/`.
- `open-text` engines are now fp16 in and out like `set-classes` (TensorRT 11 has no `--fp16`); `sam3open_text.py` writes fp16 text features. Boxes are clamped to the frame in
  both modes (the grid fix can push edge boxes past it; `open-text` used not to clamp).
- **`--mode prompt-free`** (new): no text, no class. A grid of points goes through the SAM3 tracker (its weights are in `facebook/sam3` but `Sam3Model` drops them; `Sam3TrackerModel` loads them) and returns every region it can segment, objects and stuff alike, like SAM2's automatic mask generator. The engine output is raw: `test/infer_prompt_free.py` has the host post-process (min mask area 0.3% of the frame, NMS by mask IoU with the largest mask first, drop masks inside a kept one). Honest numbers on COCO val2017 (`test/eval_prompt_free.py`): the engine built at imgsz 420 finds 27% of the ground-truth instances (IoU >= 0.5; person 15%, medium objects 50%) with 27 masks per image. An eager study at imgsz 700, 24x24 points and all 3 masks per point reaches about 62%, but the engine does not do that yet (details in CLAUDE.md). Not wired to DeepStream.
- Every engine build now reuses `weight/timing.cache` (`--timingCacheFile`): rebuilding the same ONNX went from 159 s to 30 s.
- Checked here: the PyTorch forwards of `set-classes` and `open-text` are bit-identical to the old scripts, a rebuilt `set-classes` engine passes `test/golden.py`, and `open-text` builds both engines on TensorRT 11.4. Not run here: the DeepStream pipelines.

</details>

<details>
<summary><strong>2026-10-23</strong></summary>

- **Fix box shrink at imgsz != 1008.** The checkpoint was trained only at 1008 (72x72 cells); at other sizes boxes shrank toward the top-left
  (width -10% at 420). Both `SAM3Fixed` and `SAM3Open` now patch the decoder grid to end at the trained 71/72, baked into every exported engine
  (centre fix; 1008 is unchanged). A query-box scale (`sineq`, `imgsz/1008`) that also fixes the box size (width -10.4% -> -1.8% at 420) was tested
  but is not shipped yet while it is researched further.
  Evidence: `test/BOX_SHRINK_INVESTIGATION.md`. 

</details>

<details>
<summary><strong>2026-10-22</strong></summary>

- **SAM3Open per-camera class lists** (K class slots, `--k-max`, default 5); SAM3Fixed folds classes into the batch axis.
- Configurable `imgsz` in the pipelines, WSL2 mux fix, file-source timestamps, stable labels.
- Renamed to SAM3-Edge.

</details>

<details>
<summary><strong>2026-08-13</strong></summary>

- **Fix small-imgsz model degrading (>280 should work).** 
  Tiny-imgsz model like 280 on small object will still have high-error boxes
  due to feature loss while downsizing an image, so not recommend below 280.
- **Repository architecture reconstructed.** 
  See `SAM3Fixed/` and `SAM3Open/`.
- **All generated engines now live in one shared `weight/` folder** 
  `weight/sam3_{imgsz}_{mode}_{classes}/` for SAM3Fixed,
  `weight/sam3_{imgsz}_open/` for SAM3Open (det only).
- **SAM3 DeepStream pipeline OTF model update**
  Can choose between:
   + Live prompt change -> text tokenizer re-encode -> nvinfer reload -> DeepStream
   + Specialize another SAM3Detector -> change path -> nvinfer reload -> DeepStream


</details>

<details>
<summary><strong>2026-08-10</strong></summary>

- WxH other than 1008x1008 works now with conditions: W and H must be equal and must be 14-multiple. Smaller input size leads to less accurate bounding boxes.

  | imgsz | latency | qps | quality |
  |-------|---------|-----|---------|
  | 1008  | 179ms   | 5.6 | clean |
  | 504   | 55ms    | 18.2| clean |
  | 420   | 47.5ms  | 20.9| clean |
  | 350   | 41ms    | 24.0| clean |
  | 280   | 19.3ms  | 51.8| clean at conf≥0.6 (default conf bumped 0.5→0.6, --pad bumped 0.05→0.10) |

- `specialize.py` output folder renamed `sam3_{mode}_{classes}` → `sam3_{imgsz}_{mode}_{classes}`; `--mask 0/1` arg replaced by `--mode det/seg` (`config.txt`/`config_video.txt` field renamed `mask_mode` → `mode`, plus new `imgsz` field).

</details>

<details>
<summary><strong>2026-08-06</strong></summary>

- EfficientSAM3 works now.

</details>

<details>
<summary><strong>2026-07-20</strong></summary>

- Fixed detection score calculation: SAM3's final per-query score must be gated by the scene-level presence token (matching the HF `transformers` reference implementation).

</details>

<details>
<summary><strong>2026-06-26</strong></summary>

- Dynamic batch works now.

</details>

<details>
<summary><strong>2026-06-25</strong></summary>

- Initial commit and first update to the repo.

</details>

## Notes

- All `gpu-id` properties in DeepStream must be `0`; use `CUDA_VISIBLE_DEVICES=N` to select the physical GPU.
- SAM3.1 (`weights/sam3.1_multiplex.pt`) uses a different architecture and is not yet supported by `export.py`.

---

## Which one do I want?

| | [`set-classes`](deepstream/GUIDE_set_classes.md) | [`open-text`](deepstream/GUIDE_open_text.md) | `prompt-free` |
|---|---|---|---|
| Prompt | Baked into the engine at export time | Free text, supplied at runtime | None: the tracker segments everything from a grid of points (class-less, batch 1, not wired to DeepStream yet) |
| Classes per camera | Multiple | Up to `K_MAX` (5) | none: class-less masks (`cls_id` 0) |
| Changing the prompt | Re-export (or hot-swap to another already-built engine) | Just edit `config_open_text.txt`, no re-export ever | n/a |

Setup, config reference, and manual export/recompile steps for each are in their own guide: **[deepstream/GUIDE_set_classes.md](deepstream/GUIDE_set_classes.md)** and **[deepstream/GUIDE_open_text.md](deepstream/GUIDE_open_text.md)**.

---

## How it works

`export.py` builds the engines for every mode, `deepstream/` runs them: the pipelines, their configs, the tracker config and the custom `nvinfer` plugin. The encoder/decoder
modules and the imgsz fix are shared (`export.py`). Engines go into the `weight/` folder at the repo root. Every engine outputs detections and masks.

There are two ways to give SAM3 a prompt, and a third mode that needs none (described after them):

**`--mode set-classes`** — classes are baked into the graph at export time as ONNX
constants. The resulting engine takes only an image as input and outputs
detections directly, running natively in `nvinfer` exactly like a YOLO model.
Changing classes requires re-exporting (or swapping to a different
already-exported engine on the fly, see `spec_cache.py`).

```
export.py  (run once per class set)
  SAM3 ViT backbone + FPN + DETR decoder
    └─ text class embeddings baked as ONNX constants
    └─ mask head output: 36×36 fp16 per object
  → weight/sam3_{imgsz}_set-classes_{classes}/sam3.engine  (TRT FP16)

deepstream/ds9_set_classes.py  (runtime — live RTSP cameras)
  N× nvurisrcbin (RTSP)
    → nvstreammux → nvinfer (weight/ engine, interval=4)
        → nvtracker (propagates boxes to skipped frames)
            → PerCameraFilter (filters union detections down to each camera's own classes)
                → nvmultistreamtiler → nvdsosd (boxes + masks)
                    → nvv4l2h264enc → mp4mux → output/
```

`deepstream/spec_cache.py` adds a union-prompt layer on top: multiple cameras with
different prompts all share ONE engine baked (into `weight/`) with the
union of every class needed, with `PerCameraFilter` filtering detections back
down to what each camera actually asked for by `cls_id`. Because `nvinfer`'s
`config-file-path` property is live-swappable at runtime as long as input
resolution stays the same, this also lets `ds9_set_classes.py` hot-reload a running
pipeline onto a newly-built engine without a restart, by watching `config_set_classes.txt`
for changes.

**`--mode open-text`** — open-vocabulary: the text prompt is encoded into a
tensor at runtime (`export.py` produces a separate text-encoder engine +
vision-decoder engine) instead of being baked into the graph. Never requires
re-exporting to change the prompt.

```
export.py --mode open-text  (run once, ever)
  → weight/sam3_420_open-text/text_encoder.engine     input_ids+attention_mask → text_features+text_mask
  → weight/sam3_420_open-text/vision_decoder.engine   pixel_values+text_features+text_mask → detections+masks

deepstream/ds9_open_text.py  (runtime — live cameras)
  deepstream/sam3open_text.py encodes each camera's prompt via text_encoder.engine
  (Python/TensorRT, not through nvinfer), writes text_features/text_mask
  snapshot files (one row per camera) to /dev/shm

  N× nvurisrcbin (RTSP)
    → nvstreammux → nvinfer (vision_decoder.engine, interval=5)
        │   deepstream/nvdsparsebbox_sam3.cpp's NvDsInferInitializeInputLayers reads
        │   the snapshot files once per context load, injecting camera b's
        │   prompt encoding into batch row b -- so every camera is scored
        │   against its OWN prompt in the SAME forward pass
        → nvtracker → PromptLabeler (labels each camera's boxes from its own prompt string)
            → nvmultistreamtiler → nvdsosd → nvv4l2h264enc → mp4mux → output/
```

A prompt change only needs a snapshot rewrite + `nvinfer.set({"config-file-path":
<same path>})`, which reloads the context and re-triggers
`NvDsInferInitializeInputLayers` — verified empirically that reloading the
*same* config path re-reads the snapshot files. No TensorRT engine rebuild is
ever needed for a prompt change, unlike set-classes.

**`--mode prompt-free`** — the SAM3 tracker (the SAM2-style point-prompt mask decoder inside the checkpoint) gets one point prompt per grid cell (`--grid-stride`, 225 points at imgsz 420), keeps the best of its 3 masks per point and returns them with a box (the mask's extent) and the tracker's predicted IoU. One batch-1 engine, no text, no DeepStream plugin yet. Nothing says which masks are objects and which are background, so post-process on the host (`test/infer_prompt_free.py`). Filtering on score does not work (the predicted IoU favours parts over whole objects); on a static camera, keeping only the masks that overlap motion removes the background. See CLAUDE.md for the measurements.

---

## File structure

```
SAM3-Edge/
├── export.py                    # SAM3 → ONNX → TRT: --mode set-classes | open-text | prompt-free; detections + masks, fp16
├── bench_stages.py              # per-stage throughput: decode fps, text encoder, vision decoder, baked engine
├── weight/                      # generated engines (not in git)
│   ├── sam3_{imgsz}_set-classes_{classes}/   # onnx + engine + config_infer.txt + labels.txt
│   ├── sam3_{imgsz}_open-text/               # text_encoder + vision_decoder, onnx + engine
│   ├── sam3_{imgsz}_prompt-free/             # sam3.onnx + sam3.engine, batch 1
│   └── timing.cache                          # TensorRT timing cache shared by every build
├── deepstream/                  # everything that runs the engines in DeepStream (not testable without DeepStream)
│   ├── nvdsparsebbox_sam3.cpp   # the one nvinfer plugin: output parsers + the open-text input injector (build to libnvdsinfer_sam3.so, not in git)
│   ├── test/                    # parser unit test, no DeepStream needed
│   ├── ds9_set_classes.py       # multi-cam pipeline, per-camera classes, hot-reload
│   ├── ds9_open_text.py         # multi-cam pipeline, comma-separated class list per camera, hot-reload
│   ├── spec_cache.py            # union-prompt cache: build-or-reuse a weight/ engine for a class union
│   ├── sam3open_text.py         # text encoder + non-image-input snapshot writer
│   ├── config_set_classes.txt, config_open_text.txt   # sources + per-camera prompts
│   ├── config_tracker_NvSORT_i5.yml
│   ├── GUIDE_set_classes.md, GUIDE_open_text.md
│   └── output/                  # annotated mp4 outputs
└── test/                        # box-accuracy investigation, golden.py (refactor check), infer_prompt_free.py + eval_prompt_free.py (prompt-free post-process and COCO recall)
```

---

## Disclaimer

This is a personal project. It is not affiliated with, endorsed by, or connected to any company or organization. All work here represents individual research and experimentation.

---

## License

Derivative work of SAM3 by Meta. Distributed under the [SAM License](LICENSE).

