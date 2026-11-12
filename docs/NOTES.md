# Notes

What was tried, what was measured and why things are the way they are. Usage is in the [README](../README.md) and [deepstream/GUIDE.md](../deepstream/GUIDE.md); the box-shrink evidence has its own file, [research/BOX_SHRINK.md](../research/BOX_SHRINK.md). Dev machine for the numbers below unless noted: RTX 3050 Ti 4 GB, WSL2 Ubuntu 26.04, TensorRT 11.4, Python 3.14.

## 1. TensorRT 10 vs 11, and the fp32 image input

- An `.engine` only loads in the TensorRT major version that built it. The `.onnx` is what travels.
- TensorRT 11 is strongly typed: there is no `--fp16`, the dtype of each tensor in the ONNX is the precision of the engine. So `export.py` converts the graph to FP16 (`onnx_to_fp16`). TensorRT 10 defaults to weak typing and needs `--stronglyTyped` for the same result; `run_trtexec` always passes it (a no-op on 11).
- **The image input must stay fp32.** After the move to an FP16 ONNX, `pixel_values` was fp16 too. Python callers did not care, but `nvinfer` only writes float32 images: the pipeline ran with no error and no detections. How it was found: the engine alone gave the right boxes on `bus.jpg`, so the engine was fine; the `nvinfer` log showed `INPUT kHALF pixel_values`; an engine with an fp32 input and a `Cast` behind it gave 7 people per frame. `onnx_to_fp16` now keeps `pixel_values` fp32. `text_features` (open-text) stays fp16 because the plugin copies those bytes itself.
- Lesson: a change of an engine's I/O types has to be run through its real consumer. The README of 2026-11-10 said "not run here: the DeepStream pipelines", and that is where the bug was.

## 2. BF16 instead of FP16: no

| | sign | exponent | mantissa | range |
|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | ~3.4e38 |
| FP16 | 1 | 5 | 10 | 65504 |
| BF16 | 1 | 8 | 7 | ~3.4e38 |

BF16 is FP32 without its low 16 bits: no overflow, but 3 fewer mantissa bits than FP16. (TF32 is different: a compute mode inside the tensor cores, the data stays FP32.)

`research/bf16_probe.py` runs `export.py` with a BF16 converter (fp16 in and out, BF16 inside). On TensorRT 11.4: the first build failed because `ConvTranspose` has no BF16 implementation (kept as an fp16 island between two `Cast`s); the second built, and every output was NaN (190 of 190 golden boxes lost). A guess that the `-FLT_MAX` attention-mask fill rounded to `-inf` was wrong (no such constant in the graph). Not root-caused: the next step would be to mark intermediate outputs and find the first NaN layer.

| GPU compute time | batch 1 | batch 4 |
|---|---|---|
| FP16 | 84 ms | 318 ms |
| BF16 | 106 ms | 394 ms |

Even with the NaN fixed BF16 is 26% slower here, and FP16 does not overflow (golden check passes). Worth another look only for a model that overflows in FP16.

Precision rules that stand:

Always run and test in FP16 — TensorRT engines and PyTorch tests/benchmarks alike (`model.half()`, pixel scaling kept in fp32). Never use FP32, and never `torch.autocast` over FP32 weights: FP32 is only marginally more accurate and much slower, and on the 4 GB dev GPU (RTX 3050 Ti, sm_86) autocast keeps both weight copies and spills out of VRAM (8 s/forward vs 0.2 s for a real fp16 model). Do not spend time on FP32 comparisons. FP8 needs sm_89+ and cannot be built or run here.

TensorRT 11 is strongly typed: `trtexec` has no `--fp16`, the engine precision is the dtype in the ONNX (TensorRT 10 needs `--stronglyTyped` for the same behaviour; `run_trtexec` always passes it, a no-op on 11). `export.py` exports the FP32 graph (sdpa) and converts it with `onnx_to_fp16` (onnxconverter-common; outputs and text inputs become fp16 too, `pixel_values` stays fp32 with a `Cast` behind it because `nvinfer` only writes float32 images; no shape inference because it needs ~7 GB RAM, so leftover `Cast(to=fp32)` nodes are patched there), for the `set-classes` engine and for both `open-text` engines (`sam3open_text.py` and the plugin read fp16 text features). Engine `person` @420: 100/106 boxes, IoU 0.829, ~76 ms/frame, same accuracy as the old `--fp16` engines. Exporting a `.half()` model with `attn_implementation="eager"` also works (~69 ms) but sdpa's ONNX symbolic breaks an fp16 graph (fp32 mask constants).

`run_trtexec` passes `--timingCacheFile=weight/timing.cache` (unless `--trt-args` sets its own): a rebuild reuses the tactics timed before (the ViT part of a build went 110 s -> 42 s), delete the file after a TensorRT or driver change.

Set `HF_HUB_OFFLINE=1` when exporting from the local HF cache: the `[Load]` step drops from 20-36 min to ~15 s. The machine has 7 GB RAM; the export process can be OOM-killed while `trtexec` runs (the child keeps going, check `sam3.engine`).

## 3. DeepStream 9.2 on TensorRT 11 + Python 3.14, without sudo

DeepStream 9.2 is released for Ubuntu 24.04, TensorRT 10.16, CUDA 13.2, Python 3.12 (`pyservicemaker`), installed with apt and a root `install.sh`. The dev machine has Ubuntu 26.04, GStreamer 1.28, TensorRT 11.4, CUDA 13.5, Python 3.14, no sudo and no Docker. The requirement was one environment: no second TensorRT or Python next to the shared one. `deepstream/trt11/setup.sh` does all of the below.

Only a few files depend on the versions, and they fall in two groups (found with `ldd` and `nm -D --undefined-only`):

- **They use TensorRT's C++ objects**, whose layout changed between 10 and 11, so they must be recompiled: `libnvds_infer.so` (the core of `nvinfer`) and `libnvds_inferlogger.so` (the logger object handed to TensorRT). Both have sources in the SDK (`sources/libs/`).
- **They only name the library**: the closed tracker `libnvds_nvmultiobjecttracker.so` imports three C entry points (`createInferBuilder_INTERNAL`, `createInferRuntime_INTERNAL`, `createNvOnnxParser_INTERNAL`) that TensorRT 11 still has, and NvSORT never calls them. A symlink `libnvinfer.so.10 -> libnvinfer.so.11` is enough.

Step by step:

1. **No apt:** the SDK tarball on `github.com/NVIDIA/DeepStream/releases` (1.4 GB) unpacks anywhere.
2. **No root for `install.sh`:** its main job is to replace the system `libv4l2.so.0` with NVIDIA's `libnvv4l2.so` (the `nvv4l2` decoder and encoder go through it; without it: `libv4l2: error getting capabilities: Inappropriate ioctl for device`). A `shim/` folder with that symlink at the front of `LD_LIBRARY_PATH` does it, plus `LIBV4L2_PLUGIN_DIR` (found with `strings libnvv4l2.so`).
3. **GStreamer 1.28, CUDA 13.5:** nothing to do. The GStreamer 1.x ABI is stable and CUDA 13.x keeps the name `libcudart.so.13`.
4. **`nvinfer` against TensorRT 11:** about 30 compile errors, all in the build-an-engine-from-a-model path (`IInt8Calibrator`, `BuilderFlag::kFP16/kINT8`, `platformHasFastFp16`, `ITensor::setType`, `ILayer::setPrecision`, `kEXPLICIT_BATCH`), which TensorRT 11 removed because the precision now lives in the model. `patch_nvdsinfer.py` wraps those 14 blocks in `#if NV_TENSORRT_MAJOR < 11`. Loading a prebuilt engine is untouched.
5. **A segfault at start:** after step 4 the pipeline died when `nvinfer` started. There was no `gdb`; `apt-get download gdb` needs no root, `dpkg -x` unpacks it, `LD_LIBRARY_PATH` runs it. The backtrace: `initLibNvInferPlugins` jumping to address `0x49d`, a virtual call on an object with the wrong layout. The object was `gTrtLogger`, defined (`nm -D`) in `libnvds_inferlogger.so`, still the TensorRT 10 build. Rebuilding that library fixed it. (A hand-written SIGSEGV handler printed nothing because the crash was in a GStreamer thread: the real debugger was faster than guessing.)
6. **`pyservicemaker` for Python 3.14:** pybind11 2 does not support 3.14 and pybind11 3 refuses `keep_alive` on `def_property_readonly`. `patch_pyservicemaker.py` turns the three getters into `py::cpp_function`, as the error message asks. One `g++` command builds the binding, the package is copied into `.venv`.

Everything lives in `~/.dev/SAM3-Edge-deepstream` (1.8 GB) plus the `pyservicemaker` package in `.venv`; the two stock libraries are kept next to the new ones as `*.trt10`. The pipelines use the ordinary engines in `weight/`.

Run on 2026-11-12 (imgsz 420, interval 5 + NvSORT, two sources from `test/data/pedestrians.mp4`):

- `ds9_set_classes.py`, `file://`, class `person`: 7 people per frame, scores 0.82-0.92.
- `ds9_open_text.py`, `file://`, different prompts per camera: each camera shows its own classes (after the fix in section 4).
- Over RTSP (`test/rtsp_serve.py`), config edited while running. set-classes: cam0 `person` -> `dog` changes the union, `nvinfer` moves to the `dog_person` engine without a restart; cam1 -> `dog, person` leaves the union alone and only updates the filter. open-text: cam1 `bench, person` -> `bench` only updates the filter; cam0 -> `potted plant, person` rewrites the text features and reloads the context (0.05 s).
- For a few seconds after a swap, tracks from before it keep their old class number, so labels can be wrong in the transition. Accepted, not fixed.
- With `rtsp://` (UDP) the picture broke up and boxes were lost: the fake cameras drop UDP packets on loopback. `rtspt://` (RTSP over TCP) is clean.

Not run: real cameras; masks (not visible in the output videos, old ones included); `spec_cache` building a missing class union while the pipeline runs (it calls `export.py --device cuda`, takes minutes and may not fit a 4 GB GPU). Plugins that are not used and do not load here for lack of system libraries: `nvmultiurisrcbin`, `nvinferserver`, `nvdslogger`, `ucx`.

## 4. Bugs found on 2026-11-12

- **set-classes prompt hot-swap never fired** since the merge of `SAM3Fixed`/`SAM3Open`: the watcher unpacked 6 values from a `read_config` that returns 5, and the `ValueError` was swallowed by `except Exception`. Fixed.
- **fp16 image input**: section 1. Fixed.
- **open-text with different prompts per camera gave wrong labels**: people were labelled `dog` on cam0 and `bench` on cam1, and correct when both cameras had the same prompt. The text features were injected per batch row (row b = camera b), but `nvstreammux` does not keep camera b in row b (inferred from those two runs, the batch order was not measured directly). Now every row gets the same slots, the union of all cameras' classes, and `PromptLabeler` keeps each camera's own classes. Price: `K_MAX` (5) is the number of different classes over all cameras, not per camera.
- `network-mode` in the generated nvinfer configs was 1 (INT8), now 2 (FP16). Harmless while `model-engine-file` exists.
- Two research scripts (`infer_engine_bus.py`, `eval_engine_person.py`) did not bind the `masks` output that every engine has since 2026-11-10. Fixed.

Left alone on purpose: `spec_cache.ensure_spec` calls `export.py --device cuda` (right on a large GPU, too big for 4 GB); `K_MAX`/`MAX_SOURCES` in `sam3open_text.py` must match `--k-max`/`--max-batch` of the export by hand.

## 5. Object queries (200 per class slot) are not worth cutting

The 200 learned object queries (`Sam3DetrDecoder.query_embed`, paper default `Q=200`) cannot be shrunk cheaply: truncating to the first N (tried 100/50/25 on 24 frames of `pedestrians.mp4`, imgsz 420, Q=200 as reference, conf 0.3 + NMS 0.6) kept ~83-90% / 68-82% / ~59% of the reference detections with growing score drift, and the PyTorch fp32 forward time did not change measurably (~420 ms; backbone + DETR encoder dominate). Only 85 of 200 queries ever fired on that clip but which ones depends on the image (query 87 alone took 22/236 detections), so there is no fixed "best N". Each query is conditioned on one prompt, so a shared pool across classes is not possible either; cost scales with the number of classes/slots. There is deliberately no `--num-queries` option.

## 6. Prompt-free: measurements

The engine output is raw: post-process it on the host (`test/infer_prompt_free.py: postprocess`: drop masks under `--min-area`, NMS by mask IoU 0.5 with the largest mask first, drop masks >= 90% inside a kept one, optional `--max-area`). Do not filter on score: the predicted IoU ranks parts above whole objects (person recall 0.60 -> 0.07 at score >= 0.7) and the object score is always >> 0 (no memory).

Recall = share of COCO val2017 ground-truth instances (first 200 images, 1444 instances) with a mask of IoU >= 0.5, class-less (`test/eval_prompt_free.py` for an engine, `research/prompt_free_levers.py` for the same wrapper in eager PyTorch). "Raw" is before the post-process, "post" after it.

| imgsz | points | masks kept per point | mask size | min area | raw | post | masks per image | person raw / post | small (<1% of frame) raw / post |
|---|---|---|---|---|---|---|---|---|---|
| 420 | 225 | best | 36 | 0.3% | 0.35 | 0.32 | 27 | 0.21 / 0.18 | 0.16 / 0.13 |
| 420 | 225 | best | 120 | 0.3% | 0.44 | 0.36 | 29 | 0.30 / 0.22 | 0.29 / 0.17 |
| 420 | 225 | all 3 | 36 | 0.3% | 0.51 | 0.43 | 45 | 0.47 / 0.43 | 0.27 / 0.19 |
| **420** | **225** | **all 3** | **120** | 0.3% | **0.62** | 0.48 | 52 | 0.60 / 0.50 | 0.44 / 0.23 |
| 420 | 225 | all 3 | 120 | 0.05% | 0.62 | 0.57 | 72 | 0.60 / 0.57 | 0.44 / 0.38 |
| 700 | 256 (stride 3) | all 3 | 200 | 0.05% | 0.74 | 0.63 | 58 | 0.77 / 0.70 | 0.56 / 0.43 |

- **Shipped since 2026-11-12: all 3 masks per point at the decoder's own resolution** (bold row; `imgsz / 14 * 4` = 120 at 420, so N = 675 masks). The first engine kept the best of the 3 masks at 36x36: recall 0.35. Same latency (122 ms at batch 1, the mask decoder computes the 3 masks anyway); the engine row and the eager row agree. The 3 masks of a point are part / object / whole, and the predicted IoU picks the part too often.
- The post-process default `--min-area 0.003` costs most of the small objects; 0.0005 gives 0.57 with 72 masks per image. It is a trade of recall against the number of masks; the default was left alone.
- imgsz helps more than anything else at equal point count (700 with 256 points beats 420 with 225 points everywhere), and it is what fixes the IoU >= 0.7 numbers (raw 0.61 vs 0.37). A 700 engine is not built: the eager wrapper ran out of memory on the 4 GB GPU with 625 points (stride 2), 256 points fit.
- A denser grid at 420 (900 points) reached 0.72 raw in an earlier run (before the fix below), with 2700 masks to post-process; not pursued.
- **A bug in the evaluation, fixed 2026-11-12:** the mask IoU was summed in fp16, which overflows above 65504 pixels (a mask over 37% of a 420 frame), so large objects counted as missed. Numbers quoted before that date are too low (the first engine was reported as 0.31 raw / 0.27 post; it is 0.35 / 0.32, and large objects 0.71, not ~0.4).
- Size filters alone do not remove the background (hundreds of mid-size patches); dropping masks >30% of the frame removes whole large objects and leaves their fragments. On a static camera (`pedestrians.mp4`) keeping only masks that overlap motion (median background) leaves ~7 masks per frame instead of ~110, but then plain background subtraction does most of the job.

`research/generic_prompts.py` showed that class-agnostic text prompts ("object", "thing", ...) do not work with the detector (best: recall 0.21-0.42 vs 0.90 when the prompt is the true class names, `generic_prompts.log`), which is why prompt-free uses the tracker.

## 7. Boxes at imgsz != 1008: the grid and the size patch

The checkpoint was trained at imgsz 1008 only: a 72 x 72 grid of feature cells (14 px each). Run at another grid H, the decoder is outside the one geometry it knows and the boxes come out shifted toward the top-left and too small, smoothly in H (width -10% at 420, -13% at 336, +1.5% at 1260). `export.py` puts the decoder back into the trained geometry with two patches, applied at **module import time**, baked into every exported engine, both the identity at H = 72:

1. **Grid (centre).** `Sam3DetrDecoder._get_coords` returns `arange(0,H)/(H-1) * 71/72`: the cell coordinates end at the trained 71/72 for any H. This fixes the box *centre*.
2. **Size.** The decoder learned that a box of normalized width w covers w x 72 cells. At grid H it covers w x H cells, so the size the query is told (the w, h part of the sine embedding of its reference box, `Sam3SinePositionEmbedding.encode_boxes`) is given in cells / 72: `w x H/72`, `h x H/72`. The centre is left alone. The factor is the ratio of the two grids, nothing is tuned. Switch: `export.BOX_SIZE_FIX` (research scripts turn it off to test their own variants).

Why the size patch is geometry and not a fitted constant (`research/BOX_SHRINK.md`, steps 12-13):
- only w and h matter: scaling them alone does everything the full box scale did, scaling the centre alone does nothing;
- inside the training distribution, at 1008, mis-stating the size by 72/H on purpose reproduces the error of grid H: growth for H > 72 (+1.6% width predicted, +1.7% measured at 1260), 93% of the width shrink at 840, 72% at 700, 69% at 420, 55% at 336. Most of the shrink is the trained decoder's own response to a mis-stated size; a smaller second effect remains at small grids.

Measured with real engines, `person` @420 (the same ONNX export with and without patch 2):

| | without the size patch | with it |
|---|---|---|
| COCO val2017, 300 images, 685 people: recall at score >= 0.5 | 0.514 | **0.628** |
| precision | 0.608 | **0.741** |
| AP@0.5 | 0.487 | **0.653** |
| coco128, boxes > 2% of the image: width / height error | -10.8% / -2.8% | **-2.3% / -0.4%** |
| mean IoU with the label | 0.829 | **0.869** |
| latency | 84 ms | 84 ms |

Small boxes gain the most: people 14-56 px wide were 23-28% too narrow and are now 6-8% narrow, and twice as many of the 14-28 px ones reach IoU 0.5. The price: scores are slightly lower (median -0.01, mean -0.04 on boxes both engines find). Engines exported before 2026-11-12 do not have patch 2; `test/golden/person_420.npz` was saved again with it.

Rejected along the way: letting the grid end at 1.0, content resolution, window padding, encoder RoPE/abs-pos, key sine in the 72 frame (breaks detection). What is still open: the 15-45% of the shrink below H = 50 that the mis-scaling test does not reproduce.

## 8. Sub-block pruning: `--prune`, 20% faster

A sub-block is switched off by replacing it with zero, so only the residual passes and the block is not in the ONNX at all. 106 sub-blocks = ViT 64 (32 layers x attention, mlp) + DETR encoder 18 + decoder 24. No retraining.

**What ships:** `export.py --prune` (set-classes and open-text, off by default) leaves 16 ViT sub-blocks out, `mlp:27, mlp:26, attention:24, attention:22, mlp:25, mlp:20, mlp:19, mlp:31, attention:26, attention:25, mlp:24, mlp:30, attention:28, attention:23, attention:29, mlp:29`, prints a warning and adds `_pruned` to the engine folder. In the tables below the columns are named after the number of blocks cut: `--prune` is the 16 column; 6, 10 and 14 were built to find the limit and are not options. (It was the default for a few hours on 2026-11-13 and went back to opt-in.)

### What a cut costs

`research/compare_engines.py`: the pruned engine and the unpruned one on the same 300 COCO val2017 images (685 labelled people), class `person`, imgsz 420, FP16, score threshold 0.5.

| | unpruned | 6 blocks | 10 blocks | 14 blocks | **16 blocks (`--prune`)** |
|---|---|---|---|---|---|
| **Speed**, batch 1 | 84.3 ms | 78.0 ms (-7.5%) | 72.9 ms (-13.5%) | 69.3 ms (-17.8%) | 67.2 ms (-20.3%) |
| batch 4 | 317 ms | 293 ms | 274 ms | 260 ms | 250 ms (-21%) |
| **Recall**: labelled people found with score >= 0.5 | 0.628 (430) | 0.620 (425) | 0.619 (424) | 0.613 (420) | 0.591 (405) |
| **Precision**: boxes over 0.5 that are a labelled person | 0.741 | 0.731 | 0.715 | 0.718 | 0.699 |
| **AP@0.5** (recall and precision over all thresholds) | 0.653 | 0.645 | 0.646 | 0.634 | 0.623 |
| **Same object, score change**: mean | | -0.011 | -0.021 | -0.049 | -0.067 |
| median | | -0.004 | -0.008 | -0.026 | -0.042 |
| the worst 10% of objects lose more than | | 0.053 | 0.092 | 0.152 | 0.186 |
| **Objects that fall under 0.5** (found, but now below the threshold) | | 4.8% | 7.5% | 12.1% | 14.5% |
| **Objects gone** (no box at all any more), of 627 | | 0 | 0 | 1 | 0 |
| **New boxes over 0.5**: real people the unpruned engine scored lower | | 4 | 9 | 14 | 11 |
| boxes the unpruned engine also had, under 0.5, with no label | | 23 | 49 | 62 | 68 |
| **invented** (no label, and the unpruned engine had no box there at all) | | 0 | 0 | 0 | 1 |

How to read it:
- **Speed** is about 1.3% per ViT sub-block, the same at batch 1 and 4 and at imgsz 700. Measured 2026-11-13 with the GPU between 52 and 70 C, two rounds within 1% of each other. (A first attempt right after a series of builds ran at 79-83 C and read 106 ms for the unpruned engine: this laptop throttles, let it cool before timing.)
- **The same object keeps its box and loses a little score.** With 10 blocks cut the typical object loses 0.008, one object in ten loses more than 0.09, and a few lose almost everything (the worst went from about 0.8 to 0.05). No object disappears; what happens is that 7.5% of them slip under the 0.5 threshold. That is the whole recall loss (424 instead of 430 of 685), and lowering the threshold gets most of it back.
- **Pruning does not hallucinate.** Up to 14 blocks not one box appeared where the unpruned engine had nothing (one did at 16). What does happen is the mirror image of the point above: scores move in both directions, so some boxes the unpruned engine held just under 0.5 rise over it. A few of those are real people (recall gained), most are not labelled (precision 0.741 -> 0.715): doubtful boxes that were already there, now counted.
- **Is 10 the maximum?** It is where it still costs almost nothing: AP 0.646 against 0.653, as good as 6. At 14 every number is worse (AP 0.634, 12% of the objects under the threshold, the first object lost) and at 16 it is clearly too much: recall 0.591 (25 people fewer), AP 0.623, one object in seven under the threshold and the first invented box. There is no cliff, the cost grows by about 0.005 of score per block. 10 is where almost nothing is lost; 16 is the limit of the option: 20% faster for 0.04-0.07 of score per object, to be used with a lower score threshold (at 0.5 it finds 405 people instead of 430). More than 16 was not measured as an engine; the eager search had the mean score at 0.52 with 20 blocks.
- **Lowering the threshold gives the recall back, not the precision.** `--prune` at a 0.4 threshold finds exactly as many people as the unpruned engine at 0.5 (430 of 685, recall 0.628), but with 298 boxes that match no label instead of 150 (precision 0.591 against 0.741); at 0.45: recall 0.610, precision 0.649. The scores of true and doubtful boxes do not drop by the same amount, so the shift cannot be undone by one threshold.

### At imgsz 700

The same comparison with both engines exported at 700 (`--max-batch 4`), same 300 images:

| | unpruned @700 | 16 blocks @700 | for reference: unpruned @420 | 16 blocks @420 |
|---|---|---|---|---|
| Speed, batch 1 | 190.9 ms | 150.3 ms (-21.3%) | 84.3 ms | 67.2 ms (-20.3%) |
| Recall at 0.5 | 0.815 (558) | 0.793 (543) | 0.628 (430) | 0.591 (405) |
| Precision at 0.5 | 0.780 | 0.781 | 0.741 | 0.699 |
| AP@0.5 | 0.832 | 0.823 | 0.653 | 0.623 |
| Same object, score change: mean / median | | -0.042 / -0.022 | | -0.067 / -0.042 |
| Objects that fall under 0.5 | | 10.7% | | 14.5% |
| Objects gone / boxes invented | | 0 / 0 | | 0 / 1 |

- **The low numbers at 420 were the resolution, not the pruning:** 700 finds 558 of the 685 people where 420 finds 430.
- **Pruning costs less at 700:** AP -0.009 instead of -0.030, precision unchanged, the score of an object drops by 0.04 instead of 0.07. With more cells per object the remaining layers have more to work with. At a 0.4 threshold the pruned 700 engine has recall 0.829 and precision 0.709, against 0.839 and 0.718 unpruned.
- So the pruned engine at 700 (150 ms) is far more accurate than the unpruned 420 engine (84 ms), and pruning is the cheaper way to pay for a larger imgsz. The block order was ranked at 420 and still works at 700.

### Other blocks

`research/prune_search.py` cuts each of the 106 sub-blocks alone (eager FP16, imgsz 420, classes person/car/chair) and ranks them by what happens to the unpruned detections.

- **DETR encoder and decoder: no.** Alone, several are harmless (`enc.cross_attn:2,3`, `dec.text_cross_attn:0,5`, `dec.vision_cross_attn:4`, `dec.self_attn:3`: score change within ±0.01). Together they are not: on top of 10 ViT cuts, 4 more take the mean score from 0.74 to 0.63, 8 to 0.53, 10 to 0.42. And there is nothing to gain: all 18 encoder/decoder cuts of the first study saved 1.2 ms of 84 in a one-class engine (measured: 76.1 ms with them, 77.3 ms without, scores 0.23 lower). The worst single cuts of the whole model are here too: `dec.vision_cross_attn:0` (-0.24), `enc.mlp:2` (-0.15).
- **ViT: the damage is cumulative.** No single ViT cut loses a box, and single-cut ranks do not add up, so the script also picks blocks greedily (at each step the one that costs the least score given those already cut): `mlp:21, mlp:29, mlp:28, attention:27, mlp:24, attention:26, attention:21, mlp:25, attention:24` took the mean score 0.739 -> 0.683 in 9 steps (stopped there: 5 minutes a step). That is the same slope as the shipped list (about -0.006 per block), from a partly different set: layers 19-31 are interchangeable to this precision, and no clever choice of blocks beats the slope. The early layers (0-8) and the global-attention layers 7 and 15 rank last.
- **Neck, mask head, text encoder:** they have no residual sub-blocks to switch off. The text encoder is not in the runtime path of a set-classes engine at all.

Why the decoder looked the most prunable in the eager study: only the decoder has a cluster of sub-blocks pointing the same way, `dectext:0,1,2` + `decself:1,2` (VIF 18-38); cutting the whole cluster costs about 49% of the sum of the individual errors. ViT and encoder sub-blocks are nearly independent (VIF ~1-2). The correlation matrix finds clusters but does not predict the error of cutting several stages at once (off by up to ~7x): it has to be measured.

Background, the first study (coco128, oracle prompts, PyTorch eager, imgsz 420 and 1008; its scripts were removed): about 24 of 106 sub-blocks could be cut "with no recall lost" when boxes were matched whatever their score, for 9-15% in eager mode. Real engines showed where that held (ViT) and where it hid a collapse of the scores (encoder, decoder).

Still open: the ranking at 1008 and for other classes than person, mean ablation (replace a block by its mean output instead of 0), which may give the scores back.

## 9. DeepStream pipeline gotchas (measured on an RTX 50-series, driver 610.x, DeepStream 9.1)

- **Video encoder**: use `nvv4l2h264enc`, and force `video/x-raw(memory:NVMM),format=NV12` via a `capsfilter` immediately before it — without the explicit format, GStreamer negotiates I420 (listed first in the sink template) and the encoder misreads chroma, corrupting the image (bitstream stays valid, so `ffmpeg -v error` reports zero errors — only visual inspection catches it). Also set `idrinterval` explicitly; `iframeinterval` alone gets ignored and produces too few real seek points. `nvh264enc` and `nvcudah264enc` do not work at all with a DeepStream hardware-decode source on this GPU/driver combination.
- **Tracker**: `NvSORT` (cascaded/greedy matching, no visual tracker) is cheaper than `NvDCF` and tolerates a higher `interval`, but its stock `probationAge`/`minTrackerConfidence` assume near-every-frame detection — at `interval=5` they need lowering (`probationAge=1`, `minTrackerConfidence≈0.1`) or tracks never "mature" and zero boxes get emitted (see `deepstream/config_tracker_NvSORT_i5.yml`).
- The custom parsers read detection count and layer shapes from the actual output tensor at runtime rather than assuming a fixed layout, so they stay correct across different baked class counts (set-classes) or camera counts (open-text) without recompiling.
