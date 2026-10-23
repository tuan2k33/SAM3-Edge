# Boxes shrink when SAM3 runs at `imgsz` != 1008: research log

Updated: 2026-10-23. Status: done, shipped in `SAM3Fixed/export.py` and `SAM3Open/export.py` (step 11).
Background: the starting point was the hypothesis "bug in `_get_coords`" (an earlier single-image bus experiment, removed from the repo, still in git history). This file re-checks that hypothesis against ground truth and looks for the real cause.

## 0. Summary

- The effect is **real and systematic**: for `imgsz` < 1008 predicted boxes shrink toward the top-left corner (right/bottom edges are short); for `imgsz` > 1008 they grow slightly. The amount changes **smoothly and monotonically with H = imgsz/14** and crosses 0 around H = 72-90.
- It is **not** the RPB coordinate grid (4 variants, none flattens the trend), **not** content resolution, **not** attention-window padding tokens, **not** the encoder's global RoPE or absolute position embedding.
- It **lives in the decoder** (step 8: with the encoder at H=72 and the decoder at a small H the shrink comes back intact).
- Inside the decoder, the **sine encoding of the query boxes** is what matters: multiplying the box coordinates by `H/72` before it (`sineq`) cuts the **slope over H about 4x** (width +2.04 -> +0.51 %/10 cells) and squeezes the error **at every size from 336 to 1260** into about -4.6% ... +0.4% (original: -13.1% ... +1.5%), with no change at 1008 and unchanged recall. An exponent sweep gives a smooth, monotonic dose-response at 420, 700 and 1260. The zero crossing is not at one fixed exponent (0.34-1.49 depending on size and axis), so `H/72` is a good approximation, not an exact formula.
- The `71/72` grid is **not the root cause**: it mostly shifts the box (centre) and does not fix its size; at 1008 it is a no-op. It is still useful to centre the box when used together with `sineq`.
- **Decision:** `sineq` with `a = 1` (factor `H/72`) + grid `arange/(H-1)*71/72`. Step 11 has held-out numbers and a real TensorRT FP16 engine.

## 0b. Rejected directions (raw results removed, rerun with `collect_box_pairs.py`)

None of these variants is the cause: none flattens the trend over H. First 64 images, mean over the sizes that were run (width/height: |error %|, slope: %/10 cells; `orig` at the same sizes is about 6.1 / 3.2 / +2.0). Details in the corresponding step.

| variant | idea | \|width\| / \|height\| | slope | conclusion |
|---|---|---|---|---|
| `1.0` | RPB grid ends at 1.0 | 4.2 / 1.9 | +1.86 | not flat; larger centre error |
| `cell` | RPB distances measured in cell units (H/72) | 7.1 / 3.3 | +2.75 | worse than `orig` |
| `ropecell` | global RoPE per cell (Hp/72) | 5.8 / 2.7 | +2.33 | same as `orig` |
| `peinterp` | absolute position embedding resampled from 72x72 | 5.7 / 2.8 | +2.23 | same as `orig` |
| `sinekey` | key sine position in the 72 frame | 12.6 / 8.5 | +3.84 | much worse |
| `frame72` | whole decoder in cell geometry (`sinekey`+`sineq`+`cell`) | 1.7 / 1.1 | +0.20 | flat slope, but 0.2 patch centre offset and collapsed recall; more complex without beating `sineq`+`71/72` |
| `encpad` (step 8) | encoder at H=72, decoder at small H | shrink returns | | locates the cause in the decoder |

## 1. Question and criteria

Question: why do boxes shrink when `imgsz` != 1008 (the model was trained only at 1008 = 72x72 patches of 14 px), and can it be fixed.

Criteria (set by the project owner):
1. The primary measure is the **relative width/height error** (`pred_size / gt_size - 1`) and **its trend over `H`**. A good fix flattens the trend and keeps it around 0 within about +-15%; the error need not be 0.
2. **IoU is only a reference.**
3. All runs use **FP16** (see section 2).

## 2. Method

- Data: **coco128** (COCO images with box labels), first 64 images, 226 GT boxes with area >= 2% of the image. Not in the repo (`test/data/`, git-ignored).
- Prompts: the COCO class names that have a GT box in the image, at most 4 classes per image ("oracle prompts": measures box geometry, not false positives).
- Predictions: score >= 0.5, NMS 0.6, boxes clipped to the image. Each GT box is paired with the best-IoU unused prediction of the same class (IoU >= 0.3); statistics use only pairs with IoU >= 0.5 (about 190-210 pairs per configuration).
- Error = prediction - GT, reported in **patches** (`normalised delta / (14/imgsz)`) and in **% of box size**.
- **True FP16 model** (`model.half()`, pixel `scale` kept in FP32), grid computed in FP32 then cast to FP16. FP32 + `torch.autocast` was rejected because it spills out of 4 GB VRAM (8.2 s/forward vs 0.2 s); the true-FP16 result was checked against autocast (IoU 0.803 vs 0.802, same recall, patch errors equal to 0.01).
- Collection: `collect_box_pairs.py` stores the raw **(GT, prediction) pairs** in `results_pairs/*.json` (one file per (size, variant); reruns skip existing files). Offline CPU analysis: `analyze_box_pairs.py`.
- Reading caveats: there is no perfect ground truth (COCO labels are noisy); standard errors follow the +-; 64 images is not many, so differences below ~1% are not reliable.
- A trap met on the way: `_VisionEncoder.__init__` overwrites `position_ids`/position embedding **in the shared model itself**, so wrappers for different `imgsz` built from one `Sam3Model` break each other. The scripts rebuild the wrapper for every size.

Notation: H = imgsz/14 (cells per side); `d = g_i - e` is the distance from cell coordinate `g_i` to a box edge `e` (`modeling_sam3.py`: `_get_coords` lines 1718-1719, `_get_rpb_matrix` lines 1746-1758 of transformers 5.20.2).

## 3. Research steps

### Step 1. Re-check earlier claims

- "1-class and 2-class wrappers give different boxes" **does not reproduce**: boxes are identical (0.00 px) at 1008 and 420, for both the old repeat-class wrapper and the batch-folded one. Removed from the old notes.
- "cap = 0.987654321 is the best" only held for one box of the bus image (the number was picked by hand, not derived). On 91 boxes, `71/72` is as good and equals the original grid at 1008.
- The original formula `arange(0,H)/H` is **also Meta's original code** (facebookresearch/sam3 `decoder.py`): it is the training-time convention, not a bug of the `transformers` port.

### Step 2. Is the deviation real against ground truth?

Yes. At 420, original model (patch units; positive = prediction larger than GT):

| 420 | x1 | x2 | y1 | y2 | width | height |
|---|---|---|---|---|---|---|
| `orig` | +0.15 | **-0.71** | +0.03 | **-0.49** | -0.87 (-10.4%) | -0.52 (-5.2%) |

The left/top edges are almost right, the right/bottom edges are short by about 0.5-0.7 patch. This matches the project owner's observation ("top-left corner right, w and h smaller").

### Step 3. Is it the `_get_coords`/RPB grid? (4 variants)

| Variant | Formula (`g_i`, `d`) |
|---|---|
| `orig` | `g_i = i/H` |
| `71/72` | `g_i = i/(H-1) * 71/72` |
| `1.0` | `g_i = i/(H-1)` |
| `cell` | `g_i = i/H`, `d' = (g_i - e)*H/72` before the log encoding |

Results at 420 (192-194 pairs per variant):

| 420 | width | height | x1 | x2 | y1 | y2 | IoU (ref.) |
|---|---|---|---|---|---|---|---|
| `orig` | -10.4% | -5.2% | +0.15 | -0.71 | +0.03 | -0.49 | 0.803 |
| `1.0` | -8.2% | -4.0% | +0.48 | -0.17 | +0.34 | 0.00 | 0.805 |
| `71/72` | -9.1% | -3.8% | +0.35 | -0.38 | +0.20 | -0.10 | 0.824 |
| `cell` | -11.1% | -5.1% | +0.35 | -0.63 | +0.36 | -0.20 | 0.807 |

Reading: the grid variants mostly **shift the whole box** right/down (x1 and x2 both rise 0.2-0.3 patch), **its size barely changes**. Slope of the width over `H`: `orig` +2.04, `71/72` +1.98, `1.0` +1.86, `cell` +2.75 (% per 10 cells): **no variant flattens it**. `cell` (the project owner's idea: measure distances in cell units) is even worse.

At 1008: `71/72` and `cell` are **identical to `orig`** (max difference 0.0 on every number, a self-check of the patch); `1.0` is worse (IoU 0.916 -> 0.874, every edge shifted 0.3-0.6 patch). A grid ending at 1.0 only "helps" at small sizes and hurts at the native size.

Conclusion of step 3: **the RPB grid is not the main cause.**

### Step 4. Shape of the effect over `imgsz` (original)

| imgsz | H | width % | height % | IoU (ref) |
|---|---|---|---|---|
| 336 | 24 | -13.1 | -7.3 | 0.754 |
| 420 | 30 | -10.4 | -5.2 | 0.803 |
| 504 | 36 | -8.9 | -4.8 | 0.830 |
| 560 | 40 | -7.0 | -3.9 | 0.851 |
| 672 | 48 | -5.6 | -2.9 | 0.875 |
| 700 | 50 | -5.4 | -2.2 | 0.877 |
| 840 | 60 | -2.9 | -1.0 | 0.901 |
| **1008** | 72 | **-1.1** | **-0.2** | 0.916 |
| 1260 | 90 | **+0.6** | **+1.5** | 0.914 |

Slope: width +2.04%/10 cells, height +1.27%/10 cells. The shrink is **smooth, monotonic in H, changes sign above 72**, stays within +-15% at every size but has a clear trend. Height crosses zero earlier than width.

### Step 5. Content resolution or H? (padding control)

The content keeps its low resolution but is pasted into the top-left of a grey 1008 canvas and run at H=72 with the `orig` grid; coordinates are rescaled back to the image:

| content | run on its natural grid (small H) | pasted into a 1008 canvas (H=72) |
|---|---|---|
| 336 | width -13.1%, height -7.3% | **-2.0%, +0.2%** |
| 420 | -10.4%, -5.2% | **-1.3%, +0.0%** |
| 700 | -5.4%, -2.2% | **-1.4%, 0.0%** |

Conclusion: **it is not that the object has few pixels**; the error is tied to **the number of cells H**. (The canvas costs a full 1008 forward, so it is not a "light" solution.)

### Step 6. Attention-window padding tokens? (rejected)

Hypothesis: `window_partition` pads zeros at the bottom/right when H is not divisible by 24, without a mask, corrupting the features of the far edge. Two tests:
- Error by position (420): x2 is off by **-0.60** for boxes fully inside the first full window (x2 < 0.6), and **-0.88, -0.81, -0.69** for the other groups: **uniform over the image**, not concentrated at the padded edge.
- H divisible by 24 (336: H=24, 672: H=48; no padding) still shrinks: 336 is the size with the **strongest** shrink (-13.1%), 672 lies exactly on the curve.

Conclusion: **rejected.**

### Step 7. Encoder: global RoPE and absolute position embedding (rejected)

- `ropecell`: RoPE positions of the 4 global layers multiplied by `H/72` (each cell is always 1/3 unit, as in training).
- `peinterp`: position embedding = the 72x72 map (3x3 tile of 24x24) resampled bilinearly down to HxH.

| | 420 width/height | 700 width/height | 1008 |
|---|---|---|---|
| `orig` | -10.4% / -5.2% | -5.4% / -2.2% | -1.1% / -0.2% |
| `ropecell` | -10.9% / -5.8% | -5.4% / -2.2% | same as `orig` |
| `peinterp` | -10.5% / -5.7% | -5.5% / -2.6% | same as `orig` |

Conclusion: neither component is the cause (and, as designed, both are no-ops at 1008).

### Step 8. Bisection: encoder or decoder? (`encpad`)

The encoder runs on the 1008 canvas (H=72, where it is right), the feature map is cropped to the content's HxH, and then the **decoder runs with HxH geometry** (grid, sine, RPB at HxH):

| content | normal (both at small H) | both at H=72 | **encoder H=72, decoder small H** |
|---|---|---|---|
| 336 | -13.1% / -7.3% | -2.0% / +0.2% | **-13.3% / -7.7%** |
| 420 | -10.4% / -5.2% | -1.3% / +0.0% | **-10.9% / -6.5%** |
| 700 | -5.4% / -2.2% | -1.4% / 0.0% | **-4.8% / -2.4%** |

Three sizes, one conclusion: **the error is in the decoder when H != 72, not in the ViT or the neck.**

### Step 9. Inside the decoder: the sources of position information

The decoder uses three sources: (a) the box sine -> `ref_point_head` -> `query_pos` (`encode_boxes`); (b) the key sine `fpn_pos_2` (also fed to the DETR encoder); (c) the RPB (ruled out in step 3). Three variants (run at normal sizes):
- `sinekey`: key sine in the 72 frame (`(i+1)/72`, top-left part);
- `sineq`: box coordinates entering `encode_boxes` multiplied by `H/72`;
- `frame72`: both plus the RPB in the 72 frame (`cell`).

| 420 | pairs | recall@.5 | width | height |
|---|---|---|---|---|
| `orig` | 191 | 0.845 | -10.4% | -5.2% |
| **`sineq`** | **192** | **0.850** | **-3.6%** | **-0.7%** |
| `frame72` | 130 | 0.575 | -1.6% | -1.0% |
| `sinekey` | 124 | 0.549 | -17.1% | -11.3% |

| 336 | pairs | recall@.5 | width | height |
|---|---|---|---|---|
| `orig` | 169 | 0.748 | -13.1% | -7.3% |
| **`sineq`** | **175** | **0.774** | **-4.6%** | **-2.3%** |

Reading:
- **`sineq` alone** removes most of the shrink at both resolutions, recall unchanged or slightly higher.
- **`sinekey` breaks detection** (recall 0.55): keys and queries end up in different frames; `frame72` removes the shrink but also collapses recall because it contains `sinekey`. Conclusion: **do not touch the key sine.**
- At 1008 all three variants are identical to `orig` (self-check, difference 0.0).

### Step 10. Does `sineq` flatten the curve at every size?

**(a) `sineq` at 9 sizes** (vs `orig`; slope = % per 10 cells):

| | width slope | height slope | width range | height range |
|---|---|---|---|---|
| `orig` | +2.04 | +1.27 | [-13.1, +0.6]% | [-7.3, +1.5]% |
| **`sineq`** | **+0.51** | **+0.31** | **[-4.6, -1.1]%** | **[-2.3, +0.4]%** |

| imgsz | H | `orig` width / height | `sineq` width / height | IoU `orig` -> `sineq` (ref.) |
|---|---|---|---|---|
| 336 | 24 | -13.1 / -7.3 | -4.6 / -2.3 | 0.754 -> 0.773 |
| 420 | 30 | -10.4 / -5.2 | -3.6 / -0.7 | 0.803 -> 0.824 |
| 504 | 36 | -8.9 / -4.8 | -3.3 / -1.4 | 0.830 -> 0.851 |
| 560 | 40 | -7.0 / -3.9 | -2.6 / -0.8 | 0.851 -> 0.872 |
| 672 | 48 | -5.6 / -2.9 | -2.6 / -0.4 | 0.875 -> 0.890 |
| 700 | 50 | -5.4 / -2.2 | -2.5 / -0.2 | 0.877 -> 0.895 |
| 840 | 60 | -2.9 / -1.0 | -1.4 / -0.3 | 0.901 -> 0.910 |
| 1008 | 72 | -1.1 / -0.2 | -1.1 / -0.2 (identical) | 0.916 -> 0.916 |
| 1260 | 90 | +0.6 / +1.5 | -1.2 / +0.4 | 0.914 -> 0.911 |

Reading: the error at every size is squeezed into a narrow band around 0, **including above 72** (at 1260 `orig` grows +0.6%/+1.5%, `sineq` no longer changes sign), and the curve is nearly flat from 700 up. Width keeps a residual shrink of about 2-3% in the middle (504-700). Everything is inside the acceptable +-15% band.

**(b) Exponent sweep `sineq<a>`** (the box coordinate factor is `(H/72)^a`, `a=1` is `sineq`):

| `a` | factor `(H/72)^a` at 420 / 700 / 1260 | 420 width / height | 700 | 1260 |
|---|---|---|---|---|
| 0 (`orig`) | 1 / 1 / 1 | -10.4 / -5.2 | -5.4 / -2.2 | +0.6 / +1.5 |
| 0.5 | 0.645 / 0.833 / 1.118 | -7.0 / -2.8 | -3.7 / -1.3 | -0.3 / +0.9 |
| **1** | 0.417 / 0.694 / 1.250 | -3.6 / -0.7 | -2.5 / -0.2 | -1.2 / +0.4 |
| 1.5 | 0.269 / 0.579 / 1.398 | +1.8 / +2.5 | -1.0 / +0.6 | -1.9 / 0.0 |
| 2 | 0.174 / 0.482 / 1.562 | +9.1 / +5.9 | (not run) | (not run) |

- The relation is **monotonic and smooth** at all three sizes (larger `a` always makes the box larger), so this is a real tuning knob, not a coincidence. Recall at 420 hardly changes with `a` (0.832-0.858).
- **Zero crossing of the shrink** (linear interpolation in `a`): 420 width ~ 1.33, height ~ 1.11; 700 height ~ 1.11, width has not changed sign within the swept range; 1260 width ~ 0.34, height ~ 1.49. It does not fall on one fixed `a`, but `a = 1` gives the flattest result over the three sizes together (-3.6 / -2.5 / -1.2% width, -0.7 / -0.2 / +0.4% height).

### Step 11. Final patch, held-out validation and a real engine

**(a) `sineq^a` x grid combinations** (first 64 images, mean over sizes; `|width|`/`|height|` are mean absolute errors in %, `slope` is %/10 cells):

| variant | sizes | \|width\| | \|height\| | slope | centre offset x/y (patch) | IoU (ref.) |
|---|---|---|---|---|---|---|
| `orig` | 9 | 6.11 | 3.23 | +2.04 | 0.16 / 0.16 | 0.858 |
| `71/72` | 5 | 5.12 | 2.15 | +1.98 | 0.03 / 0.06 | 0.875 |
| `sineq` (a=1) | 9 | 2.55 | 0.74 | +0.51 | 0.20 / 0.19 | 0.871 |
| **`sineq+71/72` (a=1)** | 7 | 1.88 | 0.26 | +0.24 | 0.05 / 0.06 | 0.893 |
| `sineq1.2+71/72` | 7 | 1.03 | 0.46 | -0.15 | 0.06 / 0.06 | 0.894 |
| `sineq1.4+71/72` | 7 | 1.36 | 1.00 | -0.65 | 0.07 / 0.06 | 0.893 |

`sineq` fixes the **size**, `71/72` fixes the **centre**; the two effects are independent so they add up. `a = 1` was chosen because it is the "72 grid frame" factor (box coordinates measured in cell units as in training, not a tuned number), has a small positive slope (+0.24), and `a = 1.2` is only better in mean error while it starts sloping negative (boxes grow at large H): it moves the problem instead of removing it. There is no theoretical derivation for the exponent itself.

**(b) Held-out images** (images 65-128, not used to choose anything):

| variant | sizes | \|width\| | \|height\| | slope | centre offset x/y | IoU |
|---|---|---|---|---|---|---|
| `orig` | 7 | 5.86 | 3.44 | +1.78 | 0.14 / 0.20 | 0.856 |
| `71/72` | 7 | 5.27 | 2.85 | +1.51 | 0.09 / 0.07 | 0.868 |
| **`sineq+71/72`** | 7 | 1.93 | 0.84 | -0.07 | 0.05 / 0.09 | 0.888 |

The result holds, the slope goes to ~0.

**(c) Integration.** The same "Fixes for running at imgsz != 1008" block in `SAM3Fixed/export.py` and `SAM3Open/export.py` (each file self-contained): `Sam3DetrDecoder._get_coords` -> `arange/(H-1)*71/72`; `Sam3SinePositionEmbedding.encode_boxes` multiplies the box by `box_scale = (H/72)^1.0`, with `H` taken from the feature map at export time (a constant in the graph). At 1008 the factor is 1 and the grid is the original. The eager production path equals the experiment's `sineq+71/72` (difference 0.0000); Fixed and Open agree within FP16 error.

**(d) Real TensorRT FP16 engine** (`eval_engine_person.py`, class `person` only, 420):

| engine | set | boxes | recall@.5 | width | height | IoU |
|---|---|---|---|---|---|---|
| old (no `sineq`) | first 64 | 54/59 | 0.915 | -10.9+-1.1% | -2.6% | 0.828 |
| **new** | first 64 | 57/59 | 0.966 | -3.9+-1.2% | -0.5% | 0.865 |
| old | held-out | 45/47 | 0.957 | -10.4+-1.6% | -2.5% | 0.835 |
| **new** | held-out | 45/47 | 0.957 | -1.8+-1.4% | +0.3% | 0.859 |

Matches the eager table: the patch is baked correctly into ONNX and TensorRT FP16. The sample is small (47-59 boxes), enough to see the direction only.

**(e) Bus image** (`compare_bus.py`, `person`+`bus`; reference = the model at 1008, since the image has no labels). Mean ratio of box size to the reference:

| imgsz | before (original) | grid `71/72` only | `71/72` + `sineq` (shipped) |
|---|---|---|---|
| 420 | width 0.91 / height 0.97 | 0.92 / 0.99 | **1.01 / 1.02** |
| 700 | 0.96 / 0.99 | 0.96 / 1.00 | **1.00 / 1.01** |

The grid alone fixes the height roughly but leaves the width short; `sineq` closes it.

## 4. Open questions

- **Mechanism.** Why does multiplying the box coordinates by `H/72` in the query sine encoding make box size right? Only empirical evidence so far (smooth dose-response, right direction at 3 sizes, no-op at 1008), no explanation. The zero crossing changes with size and axis (a ~ 0.34-1.49), so `H/72` is a good approximation, not necessarily the right formula.
- A residual width shrink of ~2-3% at 504-840 after `sineq`, and slightly negative x1, y1 (-0.16, -0.30 patch at 336). A small additional correction could be combined, to be tested.
- Validation on more data (only 64 images so far, COCO labels are noisy) and on boxes smaller than 2% of the image (excluded here).
- Exponents above 1.5 were only run at 420.

## 5. Reproduction

From the repo root, needs `HF_TOKEN` in `.env` and coco128 in `test/data/coco128`:

```bash
# collect (each (size, variant) is saved as soon as it finishes)
python -u test/collect_box_pairs.py --sizes 420,700 --variants orig,71/72,sineq,sineq+71/72   # rejected variants: see section 0b and the docstring
# analyse (CPU)
python test/analyze_box_pairs.py
```

Variants and size syntax: see the docstring of `collect_box_pairs.py`. Raw results: `test/results_pairs/*.json`.
