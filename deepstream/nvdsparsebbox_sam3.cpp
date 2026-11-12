/**
 * nvdsparsebbox_sam3.cpp
 *
 * The one custom nvinfer plugin for all SAM3-Edge engines (libnvdsinfer_sam3.so):
 *
 *   NvDsInferParseSAM3Det      detections [N,6]            x1 y1 x2 y2 score cls_id   (set-classes, box only)
 *   NvDsInferParseSAM3Full     detections [N,6] + masks [N,Hm,Wm] probabilities       (set-classes, with masks)
 *   NvDsInferParseSam3OpenDet  detections [N,6]            (open-text; N = K slots * 200, cls_id = the slot in the union of all cameras' classes)
 *   NvDsInferInitializeInputLayers   injects text_features [K,32,256] (the layer's dtype, fp16) + text_mask [K,32] u8 as non-image input layers (open-text)
 *
 * Detections and masks are read as float32 or float16 according to the layer's own dataType.
 * N = num_classes * Q (Q = 200), cls_id for query i is i / Q.
 *
 * open-text injection: one row per batch slot (K class slots per row, the same in every row), read from two flat snapshot files written by sam3open_text.py;
 * the row size comes from the layer dims, so K_MAX is not hardcoded here:
 *     /dev/shm/sam3open_text_features.bin   MAX_SOURCES x K x 32 x 256  f16 (the engine's dtype: the bytes are copied as they are)
 *     /dev/shm/sam3open_text_mask.bin       MAX_SOURCES x K x 32        u8   (TensorRT reports BOOL, DeepStream maps it to UINT8)
 * Called once per nvinfer context (re)load: reloading the SAME config-file-path via nvinfer.set() re-triggers it without a pipeline restart or
 * an engine rebuild. nvinfer looks it up by this EXACT symbol name (dlsym); a custom name is silently never called.
 *
 * Build (DeepStream and CUDA headers needed):
 *   g++ -shared -fPIC -O2 -std=c++20 -o libnvdsinfer_sam3.so nvdsparsebbox_sam3.cpp \
 *       -I/opt/nvidia/deepstream/deepstream/includes \
 *       -I/usr/local/cuda-13.2/targets/x86_64-linux/include
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

static const int   Q               = 200;
static const float SCORE_THRESHOLD = 0.3f;
static const float NMS_IOU_THRESH  = 0.6f;
static const int   MASK_H          = 36;
static const int   MASK_W          = 36;

// ── FP16 → FP32 (portable, no CUDA headers needed) ───────────────────────────

static float fp16_to_float(uint16_t h)
{
    uint32_t sign =  (h >> 15) & 0x1u;
    uint32_t exp  =  (h >> 10) & 0x1fu;
    uint32_t mant =   h        & 0x3ffu;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; --exp; }
            mant &= 0x3ffu;
            f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7f800000u | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

// element i of a float32 or float16 layer buffer
static float at(const NvDsInferLayerInfo &l, size_t i)
{
    return l.dataType == HALF ? fp16_to_float(((const uint16_t *)l.buffer)[i]) : ((const float *)l.buffer)[i];
}

// ── IoU ───────────────────────────────────────────────────────────────────────

static float iou_xyxy(float ax1, float ay1, float ax2, float ay2,
                      float bx1, float by1, float bx2, float by2)
{
    float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    if (inter == 0.0f) return 0.0f;
    float a = (ax2 - ax1) * (ay2 - ay1);
    float b = (bx2 - bx1) * (by2 - by1);
    return inter / (a + b - inter);
}

// ── threshold + per-class NMS ─────────────────────────────────────────────────

struct Det {
    float x1, y1, x2, y2, score;
    int   cls_id, query_idx;
};

static std::vector<Det> collect_and_nms(
    const NvDsInferLayerInfo &det, int N,
    const std::vector<float> &threshVec)
{
    std::vector<Det> candidates;
    candidates.reserve(32);

    for (int i = 0; i < N; ++i) {
        size_t o     = (size_t)i * 6;
        float score  = at(det, o + 4);
        int   cls    = (int)(at(det, o + 5) + 0.5f);

        float thresh = SCORE_THRESHOLD;
        if (!threshVec.empty())
            thresh = (cls >= 0 && cls < (int)threshVec.size())
                     ? threshVec[cls] : threshVec[0];

        if (score < thresh) continue;
        float x1 = at(det, o), y1 = at(det, o + 1);
        float x2 = at(det, o + 2), y2 = at(det, o + 3);
        if (x2 <= x1 || y2 <= y1) continue;

        candidates.push_back({x1, y1, x2, y2, score, cls, i});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const Det &a, const Det &b){ return a.score > b.score; });

    std::vector<bool> sup(candidates.size(), false);
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (sup[i]) continue;
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (sup[j] || candidates[i].cls_id != candidates[j].cls_id) continue;
            if (iou_xyxy(candidates[i].x1, candidates[i].y1,
                         candidates[i].x2, candidates[i].y2,
                         candidates[j].x1, candidates[j].y1,
                         candidates[j].x2, candidates[j].y2) > NMS_IOU_THRESH)
                sup[j] = true;
        }
    }

    std::vector<Det> out;
    for (size_t i = 0; i < candidates.size(); ++i)
        if (!sup[i]) out.push_back(candidates[i]);
    return out;
}

// ── detections only ───────────────────────────────────────────────────────────

static bool parse_det(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferParseDetectionParams   const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    const NvDsInferLayerInfo *det_layer = nullptr;
    for (const auto &l : outputLayersInfo)
        if (std::string(l.layerName) == "detections") { det_layer = &l; break; }
    if (!det_layer && !outputLayersInfo.empty())
        det_layer = &outputLayersInfo[0];
    if (!det_layer) return false;

    int N = det_layer->inferDims.d[0];
    if (N <= 0) return false;

    auto dets = collect_and_nms(*det_layer, N, detectionParams.perClassPreclusterThreshold);

    for (const auto &d : dets) {
        NvDsInferObjectDetectionInfo obj = {};
        obj.classId             = (unsigned int)d.cls_id;
        obj.detectionConfidence = d.score;
        obj.left                = d.x1;
        obj.top                 = d.y1;
        obj.width               = d.x2 - d.x1;
        obj.height              = d.y2 - d.y1;
        objectList.push_back(obj);
    }
    return true;
}

extern "C"
bool NvDsInferParseSAM3Det(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo              const &networkInfo,
    NvDsInferParseDetectionParams     const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    return parse_det(outputLayersInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseSAM3Det);

// same parser, the name deepstream/ds9_open_text.py's config_infer.txt uses
extern "C"
bool NvDsInferParseSam3OpenDet(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo              const &networkInfo,
    NvDsInferParseDetectionParams     const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    return parse_det(outputLayersInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseSam3OpenDet);

// ── detections + masks ────────────────────────────────────────────────────────

extern "C"
bool NvDsInferParseSAM3Full(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo              const &networkInfo,
    NvDsInferParseDetectionParams     const &detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>  &objectList)
{
    const NvDsInferLayerInfo *det_layer   = nullptr;
    const NvDsInferLayerInfo *masks_layer = nullptr;

    for (const auto &l : outputLayersInfo) {
        std::string name(l.layerName);
        if (name == "detections") det_layer   = &l;
        if (name == "masks")      masks_layer = &l;
    }
    if (!det_layer || !masks_layer) {
        if (outputLayersInfo.size() >= 2) {
            det_layer   = &outputLayersInfo[0];
            masks_layer = &outputLayersInfo[1];
        } else {
            return false;
        }
    }

    int N = det_layer->inferDims.d[0];
    if (N <= 0) return false;

    int mask_h = (masks_layer->inferDims.numDims >= 3) ? masks_layer->inferDims.d[1] : MASK_H;
    int mask_w = (masks_layer->inferDims.numDims >= 3) ? masks_layer->inferDims.d[2] : MASK_W;

    float net_w = (float)networkInfo.width;
    float net_h = (float)networkInfo.height;

    auto dets = collect_and_nms(*det_layer, N, detectionParams.perClassPreclusterThreshold);

    for (const auto &d : dets) {
        int cx1 = std::max(0,      (int)( d.x1 * mask_w / net_w));
        int cy1 = std::max(0,      (int)( d.y1 * mask_h / net_h));
        int cx2 = std::min(mask_w, (int)std::ceil(d.x2 * mask_w / net_w));
        int cy2 = std::min(mask_h, (int)std::ceil(d.y2 * mask_h / net_h));
        int cw  = cx2 - cx1;
        int ch  = cy2 - cy1;

        NvDsInferInstanceMaskInfo obj = {};
        obj.classId             = (unsigned int)d.cls_id;
        obj.detectionConfidence = d.score;
        obj.left                = d.x1;
        obj.top                 = d.y1;
        obj.width               = d.x2 - d.x1;
        obj.height              = d.y2 - d.y1;

        if (cw > 0 && ch > 0) {
            float *mask_crop = (float *)std::malloc(cw * ch * sizeof(float));
            if (mask_crop) {
                size_t src = (size_t)d.query_idx * mask_h * mask_w;
                for (int r = 0; r < ch; ++r)
                    for (int c = 0; c < cw; ++c)
                        mask_crop[r * cw + c] = at(*masks_layer, src + (size_t)(cy1 + r) * mask_w + cx1 + c);
                obj.mask        = mask_crop;
                obj.mask_width  = (unsigned int)cw;
                obj.mask_height = (unsigned int)ch;
                obj.mask_size   = cw * ch * sizeof(float);
            }
        }
        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseSAM3Full);

// ── SAM3Open: non-image input layers ──────────────────────────────────────────

static size_t elemSize(NvDsInferDataType t)
{
    switch (t) {
    case INT64: return 8;
    case INT32: case FLOAT: return 4;
    case HALF: return 2;
    case INT8: case UINT8: return 1;
    default: return 0;
    }
}

extern "C" bool NvDsInferInitializeInputLayers(
    std::vector<NvDsInferLayerInfo> const &inputLayers,
    NvDsInferNetworkInfo const &networkInfo,
    unsigned int maxBatchSize)
{
    unsigned int fileRows = 0, fileSlots = 0;
    FILE *mf = fopen("/dev/shm/sam3open_text_meta.txt", "r");
    if (!mf || fscanf(mf, "%u %u", &fileRows, &fileSlots) != 2) {
        if (mf) fclose(mf);
        printf("[SAM3Open] MISSING/bad /dev/shm/sam3open_text_meta.txt (write it before pipeline.start())\n");
        return false;
    }
    fclose(mf);

    for (auto const &layer : inputLayers) {
        size_t es = elemSize(layer.dataType);
        int slots = layer.inferDims.numDims > 0 ? layer.inferDims.d[0] : 0;
        if (slots <= 0) {
            printf("[SAM3Open] layer %s has unresolved slot dim (d[0]=%d)\n", layer.layerName, slots);
            return false;
        }
        // per-slot bytes, from the layer's own dims; the engine's K may be
        // smaller than the snapshot's K_MAX, so rows are re-strided here
        size_t slotBytes = layer.inferDims.numElements * es / slots;
        size_t rowBytes = slotBytes * slots;
        size_t fileRowBytes = slotBytes * fileSlots;
        std::string path = std::string("/dev/shm/sam3open_") + layer.layerName + ".bin";

        FILE *f = fopen(path.c_str(), "rb");
        if (!f) {
            printf("[SAM3Open] MISSING snapshot %s (write it before pipeline.start())\n", path.c_str());
            return false;
        }
        std::vector<char> src(fileRowBytes * fileRows);
        size_t rd = fread(src.data(), 1, src.size(), f);
        fclose(f);
        if (rd != src.size()) {
            printf("[SAM3Open] SHORT read %s: %zu/%zu bytes\n", path.c_str(), rd, src.size());
            return false;
        }
        char *dst = (char *)layer.buffer;
        memset(dst, 0, rowBytes * maxBatchSize);
        size_t copySlots = std::min((size_t)slots, (size_t)fileSlots);
        if ((size_t)slots < fileSlots)
            printf("[SAM3Open] WARNING: engine has %d class slots < snapshot K_MAX=%u, extra classes are dropped\n", slots, fileSlots);
        for (unsigned int r = 0; r < maxBatchSize && r < fileRows; ++r)
            memcpy(dst + r * rowBytes, src.data() + r * fileRowBytes, copySlots * slotBytes);
        printf("[SAM3Open] loaded %s: %u rows x %d slots x %zu bytes (file: %u slots)\n",
               layer.layerName, maxBatchSize, slots, slotBytes, fileSlots);
    }
    fflush(stdout);
    return true;
}
