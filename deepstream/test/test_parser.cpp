// g++ -std=c++20 -Wall -Wextra -I deepstream/test deepstream/test/test_parser.cpp -o /tmp/test_parser && /tmp/test_parser
#include <cassert>
#include <cstdio>

#include "../nvdsparsebbox_sam3.cpp"

static uint16_t to_half(float f)  // normal numbers only, truncating: enough for the test values
{
    uint32_t x; std::memcpy(&x, &f, 4);
    return f == 0 ? 0 : (uint16_t)(((x >> 16) & 0x8000) | ((((x >> 23) & 0xff) - 127 + 15) << 10) | ((x >> 13) & 0x3ff));
}

static NvDsInferLayerInfo layer(const char *name, NvDsInferDataType t, void *buf, unsigned n, unsigned a, unsigned b)
{
    return {t, {3, {n, a, b}, n * a * b}, name, buf};
}

int main()
{
    // x1 y1 x2 y2 score cls: 0 and 1 overlap (same class -> NMS drops the weaker), 2 is another class on the same box (kept), 3 is below the threshold
    float det[4][6] = {{10, 10, 50, 50, 0.9f, 0}, {12, 12, 52, 52, 0.8f, 0}, {10, 10, 50, 50, 0.7f, 1}, {60, 60, 90, 90, 0.1f, 0}};
    uint16_t det16[4][6]; for (int i = 0; i < 4; ++i) for (int j = 0; j < 6; ++j) det16[i][j] = to_half(det[i][j]);
    float mask[4][2][2]; uint16_t mask16[4][2][2];
    for (int i = 0; i < 4; ++i) for (int r = 0; r < 2; ++r) for (int c = 0; c < 2; ++c) { mask[i][r][c] = 0.25f * (r * 2 + c); mask16[i][r][c] = to_half(mask[i][r][c]); }
    NvDsInferNetworkInfo net = {100, 100, 3};
    NvDsInferParseDetectionParams params = {2, {}, {0.3f, 0.3f}};

    for (int half = 0; half < 2; ++half) {  // the same data as float32 and as float16 must parse alike
        std::vector<NvDsInferLayerInfo> out = {half ? layer("detections", HALF, det16, 4, 6, 1) : layer("detections", FLOAT, det, 4, 6, 1)};
        out[0].inferDims.numDims = 2;
        std::vector<NvDsInferObjectDetectionInfo> objs;
        assert(NvDsInferParseSAM3Det(out, net, params, objs));
        assert(objs.size() == 2);
        assert(objs[0].classId == 0 && std::abs(objs[0].detectionConfidence - 0.9f) < 2e-3f && std::abs(objs[0].left - 10) < 0.1f);
        assert(objs[1].classId == 1);
        std::vector<NvDsInferObjectDetectionInfo> objs2;
        assert(NvDsInferParseSam3OpenDet(out, net, params, objs2) && objs2.size() == objs.size());

        NvDsInferParseDetectionParams strict = {2, {}, {0.3f, 0.95f}};  // per-class threshold: class 1 (score 0.7) is dropped
        objs.clear();
        assert(NvDsInferParseSAM3Det(out, net, strict, objs) && objs.size() == 1 && objs[0].classId == 0);

        std::vector<NvDsInferLayerInfo> both = {out[0], half ? layer("masks", HALF, mask16, 4, 2, 2) : layer("masks", FLOAT, mask, 4, 2, 2)};
        std::vector<NvDsInferInstanceMaskInfo> inst;
        assert(NvDsInferParseSAM3Full(both, net, params, inst) && inst.size() == 2);
        // box 10..50 of 100 on a 2x2 mask: cells 0..1 -> a 1x1 crop of mask cell (0,0) = 0
        assert(inst[0].mask && inst[0].mask_width == 1 && inst[0].mask_height == 1 && std::abs(inst[0].mask[0]) < 1e-3f);
        for (auto &o : inst) std::free(o.mask);
    }
    std::puts("parser tests passed (float32 and float16)");
}
