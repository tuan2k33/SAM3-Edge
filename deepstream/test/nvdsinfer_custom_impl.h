// Just enough of DeepStream's nvdsinfer_custom_impl.h to compile and unit-test nvdsparsebbox_sam3.cpp without DeepStream (build with -I this directory).
// Never use it for the real plugin: that needs DeepStream's own headers.
#pragma once
#include <vector>

enum NvDsInferDataType { FLOAT, HALF, INT8, INT32, INT64, UINT8 };
struct NvDsInferDims { unsigned int numDims; unsigned int d[8]; unsigned int numElements; };
struct NvDsInferLayerInfo { NvDsInferDataType dataType; NvDsInferDims inferDims; const char *layerName; void *buffer; };
struct NvDsInferNetworkInfo { unsigned int width, height, channels; };
struct NvDsInferParseDetectionParams { unsigned int numClassesConfigured; std::vector<float> perClassThreshold, perClassPreclusterThreshold; };
struct NvDsInferObjectDetectionInfo { unsigned int classId; float left, top, width, height, detectionConfidence; };
struct NvDsInferInstanceMaskInfo { unsigned int classId; float left, top, width, height, detectionConfidence; float *mask; unsigned int mask_width, mask_height, mask_size; };

#define CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(f) static_assert(true, #f)
#define CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(f) static_assert(true, #f)
