#!/usr/bin/env python3
"""Make DeepStream 9.2's sources/libs/nvdsinfer compile against TensorRT 11: wrap what TensorRT 11 removed in `#if NV_TENSORRT_MAJOR < 11`.
All of it is the build-an-engine-from-a-model path (INT8 calibrator, kFP16/kINT8 builder flags, ITensor::setType, ILayer::setPrecision, kEXPLICIT_BATCH):
TensorRT 11 is strongly typed, the precision comes from the model. Loading a prebuilt engine is untouched.

    python3 patch_nvdsinfer.py <copy of sources/libs/nvdsinfer>     (line numbers are those of DeepStream 9.2.0)
"""
import os, sys

RANGES = {  # file -> [(first line, last line, text that must be on the first line)]
    "nvdsinfer_context_impl.cpp": [(60, 108, "class NvDsInferInt8Calibrator"), (2209, 2215, "int8CalibrationFilePath")],
    "nvdsinfer_model_builder.h": [(269, 272, "setInt8Calibrator"), (344, 344, "m_Int8Calibrator")],
    "nvdsinfer_model_builder.cpp": [(419, 454, "NvDsInferNetworkMode_INT8"), (456, 470, "NvDsInferNetworkMode_FP16"), (730, 732, "netDefFlags |="),
                                    (931, 941, "g_fp16FlagWasSet"), (985, 985, "setType"), (998, 998, "setType"), (1017, 1017, "setPrecision"),
                                    (1026, 1059, "NvDsInferNetworkMode_INT8"), (1061, 1074, "NvDsInferNetworkMode_FP16"), (1087, 1093, "NvDsInferNetworkMode_INT8")],
}
for name, ranges in RANGES.items():
    path = os.path.join(sys.argv[1], name)
    lines = open(path).read().split("\n")
    assert "NV_TENSORRT_MAJOR < 11" not in "\n".join(lines), f"{name} is already patched"
    for a, b, must in sorted(ranges, reverse=True):
        assert must in lines[a - 1], f"{name}:{a} is not the expected line (another DeepStream version?): {lines[a - 1]!r}"
        lines[a - 1:b] = ["#if NV_TENSORRT_MAJOR < 11"] + lines[a - 1:b] + ["#endif"]
    open(path, "w").write("\n".join(lines))
    print(f"patched {name}: {len(ranges)} blocks")
