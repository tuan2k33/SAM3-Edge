"""
Text-encoder + non-image-input-snapshot helper for the SAM3Open + DeepStream
pipeline (ds9_rtsp.py).

Runs ../weight/sam3_1008_open/text_encoder.engine directly via TensorRT --
NOT through nvinfer, which only ever runs the vision decoder. The text
encoder runs once per unique prompt (cached), here in Python, and the
result is written into two flat snapshot files that
nvdsparsebbox_sam3.cpp's NvDsInferInitializeInputLayers reads at every
nvinfer context (re)load:

    /dev/shm/sam3open_text_features.bin   MAX_SOURCES x K_MAX x 32 x 256  float32
    /dev/shm/sam3open_text_mask.bin       MAX_SOURCES x K_MAX x 32        uint8

Row b = camera b; slot k = its k-th class (a camera's prompt is a comma-
separated class list). Unused cameras/slots stay zero: a slot whose mask is
all zero scores 0 inside the graph, so it never produces detections.

MAX_SOURCES must match vision_decoder.engine's max optimization-profile
batch (currently 4 -- see SAM3Open/export.py / the --max-batch used at
trtexec build time). Rebuild the engine with a larger max batch to raise
the camera limit.
"""
import os

import numpy as np
import torch
import tensorrt as trt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
WEIGHT_DIR = os.path.join(_ROOT, "weight")

MAX_SOURCES = 4
K_MAX = 5  # class slots per camera; must match vision_decoder.engine (export.py --k-max)
CTX_LEN = 32
FEAT_DIM = 256
IMGSZ = 1008  # vision_decoder.engine was only ever exported at this size


def open_dir_for(imgsz=IMGSZ):
    return os.environ.get("SAM3OPEN_DIR") or os.path.join(WEIGHT_DIR, f"sam3_{imgsz}_open")


TEXT_ENGINE = os.path.join(open_dir_for(), "text_encoder.engine")

SNAPSHOT_FEATURES = "/dev/shm/sam3open_text_features.bin"
SNAPSHOT_MASK = "/dev/shm/sam3open_text_mask.bin"
SNAPSHOT_META = "/dev/shm/sam3open_text_meta.txt"  # "MAX_SOURCES K_MAX": file row stride for the C++ reader


class TextEncoder:
    """Loads text_encoder.engine once, encodes prompt strings on demand."""

    def __init__(self):
        from transformers.models.sam3 import Sam3Processor
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        logger = trt.Logger(trt.Logger.ERROR)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(open(TEXT_ENGINE, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self._cache = {}  # prompt string -> (text_features[32,256] f32, text_mask[32] u8)

    def encode(self, prompt: str):
        """Returns (text_features[32,256] float32, text_mask[32] uint8) numpy arrays."""
        if prompt in self._cache:
            return self._cache[prompt]
        tok = self.processor(text=prompt, return_tensors="pt")
        ids = tok["input_ids"].cuda()
        mask = tok["attention_mask"].cuda()
        self.ctx.set_input_shape("input_ids", ids.shape)
        self.ctx.set_input_shape("attention_mask", mask.shape)
        self.ctx.set_tensor_address("input_ids", ids.data_ptr())
        self.ctx.set_tensor_address("attention_mask", mask.data_ptr())
        feat = torch.zeros((1, CTX_LEN, FEAT_DIM), dtype=torch.float32, device="cuda")
        tmask = torch.zeros((1, CTX_LEN), dtype=torch.uint8, device="cuda")
        self.ctx.set_tensor_address("text_features", feat.data_ptr())
        self.ctx.set_tensor_address("text_mask", tmask.data_ptr())
        stream = torch.cuda.Stream()
        self.ctx.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        out = (feat[0].cpu().numpy().astype(np.float32), tmask[0].cpu().numpy().astype(np.uint8))
        self._cache[prompt] = out
        return out


def split_classes(prompt: str):
    """'car, truck' -> ['car', 'truck'] (empty items dropped, order kept)."""
    return [c.strip() for c in prompt.split(",") if c.strip()]


def write_snapshot(encoder: TextEncoder, camera_classes: dict):
    """camera_classes: dict cam_id (0..MAX_SOURCES-1) -> list of class strings.
    Cameras not present, or with index >= MAX_SOURCES, are left zeroed.
    Files must be fully written BEFORE the nvinfer context that reads them
    is (re)loaded -- the caller triggers that load/reload AFTER this
    returns (pipeline.start() the first time, nvinfer.set({"config-file-path":
    ...}) on later prompt changes)."""
    feat_full = np.zeros((MAX_SOURCES, K_MAX, CTX_LEN, FEAT_DIM), dtype=np.float32)
    mask_full = np.zeros((MAX_SOURCES, K_MAX, CTX_LEN), dtype=np.uint8)
    for cam, classes in camera_classes.items():
        if cam >= MAX_SOURCES:
            print(f"[sam3open_text] WARNING: cam{cam} exceeds MAX_SOURCES={MAX_SOURCES}, ignored")
            continue
        if len(classes) > K_MAX:
            print(f"[sam3open_text] WARNING: cam{cam} has {len(classes)} classes > K_MAX={K_MAX}, extra ignored")
        for k, cls in enumerate(classes[:K_MAX]):
            feat, mask = encoder.encode(cls)
            feat_full[cam, k] = feat
            mask_full[cam, k] = mask
    feat_full.tofile(SNAPSHOT_FEATURES)
    mask_full.tofile(SNAPSHOT_MASK)
    with open(SNAPSHOT_META, "w") as f:
        f.write(f"{MAX_SOURCES} {K_MAX}\n")
