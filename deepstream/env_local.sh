# source deepstream/env_local.sh   -- DeepStream 9.2 unpacked without sudo and rebuilt for TensorRT 11 + the project's Python (deepstream/trt11/setup.sh)
B=${SAM3_DS_HOME:-$HOME/.dev/SAM3-Edge-deepstream}
export DS_ROOT=$B/opt/nvidia/deepstream/deepstream-9.2
# $B/shim: libv4l2.so.0 -> NVIDIA's libnvv4l2.so (install.sh swaps the system one as root; the nvv4l2 decoder/encoder go through it), and
# libnvinfer.so.10 -> the system libnvinfer.so.11 for the closed libraries that only name it (the tracker)
export LIBV4L2_PLUGIN_DIR=$DS_ROOT/lib/libv4l/plugins
export LD_LIBRARY_PATH=$B/shim:$DS_ROOT/lib:/usr/local/cuda/lib64:$(.venv/bin/python -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))" 2>/dev/null)
export GST_PLUGIN_PATH=$DS_ROOT/lib/gst-plugins GST_REGISTRY=$B/gst-registry.bin
export DS_NVBUF_MEMTYPE=1   # WSL2
