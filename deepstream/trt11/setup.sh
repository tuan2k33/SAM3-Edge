#!/usr/bin/env bash
# DeepStream 9.2 for this machine's TensorRT 11 and the project's Python (.venv, 3.14), no sudo. DeepStream 9.2 ships for TensorRT 10.16 + Python 3.12.
# What it does, and why each step is needed: docs/NOTES.md, section 3.   Run from the repo root:  bash deepstream/trt11/setup.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
B=${SAM3_DS_HOME:-$HOME/.dev/SAM3-Edge-deepstream}
D=$B/opt/nvidia/deepstream/deepstream-9.2
PY=$ROOT/.venv/bin/python
CUDA_VER=${CUDA_VER:-$(basename "$(readlink -f /usr/local/cuda)" | sed 's/cuda-//')}
SYS=/usr/lib/x86_64-linux-gnu
mkdir -p "$B/shim" "$B/build"

# 1. the SDK tarball (public, 1.4 GB), unpacked as it is
if [ ! -d "$D" ]; then
    curl -L -o "$B/ds.tbz2" https://github.com/NVIDIA/DeepStream/releases/download/v9.2.0/deepstream_sdk_v9.2.0_x86_64.tbz2
    tar xjf "$B/ds.tbz2" -C "$B" && rm "$B/ds.tbz2"
fi

# 2. what install.sh does as root, as symlinks: NVIDIA's libv4l2 (the nvv4l2 decoder/encoder), and the TensorRT 10 names
#    the closed libraries were linked with (the tracker only imports three C entry points that TensorRT 11 still has)
ln -sf "$D/lib/libnvv4l2.so" "$B/shim/libv4l2.so.0"
ln -sf "$D/lib/libnvv4lconvert.so" "$B/shim/libv4lconvert.so.0"
for n in nvinfer nvinfer_plugin nvonnxparser; do ln -sf "$SYS/lib$n.so.11" "$B/shim/lib$n.so.10"; done

# 3. nvinfer's library and the logger object it hands to TensorRT, rebuilt from the SDK sources against TensorRT 11
#    (both hold TensorRT C++ objects, whose layout changed; the stock logger segfaults inside initLibNvInferPlugins)
for lib in nvdsinfer_logger nvdsinfer; do
    rm -rf "$B/build/$lib" && cp -r "$D/sources/libs/$lib" "$B/build/$lib"
    sed -i "s|-I *\.\./\.\./\.\./includes|-I $D/sources/includes -I $D/includes|" "$B/build/$lib/Makefile"
done
python3 "$ROOT/deepstream/trt11/patch_nvdsinfer.py" "$B/build/nvdsinfer"
for lib in nvdsinfer_logger nvdsinfer; do
    so=lib$(echo $lib | sed 's/nvdsinfer_logger/nvds_inferlogger/; s/^nvdsinfer$/nvds_infer/').so
    make -C "$B/build/$lib" CUDA_VER="$CUDA_VER" LIB_INSTALL_DIR="$D/lib/" -j4 > "$B/build/$lib.log" 2>&1 || { tail -20 "$B/build/$lib.log"; exit 1; }
    [ -f "$D/lib/$so.trt10" ] || cp "$D/lib/$so" "$D/lib/$so.trt10"     # the stock library, kept
    cp "$B/build/$lib/$so" "$D/lib/$so"
done

# 4. pyservicemaker for the project's Python: the SDK only has a cp312 wheel; pybind11 3 is the first that supports 3.14
"$PY" -m pip --version > /dev/null 2>&1 && "$PY" -m pip install -q pybind11 pyyaml || uv pip install -q --python "$PY" pybind11 pyyaml
rm -rf "$B/build/pysm" && cp -r "$D/service-maker/sources/python/src" "$B/build/pysm"
python3 "$ROOT/deepstream/trt11/patch_pyservicemaker.py" "$B/build/pysm"
mkdir -p "$B/build/dlpack/dlpack"
curl -sL -o "$B/build/dlpack/dlpack/dlpack.h" https://raw.githubusercontent.com/dmlc/dlpack/v1.0/include/dlpack/dlpack.h
pyc() { "$PY" -c "import sysconfig, pybind11; $1"; }
(cd "$B/build/pysm" && g++ -shared -fPIC -O2 -std=c++17 -DHAVE_CONFIG_H -DHAVE_X11 -Wno-deprecated-declarations -o pyservicemaker/_pydeepstream.so bindings/main.cpp \
    -Ibindings -Ibindings/docstrings -I"$(pyc 'print(pybind11.get_include())')" -I"$B/build/dlpack" -I"$(pyc 'print(sysconfig.get_paths()["include"])')" \
    $(pkg-config --cflags gstreamer-1.0) -I"$D/service-maker/includes" -I"$D/sources/includes" -I"$D/includes" \
    -L"$D/lib" -L"$(pyc 'print(sysconfig.get_config_var("LIBDIR"))')" -lnvds_service_maker -lnvds_service_maker_utils -lgstreamer-1.0 -lglib-2.0 -lgobject-2.0 \
    -lpython"$(pyc 'print(sysconfig.get_python_version())')" -Wl,-rpath,"$D/lib")
SP=$(pyc 'print(sysconfig.get_paths()["purelib"])')
rm -rf "$SP/pyservicemaker" && cp -r "$B/build/pysm/pyservicemaker" "$SP/"

# 5. the SAM3 nvinfer plugin
g++ -shared -fPIC -O2 -std=c++20 -o "$ROOT/deepstream/libnvdsinfer_sam3.so" "$ROOT/deepstream/nvdsparsebbox_sam3.cpp" -I"$D/includes" -I/usr/local/cuda/include

# 6. check
cd "$ROOT" && source deepstream/env_local.sh && rm -f "$GST_REGISTRY"
ldd -r "$D/lib/libnvds_infer.so" "$D/lib/libnvds_nvmultiobjecttracker.so" | grep -E "not found|undefined" && exit 1
gst-inspect-1.0 nvinfer > /dev/null 2>&1
"$PY" -c "import pyservicemaker, tensorrt, sys; print('ok: pyservicemaker on Python', sys.version.split()[0], '+ TensorRT', tensorrt.__version__)"
