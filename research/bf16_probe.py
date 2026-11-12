"""Can the set-classes engine be BF16 inside (fp16 in and out, the engine contract)? Runs export.py with its fp16 converter swapped for the bf16 one below.

    HF_HUB_OFFLINE=1 python research/bf16_probe.py --mode set-classes --classes person --imgsz 420 --output-dir weight/sam3_420_bf16-probe_person
    python research/bf16_probe.py --rebuild weight/sam3_420_bf16-probe_person 420     # after editing KEEP_FP16
    python test/golden.py check weight/sam3_420_bf16-probe_person/sam3.engine 420 test/golden/person_420.npz

Result on TensorRT 11.4, RTX 3050 Ti (2026-11-12): it builds once ConvTranspose stays fp16, but every detection is NaN (not root-caused) and it is slower
than fp16 (106 vs 84 ms at batch 1, 394 vs 318 ms at batch 4). FP16 stays. Notes: docs/NOTES.md
"""
import os, sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import export


def to_bf16_bits(a):
    """float32 -> the upper 16 bits, round to nearest even (bf16 = fp32 without its low 16 mantissa bits)."""
    b = np.ascontiguousarray(a, np.float32).view(np.uint32).astype(np.uint64)
    r = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)
    over = np.isfinite(a) & ((r & 0x7FFF) == 0x7F80)  # -FLT_MAX (the attention-mask fill) would round to -inf, then inf - inf = NaN in the softmax
    r[over] = (r[over] & 0x8000) | 0x7F7F
    return r


def convert_tensor(t):
    if t.data_type != TensorProto.FLOAT:
        return
    bits = to_bf16_bits(numpy_helper.to_array(t))
    t.ClearField("float_data"); t.ClearField("external_data"); t.data_location = TensorProto.DEFAULT
    t.data_type, t.raw_data = TensorProto.BFLOAT16, bits.tobytes()


KEEP_FP16 = {"ConvTranspose"}  # TensorRT 11.4 has no bf16 implementation for these: they run as fp16 islands between two Casts


def onnx_to_bf16(fp32_path, out_path):
    keep = os.path.join(os.path.dirname(out_path), "sam3_fp32.onnx")  # export.py converts in place: keep the fp32 graph for --rebuild
    if fp32_path != keep:
        os.replace(fp32_path, keep)
    m = onnx.load(keep)
    g = m.graph
    assert not any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for n in g.node for a in n.attribute), "subgraphs are not handled"
    inits = {t.name: t for t in g.initializer}
    nodes = []
    for n in g.node:
        if n.op_type not in KEEP_FP16:
            nodes.append(n); continue
        for k, x in enumerate(n.input):
            if x in inits:
                if inits[x].data_type == TensorProto.FLOAT:
                    inits[x].CopyFrom(numpy_helper.from_array(numpy_helper.to_array(inits[x]).astype(np.float16), x))
            elif x:
                nodes.append(helper.make_node("Cast", [x], [f"{n.name}_in{k}_fp16"], to=TensorProto.FLOAT16)); n.input[k] = f"{n.name}_in{k}_fp16"
        outs = list(n.output)
        n.output[:] = [o + "_fp16" for o in outs]
        nodes.append(n)
        nodes += [helper.make_node("Cast", [o + "_fp16"], [o], to=TensorProto.BFLOAT16) for o in outs]
    for t in g.initializer:
        convert_tensor(t)
    for n in nodes:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR:
                convert_tensor(a.t)
            if n.op_type == "Cast" and a.name == "to" and a.i == TensorProto.FLOAT:
                a.i = TensorProto.BFLOAT16
    del g.value_info[:]
    casts_in, casts_out = [], []
    for i in g.input:  # the engine contract: fp16 in and out, bf16 only inside
        if i.type.tensor_type.elem_type == TensorProto.FLOAT:
            i.type.tensor_type.elem_type = TensorProto.FLOAT16
            for n in nodes:
                n.input[:] = [i.name + "_bf16" if x == i.name else x for x in n.input]
            casts_in.append(helper.make_node("Cast", [i.name], [i.name + "_bf16"], to=TensorProto.BFLOAT16))
    for o in g.output:
        if o.type.tensor_type.elem_type == TensorProto.FLOAT:
            o.type.tensor_type.elem_type = TensorProto.FLOAT16
            for n in nodes:
                n.output[:] = [o.name + "_bf16" if x == o.name else x for x in n.output]
                n.input[:] = [o.name + "_bf16" if x == o.name else x for x in n.input]
            casts_out.append(helper.make_node("Cast", [o.name + "_bf16"], [o.name], to=TensorProto.FLOAT16))
    nodes = casts_in + nodes + casts_out
    del g.node[:]; g.node.extend(nodes)
    onnx.save(m, out_path)
    print(f"[BF16] {out_path}  ({os.path.getsize(out_path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    x = np.array([1.0, -2.5, 3.14159274, 1e-3, 65504.0 * 4], np.float32)
    back = (to_bf16_bits(x).astype(np.uint32) << 16).view(np.float32)
    assert np.all(np.abs(back - x) <= np.abs(x) * 2.0 ** -8), back  # 8 significant bits
    assert to_bf16_bits(np.float32([-3.4028235e38]))[0] == 0xFF7F
    if sys.argv[1] == "--rebuild":  # --rebuild DIR IMGSZ: convert DIR/sam3_fp32.onnx again (after editing KEEP_FP16) and build
        d, S = sys.argv[2], sys.argv[3]
        onnx_to_bf16(os.path.join(d, "sam3_fp32.onnx"), os.path.join(d, "sam3.onnx"))
        export.run_trtexec(os.path.join(d, "sam3.onnx"), os.path.join(d, "sam3.engine"), {"pixel_values": (f"1x3x{S}x{S}", f"4x3x{S}x{S}", f"16x3x{S}x{S}")})
    else:
        export.onnx_to_fp16 = onnx_to_bf16
        export.main()
