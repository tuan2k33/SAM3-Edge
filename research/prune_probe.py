"""What does cutting sub-blocks buy in a real TensorRT engine? Runs export.py with sub-blocks replaced by zero (the residual passes through),
so they are not traced into the ONNX at all. The eager study (docs/NOTES.md section 8) found 24 of 106 that cost no recall.

    HF_HUB_OFFLINE=1 python research/prune_probe.py [--cut vit,enc,dec | --vit attention:22,mlp:27] --mode set-classes --classes person --imgsz 420 --output-dir weight/sam3_420_pruned_person
    python test/golden.py check weight/sam3_420_pruned_person/sam3.engine 420 test/golden/person_420.npz ; python research/eval_engine_person.py weight/sam3_420_pruned_person/sam3.engine 420
"""
import os, sys

import torch, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import export

CUT = {  # stage -> (module list, {attribute: layer indices}); the sets of docs/NOTES.md section 8
    "vit": (lambda m: m.vision_encoder.backbone.layers, {"attention": (25, 26, 27), "mlp": (28, 29, 30)}),
    "enc": (lambda m: m.detr_encoder.layers, {"self_attn": (0, 1), "cross_attn": (2, 3, 4, 5)}),
    "dec": (lambda m: m.detr_decoder.layers, {"self_attn": (4, 1, 3), "text_cross_attn": (4, 2, 1, 0), "vision_cross_attn": (5, 4), "mlp": (4, 0, 5)}),
}


Zero = export.Zero


if __name__ == "__main__":
    stages = ("vit", "enc", "dec")
    if "--cut" in sys.argv:
        i = sys.argv.index("--cut"); stages = sys.argv[i + 1].split(","); del sys.argv[i:i + 2]
    if "--vit" in sys.argv:  # --vit attention:22,mlp:27,...: these ViT sub-blocks only (the ranking of research/prune_search.py)
        i = sys.argv.index("--vit"); cuts = {}
        for item in sys.argv[i + 1].split(","):
            cuts.setdefault(item.split(":")[0], []).append(int(item.split(":")[1]))
        CUT["vit"], stages = (CUT["vit"][0], cuts), ("vit",); del sys.argv[i:i + 2]
    load = export.load
    def load_pruned(args):
        model, processor = load(args)
        for s in stages:
            layers, cuts = CUT[s][0](model), CUT[s][1]
            for attr, idx in cuts.items():
                for k in idx:
                    setattr(layers[k], attr, Zero(pair=attr != "mlp"))
            print(f"[Prune] {s}: " + ", ".join(f"{a}:{','.join(map(str, i))}" for a, i in cuts.items()))
        return model, processor
    export.load = load_pruned
    export.main()
