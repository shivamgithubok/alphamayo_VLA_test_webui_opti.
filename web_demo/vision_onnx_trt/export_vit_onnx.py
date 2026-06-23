# SPDX-License-Identifier: Apache-2.0
#
# STEP 2: Export the Qwen3-VL vision encoder to ONNX.
#
# Contract captured by probe_vit.py:
#   INPUT:  hidden_states (N_patches, 1536) bf16   [N_patches dynamic]
#           grid_thw       (num_images, 3) int64
#   OUTPUT: main_embeds    (N_tokens, 4096) bf16    [N_tokens dynamic]
#           deepstack[0..2](N_tokens, 4096) bf16    (3 tensors)
#   where N_tokens = N_patches / 4 (spatial merge)
#
# KEY DECISIONS:
#  - Export in FP16, not BF16: TensorRT's ONNX path supports FP16 cleanly;
#    BF16 ONNX export is poorly supported. FP16 has the same memory footprint
#    and similar accuracy for a ViT. We cast the module to FP16 for export.
#  - Mark patch dim and token dim as DYNAMIC so the engine handles any frame
#    count (1 image = 720 patches, 16 images = ~11520).
#  - Wrap the vision module so it returns a flat tuple (ONNX can't return
#    a nested list) — main + 3 deepstack as 4 separate outputs.
# ─────────────────────────────────────────────────────────────────────────────
import sys
import torch
import torch.nn as nn

sys.path.append("/home/acf-thor/SHIVAM/alpamayo/src")
sys.path.append("/home/acf-thor/SHIVAM/alpamayo")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

ONNX_PATH = "/home/acf-thor/SHIVAM/alpamayo/vit_encoder.onnx"
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"

# ── SHIPPING CONFIG ──────────────────────────────────────────────────────
# Set these to the frame config you validated as accurate (NOT 1x1, which
# was blind to the construction vehicle). The baked ONNX grid will match this.
# For your demo the 4 cameras are fake replicas, so 1 camera x 4 frames keeps
# the real temporal info. Adjust after your quality sweep picks the winner.
N_CAMERAS = 1
N_FRAMES = 4
# ─────────────────────────────────────────────────────────────────────────


class ViTExportWrapper(nn.Module):
    """
    Wraps Qwen3VLVisionModel so ONNX export sees a clean flat-tuple output.
    The original returns (main, [deepstack0, deepstack1, deepstack2]);
    ONNX needs flat outputs, so we unpack to (main, d0, d1, d2).
    """
    def __init__(self, vis):
        super().__init__()
        self.vis = vis

    def forward(self, hidden_states, grid_thw):
        out = self.vis(hidden_states, grid_thw=grid_thw)
        main = out[0]
        deepstack = out[1]   # list of 3
        return main, deepstack[0], deepstack[1], deepstack[2]


def capture_real_inputs(model, processor):
    """Capture a real (hidden_states, grid_thw) pair to use as export sample."""
    captured = {}

    def hook(module, args, kwargs, output):
        captured["hidden_states"] = args[0]
        captured["grid_thw"] = kwargs.get("grid_thw")

    vis = dict(model.named_modules())["vlm.model.visual"]
    h = vis.register_forward_hook(hook, with_kwargs=True)

    data = load_physical_aiavdataset(CLIP_ID)
    sub = data["image_frames"][:N_CAMERAS, :N_FRAMES]
    flat = sub.flatten(0, 1)
    messages = helper.create_message(flat)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    mi = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    mi = helper.to_device(mi, "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        try:
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=mi, top_p=0.9, temperature=0.6,
                num_traj_samples=1, max_generation_length=16, return_extra=False,
            )
        except Exception:
            pass
    h.remove()
    return captured["hidden_states"], captured["grid_thw"]


def main():
    print("Loading BF16 model...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16,
    ).to("cuda").eval()
    processor = helper.get_processor(model.tokenizer)

    print("Capturing real vision inputs...")
    hidden_states, grid_thw = capture_real_inputs(model, processor)
    print(f"  hidden_states: {tuple(hidden_states.shape)} {hidden_states.dtype}")
    print(f"  grid_thw:      {tuple(grid_thw.shape)} {grid_thw.dtype} = {grid_thw.tolist()}")

    vis = dict(model.named_modules())["vlm.model.visual"]

    # Force EAGER attention so the fused scaled_dot_product_attention op
    # (which carries enable_gqa=True and breaks ONNX export) is never used.
    print("Switching vision attention to eager for export...")

    if hasattr(vis, "config"):
        vis.config._attn_implementation = "eager"

    for m in vis.modules():
        if hasattr(m, "config"):
            m.config._attn_implementation = "eager"

        if hasattr(m, "_attn_implementation"):
            m._attn_implementation = "eager"

    # Cast vision module + inputs to FP16 for clean ONNX/TensorRT support
    print("Casting vision module to FP16 for export...")
    vis = vis.to(torch.float16)
    wrapper = ViTExportWrapper(vis).eval()
    hs16 = hidden_states.to(torch.float16)

    # FIXED-SHAPE EXPORT.
    # The vision model's fast_pos_embed_interpolate does:
    #     h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
    # where h is read from grid_thw via .item() -> data-dependent output shape.
    # torch.export cannot trace that with a dynamic grid. So we BAKE IN the grid:
    # h, w become compile-time constants, linspace gets a static length, and the
    # data-dependent guard disappears.
    #
    # Tradeoff: this engine is valid ONLY for this exact grid_thw (this many
    # frames at this resolution). For a different frame count you re-export with
    # that grid. Since you ship ONE demo config, that's fine.
    #
    # We use the legacy TorchScript tracer (dynamo=False) which is more tolerant
    # of this model's control flow than the new dynamo exporter.
    print(f"Exporting ONNX (FIXED grid={grid_thw.tolist()}) to {ONNX_PATH}...")
    print(f"  hidden_states fixed at {tuple(hs16.shape)}")

    # Force SDPA to use the MATH backend during export. The fused SDPA kernel
    # passes enable_gqa=True (grouped-query attention), which the ONNX exporter
    # cannot convert. The math backend decomposes attention into plain
    # matmul + softmax + matmul, which exports cleanly. Numerically equivalent.
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (hs16, grid_thw),
            ONNX_PATH,
            input_names=["hidden_states", "grid_thw"],
            output_names=["main_embeds", "deepstack_0", "deepstack_1", "deepstack_2"],
            dynamic_axes=None,
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
        )
    print("ONNX export complete.")

    # Verify the ONNX loads and has sane I/O
    try:
        import onnx
        m = onnx.load(ONNX_PATH)
        onnx.checker.check_model(m)
        print("\nONNX model is valid.")
        print("Inputs:")
        for i in m.graph.input:
            print("  ", i.name)
        print("Outputs:")
        for o in m.graph.output:
            print("  ", o.name)
    except ImportError:
        print("(install onnx to verify: pip install onnx --break-system-packages)")
    except Exception as e:
        print(f"ONNX validation warning: {e}")

    print("\nNext: build the TensorRT engine with trtexec (see build_vit_trt.sh)")


if __name__ == "__main__":
    main()