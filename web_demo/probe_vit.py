# SPDX-License-Identifier: Apache-2.0
#
# STEP 1 of ViT TensorRT export: probe the vision encoder's real I/O contract.
#
# We cannot guess the ViT's input shape — Qwen3-VL vision towers take a
# flattened patch tensor + a grid_thw tensor (temporal/height/width grid),
# NOT raw images. Exporting ONNX with wrong dummy shapes produces a broken
# engine. This script hooks the real vision module during a real forward pass
# and records exactly what tensors flow in and out.
#
# Run this FIRST. It prints the shapes/dtypes you'll paste into the exporter.
# ─────────────────────────────────────────────────────────────────────────────
import sys
import torch

sys.path.append("/home/acf-thor/SHIVAM/alpamayo/src")
sys.path.append("/home/acf-thor/SHIVAM/alpamayo")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"


def describe(x, name):
    if isinstance(x, torch.Tensor):
        return f"{name}: Tensor shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"
    if isinstance(x, (tuple, list)):
        return f"{name}: {type(x).__name__}[{len(x)}] -> " + \
               " | ".join(describe(e, f"{name}[{i}]") for i, e in enumerate(x))
    if isinstance(x, dict):
        return f"{name}: dict keys={list(x.keys())}"
    return f"{name}: {type(x).__name__} = {x}"


def main():
    print("Loading BF16 model...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16,
    ).to("cuda").eval()
    processor = helper.get_processor(model.tokenizer)

    # Locate the vision module. From the FP8 dump we know it's vlm.model.visual
    vis = None
    for name, mod in model.named_modules():
        if name.endswith("model.visual") or name.endswith(".visual"):
            vis = mod
            vis_name = name
            break
    if vis is None:
        print("Could NOT find vision module. Dumping top-level vlm children:")
        for name, _ in model.named_children():
            print(" ", name)
        # try vlm submodule
        if hasattr(model, "vlm"):
            for name, _ in model.vlm.named_children():
                print("  vlm.", name)
        return
    print(f"\nFound vision module at: {vis_name}")
    print(f"Vision module class: {type(vis).__name__}")

    # Hook the vision module to capture its actual inputs/outputs
    captured = {}

    def hook(module, args, kwargs, output):
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["output"] = output

    h = vis.register_forward_hook(hook, with_kwargs=True)

    # Run a real forward pass with 1 camera x 1 frame (smallest) to capture shapes
    data = load_physical_aiavdataset(CLIP_ID)
    sub = data["image_frames"][:1, :1]   # (1,1,3,H,W)
    flat = sub.flatten(0, 1)
    messages = helper.create_message(flat)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    print("\nRunning forward pass to capture vision I/O...")
    torch.cuda.manual_seed_all(42)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        try:
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=0.9, temperature=0.6,
                num_traj_samples=1, max_generation_length=16, return_extra=False,
            )
        except Exception as e:
            print(f"(forward raised after hook fired, ok if shapes captured: {e})")

    h.remove()

    print("\n" + "=" * 70)
    print("VISION ENCODER I/O CONTRACT")
    print("=" * 70)
    if not captured:
        print("Hook did not fire — vision module may be called differently.")
        return

    print("\n--- POSITIONAL ARGS ---")
    for i, a in enumerate(captured.get("args", ())):
        print(" ", describe(a, f"arg[{i}]"))
    print("\n--- KEYWORD ARGS ---")
    for k, v in captured.get("kwargs", {}).items():
        print(" ", describe(v, f"kwarg['{k}']"))
    print("\n--- OUTPUT ---")
    print(" ", describe(captured.get("output"), "output"))

    print("\n" + "=" * 70)
    print("Forward signature of the vision module:")
    import inspect
    try:
        print(" ", inspect.signature(vis.forward))
    except Exception as e:
        print(f"  (couldn't introspect: {e})")
    print("=" * 70)
    print("\nPaste this output back — the exporter needs these exact shapes/dtypes.")


if __name__ == "__main__":
    main()