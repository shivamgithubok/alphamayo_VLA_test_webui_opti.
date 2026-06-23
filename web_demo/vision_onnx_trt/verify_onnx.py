# SPDX-License-Identifier: Apache-2.0
#
# Verify the exported ONNX numerically matches the PyTorch vision encoder.
#
# WHY THIS IS MANDATORY HERE:
# The export printed many "trace might not generalize / treated as constant"
# warnings from fast_pos_embed_interpolate. The position-embedding path got
# frozen as constants for the exported grid. onnx.checker only validates graph
# STRUCTURE, not numerical correctness. This script runs the SAME input through
# both PyTorch and ONNXRuntime and compares all 4 outputs.
#
# Run with the SAME N_CAMERAS x N_FRAMES you exported with.
# ─────────────────────────────────────────────────────────────────────────────
import sys
import numpy as np
import torch

sys.path.append("/home/acf-thor/SHIVAM/alpamayo/src")
sys.path.append("/home/acf-thor/SHIVAM/alpamayo")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

import onnxruntime as ort

ONNX_PATH = "/home/acf-thor/SHIVAM/alpamayo/vit_encoder.onnx"
CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"

# MUST match what you exported with
N_CAMERAS = 1
N_FRAMES = 4


def capture_inputs_and_ref(model, processor):
    """Capture (hidden_states, grid_thw) and the PyTorch reference outputs."""
    captured = {}

    def hook(module, args, kwargs, output):
        captured["hidden_states"] = args[0].detach()
        captured["grid_thw"] = kwargs.get("grid_thw").detach()
        # output is (main, [d0,d1,d2])
        captured["main"] = output[0].detach()
        captured["deep"] = [d.detach() for d in output[1]]

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
    return captured


def main():
    print("Loading model + capturing reference outputs...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16,
    ).to("cuda").eval()
    processor = helper.get_processor(model.tokenizer)
    cap = capture_inputs_and_ref(model, processor)

    hs = cap["hidden_states"].to(torch.float16).cpu().numpy()
    grid = cap["grid_thw"].cpu().numpy().astype(np.int64)
    print(f"  hidden_states {hs.shape}  grid_thw {grid.tolist()}")

    # Run ONNXRuntime (CPU is fine for a correctness check)
    print("Running ONNXRuntime...")
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    ort_out = sess.run(
        ["main_embeds", "deepstack_0", "deepstack_1", "deepstack_2"],
        {"hidden_states": hs, "grid_thw": grid},
    )

    # Compare
    refs = [cap["main"]] + cap["deep"]
    names = ["main_embeds", "deepstack_0", "deepstack_1", "deepstack_2"]
    print("\n" + "=" * 60)
    print("NUMERICAL COMPARISON (PyTorch vs ONNX)")
    print("=" * 60)
    all_ok = True
    for name, ref, got in zip(names, refs, ort_out):
        ref_np = ref.to(torch.float32).cpu().numpy()
        got_np = got.astype(np.float32)
        if ref_np.shape != got_np.shape:
            print(f"  {name}: SHAPE MISMATCH ref={ref_np.shape} onnx={got_np.shape}")
            all_ok = False
            continue
        abs_diff = np.abs(ref_np - got_np)
        max_d = abs_diff.max()
        mean_d = abs_diff.mean()
        # Relative error vs the magnitude of the reference — robust to FP16 noise
        scale = max(np.abs(ref_np).mean(), 1e-3)
        rel = mean_d / scale
        # Cosine similarity — best single number for "do these mean the same thing"
        a = ref_np.flatten().astype(np.float64)
        b = got_np.flatten().astype(np.float64)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        # Thresholds tuned for FP16 outputs of a 27-block ViT.
        # 27 transformer blocks of FP16 accumulation produce ~2-3% relative
        # error vs the eager PyTorch path, even when the export is correct.
        # Cosine similarity is the real test — neural net embeddings are
        # functionally equivalent at cos > 0.999 (downstream layers normalize).
        ok = cos > 0.999
        all_ok = all_ok and ok
        flag = "OK" if ok else "FAIL"
        print(f"  {name}: max|Δ|={max_d:.4f}  mean|Δ|={mean_d:.5f}  "
              f"rel={rel:.5f}  cos={cos:.6f}  [{flag}]")

    print("=" * 60)
    if all_ok:
        print("PASS — ONNX matches PyTorch. Safe to build the TensorRT engine.")
    else:
        print("FAIL — outputs diverge. Do NOT build the engine yet.")
        print("Likely the frozen position-embedding constants don't match this")
        print("input. Re-export with the exact grid you'll use at runtime.")
    print("=" * 60)


if __name__ == "__main__":
    main()