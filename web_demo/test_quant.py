# SPDX-License-Identifier: Apache-2.0
#
# Test the CHEAPEST lever first: does reducing frames/cameras cut the
# vision+prefill cost (which the stage profiler showed is ~98% of latency)?
#
# Your pipeline feeds 4 cameras x 4 frames = 16 images. Each becomes many
# visual tokens. This script times inference with different subsets to see
# how latency scales with image count — BEFORE you invest in TensorRT ViT export.
#
# We vary how many of the 16 flattened frames we feed by trimming the
# image_frames tensor. NOTE: this is a LATENCY probe — trajectory quality
# with fewer frames must be validated separately before shipping.
# ─────────────────────────────────────────────────────────────────────────────
import sys
import time
import torch

sys.path.append("/home/acf-thor/SHIVAM/alpamayo/src")
sys.path.append("/home/acf-thor/SHIVAM/alpamayo")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
REPEATS = 3

# Configs to test: (n_cameras, n_frames_per_camera)
# Original is (4, 4) = 16 images.
CONFIGS = [
    (4, 4),   # baseline — all cameras, all frames
    (4, 2),   # all cameras, fewer temporal frames
    (4, 1),   # all cameras, single frame each
    (2, 4),   # half the cameras, all frames
    (1, 4),   # single camera, all frames
    (1, 1),   # absolute minimum
]


def main():
    print("Loading BF16 model...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16,
    ).to("cuda").eval()
    processor = helper.get_processor(model.tokenizer)

    data = load_physical_aiavdataset(CLIP_ID)
    full_frames = data["image_frames"]   # (N_cam=4, n_frames=4, 3, H, W)
    print(f"Full image_frames shape: {tuple(full_frames.shape)}")

    def run(n_cam, n_frm):
        # Slice cameras and frames
        sub = full_frames[:n_cam, :n_frm]          # (n_cam, n_frm, 3, H, W)
        flat = sub.flatten(0, 1)                     # (n_cam*n_frm, 3, H, W)
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
        torch.cuda.manual_seed_all(42)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs, top_p=0.9, temperature=0.6,
                num_traj_samples=1, max_generation_length=64,
                return_extra=False,
            )
        torch.cuda.synchronize()
        return time.time() - t0

    # Warmup
    print("Warmup...")
    try:
        run(1, 1)
    except Exception as e:
        print(f"(warmup note: {e})")

    print("\n" + "=" * 60)
    print("FRAME/CAMERA SWEEP — does latency scale with image count?")
    print("=" * 60)
    baseline = None
    for n_cam, n_frm in CONFIGS:
        n_img = n_cam * n_frm
        try:
            runs = [run(n_cam, n_frm) for _ in range(REPEATS)]
            avg = sum(runs) / len(runs)
            if baseline is None:
                baseline = avg
            speedup = baseline / avg
            print(f"  {n_cam} cam x {n_frm} frм = {n_img:>2} imgs:  "
                  f"{avg:.2f}s   ({speedup:.2f}x vs baseline)")
        except Exception as e:
            print(f"  {n_cam} cam x {n_frm} frм = {n_img:>2} imgs:  FAILED — {e}")

    print("=" * 60)
    print("If latency drops sharply with fewer images -> vision-bound confirmed,")
    print("and trimming frames is your cheapest speedup (validate accuracy after).")
    print("If latency is flat -> the cost is fixed prefill, go to TensorRT ViT.")
    print("=" * 60)


if __name__ == "__main__":
    main()