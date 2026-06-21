# SPDX-License-Identifier: Apache-2.0
#
# Profile where the ~4s actually goes: vision-encode vs prefill vs decode.
#
# WHY THIS MATTERS BEFORE TENSORRT-LLM:
#   When you cut max_tokens 128->64, latency didn't change. That implies
#   most of your time is NOT in autoregressive decode (the thing TensorRT-LLM
#   speeds up most). This profiler confirms where the time really is, so you
#   don't spend days building a TRT-LLM engine that only fixes 20% of the time.
#
# Method: CUDA events around the coarse stages. We can't see inside the
# upstream model's private methods without editing them, so we measure:
#   (A) total end-to-end
#   (B) a decode-length sweep: run with max_tokens = 16, 32, 64, 128.
#       If time scales with tokens -> decode-bound (TRT-LLM helps a lot).
#       If time is flat -> prefill/vision-bound (TRT-LLM helps less).
# ─────────────────────────────────────────────────────────────────────────────
import sys
import time
import torch
import numpy as np

sys.path.append("/home/acf-thor/SHIVAM/alpamayo/src")
sys.path.append("/home/acf-thor/SHIVAM/alpamayo")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
TOKEN_SWEEP = [16, 32, 64, 128]
REPEATS = 3   # average over N runs per setting to reduce noise


# Load the clip data ONCE (expensive), but build FRESH model_inputs per call.
# The model mutates tokenized_data (pops "input_ids"), so each timed run needs
# its own fresh copy or the 2nd call fails with KeyError.
_DATA_CACHE = {}

def get_data():
    if CLIP_ID not in _DATA_CACHE:
        _DATA_CACHE[CLIP_ID] = load_physical_aiavdataset(CLIP_ID)
    return _DATA_CACHE[CLIP_ID]

def build_inputs(model, processor):
    data = get_data()
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    return helper.to_device(model_inputs, "cuda")


def time_run(model, processor, max_tokens):
    # Build FRESH inputs each call — the model pops "input_ids" from the dict.
    model_inputs = build_inputs(model, processor)
    torch.cuda.manual_seed_all(42)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs, top_p=0.9, temperature=0.6,
            num_traj_samples=1, max_generation_length=max_tokens,
            return_extra=False,
        )
    torch.cuda.synchronize()
    return time.time() - t0


def main():
    print("Loading BF16 model...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16,
    ).to("cuda").eval()
    processor = helper.get_processor(model.tokenizer)

    # Warmup (first call is always slower)
    print("Warmup...")
    time_run(model, processor, 16)

    print("\n" + "=" * 60)
    print("TOKEN SWEEP — does latency scale with generated tokens?")
    print("=" * 60)
    timings = {}
    for n in TOKEN_SWEEP:
        runs = [time_run(model, processor, n) for _ in range(REPEATS)]
        avg = sum(runs) / len(runs)
        timings[n] = avg
        print(f"  max_tokens={n:>4}:  {avg:.2f}s  (runs: {[f'{r:.2f}' for r in runs]})")

    # Analysis: linear fit time = base + per_token * n
    ns = np.array(TOKEN_SWEEP, dtype=float)
    ts = np.array([timings[n] for n in TOKEN_SWEEP])
    per_token, base = np.polyfit(ns, ts, 1)
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(f"  Fixed cost (vision+prefill, extrapolated to 0 tokens): {base:.2f}s")
    print(f"  Per-token decode cost: {per_token*1000:.1f}ms/token")
    print(f"  Decode cost at 128 tokens: {per_token*128:.2f}s")
    frac_fixed = base / timings[128]
    print(f"  Fixed stage is ~{frac_fixed*100:.0f}% of your 128-token latency")
    print()
    if frac_fixed > 0.6:
        print("  VERDICT: PREFILL/VISION-BOUND.")
        print("  TensorRT-LLM (decode optimizer) will only help the smaller part.")
        print("  Bigger win: optimize the vision encoder (TensorRT ONNX export of")
        print("  the ViT) and/or reduce image tokens (fewer frames / lower res).")
    elif frac_fixed < 0.35:
        print("  VERDICT: DECODE-BOUND.")
        print("  TensorRT-LLM is the right tool — most time is autoregressive decode.")
    else:
        print("  VERDICT: MIXED. Both stages matter.")
        print("  TensorRT-LLM helps decode; also look at vision encoder.")
    print("=" * 60)


if __name__ == "__main__":
    main()