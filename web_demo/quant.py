# SPDX-License-Identifier: Apache-2.0
#
# Correct FP8 quantization for Alpamayo-R1 using nvidia-modelopt.
#
# WHY THE PREVIOUS ATTEMPT FAILED:
#   save_pretrained() saved BF16 weights + orphaned _amax tensors.
#   from_pretrained() rebuilds the PLAIN architecture with no quantizer
#   modules, so the _amax scale factors have nowhere to attach and get
#   discarded ("weights not used" warning). Result: model runs in BF16.
#
# THE FIX:
#   modelopt quantization MODIFIES THE MODULE STRUCTURE (inserts TensorQuantizer
#   layers). You must save/restore that structure with modelopt's own
#   mto.save / mto.restore — NOT HF save_pretrained / from_pretrained.
#
# This script:
#   1. Loads the BF16 model
#   2. Runs calibration (a few real forward passes so _amax values are real,
#      not random — calibration is what makes FP8 accurate)
#   3. Quantizes to FP8
#   4. Saves with mto.save (preserves quantizer modules + scales together)
# ─────────────────────────────────────────────────────────────────────────────
import sys
import torch

sys.path.append("/home/acf-thor/SHIVAM/alpamayo/src")
sys.path.append("/home/acf-thor/SHIVAM/alpamayo")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto

OUT_PATH = "/home/acf-thor/SHIVAM/alpamayo/alpamayo_fp8_modelopt.pt"

# A few clip IDs for calibration — replace with real ones from your dataset
CALIB_CLIP_IDS = [
    "030c760c-ae38-49aa-9ad8-f5650a545d26",
    # add 2-4 more clip ids here for better calibration
]


def main():
    print("Loading BF16 model...")
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    processor = helper.get_processor(model.tokenizer)

    # ── Calibration forward loop ─────────────────────────────────────────
    # modelopt observes activation ranges during these passes to compute
    # the _amax scale factors. WITHOUT this, FP8 accuracy is garbage.
    def forward_loop(m):
        for clip_id in CALIB_CLIP_IDS:
            print(f"  calibrating on {clip_id}...")
            data = load_physical_aiavdataset(clip_id)
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
            model_inputs = helper.to_device(model_inputs, "cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                m.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.9, temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=64,   # short is fine for calibration
                    return_extra=False,
                )

    # ── Quantize ─────────────────────────────────────────────────────────
    print("Quantizing to FP8 (this runs the calibration loop)...")
    model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, forward_loop)

    # Print a quick summary of what got quantized
    mtq.print_quant_summary(model)

    # ── Save with modelopt (NOT save_pretrained) ─────────────────────────
    print(f"Saving modelopt state to {OUT_PATH}...")
    mto.save(model, OUT_PATH)
    print("Done. Load this with mto.restore in inference_runner.py")


if __name__ == "__main__":
    main()