# SPDX-License-Identifier: Apache-2.0
#
# FIX: torch.compile() moved from background load thread → first inference call
#      torch._dynamo compiled objects are not safe to construct on one thread
#      and call from another. Flask's background model-load thread was building
#      the compiled graph; the request thread then called it → Dynamo assertion
#      failure → Flask worker crash → ERR_EMPTY_RESPONSE on all endpoints.
#
#      Solution: load weights in background (fine, just tensor copies),
#      compile on first inference call on the request thread (safe).
#      _compile_done flag ensures compilation only runs once.
#
# All other optimisations (OPT-3..9, FIX-1, FIX-2) unchanged.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import time
import threading
import torch
import numpy as np

sys.path.append("/home/acf-thor/SHIVAM/alpamayo")
from web_demo.gpu_stats import InferenceGPUMonitor

sys.path.append("/home/acf-thor/SHIVAM/alpamayo/src")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

_model = None
_processor = None
_compile_done = False          # guards one-time compile on request thread
_gpu_lock = threading.Lock()   # one GPU inference at a time


def get_model_and_processor():
    """
    Loads weights only — safe to call from any thread.
    Does NOT compile here.
    """
    global _model, _processor
    if _model is not None:
        return _model, _processor

    print("Loading Alpamayo-R1-10B model...")
    t0 = time.time()

    _model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B",
        dtype=torch.bfloat16,
    ).to("cuda")
    _model.eval()

    _processor = helper.get_processor(_model.tokenizer)
    print(f"Model weights loaded in {time.time() - t0:.1f}s")
    return _model, _processor


def _ensure_compiled():
    """
    Called on the FIRST inference request, from the Flask request thread.
    torch.compile must run on the same thread that will call the model.
    Subsequent calls are no-ops (_compile_done flag).
    """
    global _model, _compile_done
    if _compile_done:
        return
    _compile_done = True   # set before compile so concurrent requests don't double-compile
    try:
        print("Compiling model with torch.compile (first inference only, ~30s)...")
        t0 = time.time()
        _model = torch.compile(_model, mode="reduce-overhead", fullgraph=False)
        print(f"torch.compile() done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"torch.compile() skipped: {e}")


def _build_ego_history(speed: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised ego history — no Python loop."""
    steps = torch.arange(16, dtype=torch.float32)
    x = speed * 0.1 * (steps - 15)
    ego_history_xyz = torch.zeros(1, 1, 16, 3)
    ego_history_xyz[0, 0, :, 0] = x
    ego_history_rot = torch.eye(3).view(1, 1, 1, 3, 3).expand(1, 1, 16, 3, 3).clone()
    return ego_history_xyz, ego_history_rot


def run_youtube_inference(
    image_frames,
    ego_history_speed: float = 10.0,
    num_traj_samples: int = 1,
    max_generation_length: int = 128,
):
    model, processor = get_model_and_processor()

    # Compile on first call, from this (request) thread — thread-safe
    _ensure_compiled()

    flattened_frames = image_frames.flatten(0, 1)
    messages = helper.create_message(flattened_frames)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    ego_history_xyz, ego_history_rot = _build_ego_history(ego_history_speed)
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    print(f"Running inference: speed={ego_history_speed} m/s, "
          f"samples={num_traj_samples}, max_tokens={max_generation_length}")

    t0 = time.time()
    torch.cuda.manual_seed_all(42)

    with _gpu_lock:
        gpu_monitor = InferenceGPUMonitor()
        with gpu_monitor:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.9,
                    temperature=0.6,
                    num_traj_samples=num_traj_samples,
                    max_generation_length=max_generation_length,
                    return_extra=True,
                )

    torch.cuda.synchronize()
    print(f"Inference done in {time.time() - t0:.2f}s")
    gpu_monitor.print_summary()

    cots, trajectories = _extract_results(pred_xyz, extra)
    history_waypoints = ego_history_xyz[0, 0, :, :2].numpy().tolist()

    return {
        "cots": cots,
        "trajectories": trajectories,
        "history_waypoints": history_waypoints,
    }


def run_dataset_inference(
    clip_id: str,
    num_traj_samples: int = 1,
    max_generation_length: int = 128,
):
    model, processor = get_model_and_processor()
    _ensure_compiled()

    print(f"Loading dataset clip: {clip_id}...")
    data = load_physical_aiavdataset(clip_id)

    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    t0 = time.time()
    torch.cuda.manual_seed_all(42)

    with _gpu_lock:
        gpu_monitor = InferenceGPUMonitor()
        with gpu_monitor:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.9,
                    temperature=0.6,
                    num_traj_samples=num_traj_samples,
                    max_generation_length=max_generation_length,
                    return_extra=True,
                )

    torch.cuda.synchronize()
    print(f"Inference done in {time.time() - t0:.2f}s")
    gpu_monitor.print_summary()

    cots, trajectories = _extract_results(pred_xyz, extra)
    history_waypoints = data["ego_history_xyz"][0, 0, :, :2].numpy().tolist()
    gt_trajectory = data["ego_future_xyz"][0, 0, :, :2].numpy().tolist()
    front_frames = data["image_frames"][1]

    return {
        "cots": cots,
        "trajectories": trajectories,
        "history_waypoints": history_waypoints,
        "gt_trajectory": gt_trajectory,
        "front_frames": front_frames,
    }


def _extract_results(pred_xyz: torch.Tensor, extra: dict) -> tuple[list, list]:
    cots, trajectories = [], []
    for i in range(pred_xyz.shape[2]):
        cot_text = extra["cot"][0][0][i]
        if isinstance(cot_text, bytes):
            cot_text = cot_text.decode("utf-8")
        cots.append(str(cot_text))
        waypoints = pred_xyz[0, 0, i, :, :2].cpu().numpy().tolist()
        trajectories.append(waypoints)
    return cots, trajectories