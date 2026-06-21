# SPDX-License-Identifier: Apache-2.0
import sys
import copy
import torch
import numpy as np

# Add src to path
sys.path.append("/home/acf-thor/SHIVAM/alpamayo/src")

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper

# Global model and processor references
_model = None
_processor = None

def get_model_and_processor():
    """
    Singleton-like getter for the model and processor to prevent multiple loads.
    Loads on CUDA in bfloat16.
    """
    global _model, _processor
    if _model is None:
        print("Loading Alpamayo-R1-10B model (this may take a minute on first run)...")
        _model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
        _processor = helper.get_processor(_model.tokenizer)
        print("Model and processor successfully loaded!")
    return _model, _processor

def run_youtube_inference(image_frames, ego_history_speed=10.0, num_traj_samples=3):
    """
    Runs inference using extracted YouTube frames.
    image_frames: torch.Tensor of shape (4, 4, 3, H, W)
    ego_history_speed: initial speed of the vehicle in m/s
    """
    model, processor = get_model_and_processor()
    
    # 1. Prepare Qwen-VL chat messages
    # Flatten N_cameras and num_frames: (16, 3, H, W)
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
    
    # 2. Build synthetic ego history (16 steps at 10Hz, ending at t0 = index 15)
    # x is forward coordinate: x = speed * dt * (step - 15)
    # y, z = 0
    ego_history_xyz = torch.zeros((1, 1, 16, 3))
    for i in range(16):
        ego_history_xyz[0, 0, i, 0] = ego_history_speed * 0.1 * (i - 15)
        
    ego_history_rot = torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 1, 16, 1, 1)
    
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    model_inputs = helper.to_device(model_inputs, "cuda")
    
    print(f"Running YouTube inference with speed={ego_history_speed} m/s...")
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=num_traj_samples,
            max_generation_length=256,
            return_extra=True,
        )
        
    # Extract results
    cots = []
    trajectories = []
    
    for i in range(pred_xyz.shape[2]):  # num_traj_samples
        cot_text = extra["cot"][0][0][i]
        if isinstance(cot_text, bytes):
            cot_text = cot_text.decode("utf-8")
        cots.append(str(cot_text))
        
        # Trajectories shape is: (B, num_traj_sets, num_traj_samples, T=64, 3)
        waypoints = pred_xyz[0, 0, i, :, :2].cpu().numpy().tolist()  # [x, y]
        trajectories.append(waypoints)
        
    history_waypoints = ego_history_xyz[0, 0, :, :2].numpy().tolist()
    
    return {
        "cots": cots,
        "trajectories": trajectories,
        "history_waypoints": history_waypoints
    }

def run_dataset_inference(clip_id, num_traj_samples=3):
    """
    Loads real clip data from the dataset, runs inference, and returns results.
    """
    model, processor = get_model_and_processor()
    
    print(f"Loading dataset clip: {clip_id}...")
    data = load_physical_aiavdataset(clip_id)
    
    # 1. Prepare chat messages
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
    
    print("Running dataset inference...")
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=copy.deepcopy(model_inputs),
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=num_traj_samples,
            max_generation_length=256,
            return_extra=True,
        )
        
    # Extract results
    cots = []
    trajectories = []
    
    for i in range(pred_xyz.shape[2]):  # num_traj_samples
        cot_text = extra["cot"][0][0][i]
        if isinstance(cot_text, bytes):
            cot_text = cot_text.decode("utf-8")
        cots.append(str(cot_text))
        
        waypoints = pred_xyz[0, 0, i, :, :2].cpu().numpy().tolist()  # [x, y]
        trajectories.append(waypoints)
        
    history_waypoints = data["ego_history_xyz"][0, 0, :, :2].numpy().tolist()
    gt_trajectory = data["ego_future_xyz"][0, 0, :, :2].numpy().tolist()
    
    # Also extract camera images to return to frontend (for visualization)
    # We will take camera 1 (camera_front_wide_120fov) frames
    # data["image_frames"] has shape (N_cameras=4, num_frames=4, 3, H, W)
    # We take CAMERA_FRONT_WIDE (sorted at index 1)
    front_frames = data["image_frames"][1]  # (4, 3, H, W)
    
    return {
        "cots": cots,
        "trajectories": trajectories,
        "history_waypoints": history_waypoints,
        "gt_trajectory": gt_trajectory,
        "front_frames": front_frames  # (4, 3, H, W) tensor
    }
