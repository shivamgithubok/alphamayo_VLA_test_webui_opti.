import os
import sys
import subprocess
import av
import numpy as np
import torch
from PIL import Image

# [OPT-B] Frames within this many seconds of target are "good enough" to stop
CLOSE_ENOUGH_SEC = 0.04   # ~1 frame at 25fps


def download_youtube_clip(url: str, target_sec: float, output_path: str = "temp_clip.mp4") -> float:
    """
    Downloads a short 4-second clip around target_sec. Returns start_sec.
    Unchanged from original — network I/O dominates here.
    """
    start_sec = max(0.0, target_sec - 2.0)
    end_sec = target_sec + 2.0

    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass

    yt_dlp_path = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    if not os.path.exists(yt_dlp_path):
        yt_dlp_path = "yt-dlp"

    cmd = [
        yt_dlp_path, "-f", "mp4",
        "--download-sections", f"*{start_sec:.2f}-{end_sec:.2f}",
        "--force-keyframes-at-cuts",
        "-o", output_path, url,
    ]
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print("Section download succeeded.")
            return start_sec
    except Exception as e:
        print(f"Section download error: {e}")

    print("Falling back to full low-res download...")
    cmd_fallback = [yt_dlp_path, "-f", "worst[ext=mp4]/mp4", "-o", output_path, url]
    try:
        result = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=60)
        if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")
        print("Full download fallback succeeded.")
        return 0.0
    except Exception as e:
        raise RuntimeError(f"Failed to download YouTube video: {e}")


def extract_frames(
    video_path: str,
    start_sec: float,
    target_sec: float,
    target_width: int = 336,
) -> list[Image.Image]:
    """
    Extracts 4 frames at [target_sec-0.3s, -0.2s, -0.1s, target_sec].
    Returns a list of 4 RGB PIL Images.
    """
    targets = [target_sec]
    relative_targets = [max(0.0, t - start_sec) for t in targets]
    earliest_target = relative_targets[0]

    print(f"Extracting frames near t={relative_targets}")

    # [OPT-D] Try hardware (NVDEC) decoding first, fall back to software
    container = _open_container(video_path)

    frames_found = {}   # idx → (numpy uint8 HWC, diff_seconds)

    try:
        stream = container.streams.video[0]

        # [OPT-A] Seek to just before earliest target frame
        # PyAV seek is in stream.time_base units
        seek_pts = max(0, int((earliest_target - 0.5) / stream.time_base))
        container.seek(seek_pts, stream=stream)

        for frame in container.decode(stream):
            frame_time = frame.time
            if frame_time is None:
                continue

            # [OPT-B] Early exit: stop if we're past last target + margin
            if frame_time > relative_targets[-1] + 0.5:
                break

            # [OPT-C] Decode to numpy once, reuse across all 4 target checks
            frame_np = None   # lazy — only decode if we actually match

            for idx, rel_t in enumerate(relative_targets):
                diff = abs(frame_time - rel_t)
                if idx not in frames_found or diff < frames_found[idx][1]:
                    if frame_np is None:
                        # Convert YUV→RGB once, as numpy array
                        frame_np = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
                    frames_found[idx] = (frame_np, diff)

            # [OPT-B] Stop early once all targets matched within threshold
            if (len(frames_found) == len(targets) and
                    all(frames_found[i][1] < CLOSE_ENOUGH_SEC for i in range(len(targets)))):
                break

    except Exception as e:
        print(f"Warning during frame decoding: {e}")
    finally:
        container.close()

    # Build result list — convert numpy→PIL and resize here
    result_images: list[Image.Image] = []
    for idx in range(len(targets)):
        if idx in frames_found:
            arr, _ = frames_found[idx]
            img = Image.fromarray(arr)
            # [OPT-G] Resize if needed — BILINEAR for quality, swap to NEAREST for speed
            if img.width > target_width:
                h = int(img.height * (target_width / img.width))
                img = img.resize((target_width, h), Image.Resampling.BILINEAR)
        else:
            print(f"Warning: frame {idx} not found, using black placeholder.")
            img = Image.new("RGB", (target_width, target_width), (0, 0, 0))
        result_images.append(img)

    return result_images


def prepare_input_tensors(pil_images: list[Image.Image]) -> torch.Tensor:
    """
    Converts RGB PIL Images → torch.Tensor (N_cameras=1, num_frames=1, 3, H, W) uint8.
    """
    frame_tensors = []
    for img in pil_images:
        # [OPT-E] No .convert("RGB") — extract_frames already returns RGB
        arr = np.array(img)                          # (H, W, 3) uint8
        t = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W) uint8
        frame_tensors.append(t)

    single_camera_frames = torch.stack(frame_tensors, dim=0)   # (num_frames, 3, H, W)

    # [OPT-F] pin_memory for faster CPU→GPU DMA transfer
    image_frames = (
        single_camera_frames
        .unsqueeze(0)
        .pin_memory()
    )
    return image_frames


def _open_container(video_path: str) -> av.container.InputContainer:
    """
    [OPT-D] Try to open with NVDEC hardware decoder (Thor has dual NVDEC).
    Falls back to software decoding if NVDEC is unavailable.
    Jetson Thor's NVDEC supports H.264, H.265, AV1 — most common video formats.
    """
    hw_options = {
        "hwaccel": "cuda",
        "hwaccel_output_format": "cuda",
    }
    try:
        container = av.open(video_path, options=hw_options)
        # Probe by accessing the first stream to confirm HW works
        _ = container.streams.video[0]
        print("Using NVDEC hardware video decoder.")
        return container
    except Exception:
        pass  # HW unavailable or not supported for this codec

    # Software fallback
    print("Using software video decoder (CPU).")
    return av.open(video_path)