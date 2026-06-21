# SPDX-License-Identifier: Apache-2.0
#
# FIXES applied:
#
#  [FIX-A] Return HTTP 503 immediately if an inference is already running
#          instead of queuing another thread on the GPU.
#          The frontend sim loop handles this and skips the step cleanly.
#
#  [FIX-B] Expose /api/busy endpoint so frontend can check before firing.
#
# All other logic unchanged from original app.py.
# ─────────────────────────────────────────────────────────────────────────────

import io
import os
import sys
import base64
import threading
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

sys.path.append("/home/acf-thor/SHIVAM/alpamayo")

app = Flask(__name__, template_folder="templates")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

model_loading_status = "loading"
model_loading_error = None

# [FIX-A] Track whether inference is currently running
_inference_busy = False
_inference_lock = threading.Lock()


def load_model_thread():
    global model_loading_status, model_loading_error
    try:
        from web_demo.inference_runner import get_model_and_processor
        get_model_and_processor()
        model_loading_status = "ready"
        print("Backend Model Loaded Successfully!")
    except Exception as e:
        model_loading_status = "error"
        model_loading_error = str(e)
        import traceback
        traceback.print_exc()


threading.Thread(target=load_model_thread, daemon=True).start()


def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def tensor_to_base64(t):
    arr = t.permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr)
    return pil_to_base64(img)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    global model_loading_status, model_loading_error
    try:
        parquet_path = "/home/acf-thor/SHIVAM/alpamayo/notebooks/clip_ids.parquet"
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            clip_ids = df["clip_id"].tolist()[:50]
        else:
            clip_ids = []
    except Exception as e:
        print(f"Error loading clip IDs: {e}")
        clip_ids = []

    return jsonify({
        "status": model_loading_status,
        "error": model_loading_error,
        "clip_ids": clip_ids,
        "busy": _inference_busy,   # [FIX-B] expose busy state
    })


@app.route("/api/busy", methods=["GET"])
def api_busy():
    """[FIX-B] Lightweight endpoint for frontend to check GPU availability."""
    return jsonify({"busy": _inference_busy})


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    valid_extensions = (".mp4", ".mov", ".avi", ".mkv")
    if not file.filename.lower().endswith(valid_extensions):
        return jsonify({"error": f"Invalid file type. Supported: {', '.join(valid_extensions)}"}), 400

    try:
        filename = secure_filename(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return jsonify({"filename": filename, "url": f"/uploads/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/api/process_uploaded", methods=["POST"])
def api_process_uploaded():
    global model_loading_status, _inference_busy

    if model_loading_status != "ready":
        return jsonify({"error": "Model is still loading, please wait."}), 503

    # [FIX-A] Reject immediately if GPU is already busy — don't queue
    with _inference_lock:
        if _inference_busy:
            return jsonify({"error": "busy", "retry": True}), 503
        _inference_busy = True

    try:
        data = request.json or {}
        filename = data.get("filename")
        timestamp = float(data.get("timestamp", 0.0))
        speed = float(data.get("speed", 10.0))

        if not filename:
            return jsonify({"error": "No filename provided"}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({"error": f"Video file {filename} not found"}), 404

        from web_demo.video_utils import extract_frames, prepare_input_tensors
        from web_demo.inference_runner import run_youtube_inference

        pil_images = extract_frames(filepath, start_sec=0.0, target_sec=timestamp)
        image_frames = prepare_input_tensors(pil_images)

        results = run_youtube_inference(image_frames, ego_history_speed=speed)

        base64_images = [pil_to_base64(img) for img in pil_images]
        results["images"] = base64_images

        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # [FIX-A] Always release the busy flag, even on error
        with _inference_lock:
            _inference_busy = False


if __name__ == "__main__":
    # threaded=True is Flask default — needed for background model load
    # The _gpu_lock in inference_runner.py handles GPU serialisation
    app.run(host="0.0.0.0", port=3000, debug=False, threaded=True)