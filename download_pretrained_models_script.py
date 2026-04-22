"""
Usage:
    pip install torch transformers mediapipe pillow
    python download_pretrained_models.py
"""
import os
import urllib.request
from pathlib import Path

import torch


CACHE_DIR = Path.home() / ".cache" / "yoga_demo"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MEDIAPIPE_URL = ("https://storage.googleapis.com/mediapipe-models/"
                 "pose_landmarker/pose_landmarker_full/float16/latest/"
                 "pose_landmarker_full.task")
MEDIAPIPE_PATH = CACHE_DIR / "pose_landmarker_full.task"

RTDETR_HF   = "PekingU/rtdetr_r50vd_coco_o365"
VITPOSE_HF  = "yonigozlan/synthpose-vitpose-base-hf"


def load_mediapipe():
    """Download .task file if missing, then build a PoseLandmarker."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    if not MEDIAPIPE_PATH.exists():
        print(f"  downloading {MEDIAPIPE_URL}")
        urllib.request.urlretrieve(MEDIAPIPE_URL, MEDIAPIPE_PATH)
    size_mb = MEDIAPIPE_PATH.stat().st_size / 1024**2
    print(f"  cached at {MEDIAPIPE_PATH}  ({size_mb:.1f} MB)")

    landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MEDIAPIPE_PATH)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
        )
    )
    return landmarker


def load_hf_pair(repo_id, model_cls, device):
    """Download (cached by HF) + load a processor/model pair onto `device`."""
    from transformers import AutoProcessor
    print(f"  fetching {repo_id}")
    processor = AutoProcessor.from_pretrained(repo_id)
    model     = model_cls.from_pretrained(repo_id).to(device).eval()
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  loaded  {repo_id}  ({n_params:.1f}M params, on {device})")
    return processor, model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    models = {}

    print("[1/3] MediaPipe PoseLandmarker  (google cloud storage)")
    models["mediapipe_pose"] = load_mediapipe()

    from transformers import RTDetrForObjectDetection, VitPoseForPoseEstimation
    print("\n[2/3] RTDetr person detector   (hugging face)")
    models["rtdetr_proc"], models["rtdetr_model"] = load_hf_pair(
        RTDETR_HF, RTDetrForObjectDetection, device)

    print("\n[3/3] SynthPose ViTPose-base   (hugging face)")
    models["vitpose_proc"], models["vitpose_model"] = load_hf_pair(
        VITPOSE_HF, VitPoseForPoseEstimation, device)

    print(f"\nAll 3 pre-trained models loaded into memory: "
          f"{list(models.keys())}")
    return models


if __name__ == "__main__":
    main()
