import os
import time
import torch
from ultralytics import YOLO


def load_state_dict_from_any(checkpoint_path: str) -> dict:
    obj = torch.load(checkpoint_path, map_location="cpu")
    # Case 1: pure state_dict with keys like 'model.backbone.*'
    if isinstance(obj, dict) and obj and all(isinstance(k, str) for k in obj.keys()):
        # If keys already look like 'model.xxx', use directly
        if all(k.startswith("model.") for k in obj.keys()):
            return obj
        # If keys look like plain layer names, map to 'model.' prefix
        if not any(k.startswith("model.") for k in obj.keys()):
            return {f"model.{k}": v for k, v in obj.items()}
    # Case 2: Ultralytics checkpoint containing 'model' module
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        if hasattr(model, "state_dict"):
            sd = model.state_dict()
            return {f"model.{k}": v for k, v in sd.items()}
    raise RuntimeError("Unsupported checkpoint format for conversion")


def convert_to_ultralytics(src_weights: str, dst_weights: str, base_yaml: str = "yolo11n.yaml") -> None:
    state_dict = load_state_dict_from_any(src_weights)
    yolo = YOLO(base_yaml)
    missing, unexpected = yolo.model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[convert] missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"[convert] unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")

    payload = {
        "model": yolo.model,
        "ema": None,
        "train_args": {},
        "epoch": -1,
        "best_fitness": 0.0,
        "yaml": base_yaml,
        "names": getattr(yolo.model, "names", None),
        "nc": getattr(yolo.model, "nc", None),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(dst_weights), exist_ok=True)
    torch.save(payload, dst_weights)
    print(f"[convert] saved Ultralytics checkpoint -> {dst_weights}")


if __name__ == "__main__":
    src = \
        "/home/huangmanling/ultralytics/runs/polyp_y11n_kdpre_dinov2/weights/best.pt"
    dst = \
        "/home/huangmanling/ultralytics/runs/polyp_y11n_kdpre_dinov2/weights/best_ultra.pt"
    convert_to_ultralytics(src, dst, base_yaml="yolo11n.yaml")


