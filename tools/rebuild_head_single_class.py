import time
import torch
from ultralytics import YOLO


def rebuild_single_class(src_ultra_pt: str, dst_ultra_pt: str, base_yaml: str = "yolo11n.yaml") -> None:
    # Load source Ultralytics checkpoint (may have wrong nc/names)
    ckpt = torch.load(src_ultra_pt, map_location="cpu", weights_only=False)

    # Build a fresh YOLO model with 1 class
    model = YOLO(base_yaml)
    # Force single class in yaml/meta
    if hasattr(model, 'model') and hasattr(model.model, 'yaml'):
        model.model.yaml['nc'] = 1
    model.model.nc = 1
    model.model.names = {0: 'polyp'}

    # Load weights from source with strict=False so head (cls) can be reinitialized for nc=1
    src_model = ckpt.get('model', None)
    if src_model is not None:
        state = src_model.state_dict()
    else:
        # Fallback: if only weights in 'model.*' format
        state = {k.replace('model.', ''): v for k, v in ckpt.items() if isinstance(k, str) and k.startswith('model.')}
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    print(f"loaded with missing={len(missing)} unexpected={len(unexpected)} (head will be reinitialized)")

    # Save a clean Ultralytics checkpoint with correct nc/names
    payload = {
        'model': model.model,
        'ema': None,
        'train_args': {},
        'epoch': -1,
        'best_fitness': 0.0,
        'yaml': base_yaml,
        'names': model.model.names,
        'nc': model.model.nc,
        'date': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    torch.save(payload, dst_ultra_pt)
    print('saved ->', dst_ultra_pt)


if __name__ == '__main__':
    src = '/home/huangmanling/ultralytics/runs/polyp_y11n_kdpre_dinov2/weights/best_ultra.pt'
    dst = '/home/huangmanling/ultralytics/runs/polyp_y11n_kdpre_dinov2/weights/best_ultra_polyp_fixed.pt'
    rebuild_single_class(src, dst, base_yaml='yolo11n.yaml')


