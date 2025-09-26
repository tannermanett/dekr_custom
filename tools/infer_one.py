# ---------------------------------------------------------------------------
# Single-image DEKR inference + visualization (per-keypoint drawing, debug)
# ---------------------------------------------------------------------------

from __future__ import annotations
import argparse, os, json
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T

# Add DEKR's lib/ to sys.path (same behavior as other tools via _init_paths)
import _init_paths  # noqa: F401

import models
from config import cfg, update_config
from core.inference import get_multi_stage_outputs, aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from utils.transforms import (
    resize_align_multi_scale,
    get_final_preds,
    get_multi_scale_size,
)

def parse_args():
    p = argparse.ArgumentParser("One-image DEKR inference (draw keypoints)")
    p.add_argument("--cfg", required=True, type=str, help="Path to YAML config")
    p.add_argument("--ckpt", required=True, type=str, help="Checkpoint .pth(.tar)")
    p.add_argument("--img", required=True, type=str, help="Input image path")
    p.add_argument("--out", required=True, type=str, help="Output visualization path")
    p.add_argument("--scales", type=str, default="1", help="Comma separated scales, e.g. 0.5,1,2")
    p.add_argument("--nms-thre", type=float, default=0.05, help="NMS threshold (TEST.NMS_THRE)")
    p.add_argument("--match-hmp", action="store_true", help="Enable TEST.MATCH_HMP")
    p.add_argument("--flip-test", action="store_true", help="Enable TEST.FLIP_TEST")
    p.add_argument("--th", type=float, default=None,
               help="Global detection threshold (maps to TEST.DETECTION_THRESHOLD)")

    p.add_argument("--kp-th", type=float, default=0.05, help="Min per-keypoint conf to draw")
    p.add_argument("--radius", type=int, default=5, help="circle radius")
    p.add_argument("--thick", type=int, default=2, help="circle thickness")
    p.add_argument("--draw-idx", action="store_true", help="draw joint indices")
    p.add_argument("--save-json", type=str, default="", help="Optional: path to save raw [P,J,3] keypoints JSON")
    p.add_argument("--debug", action="store_true", help="Verbose debug prints")
    # allow extra yacs overrides like: TEST.IMAGES_PER_GPU 1
    p.add_argument("opts", nargs=argparse.REMAINDER, help="Extra cfg overrides (KEY VALUE pairs)")
    return p.parse_args()

def _input_wh(sz):
    """
    Accept either an int (square) or a [W, H] / (W, H) pair.
    Return (W_in, H_in).
    """
    if isinstance(sz, (list, tuple)) and len(sz) == 2:
        return int(sz[0]), int(sz[1])
    v = int(sz)
    return v, v

def to_PxJx3(poses_any) -> np.ndarray:
    """
    Normalize poses into a numpy array [P, J, 3] (x,y,conf).
    Handles DEKR's list/np outputs.
    """
    if poses_any is None:
        return np.zeros((0, 0, 3), dtype=np.float32)

    arr = np.asarray(poses_any, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr

    if isinstance(poses_any, (list, tuple)):
        flat = []
        for p in poses_any:
            p = np.asarray(p, dtype=np.float32)
            if p.ndim == 2 and p.shape[1] == 3:
                flat.append(p)
            elif p.ndim == 1:
                J = len(p) // 3
                flat.append(p.reshape(J, 3))
        if flat:
            return np.stack(flat, axis=0)
    return np.zeros((0, 0, 3), dtype=np.float32)

def draw_keypoints_like_strong(img_bgr: np.ndarray,
                               poses_PxJx3: np.ndarray,
                               kp_th: float = 0.05,
                               radius: int = 5,
                               thick: int = 2,
                               draw_idx: bool = False) -> tuple[np.ndarray, dict]:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    ppl_drawn = 0
    kps_drawn = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    for pid in range(len(poses_PxJx3)):
        ppl_drawn += 1
        P = poses_PxJx3[pid]
        J = P.shape[0]
        for j in range(J):
            x, y, c = float(P[j, 0]), float(P[j, 1]), float(P[j, 2])
            if c < kp_th:
                continue
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            cx, cy = int(round(x)), int(round(y))
            cv2.circle(out, (cx, cy), radius, (0, 255, 0), thick)
            if draw_idx:
                cv2.putText(out, str(j), (cx + 2, cy - 2),
                            font, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            kps_drawn += 1

    stats = {"people": ppl_drawn, "keypoints": kps_drawn, "kp_th": kp_th}
    cv2.putText(out,
                f"people:{ppl_drawn}  kps:{kps_drawn}  kp_th:{kp_th}",
                (10, 25), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    return out, stats

def _count_pose_proposals(poses_obj) -> int:
    """
    poses is a list-like of stage aggregates; be defensive.
    We'll try to sum number of person proposals across entries.
    """
    try:
        total = 0
        for p in poses_obj:
            p_arr = np.asarray(p)
            if p_arr.ndim >= 2:
                total += p_arr.shape[0]
        return total
    except Exception:
        return 0

def main():
    args = parse_args()
    update_config(cfg, args)

    # Device + cuDNN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # Build model
    model = eval("models." + cfg.MODEL.NAME + ".get_pose_net")(cfg, is_train=False)
    print(f"=> loading model from {args.ckpt}")
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).to(device)
    model.eval()

    # Load image
    assert os.path.isfile(args.img), f"Image not found: {args.img}"
    img_bgr = cv2.imread(args.img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img_bgr is None:
        raise RuntimeError(f"cv2.imread failed for {args.img}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Torch transforms
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # size at scale 1.0 (reference for unwarping preds back to original image)
    base_size, center, scale = get_multi_scale_size(
        img_rgb, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
    )

    with torch.no_grad():
        heatmap_sum = 0
        poses = []
        center_1x, scale_resized_1x = None, None

        # multi-scale testing
        for sc in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
            image_resized, center_s, scale_resized_s = resize_align_multi_scale(
                img_rgb, cfg.DATASET.INPUT_SIZE, sc, 1.0
            )
            if sc == 1.0:
                center_1x, scale_resized_1x = center_s, scale_resized_s

            inp = transforms(image_resized).unsqueeze(0).to(device)
            heatmap, posemap = get_multi_stage_outputs(
                cfg, model, inp, cfg.TEST.FLIP_TEST
            )
            heatmap_sum, poses = aggregate_results(
                cfg, heatmap_sum, poses, heatmap, posemap, sc
            )

        heatmap_avg = heatmap_sum / len(cfg.TEST.SCALE_FACTOR)
        poses_list, scores = pose_nms(cfg, heatmap_avg, poses)

        if len(scores) == 0:
            print("No detections found.")
            poses_PxJx3 = np.zeros((0, 0, 3), dtype=np.float32)
            vis = img_bgr
        else:
            if cfg.TEST.MATCH_HMP:
                poses_list = match_pose_to_heatmap(cfg, poses_list, heatmap_avg)

            # ✅ use center + scale from scale=1.0 (exactly like valid.py)
            final_poses = get_final_preds(poses_list, center_1x, scale_resized_1x, base_size)

            # normalize to [P,J,3]
            poses_PxJx3 = to_PxJx3(final_poses)

            print(f"Found {len(poses_PxJx3)} instance(s).")

            vis, stats = draw_keypoints_like_strong(
                img_bgr, poses_PxJx3,
                kp_th=args.kp_th, radius=args.radius,
                thick=args.thick, draw_idx=args.draw_idx
            )
            print(f"Drawn: {stats}")

    # Save outputs
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, vis)
    print(f"Saved visualization → {args.out}")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(poses_PxJx3.tolist(), f)
        print(f"Saved keypoints JSON → {args.save_json}")


if __name__ == "__main__":
    main()
