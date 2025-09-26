import os
import cv2
import torch
import numpy as np

from config import cfg
from config import update_config
import models
from utils.transforms import get_affine_transform
from utils.vis import save_valid_image

# --- CONFIG ---
cfg_file = "experiments/rink/w32/w32_rink_512.yaml"
ckpt_path = "output/coco_kpt/hrnet_dekr/w32_rink_512/model_best.pth.tar"
img_dir = "data/rink/images/val2017"
out_dir = "output/rink_val_vis"
os.makedirs(out_dir, exist_ok=True)

# --- LOAD CONFIG ---
update_config(cfg, cfg_file)

# --- LOAD MODEL ---
pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)
state_dict = torch.load(ckpt_path, map_location="cpu")
pose_model.load_state_dict(state_dict['state_dict'])
pose_model.cuda()
pose_model.eval()

# --- INFERENCE LOOP ---
input_size = cfg.DATASET.INPUT_SIZE
for fname in sorted(os.listdir(img_dir)):
    if not fname.lower().endswith(('.jpg', '.png')):
        continue
    img_path = os.path.join(img_dir, fname)
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        continue

    c = np.array([orig_img.shape[1] / 2, orig_img.shape[0] / 2])
    s = max(orig_img.shape[0], orig_img.shape[1]) / 200
    trans = get_affine_transform(c, s, 0, (input_size, input_size))

    inp = cv2.warpAffine(orig_img, trans, (input_size, input_size), flags=cv2.INTER_LINEAR)
    inp = inp[:, :, ::-1].astype(np.float32) / 255.0
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).cuda()

    with torch.no_grad():
        outputs = pose_model(inp)

    # Save visualization (simple)
    vis_img = orig_img.copy()
    save_valid_image(outputs, c, s, out_dir, fname, vis_img, dataset=cfg.DATASET.DATASET)

print(f"Done! Visual results saved to {out_dir}")
