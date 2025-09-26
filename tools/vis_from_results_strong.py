# tools/vis_from_results_strong.py
import os, json, argparse, cv2
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--ann', required=True, help='GT person_keypoints_val2017.json')
ap.add_argument('--res', required=True, help='results JSON from valid.py')
ap.add_argument('--img-root', required=True, help='folder with val2017 images')
ap.add_argument('--out', required=True, help='output folder')
ap.add_argument('--th', type=float, default=0.01, help='min detection score (per person)')
ap.add_argument('--kp-th', type=float, default=0.25, help='min per-keypoint score to draw')
ap.add_argument('--use-gt-mask', action='store_true',
               help='hide keypoints where GT visibility==0 (if present)')
ap.add_argument('--radius', type=int, default=5)
ap.add_argument('--thick', type=int, default=2)
ap.add_argument('--draw-idx', dest='draw_idx', action='store_true')
ap.add_argument('--max', type=int, default=100000)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)

# ----- load GT (to get file names, and optional vis mask) -----
ann = json.load(open(args.ann, 'r'))
id2name = {im['id']: im['file_name'] for im in ann['images']}

# For optional GT masking: image_id -> first annotation's vis mask (0/1/2)
gt_vis = {}
if args.use_gt_mask:
    for a in ann.get('annotations', []):
        kid = a['image_id']
        kps = a.get('keypoints', [])
        if not kps: 
            continue
        vis = [kps[i+2] for i in range(0, len(kps), 3)]
        gt_vis.setdefault(kid, vis)

def resolve_path(img_root, fn):
    cands = [
        os.path.join(img_root, fn),
        os.path.join(img_root, os.path.basename(fn)),
        os.path.join(os.path.dirname(img_root), fn),
        os.path.join(os.path.dirname(os.path.dirname(img_root)), fn),
        fn,
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return None

# ----- load results and group by image -----
res = json.load(open(args.res, 'r'))
by_img = defaultdict(list)
for r in res:
    if r.get('score', 0.0) >= args.th:
        by_img[int(r['image_id'])].append(r)

total_imgs = total_people = total_kps = 0
for img_id, dets in by_img.items():
    fn = id2name.get(img_id)
    if fn is None:
        continue

    path = resolve_path(args.img_root, fn)
    if path is None:
        print(f"[skip] not found: {fn}")
        continue

    img = cv2.imread(path)
    if img is None:
        print(f"[skip] failed to read: {path}")
        continue
    h, w = img.shape[:2]

    vis_mask = gt_vis.get(img_id) if args.use_gt_mask else None

    ppl_drawn = 0
    kps_drawn = 0
    for det in dets:
        ppl_drawn += 1
        kps = det['keypoints']  # flat [x1,y1,score1, x2,y2,score2, ...]
        n = len(kps) // 3

        for j in range(n):
            x, y, s = kps[3*j], kps[3*j+1], kps[3*j+2]

            # (A) per-keypoint score filter
            if s < args.kp_th:
                continue

            # (B) optional GT mask (hide off-screen/unlabeled in GT)
            if vis_mask is not None and j < len(vis_mask) and vis_mask[j] == 0:
                continue

            # (C) inside image
            if x < 0 or y < 0 or x >= w or y >= h:
                continue

            cx, cy = int(round(x)), int(round(y))
            cv2.circle(img, (cx, cy), args.radius, (0, 255, 0), args.thick)
            if args.draw_idx:
                cv2.putText(img, str(j), (cx+2, cy-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
            kps_drawn += 1

    cv2.putText(img, f"people:{ppl_drawn}  kps:{kps_drawn}  th:{args.th}  kp_th:{args.kp_th}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    outp = os.path.join(args.out, os.path.basename(path))
    cv2.imwrite(outp, img)
    total_imgs += 1
    total_people += ppl_drawn
    total_kps += kps_drawn
    if total_imgs >= args.max:
        break

print(f"[done] wrote {total_imgs} images -> {args.out}")
print(f"  people drawn: {total_people}   keypoints drawn: {total_kps}")
