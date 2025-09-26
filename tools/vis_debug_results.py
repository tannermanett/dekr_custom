import os, json, argparse, cv2
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--ann', required=True)
ap.add_argument('--res', required=True)
ap.add_argument('--img-root', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)

# Load GT annotations
ann = json.load(open(args.ann, 'r'))
id2name = {im['id']: im['file_name'] for im in ann['images']}

# Load predictions
res = json.load(open(args.res, 'r'))
by_img = defaultdict(list)
for r in res:
    by_img[r['image_id']].append(r)

count = 0
for img_id, dets in by_img.items():
    fn = id2name.get(img_id)
    if fn is None:
        continue
    path = os.path.join(args.img_root, fn)
    img = cv2.imread(path)
    if img is None:
        continue

    for det in dets:
        kps = det['keypoints']
        score = det.get('score', -1)
        print(f"Image {fn}, score {score:.3f}, num kps {len(kps)//3}")
        n = len(kps) // 3
        for j in range(n):
            x, y, v = kps[3*j], kps[3*j+1], kps[3*j+2]
            color = (0, 0, 255) if v == 0 else (0, 255, 0)
            cv2.circle(img, (int(round(x)), int(round(y))), 3, color, -1)

    outp = os.path.join(args.out, fn)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    cv2.imwrite(outp, img)
    count += 1

print(f"Wrote {count} images to {args.out}")
