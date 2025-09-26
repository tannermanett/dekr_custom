import os, json, argparse, cv2
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument('--ann', required=True, help='COCO ann JSON (GT)')
ap.add_argument('--res', required=True, help='results JSON from valid.py')
ap.add_argument('--img-root', required=True, help='val2017 images folder')
ap.add_argument('--out', required=True, help='output folder for visualizations')
ap.add_argument('--th', type=float, default=0.3, help='min detection score')
ap.add_argument('--max', type=int, default=100000, help='max images to dump')
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)

# image_id -> file_name from GT annotations
ann = json.load(open(args.ann, 'r'))
id2name = {im['id']: im['file_name'] for im in ann['images']}

# predictions grouped by image
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
        if det.get('score', 0) < args.th:
            continue
        kps = det['keypoints']  # x1,y1,v1, x2,y2,v2, ...
        n = len(kps) // 3
        for j in range(n):
            x, y, v = kps[3*j], kps[3*j+1], kps[3*j+2]
            if v > 0:
                cv2.circle(img, (int(round(x)), int(round(y))), 2, (0,255,0), -1)

    outp = os.path.join(args.out, fn)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    cv2.imwrite(outp, img)
    count += 1
    if count >= args.max:
        break

print(f"Wrote {count} images to {args.out}")
