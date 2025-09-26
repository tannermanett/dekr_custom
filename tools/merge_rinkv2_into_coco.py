import argparse
import os
import sys
import json
import shutil
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET


def load_coco(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_coco(data: Dict, path: str) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


def index_existing(coco: Dict) -> Tuple[Dict[str, int], int, int]:
    file_to_image_id: Dict[str, int] = {}
    max_image_id = 0
    max_ann_id = 0

    for img in coco.get('images', []):
        file_to_image_id[img['file_name']] = int(img['id'])
        max_image_id = max(max_image_id, int(img['id']))
    for ann in coco.get('annotations', []):
        max_ann_id = max(max_ann_id, int(ann['id']))

    return file_to_image_id, max_image_id, max_ann_id


def parse_cvat(xml_path: str) -> List[ET.Element]:
    root = ET.parse(xml_path).getroot()
    return list(root.findall('image'))


def collect_keypoints_for_image(image_elem: ET.Element, num_joints: int) -> List[float]:
    keypoints = [0.0] * (num_joints * 3)
    points_elems = image_elem.findall('points')
    for pe in points_elems:
        label = pe.get('label')
        coords = pe.get('points')
        if label is None or coords is None:
            continue
        try:
            idx = int(label)
        except Exception:
            continue
        if idx < 0 or idx >= num_joints:
            continue
        try:
            x_str, y_str = coords.split(',')
            x = float(x_str)
            y = float(y_str)
        except Exception:
            continue
        base = idx * 3
        keypoints[base + 0] = x
        keypoints[base + 1] = y
        keypoints[base + 2] = 2.0  # visible by default
    return keypoints


def bbox_from_keypoints(keypoints: List[float]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for i in range(0, len(keypoints), 3):
        v = keypoints[i + 2]
        if v > 0:
            xs.append(keypoints[i])
            ys.append(keypoints[i + 1])
    if not xs or not ys:
        return 0.0, 0.0, 0.0, 0.0
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return x_min, y_min, max(0.0, x_max - x_min), max(0.0, y_max - y_min)


def ensure_image_present(src_root: str, image_name: str, dst_root: str) -> Optional[str]:
    src_path = os.path.join(src_root, image_name)
    if not os.path.isfile(src_path):
        # try nested folders (recursive search is expensive; keep simple)
        for dirpath, _, filenames in os.walk(src_root):
            if image_name in filenames:
                src_path = os.path.join(dirpath, image_name)
                break
        else:
            return None
    os.makedirs(dst_root, exist_ok=True)
    dst_path = os.path.join(dst_root, image_name)
    if not os.path.isfile(dst_path):
        shutil.copy2(src_path, dst_path)
    return dst_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Merge CVAT rinkv2.xml point annotations into COCO train2017 JSON.')
    parser.add_argument('--xml', type=str, default=os.path.join('data', 'rink', 'annotations', 'rinkv2.xml'))
    parser.add_argument('--existing-json', type=str, default=os.path.join('data', 'rink', 'annotations', 'person_keypoints_train2017.json'))
    parser.add_argument('--out-json', type=str, default=os.path.join('data', 'rink', 'annotations', 'person_keypoints_train2017.merged.json'))
    parser.add_argument('--images-src-root', type=str, default=os.path.join('data', 'rink', 'images', 'vip_htd_all_frames'))
    parser.add_argument('--images-dst-root', type=str, default=os.path.join('data', 'rink', 'images', 'train2017'))
    parser.add_argument('--num-joints', type=int, default=56)
    parser.add_argument('--category-id', type=int, default=None, help='If None, inferred from existing JSON categories[0].id')
    parser.add_argument('--skip-copy', action='store_true', help='Do not copy images into train2017 (assume already present).')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of XML images to merge (for quick tests).')
    args = parser.parse_args(argv)

    if not os.path.isfile(args.xml):
        print(f'XML not found: {args.xml}', file=sys.stderr)
        return 2
    if not os.path.isfile(args.existing_json):
        print(f'Existing COCO JSON not found: {args.existing_json}', file=sys.stderr)
        return 2

    coco = load_coco(args.existing_json)
    file_to_image_id, max_image_id, max_ann_id = index_existing(coco)

    # determine category_id
    category_id = args.category_id
    if category_id is None:
        cats = coco.get('categories', [])
        if not cats:
            print('No categories in existing JSON; cannot infer category_id.', file=sys.stderr)
            return 2
        category_id = int(cats[0].get('id', 1))

    xml_images = parse_cvat(args.xml)
    if args.limit is not None:
        xml_images = xml_images[:max(0, args.limit)]

    new_images: List[Dict] = []
    new_annotations: List[Dict] = []
    skipped_existing = 0
    missing_images: List[str] = []

    for image_elem in xml_images:
        name = image_elem.get('name') or ''
        if not name:
            continue
        if name in file_to_image_id:
            skipped_existing += 1
            continue

        width = int(float(image_elem.get('width') or 0))
        height = int(float(image_elem.get('height') or 0))

        # ensure image in train2017
        if not args.skip_copy:
            dst_path = ensure_image_present(args.images_src_root, name, args.images_dst_root)
            if dst_path is None:
                missing_images.append(name)
                # still register metadata so training can warn, but skip ann
                continue

        # allocate new ids
        max_image_id += 1
        image_id = max_image_id

        # image record
        new_images.append({
            'id': image_id,
            'file_name': name,
            'width': width,
            'height': height,
        })

        # build keypoints for this image
        kpts = collect_keypoints_for_image(image_elem, args.num_joints)
        num_vis = sum(1 for i in range(2, len(kpts), 3) if kpts[i] > 0)
        x, y, w, h = bbox_from_keypoints(kpts)
        area = float(w * h)

        max_ann_id += 1
        ann_id = max_ann_id
        new_annotations.append({
            'id': ann_id,
            'image_id': image_id,
            'category_id': category_id,
            'num_keypoints': int(num_vis),
            'keypoints': [float(v) for v in kpts],
            'iscrowd': 0,
            'bbox': [float(x), float(y), float(w), float(h)],
            'area': area,
            'segmentation': [],
        })

    # merge into existing COCO dict
    coco.setdefault('images', []).extend(new_images)
    coco.setdefault('annotations', []).extend(new_annotations)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    save_coco(coco, args.out_json)

    print(f'Merged {len(new_images)} images and {len(new_annotations)} annotations.')
    print(f'Skipped {skipped_existing} images already present in existing JSON.')
    print(f'Wrote merged JSON: {os.path.abspath(args.out_json)}')
    if missing_images:
        print(f'Warning: {len(missing_images)} source images not found under {args.images_src_root}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


