#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import sys
from typing import Dict, List, Optional, Tuple


def list_images(images_dir: str) -> set:
    if not os.path.isdir(images_dir):
        print(f"[ERROR] Images directory not found: {images_dir}")
        return set()
    return {f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')}


_PATTERNS = [
    # 001_frame_00119.jpg → (num=1, frame=119)
    (re.compile(r'^(?P<num>\d{3})_frame_(?P<frame>\d{5})\.jpg$', re.IGNORECASE), 'nnn'),
    # part1_frame_00119.jpg → (num=1, frame=119)
    (re.compile(r'^part(?P<num>\d+)_frame_(?P<frame>\d{5})\.jpg$', re.IGNORECASE), 'part'),
    # frame_00119.jpg → (num=None, frame=119)
    (re.compile(r'^frame_(?P<frame>\d{5})\.jpg$', re.IGNORECASE), 'bare'),
]


def parse_name(name: str) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    for rx, kind in _PATTERNS:
        m = rx.match(name)
        if m:
            num = int(m.group('num')) if 'num' in m.groupdict() else None
            frame = int(m.group('frame')) if 'frame' in m.groupdict() else None
            return num, frame, kind
    return None, None, None


def format_name_part(num: int, frame: int) -> str:
    return f"part{num}_frame_{frame:05d}.jpg"


def format_name_nnn(num: int, frame: int) -> str:
    return f"{num:03d}_frame_{frame:05d}.jpg"


def propose_alternates(name: str) -> List[str]:
    num, frame, kind = parse_name(name)
    if frame is None:
        return []
    alts: List[str] = []
    if kind == 'nnn' and num is not None:
        alts.append(format_name_part(num, frame))
        alts.append(format_name_nnn(num, frame))  # identity/consistency
    elif kind == 'part' and num is not None:
        alts.append(format_name_nnn(num, frame))
        alts.append(format_name_part(num, frame))  # identity/consistency
    elif kind == 'bare':
        # Try part1 and 001 as plausible defaults
        alts.append(format_name_part(1, frame))
        alts.append(format_name_nnn(1, frame))
    else:
        # Unknown pattern; try to salvage a 5-digit frame id anywhere in name
        m = re.search(r'(\d{5})', name)
        if m:
            frame = int(m.group(1))
            alts.append(format_name_part(1, frame))
            alts.append(format_name_nnn(1, frame))
    # Deduplicate while preserving order
    seen = set()
    unique_alts: List[str] = []
    for a in alts:
        if a not in seen:
            seen.add(a)
            unique_alts.append(a)
    return unique_alts


def load_json(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def backup_file(path: str) -> str:
    base = path + '.bak'
    bak = base
    idx = 1
    while os.path.exists(bak):
        idx += 1
        bak = base + str(idx)
    shutil.copy2(path, bak)
    return bak


def process_split(root: str, split: str, apply: bool) -> Tuple[int, int, int]:
    ann_path = os.path.join(root, 'annotations', f'person_keypoints_{split}.json')
    img_dir = os.path.join(root, 'images', split)

    print(f"\n=== Split: {split} ===")
    print(f"[INFO] Annotations: {ann_path}")
    print(f"[INFO] Images dir : {img_dir}")

    if not os.path.isfile(ann_path):
        print(f"[WARN] Missing annotations file: {ann_path}")
        return 0, 0, 0

    fs_files = list_images(img_dir)
    data = load_json(ann_path)
    images = data.get('images', [])

    missing = []  # (idx, old_name, suggestions)
    unchanged = 0
    changed = 0

    fs_set = set(fs_files)

    # Build quick index by alt name to ensure uniqueness decisions
    def choose_existing(alts: List[str]) -> Optional[str]:
        for a in alts:
            if a in fs_set:
                return a
        return None

    for i, img in enumerate(images):
        old = img.get('file_name')
        if old in fs_set:
            unchanged += 1
            continue
        alts = propose_alternates(old)
        match = choose_existing(alts)
        if match is None:
            # Last resort: if exactly one candidate in fs has same frame digits
            frame_digits = re.findall(r'(\d{5})', old or '')
            candidate = None
            if frame_digits:
                target = frame_digits[-1]
                cands = [f for f in fs_files if target in f]
                if len(cands) == 1:
                    candidate = cands[0]
            if candidate is not None and candidate not in alts:
                alts.append(candidate)
                if candidate in fs_set:
                    match = candidate

        if match:
            missing.append((i, old, match))
        else:
            missing.append((i, old, None))

    print(f"[INFO] In-annotation images: {len(images)}")
    print(f"[INFO] Files on disk      : {len(fs_files)}")
    print(f"[INFO] Already matching  : {unchanged}")
    print(f"[INFO] Need fixes        : {sum(1 for _, _, m in missing if m)} (resolvable)")
    print(f"[INFO] Unresolved        : {sum(1 for _, _, m in missing if not m)}")

    preview_limit = 30
    shown = 0
    for idx, old, match in missing:
        if match and shown < preview_limit:
            print(f"  FIX: {old}  ->  {match}")
            shown += 1
    for idx, old, match in missing:
        if not match and shown < preview_limit:
            print(f"  TODO: {old}  ->  <no match>")
            shown += 1
    if shown >= preview_limit:
        print(f"  ... truncated preview ...")

    if apply and missing:
        # Backup then apply only resolvable matches
        bak = backup_file(ann_path)
        print(f"[INFO] Backed up annotations to: {bak}")
        changes = 0
        for idx, old, match in missing:
            if match:
                images[idx]['file_name'] = match
                changes += 1
        with open(ann_path, 'w') as f:
            json.dump(data, f)
        print(f"[INFO] Wrote updated annotations: {ann_path} (changed {changes})")
        changed = changes

    return len(images), unchanged, changed


def main():
    ap = argparse.ArgumentParser(description='Normalize/align COCO file_name entries to disk files')
    ap.add_argument('--root', default='data/rink', help='Dataset root containing images/ and annotations/')
    ap.add_argument('--apply', action='store_true', help='Apply changes to annotations (with .bak backups)')
    ap.add_argument('--splits', nargs='*', default=['train2017', 'val2017'], help='Splits to process')
    args = ap.parse_args()

    total_images = 0
    total_unchanged = 0
    total_changed = 0
    for split in args.splits:
        n, u, c = process_split(args.root, split, args.apply)
        total_images += n
        total_unchanged += u
        total_changed += c

    print("\n=== Summary ===")
    print(f"Images total   : {total_images}")
    print(f"Already aligned: {total_unchanged}")
    print(f"Changed        : {total_changed}{' (applied)' if args.apply else ' (dry-run)'}")


if __name__ == '__main__':
    main()


