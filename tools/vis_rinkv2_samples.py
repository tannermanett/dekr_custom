import argparse
import os
import sys
import glob
import random
import xml.etree.ElementTree as ET

from typing import Dict, List, Optional, Tuple

try:
    import cv2  # type: ignore
except Exception as cv2_import_error:  # pragma: no cover
    cv2 = None  # type: ignore


def parse_label_colors(annotations_root: ET.Element) -> Dict[int, Tuple[int, int, int]]:
    """Extract label id to BGR color mapping from CVAT XML meta section.

    If a label color is missing or unparsable, the label will be omitted from the map.
    """
    label_to_bgr: Dict[int, Tuple[int, int, int]] = {}
    meta = annotations_root.find("meta")
    if meta is None:
        return label_to_bgr
    project = meta.find("project")
    if project is None:
        return label_to_bgr
    labels = project.find("labels")
    if labels is None:
        return label_to_bgr

    for label_elem in labels.findall("label"):
        name_elem = label_elem.find("name")
        color_elem = label_elem.find("color")
        if name_elem is None or color_elem is None:
            continue
        try:
            label_id = int(name_elem.text.strip())  # type: ignore[arg-type]
        except Exception:
            continue
        color_text = (color_elem.text or "").strip()
        if not color_text.startswith("#") or len(color_text) not in (4, 7):
            continue
        try:
            # Expand short hex like #abc to #aabbcc
            if len(color_text) == 4:
                r_hex = color_text[1] * 2
                g_hex = color_text[2] * 2
                b_hex = color_text[3] * 2
            else:
                r_hex = color_text[1:3]
                g_hex = color_text[3:5]
                b_hex = color_text[5:7]
            r = int(r_hex, 16)
            g = int(g_hex, 16)
            b = int(b_hex, 16)
            # OpenCV uses BGR
            label_to_bgr[label_id] = (b, g, r)
        except Exception:
            continue

    return label_to_bgr


def deterministic_color_for_label(label_id: int) -> Tuple[int, int, int]:
    """Fallback BGR color for a label id when not present in meta colors."""
    rnd = random.Random(label_id)
    return (rnd.randint(0, 255), rnd.randint(0, 255), rnd.randint(0, 255))


def parse_cvat_images(annotations_root: ET.Element) -> List[ET.Element]:
    """Return list of <image> elements from a CVAT XML annotations root."""
    return list(annotations_root.findall("image"))


def find_image_path(images_root: str, image_name: str) -> Optional[str]:
    """Try to find the image path either directly under root or via recursive search."""
    direct_path = os.path.join(images_root, image_name)
    if os.path.isfile(direct_path):
        return direct_path
    # Fallback recursive search
    matches = glob.glob(os.path.join(images_root, "**", image_name), recursive=True)
    return matches[0] if matches else None


def draw_points(
    image,
    point_elements: List[ET.Element],
    label_to_bgr: Dict[int, Tuple[int, int, int]],
    radius: int = 4,
    thickness: int = -1,
    draw_text: bool = True,
    font_scale: float = 0.5,
    text_thickness: int = 1,
):
    """Draw CVAT <points> annotations onto an OpenCV image."""
    for pe in point_elements:
        label_attr = pe.get("label")
        coords_attr = pe.get("points")
        if coords_attr is None:
            continue
        try:
            label_id = int(label_attr) if label_attr is not None else -1
        except Exception:
            label_id = -1

        try:
            # CVAT points for type "points" are a single pair "x,y"
            x_str, y_str = coords_attr.split(",")
            x = int(round(float(x_str)))
            y = int(round(float(y_str)))
        except Exception:
            continue

        color = label_to_bgr.get(label_id, deterministic_color_for_label(label_id))

        cv2.circle(image, (x, y), radius, color, thickness)
        if draw_text:
            label_text = str(label_id) if label_id >= 0 else "?"
            # Draw a thin outline for readability
            cv2.putText(
                image,
                label_text,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness=text_thickness + 2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                image,
                label_text,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness=text_thickness,
                lineType=cv2.LINE_AA,
            )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a few CVAT point annotations (rinkv2.xml) overlaid on frames."
        )
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=os.path.join("data", "rink", "annotations", "rinkv2.xml"),
        help="Path to CVAT XML annotations file.",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default=os.path.join("data", "rink", "images", "vip_htd_all_frames"),
        help="Root directory containing the frames.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join("output", "rink_v2_vis_samples"),
        help="Directory to save visualizations.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=12,
        help="Number of images to visualize.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle images before sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is set.",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Disable drawing label id text next to points.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=5,
        help="Circle radius for points.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=-1,
        help="Thickness for circle (-1 fills).",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.5,
        help="Font scale for labels.",
    )
    parser.add_argument(
        "--text-thickness",
        type=int,
        default=1,
        help="Text thickness for labels.",
    )

    args = parser.parse_args(argv)

    if cv2 is None:
        print(
            "OpenCV (cv2) is not available. Please install opencv-python to use this script.",
            file=sys.stderr,
        )
        return 2

    if not os.path.isfile(args.xml):
        print(f"XML not found: {args.xml}", file=sys.stderr)
        return 2
    if not os.path.isdir(args.images_root):
        print(f"Images root not found: {args.images_root}", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

    tree = ET.parse(args.xml)
    root = tree.getroot()

    label_to_bgr = parse_label_colors(root)

    image_elements = parse_cvat_images(root)
    if not image_elements:
        print("No <image> elements found in annotations.", file=sys.stderr)
        return 1

    selectable = list(image_elements)
    if args.shuffle:
        rnd = random.Random(args.seed)
        rnd.shuffle(selectable)

    num = max(1, min(args.num_samples, len(selectable)))
    chosen = selectable[:num]

    saved_count = 0
    missing_images: List[str] = []

    for image_elem in chosen:
        image_name = image_elem.get("name") or ""
        if not image_name:
            continue
        image_path = find_image_path(args.images_root, image_name)
        if image_path is None:
            missing_images.append(image_name)
            continue

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            missing_images.append(image_name)
            continue

        point_elements = image_elem.findall("points")
        draw_points(
            img,
            point_elements,
            label_to_bgr,
            radius=args.radius,
            thickness=args.thickness,
            draw_text=not args.no_text,
            font_scale=args.font_scale,
            text_thickness=args.text_thickness,
        )

        out_path = os.path.join(args.out_dir, image_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        ok = cv2.imwrite(out_path, img)
        if ok:
            saved_count += 1

    print(
        f"Saved {saved_count}/{num} visualizations to: {os.path.abspath(args.out_dir)}"
    )
    if missing_images:
        print(
            f"Warning: {len(missing_images)} images not found under '{args.images_root}'."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


