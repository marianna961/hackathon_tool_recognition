from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


# SAM setup 
def build_sam_generator(
    checkpoint: Path,
    model_type: str = "vit_b",
    device: str = "cpu",
) -> SamAutomaticMaskGenerator:
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device)
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=25,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.90,
        min_mask_region_area=500,
        output_mode="binary_mask",
    )


# mask processing 
def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / float(a_area + b_area - inter)


def _mask_props(mask: np.ndarray) -> Tuple[int, int, int, int, int, np.ndarray]:
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    area = int(xs.size)
    return x1, y1, x2, y2, area, np.array([xs.mean(), ys.mean()], dtype=np.float32)


def select_and_merge_masks(
    masks: Sequence[Dict],
    img_shape: Tuple[int, int, int],
    *,
    min_pixels: int = 100,
    max_area_frac: float = 0.60,
    center_max_dist: float = 300.0,
    aspect_ratio_min: float = 1.6,
    merge_iou_thresh: float = 0.15,
) -> Optional[np.ndarray]:
    h, w = img_shape[:2]
    img_area = w * h
    cx, cy = w * 0.5, h * 0.5

    cand: List[Dict] = []
    for md in masks:
        m = md["segmentation"]
        if m.sum() < min_pixels:
            continue
        x1, y1, x2, y2, area, center = _mask_props(m)
        if area > max_area_frac * img_area or x1 <= 1 or y1 <= 1 or x2 >= w - 2 or y2 >= h - 2:
            continue
        ar = max((x2 - x1) / max(1.0, (y2 - y1)), (y2 - y1) / max(1.0, (x2 - x1)))
        if ar < aspect_ratio_min:
            continue
        if np.hypot(center[0] - cx, center[1] - cy) > center_max_dist:
            continue
        cand.append({"mask": m, "bbox": (x1, y1, x2, y2), "score": area * ar})

    if not cand:
        return None

    cand.sort(key=lambda d: d["score"], reverse=True)
    merged = np.zeros((h, w), dtype=np.uint8)

    for i, c in enumerate(cand):
        if (_iou_xyxy(c["bbox"], c["bbox"]) == 0) and False:  # placeholder to keep structure compact
            pass
        if merged.any():
            # merge candidates that overlap sufficiently with current merged bbox
            mx = np.where(merged > 0)
            mb = (int(mx[1].min()), int(mx[0].min()), int(mx[1].max()), int(mx[0].max()))
            if _iou_xyxy(mb, c["bbox"]) < merge_iou_thresh:
                continue
        merged |= c["mask"].astype(np.uint8)

    # light dilation to close tiny gaps
    merged = cv2.dilate(merged, np.ones((2, 2), np.uint8), iterations=1)
    return merged


# image pipeline 

def resize_max_side(img: np.ndarray, max_size: int = 1024) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    if w >= h:
        nw, nh = max_size, int(h * (max_size / w))
    else:
        nh, nw = max_size, int(w * (max_size / h))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def process_image(
    path: Path,
    mask_gen: SamAutomaticMaskGenerator,
    out_dir: Path,
    max_size: int = 1024,
) -> bool:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return False

    orig_shape = bgr.shape
    img = resize_max_side(bgr, max_size)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    masks = mask_gen.generate(rgb)
    if not masks:
        return False

    merged = select_and_merge_masks(masks, img.shape)
    if merged is None:
        return False

    if img.shape != orig_shape:
        merged = cv2.resize(merged, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

    alpha = (merged > 0).astype(np.uint8) * 255
    b, g, r = cv2.split(bgr)
    rgba = cv2.merge((b, g, r, alpha))

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{path.stem}_sticker.png"), rgba)
    return True



def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([p for p in root.rglob("*") if p.suffix in exts and p.is_file()])


def main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser("SAM sticker extractor")
    ap.add_argument("--src", required=True, type=Path, help="sources")
    ap.add_argument("--dst", required=True, type=Path, help="RGBA-sticers")
    ap.add_argument("--checkpoint", required=True, type=Path, help="SAM (sam_vit_b_01ec64.pth)")
    ap.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-size", type=int, default=1024)
    args = ap.parse_args(argv)

    gen = build_sam_generator(args.checkpoint, args.model_type, args.device)
    images = list_images(args.src)
    if not images:
        print("no images found")
        return 1

    ok = 0
    total = len(images)
    for i, p in enumerate(images, 1):
        ok += int(process_image(p, gen, args.dst, max_size=args.max_size))
        if i % 10 == 0 or i == total:
            print(f"{i}/{total} processed, ok={ok}")

    print(f"done: {ok}/{total} saved -> {args.dst.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
