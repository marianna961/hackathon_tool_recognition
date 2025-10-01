from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict

import cv2
import numpy as np
import yaml
from tqdm.auto import tqdm

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass
class TemplateAsset:
    name: str
    image: np.ndarray
    mask: np.ndarray
    gray: np.ndarray
    src_path: Path


def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES and p.is_file()])


def load_template_assets(root: Path) -> List[TemplateAsset]:
    out: List[TemplateAsset] = []
    for p in list_images(root):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            alpha = (img > 0).astype(np.uint8) * 255
        elif img.shape[2] == 4:
            bgr, alpha = img[:, :, :3], img[:, :, 3]
            alpha = (alpha > 0).astype(np.uint8) * 255
        else:
            bgr = img
            gtmp = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            _, alpha = cv2.threshold(gtmp, 5, 255, cv2.THRESH_BINARY)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        out.append(TemplateAsset(p.stem, bgr, alpha, gray, p))
    return out


def load_backgrounds(root: Optional[Path]) -> List[np.ndarray]:
    if root is None:
        return []
    imgs = []
    for p in list_images(root):
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is not None:
            imgs.append(im)
    return imgs


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_homography(tshape: Tuple[int, int], out_size: Tuple[int, int],
                      scale_rng: Tuple[float, float], rot_rng_deg: Tuple[float, float],
                      persp_amp: float, trans_margin: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = tshape
    ow, oh = out_size
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    s = random.uniform(*scale_rng)
    ang = math.radians(random.uniform(*rot_rng_deg))
    c, sng = math.cos(ang), math.sin(ang)
    hw, hh = (w * s) / 2, (h * s) / 2
    base = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
    R = np.array([[c, -sng], [sng, c]], dtype=np.float32)
    rot = base @ R.T
    mx, my = trans_margin * ow, trans_margin * oh
    cx, cy = random.uniform(mx, ow - mx), random.uniform(my, oh - my)
    rot[:, 0] += cx
    rot[:, 1] += cy
    amp = persp_amp * min(ow, oh)
    dest = rot + np.random.uniform(-amp, amp, size=(4, 2)).astype(np.float32)
    dest[:, 0] = np.clip(dest[:, 0], 0, ow - 1)
    dest[:, 1] = np.clip(dest[:, 1], 0, oh - 1)
    H = cv2.getPerspectiveTransform(src, dest)
    return H.astype(np.float32), dest


def warp_template(t: TemplateAsset, H: np.ndarray, out_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    ow, oh = out_size
    img = cv2.warpPerspective(t.image, H, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    msk = cv2.warpPerspective(t.mask, H, (ow, oh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    msk = (msk > 0).astype(np.uint8) * 255
    return img, msk


def overlay(bg: np.ndarray, obj: np.ndarray, msk: np.ndarray) -> np.ndarray:
    out = bg.copy()
    m = msk.astype(bool)
    out[m] = obj[m]
    return out


def random_background(bgs: List[np.ndarray], size: Tuple[int, int]) -> np.ndarray:
    w, h = size
    if bgs:
        # fit with padding
        im = random.choice(bgs)
        ih, iw = im.shape[:2]
        s = min(w / iw, h / ih)
        nw, nh = int(iw * s), int(ih * s)
        resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((h, w, 3), np.uint8)
        x0, y0 = (w - nw) // 2, (h - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas
    noise = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (5, 5), 0)


def add_occlusions(img: np.ndarray, msk: np.ndarray, max_count: int, max_frac: float) -> np.ndarray:
    if max_count <= 0:
        return img
    ys, xs = np.where(msk > 0)
    if len(xs) == 0:
        return img
    y0, x0, y1, x1 = ys.min(), xs.min(), ys.max(), xs.max()
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    out = img.copy()
    for _ in range(random.randint(0, max_count)):
        f = random.uniform(0.05, max_frac)
        ow, oh = max(1, int(bw * f)), max(1, int(bh * f))
        xx = random.randint(max(0, x0 - ow), min(img.shape[1] - ow, x1))
        yy = random.randint(max(0, y0 - oh), min(img.shape[0] - oh, y1))
        roi = out[yy:yy + oh, xx:xx + ow]
        overlay = np.full_like(roi, np.random.randint(0, 256, size=3, dtype=np.uint8))
        alpha = random.uniform(0.6, 1.0)
        out[yy:yy + oh, xx:xx + ow] = cv2.addWeighted(roi, 1 - alpha, overlay, alpha, 0)
    return out


class FeatureMatcher:
    def __init__(self, feature: str, nfeatures: int = 400, ratio: float | None = None, ransac_reproj: float = 5.0):
        feature = feature.upper()
        if feature == "SIFT":
            self.det = cv2.SIFT_create(nfeatures=nfeatures)
            self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
            self.ratio = 0.75 if ratio is None else float(ratio)
            self.float_des = True
        else:
            self.det = cv2.ORB_create(nfeatures=nfeatures, scaleFactor=1.2, nlevels=8)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.ratio = 0.90 if ratio is None else float(ratio)
            self.float_des = False
        self.reproj = float(ransac_reproj)

    def match(self, tpl_gray: np.ndarray, tpl_mask: np.ndarray, scene_gray: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        kp1, des1 = self.det.detectAndCompute(tpl_gray, tpl_mask)
        kp2, des2 = self.det.detectAndCompute(scene_gray, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None
        if self.float_des:
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)
        knn = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)
        if len(good) < 4:
            return None
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.reproj)
        return (H.astype(np.float32), mask.astype(np.uint8)) if H is not None and mask is not None else None


# YOLO-Seg 
def mask_to_polygons(mask: np.ndarray, approx_eps_rel: float = 0.002) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if not contours:
        return []
    c = max(contours, key=cv2.contourArea)
    per = cv2.arcLength(c, True)
    eps = max(1.0, approx_eps_rel * per)
    approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
    return [approx] if len(approx) >= 3 else []


def poly_to_yolo_line(poly: np.ndarray, w: int, h: int, cls_id: int, max_pts: int = 300) -> str:
    if len(poly) > max_pts:
        idx = np.linspace(0, len(poly) - 1, num=max_pts, dtype=int)
        poly = poly[idx]
    xs = np.clip(poly[:, 0] / float(w), 0.0, 1.0)
    ys = np.clip(poly[:, 1] / float(h), 0.0, 1.0)
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
    return f"{cls_id} {coords}"


def save_yolo_label(path: Path, mask: np.ndarray, w: int, h: int, cls_id: int) -> bool:
    polys = mask_to_polygons(mask)
    if not polys:
        return False
    lines = [poly_to_yolo_line(p, w, h, cls_id) for p in polys]
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def write_data_yaml(root: Path, names: List[str]) -> None:
    data = {"path": str(root), "train": "images/train", "val": "images/val", "names": {i: n for i, n in enumerate(names)}}
    (root / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


# Pipeline 
def build_class_names(templates_dir: Path, from_subdirs: bool, single_name: str) -> Tuple[List[str], Dict[str, int]]:
    if from_subdirs:
        names = sorted([p.name for p in templates_dir.iterdir() if p.is_dir()]) or [single_name]
    else:
        names = [single_name]
    return names, {n: i for i, n in enumerate(names)}


def class_id_for(t: TemplateAsset, name2id: Dict[str, int], from_subdirs: bool) -> int:
    return name2id[t.src_path.parent.name] if from_subdirs else 0


def generate(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    templates = load_template_assets(Path(args.templates))
    bgs = load_backgrounds(Path(args.backgrounds)) if args.backgrounds else []

    names, name2id = build_class_names(Path(args.templates), args.classes_from_subdirs, args.class_name)
    W, H = args.image_width, args.image_height
    OUT = Path(args.output)
    for p in (OUT / "images" / "train", OUT / "images" / "val", OUT / "labels" / "train", OUT / "labels" / "val"):
        ensure_dir(p)

    matcher = FeatureMatcher(args.feature, args.nfeatures)
    min_inliers = args.min_inliers
    val_frac = args.val_frac

    made = 0
    attempts = 0
    max_attempts = args.num_samples * 10
    pbar = tqdm(total=args.num_samples, desc="generate", unit="scene")

    while made < args.num_samples and attempts < max_attempts:
        attempts += 1
        t = random.choice(templates)

        H_gt, _ = sample_homography(t.gray.shape, (W, H), args.scale_range, args.rotation_range, args.perspective, args.translation_margin)
        obj, msk = warp_template(t, H_gt, (W, H))
        bg = random_background(bgs, (W, H))
        scene = overlay(bg, obj, msk)
        scene = add_occlusions(scene, msk, args.max_occlusions, args.max_occlusion_fraction)

        res = matcher.match(t.gray, t.mask, cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY))
        if res is None:
            continue
        _, inlier_mask = res
        if int(np.count_nonzero(inlier_mask)) < min_inliers:
            continue

        split = "val" if random.random() < val_frac else "train"
        stem = f"scene_{made:05d}"
        img_path = OUT / "images" / split / f"{stem}.png"
        lbl_path = OUT / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(img_path), scene)

        cid = class_id_for(t, name2id, args.classes_from_subdirs)
        if not save_yolo_label(lbl_path, msk, W, H, cid):
            continue

        made += 1
        pbar.update(1)
        pbar.set_postfix_str(f"accept={made/max(1, attempts):.0%}")

    pbar.close()
    if made < args.num_samples:
        raise RuntimeError(f"need {args.num_samples}, got {made} after {attempts} attempts")

    write_data_yaml(OUT, names)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser("Synthetic YOLO-Seg generator (concise)")
    ap.add_argument("--templates", required=True, help="Папка с шаблонами (подпапки=классы при --classes-from-subdirs)")
    ap.add_argument("--backgrounds", default=None, help="Папка с фонами (опционально)")
    ap.add_argument("--output", default="synthetic_dataset", help="Выходной каталог")
    ap.add_argument("--num-samples", type=int, default=200, help="Сколько сцен сгенерировать")
    ap.add_argument("--image-width", type=int, default=1024)
    ap.add_argument("--image-height", type=int, default=768)
    ap.add_argument("--feature", choices=["ORB", "SIFT"], default="ORB")
    ap.add_argument("--nfeatures", type=int, default=400)
    ap.add_argument("--min-inliers", type=int, default=15)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--class-name", type=str, default="object")
    ap.add_argument("--classes-from-subdirs", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # геометрия и окклюзии
    ap.add_argument("--scale-range", type=float, nargs=2, default=(0.5, 1.6), metavar=("MIN", "MAX"))
    ap.add_argument("--rotation-range", type=float, nargs=2, default=(-40.0, 40.0), metavar=("MIN", "MAX"))
    ap.add_argument("--perspective", type=float, default=0.06)
    ap.add_argument("--translation-margin", type=float, default=0.08)
    ap.add_argument("--max-occlusions", type=int, default=1)
    ap.add_argument("--max-occlusion-fraction", type=float, default=0.25)

    return ap.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    generate(args)

if __name__ == "__main__":
    main()
