"""Estimate field coordinates of a marked pixel from a photographed task map.

The PDF map artwork is registered to each photo; results remain survey estimates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

REFERENCE = Path(__file__).with_name("map_reference.png")


def locate(
    photo_path: Path,
    pixels: list[tuple[float, float]],
    anchors: list[tuple[float, float, float, float]] | None = None,
) -> tuple[list[tuple[float, float]], int, float]:
    # OpenCV's imread can fail on Windows paths containing Chinese characters.
    reference = cv2.imdecode(np.fromfile(REFERENCE, dtype=np.uint8), cv2.IMREAD_COLOR)
    photo = cv2.imdecode(np.fromfile(photo_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if reference is None or photo is None:
        raise ValueError("Map reference or photo could not be read")
    if anchors:
        if len(anchors) < 4:
            raise ValueError("At least four reference/photo anchor pairs are required")
        src = np.float32([[a[0], a[1]] for a in anchors]).reshape(-1, 1, 2)
        dst = np.float32([[a[2], a[3]] for a in anchors]).reshape(-1, 1, 2)
        matrix, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 8)
    else:
        sift = cv2.SIFT_create(nfeatures=3000)
        kp_ref, desc_ref = sift.detectAndCompute(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None)
        kp_photo, desc_photo = sift.detectAndCompute(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), None)
        if desc_ref is None or desc_photo is None:
            raise ValueError("Not enough map features in photo")
        pairs = cv2.BFMatcher().knnMatch(desc_ref, desc_photo, k=2)
        good = [first for first, second in pairs if first.distance < 0.7 * second.distance]
        if len(good) < 20:
            raise ValueError(f"Only {len(good)} reliable map matches")
        src = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_photo[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 7)
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if matrix is None or inliers < (4 if anchors else 20):
        raise ValueError(f"Map registration failed ({inliers} inliers)")
    error = np.linalg.norm(cv2.perspectiveTransform(src, matrix) - dst, axis=2).ravel()
    median_px = float(np.median(error[inlier_mask.ravel().astype(bool)]))
    inverse = np.linalg.inv(matrix)
    marked = np.float32(pixels).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(marked, inverse).reshape(-1, 2)
    height, width = reference.shape[:2]
    cm = [(float(u) * 150 / width, (height - float(v)) * 240 / height) for u, v in mapped]
    return cm, inliers, median_px


def main() -> int:
    parser = argparse.ArgumentParser(description="Map photo pixel to approximate field coordinates")
    parser.add_argument("photo", type=Path)
    parser.add_argument("--pixel", nargs=2, type=float, action="append", metavar=("X", "Y"), required=True)
    parser.add_argument("--anchor", nargs=4, type=float, action="append",
                        metavar=("REF_X", "REF_Y", "PHOTO_X", "PHOTO_Y"),
                        help="Corresponding printed-map and photo pixels; use four or more to disambiguate repeated art")
    args = parser.parse_args()
    result, inliers, median = locate(args.photo, [tuple(p) for p in args.pixel],
                                     [tuple(a) for a in args.anchor] if args.anchor else None)
    print(f"map anchors/matches={inliers}; median image reprojection error={median:.2f} px")
    for pixel, cm in zip(args.pixel, result):
        print(f"photo pixel {pixel} -> field ({cm[0]:.1f}, {cm[1]:.1f}) cm")
    print("Photo-derived positions are estimates; keep calibration flags false until physically checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
