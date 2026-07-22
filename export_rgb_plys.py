#!/usr/bin/env python3

import argparse
from pathlib import Path

from view_ply import build_point_cloud, load_gaussian_ply, save_rgb_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch export LangScene-X Gaussian point clouds to standard RGB PLY files.",
    )
    parser.add_argument(
        "--point-cloud-root",
        type=str,
        default="demo/output_real_openseg/point_cloud",
        help="Root directory containing iteration_*/point_cloud.ply outputs.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="*",
        default=[],
        help="Optional list of iterations to export, for example: --iterations 5000 10000 12000",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="point_cloud.ply",
        help="Input PLY filename inside each iteration directory.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="point_cloud_rgb.ply",
        help="Output RGB PLY filename written inside each iteration directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing RGB PLY files.",
    )
    return parser.parse_args()


def list_iteration_dirs(root: Path, iterations: list[int]) -> list[Path]:
    if iterations:
        return [root / f"iteration_{it}" for it in sorted(set(iterations))]

    return sorted(
        path
        for path in root.glob("iteration_*")
        if path.is_dir()
    )


def main() -> None:
    args = parse_args()
    root = Path(args.point_cloud_root)
    if not root.exists():
        raise FileNotFoundError(f"Point cloud root not found: {root}")

    iteration_dirs = list_iteration_dirs(root, args.iterations)
    if not iteration_dirs:
        raise RuntimeError(f"No iteration directories found under: {root}")

    exported = 0
    skipped = 0

    for iteration_dir in iteration_dirs:
        input_path = iteration_dir / args.input_name
        output_path = iteration_dir / args.output_name

        if not input_path.exists():
            print(f"[skip] missing input: {input_path}")
            skipped += 1
            continue

        if output_path.exists() and not args.overwrite:
            print(f"[skip] already exists: {output_path}")
            skipped += 1
            continue

        xyz, rgb = load_gaussian_ply(input_path)
        pcd = build_point_cloud(xyz, rgb)
        save_rgb_ply(output_path, pcd)

        color_status = "approx_rgb" if rgb is not None else "geometry_only"
        print(
            f"[ok] {iteration_dir.name}: points={len(xyz)} color={color_status} "
            f"-> {output_path}"
        )
        exported += 1

    print(f"Done. exported={exported} skipped={skipped}")


if __name__ == "__main__":
    main()
