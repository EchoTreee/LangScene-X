#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData


C0 = 0.28209479177387814


def sh_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    rgb = f_dc * C0 + 0.5
    return np.clip(rgb, 0.0, 1.0)


def load_gaussian_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    names = vertex.data.dtype.names

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)

    rgb = None
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        f_dc = np.stack(
            [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]],
            axis=1,
        ).astype(np.float64)
        rgb = sh_dc_to_rgb(f_dc)
    elif {"red", "green", "blue"}.issubset(names):
        rgb = np.stack(
            [vertex["red"], vertex["green"], vertex["blue"]],
            axis=1,
        ).astype(np.float64)
        if rgb.max() > 1.0:
            rgb /= 255.0

    return xyz, rgb


def build_point_cloud(xyz: np.ndarray, rgb: np.ndarray | None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def save_rgb_ply(path: Path, pcd: o3d.geometry.PointCloud) -> None:
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize LangScene-X Gaussian .ply point clouds with approximate RGB colors.",
    )
    parser.add_argument(
        "ply_path",
        nargs="?",
        default="demo/output_real_openseg/point_cloud/iteration_12000/point_cloud.ply",
        help="Path to the input .ply file.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help="Rendered point size in the Open3D viewer.",
    )
    parser.add_argument(
        "--save-rgb-ply",
        type=str,
        default="",
        help="Optional output path for exporting a standard RGB .ply.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip the Open3D window and only print stats or export RGB PLY.",
    )
    return parser.parse_args()


def has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def main() -> None:
    args = parse_args()
    ply_path = Path(args.ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    xyz, rgb = load_gaussian_ply(ply_path)
    pcd = build_point_cloud(xyz, rgb)

    print(f"Loaded: {ply_path}")
    print(f"Points: {len(xyz)}")
    if rgb is None:
        print("Color: none found; visualizing geometry only")
    else:
        print("Color: approximated from f_dc_* coefficients")

    if args.save_rgb_ply:
        out_path = Path(args.save_rgb_ply)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_rgb_ply(out_path, pcd)
        print(f"Saved RGB PLY: {out_path}")

    if args.no_gui:
        return

    if not has_display():
        print("No GUI display detected; skipping Open3D window.", file=sys.stderr)
        print(
            "Use --no-gui to suppress this message, or pass --save-rgb-ply "
            "to export a standard RGB point cloud for MeshLab/CloudCompare.",
            file=sys.stderr,
        )
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LangScene-X PLY Viewer")
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = float(args.point_size)
    render_option.background_color = np.array([0.08, 0.08, 0.08])

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
