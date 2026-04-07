from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass(frozen=True)
class CrateParams:
    nx: int = 3
    ny: int = 2
    bottle_diameter_mm: float = 85.0
    side_clearance_mm: float = 3.5
    divider_t_mm: float = 3.0
    wall_t_mm: float = 2.5
    post_t_mm: float = 12.0
    bottom_t_mm: float = 3.0
    rim_h_mm: float = 14.0
    mid_rail_h_mm: float = 8.0
    edge_gap_x_mm: float = 6.5
    edge_gap_y_mm: float = 8.0
    overall_h_mm: float = 330.0

    @property
    def cell_mm(self) -> float:
        return self.bottle_diameter_mm + 2.0 * self.side_clearance_mm

    @property
    def inner_l_mm(self) -> float:
        return self.nx * self.cell_mm + (self.nx - 1) * self.divider_t_mm + 2.0 * self.edge_gap_x_mm

    @property
    def inner_w_mm(self) -> float:
        return self.ny * self.cell_mm + (self.ny - 1) * self.divider_t_mm + 2.0 * self.edge_gap_y_mm

    @property
    def outer_l_mm(self) -> float:
        return self.inner_l_mm + 2.0 * self.wall_t_mm

    @property
    def outer_w_mm(self) -> float:
        return self.inner_w_mm + 2.0 * self.wall_t_mm

    @property
    def outer_h_mm(self) -> float:
        return self.overall_h_mm


def box_from_bounds(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> trimesh.Trimesh:
    extents = [x1 - x0, y1 - y0, z1 - z0]
    center = [(x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0]
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def build_crate_mesh(p: CrateParams) -> trimesh.Trimesh:
    parts: list[trimesh.Trimesh] = []

    x_max = p.outer_l_mm
    y_max = p.outer_w_mm
    z_max = p.outer_h_mm
    z_mid0 = z_max * 0.52
    z_mid1 = z_mid0 + p.mid_rail_h_mm
    z_rim0 = z_max - p.rim_h_mm

    base_rail_w = 12.0
    # Base perimeter frame.
    parts.append(box_from_bounds(0.0, x_max, 0.0, base_rail_w, 0.0, p.bottom_t_mm))
    parts.append(box_from_bounds(0.0, x_max, y_max - base_rail_w, y_max, 0.0, p.bottom_t_mm))
    parts.append(box_from_bounds(0.0, base_rail_w, base_rail_w, y_max - base_rail_w, 0.0, p.bottom_t_mm))
    parts.append(box_from_bounds(x_max - base_rail_w, x_max, base_rail_w, y_max - base_rail_w, 0.0, p.bottom_t_mm))

    # Base runners under bottle rows and columns.
    x_in0 = p.wall_t_mm
    x_in1 = x_max - p.wall_t_mm
    y_in0 = p.wall_t_mm
    y_in1 = y_max - p.wall_t_mm
    runner_w = 10.0
    for j in range(p.ny):
        y_center = y_in0 + p.edge_gap_y_mm + j * (p.cell_mm + p.divider_t_mm) + p.cell_mm / 2.0
        parts.append(
            box_from_bounds(
                x_in0,
                x_in1,
                y_center - runner_w / 2.0,
                y_center + runner_w / 2.0,
                0.0,
                p.bottom_t_mm,
            )
        )
    for i in range(p.nx):
        x_center = x_in0 + p.edge_gap_x_mm + i * (p.cell_mm + p.divider_t_mm) + p.cell_mm / 2.0
        parts.append(
            box_from_bounds(
                x_center - runner_w / 2.0,
                x_center + runner_w / 2.0,
                y_in0,
                y_in1,
                0.0,
                p.bottom_t_mm,
            )
        )

    # Corner posts.
    z_post0 = p.bottom_t_mm
    corners = [
        (0.0, p.post_t_mm, 0.0, p.post_t_mm),
        (x_max - p.post_t_mm, x_max, 0.0, p.post_t_mm),
        (0.0, p.post_t_mm, y_max - p.post_t_mm, y_max),
        (x_max - p.post_t_mm, x_max, y_max - p.post_t_mm, y_max),
    ]
    for x0, x1, y0, y1 in corners:
        parts.append(box_from_bounds(x0, x1, y0, y1, z_post0, z_max))

    # Mid rails.
    parts.append(box_from_bounds(0.0, x_max, 0.0, p.wall_t_mm, z_mid0, z_mid1))
    parts.append(box_from_bounds(0.0, x_max, y_max - p.wall_t_mm, y_max, z_mid0, z_mid1))
    parts.append(box_from_bounds(0.0, p.wall_t_mm, p.wall_t_mm, y_max - p.wall_t_mm, z_mid0, z_mid1))
    parts.append(box_from_bounds(x_max - p.wall_t_mm, x_max, p.wall_t_mm, y_max - p.wall_t_mm, z_mid0, z_mid1))

    # Top rim.
    parts.append(box_from_bounds(0.0, x_max, 0.0, p.wall_t_mm, z_rim0, z_max))
    parts.append(box_from_bounds(0.0, x_max, y_max - p.wall_t_mm, y_max, z_rim0, z_max))
    parts.append(box_from_bounds(0.0, p.wall_t_mm, p.wall_t_mm, y_max - p.wall_t_mm, z_rim0, z_max))
    parts.append(box_from_bounds(x_max - p.wall_t_mm, x_max, p.wall_t_mm, y_max - p.wall_t_mm, z_rim0, z_max))

    # Vertical side stiles between rails for torsional stiffness.
    style_positions_x = [x_max * 0.22, x_max * 0.50, x_max * 0.78]
    for xc in style_positions_x:
        parts.append(box_from_bounds(xc - p.wall_t_mm / 2.0, xc + p.wall_t_mm / 2.0, 0.0, p.wall_t_mm, z_post0, z_rim0))
        parts.append(
            box_from_bounds(
                xc - p.wall_t_mm / 2.0,
                xc + p.wall_t_mm / 2.0,
                y_max - p.wall_t_mm,
                y_max,
                z_post0,
                z_rim0,
            )
        )

    style_positions_y = [y_max * 0.30, y_max * 0.70]
    for yc in style_positions_y:
        parts.append(box_from_bounds(0.0, p.wall_t_mm, yc - p.wall_t_mm / 2.0, yc + p.wall_t_mm / 2.0, z_post0, z_rim0))
        parts.append(
            box_from_bounds(
                x_max - p.wall_t_mm,
                x_max,
                yc - p.wall_t_mm / 2.0,
                yc + p.wall_t_mm / 2.0,
                z_post0,
                z_rim0,
            )
        )

    # Internal dividers.
    z_div_top = z_max - p.rim_h_mm
    y_div0 = y_in0 + p.edge_gap_y_mm
    y_div1 = y_in1 - p.edge_gap_y_mm
    for i in range(1, p.nx):
        x0 = x_in0 + p.edge_gap_x_mm + i * p.cell_mm + (i - 1) * p.divider_t_mm
        x1 = x0 + p.divider_t_mm
        parts.append(box_from_bounds(x0, x1, y_div0, y_div1, p.bottom_t_mm, z_div_top))

    x_div0 = x_in0 + p.edge_gap_x_mm
    x_div1 = x_in1 - p.edge_gap_x_mm
    for j in range(1, p.ny):
        y0 = y_in0 + p.edge_gap_y_mm + j * p.cell_mm + (j - 1) * p.divider_t_mm
        y1 = y0 + p.divider_t_mm
        parts.append(box_from_bounds(x_div0, x_div1, y0, y1, p.bottom_t_mm, z_div_top))

    return trimesh.util.concatenate(parts)


def render_mesh(mesh: trimesh.Trimesh, output_png: Path) -> None:
    fig = plt.figure(figsize=(8.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    triangles = mesh.vertices[mesh.faces]
    poly = Poly3DCollection(
        triangles,
        facecolors="#B9D5EA",
        edgecolors="#1D3C54",
        linewidths=0.15,
        alpha=1.0,
    )
    ax.add_collection3d(poly)

    bounds = mesh.bounds
    mins = bounds[0]
    maxs = bounds[1]
    center = (mins + maxs) / 2.0
    max_range = (maxs - mins).max() / 2.0
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(0.0, center[2] + max_range)
    ax.view_init(elev=24, azim=37)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_png, dpi=240, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    params = CrateParams()
    mesh = build_crate_mesh(params)

    obj_path = out_dir / "crate_1l_lightweight.obj"
    stl_path = out_dir / "crate_1l_lightweight.stl"
    png_path = out_dir / "crate_1l_render.png"
    dims_path = out_dir / "crate_dimensions.txt"

    mesh.export(obj_path)
    mesh.export(stl_path)
    render_mesh(mesh, png_path)

    dims_text = (
        f"Outer LxWxH (mm): {params.outer_l_mm:.1f} x {params.outer_w_mm:.1f} x {params.outer_h_mm:.1f}\n"
        f"Inner LxW (mm): {params.inner_l_mm:.1f} x {params.inner_w_mm:.1f}\n"
        f"Cell size (mm): {params.cell_mm:.1f} x {params.cell_mm:.1f}\n"
        f"Grid: {params.nx} x {params.ny} for 1 L bottles (diameter ~{params.bottle_diameter_mm:.0f} mm)\n"
    )
    dims_path.write_text(dims_text, encoding="utf-8")
    print(f"Generated: {obj_path.name}, {stl_path.name}, {png_path.name}, {dims_path.name}")


if __name__ == "__main__":
    main()
