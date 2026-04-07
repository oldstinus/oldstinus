from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from vector_core import aggregate_vector_dataset, build_awac_sea_level_overlay, plot_vector_overview


def main(
    vector_folder: str | Path = ".",
    wave_path: str | Path = "Herculesa660602_23092015_14112015.wap",
    overlay_month: int = 10,
    overlay_day: int = 14,
) -> dict:
    result = aggregate_vector_dataset(vector_folder)
    overlay = build_awac_sea_level_overlay(wave_path, overlay_month, overlay_day, len(result["surface_vec"]))

    fig_velocity, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(result["vec_e_date"], "r")
    axes[0].set_title("Vector - Velocity East - date")
    axes[1].plot(result["vec_n_date"], "b")
    axes[1].set_title("Vector - Velocity North - date")
    axes[2].plot(result["vec_u_date"], "m")
    axes[2].set_title("Vector - Velocity Up - date")

    fig_pressure, axes_pressure = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes_pressure[0].plot(result["press_date"], "r")
    axes_pressure[0].set_title("Pressure")
    axes_pressure[1].plot(result["surface_vec"], "b")
    axes_pressure[1].set_title("Surface - vec")

    fig_direction, axis_direction = plt.subplots(figsize=(12, 4))
    axis_direction.plot(result["alfa_dir_date"])
    axis_direction.set_ylim(0, 360)
    axis_direction.set_title("Velocity Direction")

    fig_surface, axis_surface = plt.subplots(figsize=(12, 4))
    axis_surface.plot(result["surface_vec"], label="Vector")
    axis_surface.plot(overlay, "r:+", label="AWAC")
    axis_surface.set_title("Sea Level AWAC vs Vec")
    axis_surface.legend()

    result["overview_figure"] = plot_vector_overview(result, overlay)
    result["velocity_figure"] = fig_velocity
    result["pressure_figure"] = fig_pressure
    result["direction_figure"] = fig_direction
    result["surface_figure"] = fig_surface
    result["awac_overlay"] = overlay
    return result


if __name__ == "__main__":
    main()
    plt.show()
