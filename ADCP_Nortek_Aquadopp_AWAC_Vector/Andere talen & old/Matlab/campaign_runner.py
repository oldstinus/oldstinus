from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from nortek_core import (
    figure_depth_averaged_horizontal_velocity,
    figure_ts_depth_averaged_velocity_components,
    figure_ts_instrument_orientation,
    figure_ts_velocity_norm_profile,
    load_deployment_files,
    read_output_parameters,
)


def run_campaign_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs = []
    for entry in entries:
        dpl = load_deployment_files(entry["dpl_param"])
        out = read_output_parameters(dpl, entry["out_param"])
        runs.append(
            {
                "deployment": dpl,
                "output": out,
                "instrument_orientation": figure_ts_instrument_orientation(dpl, out),
                "depth_averaged_components": figure_ts_depth_averaged_velocity_components(dpl, out),
                "velocity_norm_profile": figure_ts_velocity_norm_profile(
                    dpl,
                    out,
                    entry["days_per_subplot"],
                    entry.get("min_depth_display", np.nan),
                    entry.get("max_depth_display", np.nan),
                    entry.get("max_velocity_display", np.nan),
                ),
                "horizontal_velocity": figure_depth_averaged_horizontal_velocity(
                    dpl,
                    out,
                    entry.get("ebb_flood", False),
                    entry.get("max_velocity_display", np.nan),
                    entry.get("ellipse_bounding_box_display", [np.nan, np.nan, np.nan, np.nan]),
                ),
            }
        )
    return runs


def show() -> None:
    plt.show()
