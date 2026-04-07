from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from vector_core import aggregate_vector_dataset, build_awac_sea_level_overlay


def main(
    vector_folder: str | Path = ".",
    wave_path: str | Path = "Herculesa660602_23092015_14112015.wap",
    month: int = 10,
    day: int = 13,
) -> plt.Figure:
    result = aggregate_vector_dataset(vector_folder)
    overlay = build_awac_sea_level_overlay(wave_path, month, day, len(result["surface_vec"]))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(result["surface_vec"], label="Vector")
    ax.plot(overlay, "r:+", label="AWAC")
    ax.set_ylabel("depth (m)")
    ax.legend()
    return fig


if __name__ == "__main__":
    main()
    plt.show()
