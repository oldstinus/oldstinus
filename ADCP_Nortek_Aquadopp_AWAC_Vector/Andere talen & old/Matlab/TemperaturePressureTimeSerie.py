from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from awac_core import plot_temperature_pressure_time_series


def main(path: str | Path = "Herculesa660602_23092015_14112015_p.sen") -> tuple[plt.Figure, plt.Figure]:
    figures = plot_temperature_pressure_time_series(path)
    plt.show()
    return figures


if __name__ == "__main__":
    main()
