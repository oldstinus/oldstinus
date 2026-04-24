from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from awac_core import plot_wave_day_summary


def main(path: str | Path = "Herculesa660602_23092015_14112015.wap", month: int = 10, day: int = 14) -> plt.Figure:
    return plot_wave_day_summary(path, month, day)


if __name__ == "__main__":
    main()
    plt.show()
