from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(base_name: str = "Herculesa660602_23092015_14112015_p") -> plt.Figure:
    beams = [np.loadtxt(Path(f"{base_name}.v{i}"), ndmin=2) for i in range(1, 4)]
    fig, axes = plt.subplots(8, 1, figsize=(12, 14), sharex=True)
    colors = ("k", "r", "b")
    for cell_index, axis in enumerate(axes):
        for beam_index, beam in enumerate(beams):
            axis.plot(beam[:, cell_index], colors[beam_index])
        axis.set_ylabel(f"Cell {cell_index + 1}\n(m/s)")
    axes[-1].set_xlabel("sample")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
