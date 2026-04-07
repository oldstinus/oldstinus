from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_awac_comparison(dir_aw: np.ndarray, dir_vec: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    axes[0].plot(dir_aw, ".")
    axes[0].set_ylim(0, 360)
    axes[0].set_ylabel("dir awac")
    axes[1].plot(np.arange(len(dir_vec)), dir_vec, ".")
    axes[1].set_ylim(0, 360)
    axes[1].set_ylabel("dir vec")
    axes[2].plot(dir_aw[: min(len(dir_aw), len(dir_vec))], dir_vec[: min(len(dir_aw), len(dir_vec))], ".")
    axes[2].set_xlabel("awac dir")
    axes[2].set_ylabel("vec dir")
    return fig


if __name__ == "__main__":
    raise SystemExit("Import plot_awac_comparison(...) and provide the direction arrays explicitly.")
