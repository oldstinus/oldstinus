#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detecteer automatisch wandcellen in Aquadopp-profielen op basis van
amplitude- en correlatie-signaturen.

Werkwijze per beam en per ensemble:
1) Zoek de eerste range-cel waar amplitude een sprong maakt (dA groot +).
2) Vereis tegelijk correlatie-afname (lage absolute correlatie en/of dC negatief).
3) Markeer vanaf die cel alle verdere cellen als wandbeinvloed.

Inputbestanden (zelfde basename):
- .hdr (voor Number of measurements / Number of cells)
- .a1 .a2 .a3 (amplitude per beam)
- .c1 .c2 .c3 (correlatie per beam)

Output:
- <basename>_wanddetectie_mask.csv
- <basename>_wanddetectie_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Tuple

import numpy as np


def parse_hdr_counts(hdr_path: str) -> Tuple[int, int]:
    text = open(hdr_path, "r", encoding="utf-8", errors="ignore").read()

    n_meas_m = re.search(r"Number of measurements\s+(\d+)", text, re.IGNORECASE)
    n_cells_m = re.search(r"Number of cells\s+(\d+)", text, re.IGNORECASE)
    if not n_meas_m or not n_cells_m:
        raise ValueError("Kon Number of measurements / Number of cells niet vinden in .hdr.")

    return int(n_meas_m.group(1)), int(n_cells_m.group(1))


def load_matrix(path: str, n_meas: int, n_cells: int) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] != n_cells:
        raise ValueError(f"{os.path.basename(path)}: verwacht {n_cells} kolommen, kreeg {arr.shape[1]}.")
    if arr.shape[0] != n_meas:
        raise ValueError(f"{os.path.basename(path)}: verwacht {n_meas} rijen, kreeg {arr.shape[0]}.")
    return arr.astype(float)


def smooth_axis1(x: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    if kernel_size <= 1:
        return x.copy()
    k = np.ones(kernel_size, dtype=float) / float(kernel_size)
    out = np.empty_like(x, dtype=float)
    for i in range(x.shape[0]):
        out[i, :] = np.convolve(x[i, :], k, mode="same")
    return out


def detect_wall_for_beam(
    amp: np.ndarray,
    corr: np.ndarray,
    amp_jump_min: float,
    corr_max: float,
    corr_drop_min: float,
    smooth_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return:
    - hit_idx: eerste wandcel-index per ensemble (-1 als niet gevonden)
    - wall_mask: bool matrix [n_meas, n_cells], True = wandbeinvloed
    """
    a = smooth_axis1(amp, smooth_k)
    c = smooth_axis1(corr, smooth_k)

    dA = np.diff(a, axis=1)
    dC = np.diff(c, axis=1)

    n_meas, n_cells = a.shape
    hit_idx = np.full(n_meas, -1, dtype=int)
    wall_mask = np.zeros((n_meas, n_cells), dtype=bool)

    for t in range(n_meas):
        cond = (
            (dA[t, :] >= amp_jump_min)
            & (
                (c[t, 1:] <= corr_max)
                | (dC[t, :] <= -corr_drop_min)
            )
        )
        idx = np.flatnonzero(cond)
        if idx.size == 0:
            continue

        hit = int(idx[0] + 1)
        hit_idx[t] = hit
        wall_mask[t, hit:] = True

    return hit_idx, wall_mask


def write_mask_csv(path: str, wall_any: np.ndarray, beam_hits: np.ndarray) -> None:
    n_meas, n_cells = wall_any.shape
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        header = ["ensemble", "beam1_hit_cell", "beam2_hit_cell", "beam3_hit_cell"]
        header.extend([f"wall_cell_{i+1}" for i in range(n_cells)])
        w.writerow(header)

        for t in range(n_meas):
            row = [t, int(beam_hits[0, t]), int(beam_hits[1, t]), int(beam_hits[2, t])]
            row.extend(int(v) for v in wall_any[t, :])
            w.writerow(row)


def write_summary_csv(path: str, wall_any: np.ndarray, beam_hits: np.ndarray) -> None:
    n_meas, n_cells = wall_any.shape
    frac_per_cell = wall_any.mean(axis=0)
    valid_hits = beam_hits >= 0

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["metric", "value"])
        w.writerow(["ensembles", n_meas])
        w.writerow(["cells", n_cells])
        w.writerow(["wall_fraction_any_cell", float(wall_any.any(axis=1).mean())])
        w.writerow(["beam1_hit_fraction", float(valid_hits[0, :].mean())])
        w.writerow(["beam2_hit_fraction", float(valid_hits[1, :].mean())])
        w.writerow(["beam3_hit_fraction", float(valid_hits[2, :].mean())])
        w.writerow([])
        w.writerow(["cell", "wall_fraction"])
        for i in range(n_cells):
            w.writerow([i + 1, float(frac_per_cell[i])])


def main() -> None:
    p = argparse.ArgumentParser(
        description="Automatische wandcel-detectie (Aquadopp) op amplitude + correlatie."
    )
    p.add_argument(
        "base_prefix",
        help="Basispad zonder extensie, bv. Projecten/.../zink301",
    )
    p.add_argument("--amp-jump-min", type=float, default=2.5, help="Minimale amplitudesprong per cel.")
    p.add_argument("--corr-max", type=float, default=60.0, help="Maximale correlatie op wand-hit cel.")
    p.add_argument("--corr-drop-min", type=float, default=8.0, help="Minimale correlatiedaling per cel.")
    p.add_argument("--smooth-k", type=int, default=3, help="Glijdende gemiddelde kernel over cellen.")
    p.add_argument(
        "--min-beams",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Minimaal aantal beams dat wand moet aangeven voor gecombineerde mask.",
    )
    args = p.parse_args()

    base = args.base_prefix
    hdr = base + ".hdr"
    a_paths = [base + ".a1", base + ".a2", base + ".a3"]
    c_paths = [base + ".c1", base + ".c2", base + ".c3"]

    missing = [p for p in [hdr, *a_paths, *c_paths] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Ontbrekende bestanden:\n- " + "\n- ".join(missing) + "\n\n"
            "Let op: dit script vereist expliciete correlatiebestanden (.c1/.c2/.c3)."
        )

    n_meas, n_cells = parse_hdr_counts(hdr)
    amps = [load_matrix(pth, n_meas, n_cells) for pth in a_paths]
    corrs = [load_matrix(pth, n_meas, n_cells) for pth in c_paths]

    beam_hits = np.full((3, n_meas), -1, dtype=int)
    beam_masks = np.zeros((3, n_meas, n_cells), dtype=bool)

    for b in range(3):
        hit_idx, mask = detect_wall_for_beam(
            amps[b],
            corrs[b],
            amp_jump_min=args.amp_jump_min,
            corr_max=args.corr_max,
            corr_drop_min=args.corr_drop_min,
            smooth_k=max(1, int(args.smooth_k)),
        )
        beam_hits[b, :] = hit_idx
        beam_masks[b, :, :] = mask

    wall_votes = beam_masks.sum(axis=0)
    wall_any = wall_votes >= int(args.min_beams)

    out_mask = base + "_wanddetectie_mask.csv"
    out_summary = base + "_wanddetectie_summary.csv"
    write_mask_csv(out_mask, wall_any, beam_hits)
    write_summary_csv(out_summary, wall_any, beam_hits)

    print(f"Klaar. Output:")
    print(f"- {out_mask}")
    print(f"- {out_summary}")


if __name__ == "__main__":
    main()

