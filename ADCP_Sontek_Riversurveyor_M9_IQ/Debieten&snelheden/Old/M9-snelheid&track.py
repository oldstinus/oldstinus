#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Grafische interface voor het laden van een SonTek / QRev .mat bestand en
het genereren van twee figuren:

- een doorsnede van stroomsnelheden (afstand langs het traject vs. diepte)
  met de kleur gebaseerd op de snelheid; de bodem wordt opgevuld met grijs
- een horizontale boottrack (X vs. Y) gebaseerd op de bottom‑track

Gebruikersinterface
-------------------
Bij het uitvoeren van dit script verschijnt een dialoogvenster waarin je het
`.mat`‑bestand kunt selecteren. Na selectie worden de gegevens verwerkt en
worden er twee figuren opgeslagen in dezelfde map als het .mat‑bestand.

Afhankelijkheden
----------------
Deze script gebruikt uitsluitend standaardbibliotheken (``tkinter``) en
``numpy``, ``scipy`` en ``matplotlib``. Zorg ervoor dat deze geïnstalleerd
zijn (bijvoorbeeld via ``pip install numpy scipy matplotlib``).

Gebruik
-------
Run het script vanuit de terminal::

    python qrev_gui_plot.py

Er wordt een file‑dialog geopend waarin je het .mat‑bestand kiest. Daarna
worden ``_section.png`` en ``_track.png`` aangemaakt naast het gekozen
bestand. De limiet van de kleurenbalk kun je aanpassen via de variabele
``VMAX`` bovenaan het script.
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox


# ======== Algemene instellingen ========
# Maximumsnelheid op de kleurenbalk (m/s)
VMAX = 4.0


def load_qrev_mat(path):
    """Laadt de relevante structuren uit een SonTek / QRev .mat bestand.

    Parameters
    ----------
    path : str
        Volledig pad naar het .mat bestand.

    Returns
    -------
    dict
        Dictionary met de structuren ``WaterTrack``, ``BottomTrack``,
        ``GPS``, ``System`` en ``Setup``.
    """
    mat = sio.loadmat(path, squeeze_me=False, struct_as_record=False)

    wt = mat["WaterTrack"][0, 0]
    bt = mat["BottomTrack"][0, 0]
    gps = mat.get("GPS", None)
    system = mat["System"][0, 0]
    setup = mat["Setup"][0, 0]

    return {
        "WaterTrack": wt,
        "BottomTrack": bt,
        "GPS": gps,
        "System": system,
        "Setup": setup,
    }


def compute_boat_track(bottom_track, system):
    """Integreert bottom‑track snelheden tot relatieve X, Y coördinaten.

    QRev gebruikt hiervoor de ENU componenten van ``BT_Vel`` en de tijdstap
    ``System.Step``. De accumulatie levert de relatieve boottrack in meter.

    Parameters
    ----------
    bottom_track : mat_struct
        Structure met veld ``BT_Vel`` (E,N,Up,Err) en ``VB_Depth``.
    system : mat_struct
        Structure met veld ``Step`` (tijdstap in s).

    Returns
    -------
    x, y : ndarray
        Relatieve coördinaten in meter van lengte ``n_ensembles``.
    """
    bt_vel = np.squeeze(bottom_track.BT_Vel)
    step = np.squeeze(system.Step)

    # Zorg dat step een vector is van lengte n_ensembles
    if step.ndim == 0:
        step = np.full(bt_vel.shape[0], float(step))

    dx = bt_vel[:, 0] * step
    dy = bt_vel[:, 1] * step

    x = np.cumsum(dx)
    y = np.cumsum(dy)

    # Hercentreer naar (0,0)
    x -= x[0]
    y -= y[0]

    return x, y


def compute_transect_distance(x, y):
    """Bereken cumulatieve afstand langs de boottrack.

    Parameters
    ----------
    x, y : ndarray
        Relatieve coördinaten in meter.

    Returns
    -------
    dist : ndarray
        Cumulatieve afstand in meter.
    """
    dist = np.zeros_like(x)
    if len(x) > 1:
        dd = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        dist[1:] = np.cumsum(dd)
    return dist


def compute_cell_depths(system, setup, n_cells):
    """Bereken diepte van cellen t.o.v. wateroppervlak.

    Parameters
    ----------
    system : mat_struct
        Structure met ``Cell_Start`` en ``Cell_Size``.
    setup : mat_struct
        Structure met ``sensorDepth`` (diepte transducer onder wateroppervlak).
    n_cells : int
        Aantal snelheidscellen.

    Returns
    -------
    depths : ndarray
        Diepte (m) naar midden van elke cel.
    """
    cell_start = float(system.Cell_Start[0, 0])
    cell_size = float(system.Cell_Size[0, 0])
    sensor_depth = float(setup.sensorDepth[0, 0])
    cell_indices = np.arange(n_cells, dtype=float)
    depths = sensor_depth + cell_start + cell_size * cell_indices
    return depths


def compute_speed_section(water_track, bottom_track, system, setup):
    """Genereer alle data voor de doorsnede‑ en boottrack‑figuren.

    Parameters
    ----------
    water_track : mat_struct
    bottom_track : mat_struct
    system : mat_struct
    setup : mat_struct

    Returns
    -------
    dist : ndarray
        Afstand langs de track (m) per ensemble.
    depths : ndarray
        Diepte (m) per cel.
    speed_masked : ndarray
        Gemaskeerde snelheidsmatrix (cells × ensembles).
    vb_depth : ndarray
        Waterdiepte per ensemble (m).
    x, y : ndarray
        Boottrack coördinaten (m).
    """
    vel = np.squeeze(water_track.Velocity)  # (n_cells, 4, n_ensembles)
    east = vel[:, 0, :]
    north = vel[:, 1, :]
    speed = np.sqrt(east**2 + north**2)
    n_cells, n_ens = speed.shape

    vb_depth = np.squeeze(bottom_track.VB_Depth)
    depths = compute_cell_depths(system, setup, n_cells)

    speed_masked = speed.copy()
    for j in range(n_ens):
        mask = depths > vb_depth[j]
        speed_masked[mask, j] = np.nan

    x, y = compute_boat_track(bottom_track, system)
    dist = compute_transect_distance(x, y)
    return dist, depths, speed_masked, vb_depth, x, y


def plot_velocity_section(dist, depths, speed, vb_depth, vmax=VMAX, outfile=None):
    """Maak een figure van de stroomsnelheid doorsnede en sla op.

    Parameters
    ----------
    dist : ndarray
        Afstand langs de track.
    depths : ndarray
        Diepte van cellen.
    speed : ndarray
        Gemaskeerde snelheden.
    vb_depth : ndarray
        Diepte van de bodem per ensemble.
    vmax : float
        Bovenlimiet van de kleurenbalk.
    outfile : str or None
        Pad waar de figuur wordt opgeslagen. Indien ``None`` wordt niet opgeslagen.
    """
    fig, ax = plt.subplots(figsize=(16, 3))
    extent = [dist.min(), dist.max(), depths.max(), depths.min()]
    im = ax.imshow(
        speed,
        extent=extent,
        origin="upper",
        aspect="auto",
        vmin=0.0,
        vmax=vmax,
    )
    ax.fill_between(dist, vb_depth, depths.max(), color="0.5", zorder=1)
    ax.set_xlabel("Afstand langs traject (m)")
    ax.set_ylabel("Diepte (m)")
    ax.set_ylim(depths.max(), 0)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.2)
    cbar.set_label("Snelheid (m/s)")
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=200)
    return fig, ax


def plot_boat_track(x, y, outfile=None):
    """Maak een figure van de boottrack en sla op.

    Parameters
    ----------
    x, y : ndarray
        Relatieve coördinaten.
    outfile : str or None
        Pad waar de figuur wordt opgeslagen.
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, y, "-", linewidth=0.8)
    ax.set_xlabel("Track X (m)")
    ax.set_ylabel("Track Y (m)")
    ax.grid(True, linestyle="--", linewidth=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=200)
    return fig, ax


def process_mat_file(path):
    """Verwerk het gekozen .mat bestand en schrijf figuren weg.

    Parameters
    ----------
    path : str
        Volledig pad naar het .mat bestand.

    Returns
    -------
    tuple
        (section_path, track_path) de paden naar de gegenereerde figuren.
    """
    structs = load_qrev_mat(path)
    dist, depths, speed, vb_depth, x, y = compute_speed_section(
        structs["WaterTrack"],
        structs["BottomTrack"],
        structs["System"],
        structs["Setup"],
    )
    base, _ = os.path.splitext(path)
    section_path = base + "_section.png"
    track_path = base + "_track.png"
    plot_velocity_section(dist, depths, speed, vb_depth, vmax=VMAX, outfile=section_path)
    plot_boat_track(x, y, outfile=track_path)
    return section_path, track_path


def main():
    """Start het Tk‑inter dialoog en verwerk het geselecteerde bestand."""
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecteer een QRev .mat bestand",
        filetypes=[("MAT bestanden", "*.mat"), ("Alle bestanden", "*.*")],
    )
    if not file_path:
        return
    try:
        section_path, track_path = process_mat_file(file_path)
    except Exception as exc:
        messagebox.showerror("Fout", f"Er is een fout opgetreden:\n{exc}")
        raise
    else:
        messagebox.showinfo(
            "Klaar",
            (
                f"Figuur gemaakt:\n\n"
                f"- Doorsnede: {os.path.basename(section_path)}\n"
                f"- Boottrack: {os.path.basename(track_path)}\n\n"
                f"Deze bestanden zijn opgeslagen in dezelfde map als het .mat bestand."
            ),
        )
    finally:
        root.destroy()


if __name__ == "__main__":
    main()