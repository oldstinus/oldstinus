import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import Delaunay

def read_sum_lines(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Vind header
    data_start = None
    for i, line in enumerate(lines):
        if 'Track' in line and 'Depth' in line:
            header = [c.strip() for c in line.strip().split(',')]
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("Geen header met 'Track' en 'Depth' gevonden in .sum bestand.")

    def col_idx(name):
        for i, c in enumerate(header):
            if c.startswith(name): return i
        raise ValueError(f"Kolom '{name}' niet gevonden in header ({header})")

    # Kolomindices
    try:
        track_idx = col_idx('Track')
        for zname in ['BT_Depth (m)', 'Depth (m)']:
            try:
                depth_idx = col_idx(zname)
                break
            except ValueError:
                continue
        else:
            raise ValueError("Geen kolom voor diepte onder instrument gevonden.")
        for dname in ['Direction (deg)', 'Boat_Direction (deg)', 'Heading (deg)', 'Boat_Direction']:
            try:
                dir_idx = col_idx(dname)
                break
            except ValueError:
                continue
        else:
            raise ValueError("Geen kolom voor gevaren richting gevonden.")
    except Exception as e:
        raise ValueError(f"Kan kolommen niet vinden: {e}")

    # Data lezen per lijn
    lines_data = []
    current = []
    last_track = None
    for line in lines[data_start:]:
        if not line.strip() or line.startswith('%'):
            continue
        parts = [x.strip() for x in line.strip().split(',')]
        if len(parts) != len(header): continue
        try:
            track = float(parts[track_idx])
            richting = float(parts[dir_idx])
            z = float(parts[depth_idx])
            # Nieuw profiel: track springt terug (of bijna terug) naar 0
            if last_track is not None and track < 1.0 and last_track > 2.0 and len(current) > 0:
                lines_data.append(current)
                current = []
            current.append([track, richting, z])
            last_track = track
        except Exception:
            continue
    if current:
        lines_data.append(current)

    return lines_data

def lines_to_xyz(lines_data):
    # Vraag gebruiker om X0, Y0 voor elke lijn
    root = tk.Tk(); root.withdraw()
    all_X, all_Y, all_Z = [], [], []
    for i, line in enumerate(lines_data):
        tracks = np.array([row[0] for row in line])
        richtingen = np.array([row[1] for row in line])
        dieptes = np.array([row[2] for row in line])
        x0 = simpledialog.askfloat(f"Lijn {i+1}", f"Geef X0 voor lijn {i+1} (meter):", initialvalue=0.0)
        y0 = simpledialog.askfloat(f"Lijn {i+1}", f"Geef Y0 voor lijn {i+1} (meter):", initialvalue=0.0)
        richting_rad = np.deg2rad(richtingen)
        X = x0 + tracks * np.sin(richting_rad)
        Y = y0 + tracks * np.cos(richting_rad)
        Z = -dieptes
        all_X.append(X)
        all_Y.append(Y)
        all_Z.append(Z)
    X = np.concatenate(all_X)
    Y = np.concatenate(all_Y)
    Z = np.concatenate(all_Z)
    return X, Y, Z

def plot_bathymetry_sum(filepath):
    lines_data = read_sum_lines(filepath)
    X, Y, Z = lines_to_xyz(lines_data)

    # Interpoleer op grid
    xi = np.linspace(np.nanmin(X), np.nanmax(X), 120)
    yi = np.linspace(np.nanmin(Y), np.nanmax(Y), 120)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((X, Y), Z, (Xi, Yi), method='cubic')

    # Masker buiten convex hull
    hull = Delaunay(np.column_stack((X, Y)))
    flat_grid = np.column_stack((Xi.ravel(), Yi.ravel()))
    mask = hull.find_simplex(flat_grid) >= 0
    Zi_flat = Zi.ravel()
    Zi_flat[~mask] = np.nan
    Zi = Zi_flat.reshape(Xi.shape)

    # Plot bathymetrie
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xi, Yi, Zi, cmap='turbo' if 'turbo' in plt.colormaps() else 'rainbow', linewidth=0)
    cb = fig.colorbar(surf, ax=ax, shrink=0.5)
    cb.set_label("Bodemhoogte (m t.o.v. instrument)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Hoogte bodem (negatief = diepte)")
    ax.set_title("Bathymetrie onder rastertrack (per lijn startpositie opgegeven)")
    plt.tight_layout()
    plt.show()

def main():
    root = tk.Tk()
    root.withdraw()
    file = filedialog.askopenfilename(title="Kies een .sum bestand", filetypes=[("SUM files","*.sum")])
    if not file: return
    try:
        plot_bathymetry_sum(file)
    except Exception as e:
        messagebox.showerror("Fout bij plotten", str(e))

if __name__ == "__main__":
    main()
