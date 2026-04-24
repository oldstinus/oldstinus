import os
import numpy as np
import pandas as pd
import scipy.io
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

############################################
# 1. Helpers: mat_struct -> dict
############################################
def _todict(matobj):
    d = {}
    if not hasattr(matobj, "_fieldnames"):
        return matobj
    for field in matobj._fieldnames:
        elem = getattr(matobj, field)
        if isinstance(elem, scipy.io.matlab.mat_struct):
            d[field] = _todict(elem)
        elif isinstance(elem, np.ndarray):
            d[field] = _tolist(elem)
        else:
            d[field] = elem
    return d

def _tolist(ndarray):
    if not isinstance(ndarray, np.ndarray):
        return ndarray
    elem_list = []
    for elem in ndarray:
        if isinstance(elem, scipy.io.matlab.mat_struct):
            elem_list.append(_todict(elem))
        elif isinstance(elem, np.ndarray):
            elem_list.append(_tolist(elem))
        else:
            elem_list.append(elem)
    return elem_list

def _check_keys(d):
    for key in list(d.keys()):
        if key.startswith("__"):
            continue
        if isinstance(d[key], scipy.io.matlab.mat_struct):
            d[key] = _todict(d[key])
        elif isinstance(d[key], np.ndarray):
            d[key] = _tolist(d[key])
    return d

def loadmat(filepath):
    mat_data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    mat_data = _check_keys(mat_data)
    return mat_data

############################################
# 2. CSV-export
############################################
def save_structure_to_csv(struct_data, struct_name, out_dir, base_filename):
    """
    Probeer struct_data direct in DataFrame te gieten.
    Lukt dat niet, dan elk subveld, of scalar fallback.
    """
    import traceback
    try:
        df = pd.DataFrame(struct_data)
        out_path = os.path.join(out_dir, f"{base_filename}_{struct_name}.csv")
        df.to_csv(out_path, index=False)
        print(f"[OK] CSV voor '{struct_name}': {out_path}")
    except Exception as e:
        if isinstance(struct_data, dict):
            print(f"[!] Kan '{struct_name}' niet direct exporteren als CSV: {e}")
            for key, value in struct_data.items():
                sub_name = f"{struct_name}_{key}"
                try:
                    df_sub = pd.DataFrame(value)
                    out_path = os.path.join(out_dir, f"{base_filename}_{sub_name}.csv")
                    df_sub.to_csv(out_path, index=False)
                    print(f"[OK] CSV veld '{key}' in '{struct_name}': {out_path}")
                except Exception as ex_sub:
                    # fallback scalar
                    if isinstance(value, (str, int, float)):
                        out_path = os.path.join(out_dir, f"{base_filename}_{sub_name}.csv")
                        df_scalar = pd.DataFrame({"Value": [value]})
                        df_scalar.to_csv(out_path, index=False)
                        print(f"[OK] CSV scalar veld '{key}' in '{struct_name}': {out_path}")
                    else:
                        print(f"[!] Kan veld '{key}' van '{struct_name}' niet exporteren: {ex_sub}")
        elif isinstance(struct_data, (str, int, float)):
            out_path = os.path.join(out_dir, f"{base_filename}_{struct_name}.csv")
            df_scalar = pd.DataFrame({"Value": [struct_data]})
            df_scalar.to_csv(out_path, index=False)
            print(f"[OK] CSV scalar veld '{struct_name}': {out_path}")
        else:
            print(f"[!] Kan '{struct_name}' niet exporteren: {e}")

def process_mat_file(mat_filepath):
    data = loadmat(mat_filepath)
    mat_dir = os.path.dirname(mat_filepath)
    base_fname = os.path.splitext(os.path.basename(mat_filepath))[0]
    for key in data:
        if key.startswith("__"):
            continue
        save_structure_to_csv(data[key], key, mat_dir, base_fname)
    return data, mat_dir, base_fname

############################################
# 3. Interpolated Curtain-Plot
############################################
def create_curtain_plot_interpolated(data, Nx=300, Nz=300, method='linear'):
    """
    Maakt een 2D "opgevulde" curtain-plot (afstand vs. diepte) met interpolatie.
      - We halen points (dist_1d, depth_1d, speed_1d) uit 
        ADCP-data van transducer tot bodem.
      - We maken een regelmatig grid in (x, y) = (afstand, diepte) 
        en gebruiken griddata om speed te interpoleren.
      - Nx, Nz => resolutie in horizontale en verticale richting.
      - method = 'linear' (of 'cubic', 'nearest').

    Vereist in data:
      - Summary["Track"] => (NS,2)
      - System["Cell_Start"], System["Cell_Size"]
      - BottomTrack["BT_Depth"]
      - WaterTrack["Velocity"] => (NC,4,NS) 
         (u=..., v=..., w=..., etc.)

    Let op: 
      - Diepte = cell_start + (index+0.5)*cell_size
      - Afkappen op BT_Depth (bodem)
      - Horiz. afstand = cumulatief over track
    """
    # ---------------------------
    # Check structuren
    # ---------------------------
    req = ["Summary", "System", "BottomTrack", "WaterTrack"]
    for r in req:
        if r not in data:
            print(f"[!] '{r}' ontbreekt in data; geen plot.")
            return
    summary = data["Summary"]
    system  = data["System"]
    btrack  = data["BottomTrack"]
    wtrack  = data["WaterTrack"]

    try:
        track = np.array(summary["Track"])            # (NS,2)
        cell_start = np.array(system["Cell_Start"]).squeeze()  # (NS,)
        cell_size  = np.array(system["Cell_Size"]).squeeze()   # (NS,)
        bt_depth   = np.array(btrack["BT_Depth"]).squeeze()    # (NS,)
        velocity   = np.array(wtrack["Velocity"])              # (NC,4,NS)
    except KeyError as e:
        print(f"[!] Vereist veld ontbreekt: {e}. Geen plot.")
        return

    NS = velocity.shape[2]
    NC = velocity.shape[0]

    # ---------------------------
    # 1. Afstand langs route
    # ---------------------------
    x = track[:,0]
    y = track[:,1]
    dist = np.zeros(NS)
    for i in range(1, NS):
        dist[i] = dist[i-1] + np.hypot(x[i]-x[i-1], y[i]-y[i-1])

    # ---------------------------
    # 2. Data samenvoegen in 1D arrays
    #    dist_1d, depths_1d, speed_1d
    # ---------------------------
    dist_all   = []
    depth_all  = []
    speed_all  = []

    for i in range(NS):
        u_i = velocity[:,0,i]  # horizontale component 1
        v_i = velocity[:,1,i]  # horizontale component 2
        speed_i = np.sqrt(u_i**2 + v_i**2)

        z_raw = cell_start[i] + (np.arange(NC)+0.5)*cell_size[i]
        valid = z_raw < bt_depth[i]
        z_raw = z_raw[valid]
        speed_i = speed_i[valid]
        if len(z_raw) < 1:
            continue
        # sample i => dist[i], z_raw, speed_i
        # we plakken in 1D-lists
        dist_all.append(np.full_like(z_raw, dist[i]))
        depth_all.append(z_raw)
        speed_all.append(speed_i)

    if len(dist_all) == 0:
        print("[!] Geen bruikbare data om te plotten.")
        return

    dist_1d  = np.concatenate(dist_all)
    depth_1d = np.concatenate(depth_all)
    speed_1d = np.concatenate(speed_all)

    # ---------------------------
    # 3. 2D Grid en Interpolatie
    # ---------------------------
    # Definieer uniform raster in X=afstand, Y=diepte
    x_min, x_max = dist_1d.min(), dist_1d.max()
    z_min, z_max = depth_1d.min(), depth_1d.max()

    # Bouw linspace en meshgrid
    dist_grid = np.linspace(x_min, x_max, Nx)
    depth_grid = np.linspace(z_min, z_max, Nz)
    # (X2d, Y2d) => shape (Nz, Nx)
    X2d, Y2d = np.meshgrid(dist_grid, depth_grid)

    # Interpoleer
    points = np.column_stack((dist_1d, depth_1d))  # (N,2)
    vals   = speed_1d                              # (N,)
    speed_grid = griddata(points, vals, (X2d, Y2d), method=method)

    # ---------------------------
    # 4. Plot
    # ---------------------------
    plt.figure(figsize=(10,6))
    # pcolormesh verwacht X2d, Y2d, speed_grid (shape (Nz, Nx))
    c = plt.pcolormesh(X2d, Y2d, speed_grid, cmap='viridis', shading='auto')
    plt.gca().invert_yaxis()
    plt.xlabel("Afstand langs route (m)")
    plt.ylabel("Diepte (m)")
    plt.title("Interpolated Curtain-Plot: horizontale snelheid")
    cb = plt.colorbar(c, label="m/s")
    plt.show()

############################################
# 4. main()
############################################
def main():
    root = tk.Tk()
    root.withdraw()

    mat_filepath = filedialog.askopenfilename(
        title="Selecteer een .mat bestand",
        filetypes=[("MAT files", "*.mat")]
    )
    if not mat_filepath:
        print("Geen bestand geselecteerd.")
        return
    
    print(f"Geselecteerd bestand:\n  {mat_filepath}")

    # (A) Laad & exporteer CSV
    data_dict, mat_dir, base_fname = process_mat_file(mat_filepath)

    # (B) Maak interpolated curtain-plot
    create_curtain_plot_interpolated(data_dict, Nx=300, Nz=300, method='linear')


if __name__ == "__main__":
    main()
