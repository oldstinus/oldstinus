import os
import numpy as np
import pandas as pd
import scipy.io
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

############################################
# 1. Hulpfuncties: mat_struct -> dict
############################################
def _todict(matobj):
    """
    Converteer één MATLAB mat_struct -> Python dict (recursief).
    """
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
    """
    Converteer numpy.ndarray -> (geneste) Python-lijsten.
    """
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
    """
    Loop door top-level keys, zet mat_struct -> dict.
    """
    for key in list(d.keys()):
        if key.startswith("__"):
            continue
        if isinstance(d[key], scipy.io.matlab.mat_struct):
            d[key] = _todict(d[key])
        elif isinstance(d[key], np.ndarray):
            d[key] = _tolist(d[key])
    return d

def loadmat(filepath):
    """
    Laadt de .mat file en converteert alle MATLAB-structuren naar Python dicts.
    """
    mat_data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    mat_data = _check_keys(mat_data)
    return mat_data

############################################
# 2. CSV-export
############################################
def save_structure_to_csv(struct_data, struct_name, out_dir, base_filename):
    """
    Probeert struct_data direct naar CSV te schrijven. 
    Lukt dat niet, probeer subvelden of fallback op 1×1 CSV als scalar.
    """
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
                    # Scalar/string fallback?
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
    """
    1) .mat -> dict
    2) Exporteer top-level structuren -> CSV
    3) Retourneert (data, mat_dir, base_fname)
    """
    data = loadmat(mat_filepath)
    mat_dir = os.path.dirname(mat_filepath)
    base_fname = os.path.splitext(os.path.basename(mat_filepath))[0]

    # Export per top-level key
    for key in data:
        if key.startswith("__"):
            continue
        save_structure_to_csv(data[key], key, mat_dir, base_fname)
    return data, mat_dir, base_fname

############################################
# 3. 3D Curtain-Plot met colorbar
############################################
def create_3d_surface_plot(data, interp_factor=2):
    """
    Maakt een 3D curtain-plot in (X,Y,Z) met kleur op basis van horizontale snelheid:
      - X = track[i,0]
      - Y = track[i,1]
      - Z = diepte (cel-centra) tot BT_Depth
      - Speed = sqrt(u^2 + v^2)
    
    interp_factor: extra interpolatie tussen de cellen (verticaal).

    Vereist: 
      data["Summary"]["Track"] => (NS,2)
      data["System"]["Cell_Start"], data["System"]["Cell_Size"] => (NS,)
      data["BottomTrack"]["BT_Depth"] => (NS,)
      data["WaterTrack"]["Velocity"] => (NC,4,NS)
    """
    # Check struct
    req_structs = ["Summary", "System", "BottomTrack", "WaterTrack"]
    for rs in req_structs:
        if rs not in data:
            print(f"[!] '{rs}' niet gevonden in data; skip 3D-plot.")
            return

    summary = data["Summary"]
    system  = data["System"]
    btrack  = data["BottomTrack"]
    wtrack  = data["WaterTrack"]

    # Haal arrays op
    try:
        track    = np.array(summary["Track"])              # (NS,2)
        cstart   = np.array(system["Cell_Start"]).squeeze()# (NS,)
        csize    = np.array(system["Cell_Size"]).squeeze() # (NS,)
        bt_depth = np.array(btrack["BT_Depth"]).squeeze()  # (NS,)
        velocity = np.array(wtrack["Velocity"])            # (NC,4,NS)
    except KeyError as e:
        print(f"[!] Vereist veld ontbreekt: {e}")
        return

    NS = velocity.shape[2]   # #samples
    NC = velocity.shape[0]   # #cellen

    # horizontale snelheid
    u = velocity[:, 0, :]  # shape (NC, NS)
    v = velocity[:, 1, :]  # shape (NC, NS)
    speed_raw = np.sqrt(u**2 + v**2)  # (NC, NS)

    # Bepaal de "verticale resolutie" na interp:
    nVert = (NC - 1)*interp_factor + 1

    # Prepare 2D arrays: (NS, nVert)
    #  X2d[i, c], Y2d[i, c], Z2d[i, c], S2d[i, c]
    X2d = np.full((NS, nVert), np.nan)
    Y2d = np.full((NS, nVert), np.nan)
    Z2d = np.full((NS, nVert), np.nan)
    S2d = np.full((NS, nVert), np.nan)

    for i in range(NS):
        z_raw = cstart[i] + (np.arange(NC)+0.5)*csize[i]
        s_raw = speed_raw[:, i]
        # Afkappen tot BT_Depth
        valid = z_raw < bt_depth[i]
        z_raw = z_raw[valid]
        s_raw = s_raw[valid]
        if len(z_raw) < 2:
            continue

        if interp_factor > 1:
            # vertical interpolation
            newz = np.linspace(z_raw[0], z_raw[-1], (len(z_raw)-1)*interp_factor + 1)
            news = np.interp(newz, z_raw, s_raw)
            z_use = newz
            s_use = news
        else:
            z_use = z_raw
            s_use = s_raw

        npts = len(z_use)
        X2d[i, :npts] = track[i,0]
        Y2d[i, :npts] = track[i,1]
        Z2d[i, :npts] = z_use
        S2d[i, :npts] = s_use

    # Transponeer => (nVert, NS)
    Xplot = X2d.T
    Yplot = Y2d.T
    Zplot = Z2d.T
    Splot = S2d.T

    # 3D-plot
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')

    # Norm + colormap
    s_min = np.nanmin(Splot)
    s_max = np.nanmax(Splot)
    norm = Normalize(vmin=s_min, vmax=s_max)
    facecolors = cm.viridis(norm(Splot))

    # We tekenen Z negatief, zodat "boven" = Z=0, en "diepte" = negatief
    # i.p.v. invert_zaxis() te gebruiken
    surf = ax.plot_surface(
        Xplot, Yplot, -Zplot,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Diepte (m)")
    ax.set_title("3D Curtain: horizontale snelheid")

    # colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    sm.set_array([])  # Lege array, we gebruiken alleen de norm
    cb = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cb.set_label("Snelheid (m/s)")

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

    # (A) Laad & Exporteer CSV
    data_dict, mat_dir, base_fname = process_mat_file(mat_filepath)

    # (B) 3D curtain-plot
    create_3d_surface_plot(data_dict, interp_factor=2)

if __name__ == "__main__":
    main()
