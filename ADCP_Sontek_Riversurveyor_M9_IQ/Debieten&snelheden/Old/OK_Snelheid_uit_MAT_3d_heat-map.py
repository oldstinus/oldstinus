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
    """Laadt een .mat-bestand en zet MATLAB-structen om naar Python-dicts."""
    mat_data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    mat_data = _check_keys(mat_data)
    return mat_data

############################################
# 2. CSV-export
############################################
def save_structure_to_csv(struct_data, struct_name, out_dir, base_filename):
    """
    Probeert 'struct_data' direct in DataFrame te gieten. 
    Lukt dat niet, dan subvelden of scalar fallback.
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
                    # Fallback: scalar/string
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

def export_top_level_to_csv(data_dict, mat_filepath):
    """
    Schrijf alle top-level structuren van data_dict -> CSV in map van mat_filepath.
    """
    mat_dir = os.path.dirname(mat_filepath)
    base_fname = os.path.splitext(os.path.basename(mat_filepath))[0]
    for key in data_dict:
        if key.startswith("__"):
            continue
        save_structure_to_csv(data_dict[key], key, mat_dir, base_fname)

############################################
# 3. ADCP-data extract (3D arrays)
############################################
def extract_3d_arrays(data_dict, interp_factor=1):
    """
    Bouw (Xplot, Yplot, Zplot, Splot, start_time) uit de .mat data.
    - Xplot, Yplot, Zplot, Splot shape => (nVert, NS)
    - start_time => bv. min(System["Time"])
    
    Vereist:
      data_dict["Summary"]["Track"] => (NS,2)
      data_dict["System"]["Time"], ["Cell_Start"], ["Cell_Size"]
      data_dict["BottomTrack"]["BT_Depth"]
      data_dict["WaterTrack"]["Velocity"] => (NC,4,NS)
    """
    reqs = ["Summary", "System", "BottomTrack", "WaterTrack"]
    for r in reqs:
        if r not in data_dict:
            raise ValueError(f"Struct '{r}' ontbreekt in data.")
    summary = data_dict["Summary"]
    system  = data_dict["System"]
    btrack  = data_dict["BottomTrack"]
    wtrack  = data_dict["WaterTrack"]

    # Arrays ophalen
    track    = np.array(summary["Track"])               # (NS,2)
    cstart   = np.array(system["Cell_Start"]).squeeze() # (NS,)
    csize    = np.array(system["Cell_Size"]).squeeze()  # (NS,)
    bt_depth = np.array(btrack["BT_Depth"]).squeeze()   # (NS,)
    velocity = np.array(wtrack["Velocity"])             # (NC,4,NS)

    # Tijdsdata
    if "Time" not in system:
        raise ValueError("System['Time'] ontbreekt, kan geen offset berekenen.")
    time_arr = np.array(system["Time"]).squeeze()
    start_time = np.min(time_arr)

    NC, four, NS = velocity.shape
    if four < 2:
        raise ValueError("Velocity array heeft minder dan 2 velocity-componenten.")

    # horizontale snelheid
    u = velocity[:,0,:]  # (NC, NS)
    v = velocity[:,1,:]
    speed = np.sqrt(u**2 + v**2)  # (NC, NS)

    nVert = (NC-1)*interp_factor + 1

    X2d = np.full((NS, nVert), np.nan)
    Y2d = np.full((NS, nVert), np.nan)
    Z2d = np.full((NS, nVert), np.nan)
    S2d = np.full((NS, nVert), np.nan)

    for i in range(NS):
        z_raw = cstart[i] + (np.arange(NC)+0.5)*csize[i]
        s_raw = speed[:, i]
        # knip tot bodem
        valid = z_raw < bt_depth[i]
        z_raw = z_raw[valid]
        s_raw = s_raw[valid]
        if len(z_raw) < 2:
            continue

        if interp_factor > 1:
            z_new = np.linspace(z_raw[0], z_raw[-1], (len(z_raw)-1)*interp_factor + 1)
            s_new = np.interp(z_new, z_raw, s_raw)
            z_use = z_new
            s_use = s_new
        else:
            z_use = z_raw
            s_use = s_raw

        npts = len(z_use)
        X2d[i,:npts] = track[i,0]
        Y2d[i,:npts] = track[i,1]
        Z2d[i,:npts] = z_use
        S2d[i,:npts] = s_use

    # transpose => (nVert, NS)
    Xplot = X2d.T
    Yplot = Y2d.T
    Zplot = Z2d.T
    Splot = S2d.T

    return Xplot, Yplot, Zplot, Splot, start_time

############################################
# 4. 3D Plot: meerdere profielen met offset
############################################
def plot_multiple_profiles_3d(profile_list, time_scale=0.1):
    """
    Plot alle profielen in 1 3D-figuur. Elk profiel wordt in Y-richting
    verschoven volgens (start_time - ref_start_time)*time_scale.

    profile_list is een list van tuples: 
       [(Xplot, Yplot, Zplot, Splot, start_t), ...]

    time_scale => meter per seconde (of andere factor)
    """
    if not profile_list:
        print("[!] Geen profielen beschikbaar om te plotten.")
        return

    # Min/max speed bepalen
    speeds_all = []
    for (Xp, Yp, Zp, Sp, t0) in profile_list:
        valids = Sp[~np.isnan(Sp)]
        if valids.size > 0:
            speeds_all.append(valids)
    if not speeds_all:
        print("[!] Geen geldige snelheden in data.")
        return

    s_min = min(arr.min() for arr in speeds_all)
    s_max = max(arr.max() for arr in speeds_all)
    norm = Normalize(vmin=s_min, vmax=s_max)
    cmap = cm.viridis

    # referentie: de eerste start_time in de lijst
    ref_time = profile_list[0][4]

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    for (Xp, Yp, Zp, Sp, t0) in profile_list:
        dt = t0 - ref_time
        offset = dt * time_scale

        # Y verschuiven
        Yp_off = Yp + offset

        facecolors = cmap(norm(Sp))
        surf = ax.plot_surface(
            Xp, Yp_off, -Zp,
            facecolors=facecolors,
            rstride=1, cstride=1,
            linewidth=0, shade=False
        )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cb.set_label("Horiz. snelheid (m/s)")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y + offset (m)")
    ax.set_zlabel("Diepte (m)")
    ax.set_title(f"Meerdere profielen in 3D (tijd-offset * {time_scale})")
    plt.show()

############################################
# 5. GUI met drie knoppen
############################################
class MultiProfileGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-profile ADCP GUI")
        self.mat_files = []  # paden naar .mat bestanden
        self.profile_list = []  # (X,Y,Z,S, start_time) per bestand

        # Buttons
        btn_select = tk.Button(root, text="Selecteer .mat bestanden", command=self.select_mat_files)
        btn_select.pack(pady=5)

        btn_export = tk.Button(root, text="Exporteer naar CSV", command=self.export_csv)
        btn_export.pack(pady=5)

        btn_plot = tk.Button(root, text="Plot 3D Offset", command=self.plot_3d_offset)
        btn_plot.pack(pady=5)

    def select_mat_files(self):
        """Vraag de gebruiker om meerdere .mat files te selecteren."""
        files = filedialog.askopenfilenames(
            title="Selecteer meerdere .mat bestanden",
            filetypes=[("MAT files", "*.mat")]
        )
        if files:
            self.mat_files = list(files)
            print("Geselecteerde bestanden:")
            for f in self.mat_files:
                print("  ", f)

    def export_csv(self):
        """Laad elk .mat bestand en exporteer top-level structs naar CSV."""
        if not self.mat_files:
            print("[!] Geen bestanden geselecteerd.")
            return
        for fpath in self.mat_files:
            print(f"--- Verwerk: {fpath} ---")
            try:
                data_dict = loadmat(fpath)
                export_top_level_to_csv(data_dict, fpath)
                print("[OK] CSV export voltooid.")
            except Exception as e:
                print(f"[!] Fout bij exporteren: {e}")

    def plot_3d_offset(self):
        """
        Laad elk .mat bestand, bouw 3D curtain arrays,
        en plot ze in één figuur met tijd-offset.
        """
        if not self.mat_files:
            print("[!] Geen bestanden geselecteerd.")
            return

        self.profile_list.clear()

        for fpath in self.mat_files:
            print(f"--- Verwerk voor 3D-plot: {fpath} ---")
            try:
                data_dict = loadmat(fpath)
                Xp, Yp, Zp, Sp, start_t = extract_3d_arrays(data_dict, interp_factor=2)
                self.profile_list.append((Xp, Yp, Zp, Sp, start_t))
            except Exception as e:
                print(f"[!] Fout bij bouwen 3D arrays voor {fpath}: {e}")

        if self.profile_list:
            plot_multiple_profiles_3d(self.profile_list, time_scale=0.1)
        else:
            print("[!] Geen valide profielen om te plotten.")


def main():
    root = tk.Tk()
    app = MultiProfileGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
