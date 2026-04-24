import os
import csv
import numpy as np
import scipy.io
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates

############################################
# Helpers voor .mat inlezen (MATLAB struct omzetten naar dict)
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
    """
    Laadt een .mat-bestand en zet MATLAB-structen om naar Python-dicts.
    """
    mat_data = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
    mat_data = _check_keys(mat_data)
    return mat_data

############################################
# Conversiefunctie: MATLAB datenum naar datetime met aangepaste offset
############################################
def _try_matlab_datenum(datenum_value):
    """
    Probeert een MATLAB datenum naar datetime om te zetten.
    Geeft None terug als de input buiten bereik valt.
    """
    if not np.isfinite(datenum_value):
        return None
    ordinal = int(np.floor(datenum_value))
    if ordinal < 1 or ordinal > datetime.max.toordinal():
        return None
    frac = datenum_value - ordinal
    try:
        return datetime.fromordinal(ordinal) + timedelta(days=frac) - timedelta(days=366)
    except (OverflowError, ValueError):
        return None


def _try_unix_timestamp(time_value, scale):
    """
    Probeert Unix tijd te converteren met schaal:
    scale=1 (s), 1e3 (ms), 1e6 (us), 1e9 (ns).
    """
    try:
        return datetime.fromtimestamp(time_value / scale)
    except (OverflowError, OSError, ValueError):
        return None


def _try_seconds_since_2000(time_value):
    """
    Probeert SonTek/QRev-stijl tijd: seconden sinds 2000-01-01.
    """
    try:
        return datetime(2000, 1, 1) + timedelta(seconds=time_value)
    except (OverflowError, ValueError):
        return None


def _is_reasonable_datetime(dt):
    return datetime(2000, 1, 1) <= dt <= datetime(2100, 1, 1)


def matlab_datenum_to_datetime(matlab_datenum):
    """
    Zet een tijdwaarde uit MAT-data om naar datetime.
    Ondersteunt:
      - MATLAB datenum
      - MATLAB datenum met extra offset (2.815.439)
      - Unix epoch tijd in s/ms/us/ns
    """
    value = float(np.asarray(matlab_datenum).squeeze())
    if not np.isfinite(value):
        raise ValueError(f"Ongeldige tijdwaarde: {matlab_datenum}")

    candidates = [
        _try_matlab_datenum(value - 2815439),
        _try_matlab_datenum(value),
        _try_seconds_since_2000(value),
        _try_unix_timestamp(value, 1.0),
        _try_unix_timestamp(value, 1e3),
        _try_unix_timestamp(value, 1e6),
        _try_unix_timestamp(value, 1e9),
    ]

    for dt in candidates:
        if dt is not None and _is_reasonable_datetime(dt):
            return dt

    for dt in candidates:
        if dt is not None:
            return dt

    raise ValueError(f"Kon tijdwaarde niet converteren: {matlab_datenum}")


def _last_finite_value(values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(finite[-1])


def _midpoint_time_value(time_values):
    arr = np.asarray(time_values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Geen geldige tijdwaarden gevonden in System['Time'].")
    return float((finite.min() + finite.max()) / 2.0)

############################################
# Debiet-berekening per transect
############################################
def calculate_discharge(mat_filepath):
    """
    Leest voor een MAT-file het RiverSurveyor-totaaldebiet uit
    Summary["Total_Q"] en koppelt dat aan het midden van de transecttijd.

    RiverSurveyor bewaart in Summary["Total_Q"] een cumulatieve reeks.
    De laatste geldige waarde stemt overeen met de gerapporteerde "Total Q"
    in de RiverSurveyor-software en in het .dis-bestand.
      
    Retourneert:
       transect_time, total_discharge
    """
    # Inlezen MAT-file
    data_dict = loadmat(mat_filepath)
    reqs = ["Summary", "System"]
    for r in reqs:
        if r not in data_dict:
            raise ValueError(f"Struct '{r}' ontbreekt in data.")
    summary = data_dict["Summary"]
    system  = data_dict["System"]

    total_discharge = _last_finite_value(summary.get("Total_Q"))
    if total_discharge is None:
        raise ValueError("Summary['Total_Q'] bevat geen geldige waarden.")

    time_arr = np.array(system["Time"]).squeeze()
    transect_time = _midpoint_time_value(time_arr)

    return transect_time, float(total_discharge)

############################################
# GUI: Tabel met geselecteerde MAT-files en optie tot activeren/deactiveren
############################################
class MatFilesTableGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Debiet per transect â€“ MAT-files")
        self.file_vars = []  # lijst van tuples: (filepath, BooleanVar)
        self.create_widgets()

    def create_widgets(self):
        # Button om bestanden te selecteren
        btn_select = tk.Button(self.master, text="Selecteer MAT-files", command=self.select_files)
        btn_select.pack(pady=5)

        # Frame waarin de file-checkbuttons komen (met scrollbar)
        self.files_frame = tk.Frame(self.master)
        self.files_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas met scrollbar (voor langere lijsten)
        self.canvas = tk.Canvas(self.files_frame)
        self.scrollbar = tk.Scrollbar(self.files_frame, orient="vertical", command=self.canvas.yview)
        self.inner_frame = tk.Frame(self.canvas)

        self.inner_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Button om de lijn-tijdsgrafiek te plotten
        btn_plot = tk.Button(self.master, text="Plot Lijn-Tijdsgrafiek Debiet", command=self.plot_time_series)
        btn_plot.pack(pady=5)

    def select_files(self):
        files = filedialog.askopenfilenames(title="Selecteer MAT-files", filetypes=[("MAT-files", "*.mat")])
        if files:
            self.file_vars = []  # reset de lijst
            # Verwijder eerdere checkbuttons
            for widget in self.inner_frame.winfo_children():
                widget.destroy()
            # Maak voor iedere file een checkbutton aan
            for f in files:
                var = tk.BooleanVar(value=True)
                cb = tk.Checkbutton(self.inner_frame, text=os.path.basename(f), variable=var)
                cb.pack(anchor="w")
                self.file_vars.append((f, var))

    def plot_time_series(self):
        times = []
        discharges = []
        file_names = []
        file_paths = []
        for f, var in self.file_vars:
            if var.get():
                try:
                    t, Q = calculate_discharge(f)
                    if not np.isfinite(Q):
                        print(f"Waarschuwing: debiet is ongeldig voor {f} en wordt overgeslagen.")
                        continue
                    times.append(t)
                    discharges.append(Q)
                    file_names.append(os.path.basename(f))
                    file_paths.append(f)
                except Exception as e:
                    print(f"Fout bij verwerken van {f}: {e}")
        if not times:
            print("Geen actieve bestanden met valide data gevonden.")
            return

        # Sorteer de metingen op tijd
        times = np.array(times)
        discharges = np.array(discharges)
        file_names = np.array(file_names, dtype=object)
        file_paths = np.array(file_paths, dtype=object)
        sort_idx = np.argsort(times)
        times_sorted = times[sort_idx]
        discharges_sorted = discharges[sort_idx]
        file_names_sorted = file_names[sort_idx]
        file_paths_sorted = file_paths[sort_idx]

        # Converteer tijdwaarden naar datetime-objecten; sla ongeldige waarden over
        times_dt = []
        discharges_dt = []
        file_names_dt = []
        file_paths_dt = []
        raw_times_dt = []
        for t, q, fname, fpath in zip(times_sorted, discharges_sorted, file_names_sorted, file_paths_sorted):
            try:
                times_dt.append(matlab_datenum_to_datetime(t))
                discharges_dt.append(q)
                file_names_dt.append(fname)
                file_paths_dt.append(fpath)
                raw_times_dt.append(t)
            except Exception as e:
                print(f"Waarschuwing: tijdwaarde overgeslagen ({t}): {e}")

        if not times_dt:
            print("Geen converteerbare tijdwaarden gevonden.")
            return

        # Exporteer dezelfde grafiekdata als CSV
        try:
            source_dirs = {os.path.dirname(p) for p in file_paths_dt if p}
            output_dir = source_dirs.pop() if len(source_dirs) == 1 else os.getcwd()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(output_dir, f"debiet_tijdserie_{timestamp}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    "bestand",
                    "pad",
                    "tijd_iso",
                    "raw_time_value",
                    "debiet_m3s",
                    "conventie",
                ])
                for fname, fpath, dt, raw_t, q in zip(file_names_dt, file_paths_dt, times_dt, raw_times_dt, discharges_dt):
                    writer.writerow([
                        fname,
                        fpath,
                        dt.strftime("%Y-%m-%d %H:%M:%S"),
                        f"{float(raw_t):.12g}",
                        f"{float(q):.12g}",
                        "RiverSurveyor Summary.Total_Q",
                    ])
            print(f"CSV export opgeslagen: {csv_path}")
        except Exception as e:
            print(f"Waarschuwing: CSV export mislukt: {e}")

        # Plot de lijn-tijdsgrafiek met datum/tijd op de x-as
        fig, ax = plt.subplots()
        line, = ax.plot(times_dt, np.array(discharges_dt), marker="o", linestyle="-")
        line.set_pickradius(8)
        ax.set_xlabel("Datum en tijd")
        ax.set_ylabel("Total Q (m3/s)")
        ax.set_title("Tijdserie van Total Q per transect")
        ax.grid(True)

        annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        annot.set_visible(False)

        xdata = line.get_xdata()
        ydata = line.get_ydata()

        def update_annot(ind):
            idx = ind["ind"][0]
            annot.xy = (xdata[idx], ydata[idx])
            annot.set_text(
                f"Bestand: {file_names_dt[idx]}\n"
                f"Tijd: {times_dt[idx]:%Y-%m-%d %H:%M:%S}\n"
                f"Total Q: {discharges_dt[idx]:+.3f} m3/s"
            )

        def hover(event):
            visible = annot.get_visible()
            if event.inaxes == ax:
                contains, ind = line.contains(event)
                if contains:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                elif visible:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
            elif visible:
                annot.set_visible(False)
                fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        # Formatteren van de x-as zodat datum en tijd goed leesbaar zijn
        fig.autofmt_xdate()
        date_format = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
        ax.xaxis.set_major_formatter(date_format)

        plt.show()

############################################
# Main
############################################
def main():
    root = tk.Tk()
    app = MatFilesTableGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
