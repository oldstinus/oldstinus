import argparse
import os
import re
import tkinter as tk
from datetime import datetime, timedelta
from tkinter import filedialog, messagebox, simpledialog, ttk

import chardet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector, Slider
from scipy.stats import linregress

plt.ion()

CALIBRATION_MODES = {
    "no_calibration": "Geen extra kalibratie",
    "diver_only": "Kalibreer enkel de Diver met de referentiesensor",
    "diver_and_hf": "Kalibreer Diver en OSSI HF met de referentiesensor",
}

PRESSURE_CAP = 2100

df_wave_global = pd.DataFrame()
df_mon_global = pd.DataFrame()
df_reference_global = pd.DataFrame()
calibration_reports_global = []


def select_directory(title="Selecteer map"):
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title=title)


def get_offset_input(label):
    root = tk.Tk()
    root.withdraw()
    try:
        value = simpledialog.askfloat(
            f"Voer {label} offset in",
            f"Voer de {label} offset in:",
            initialvalue=0.0,
        )
        return value if value is not None else 0.0
    except ValueError:
        messagebox.showerror("Ongeldige invoer", "Voer een geldig getal in.")
        return 0.0


def get_file_creation_time(path):
    try:
        return datetime.fromtimestamp(os.path.getctime(path))
    except OSError:
        return datetime.now()


def detect_file_encoding(path):
    with open(path, "rb") as handle:
        return chardet.detect(handle.read()).get("encoding") or "utf-8"


def dataset_selection_gui(default_mode="diver_only"):
    result = {"wave": False, "mon": False, "ref": False, "mode": default_mode}

    root = tk.Tk()
    root.title("Datasets en kalibratiemodus")

    use_wave = tk.BooleanVar(value=True)
    use_mon = tk.BooleanVar(value=True)
    use_ref = tk.BooleanVar(value=True)
    mode_var = tk.StringVar(value=default_mode)

    ttk.Label(root, text="Kies datasets").pack(anchor="w", padx=12, pady=(12, 4))
    ttk.Checkbutton(root, text="OSSI HF druksensor", variable=use_wave).pack(anchor="w", padx=16)
    ttk.Checkbutton(root, text="Diver (.mon)", variable=use_mon).pack(anchor="w", padx=16)
    ttk.Checkbutton(root, text="Referentiesensor (CSV)", variable=use_ref).pack(anchor="w", padx=16)

    ttk.Separator(root, orient="horizontal").pack(fill="x", padx=12, pady=10)
    ttk.Label(root, text="Kalibratie met referentiedruksensor").pack(anchor="w", padx=12, pady=(0, 4))

    for key in ("diver_only", "diver_and_hf", "no_calibration"):
        ttk.Radiobutton(root, text=CALIBRATION_MODES[key], value=key, variable=mode_var).pack(
            anchor="w", padx=16
        )

    def submit():
        result["wave"] = use_wave.get()
        result["mon"] = use_mon.get()
        result["ref"] = use_ref.get()
        result["mode"] = mode_var.get()
        root.destroy()

    ttk.Button(root, text="Verder", command=submit).pack(pady=12)
    root.mainloop()
    return result["wave"], result["mon"], result["ref"], result["mode"]


def process_wave_file(path, pressure_offset=0.0):
    try:
        data = pd.read_csv(path, encoding="latin1", skiprows=10, header=None)
        filtered = data[data[1] == "C1"].copy()
        pressure = pd.to_numeric(filtered[2], errors="coerce") * 1000 + pressure_offset
        start_time = get_file_creation_time(path)
        datetimes = [start_time + timedelta(seconds=i / 8) for i in range(len(pressure))]
        return pd.DataFrame({"Datetime": datetimes, "Pressure": pressure})
    except Exception as exc:
        print(f"Fout bij HF-bestand {path}: {exc}")
        return pd.DataFrame(columns=["Datetime", "Pressure"])


def read_mon_file(path):
    encoding = detect_file_encoding(path)
    lines = []
    with open(path, "r", encoding=encoding) as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if "END OF DATA" in line:
                break
            if index >= 54 and line:
                lines.append(line)
    return lines


def parse_mon_data(lines, pressure_offset=0.0, time_offset=0.0):
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) < 3 or "/" not in parts[0] or ":" not in parts[1]:
            continue
        try:
            dt = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y/%m/%d %H:%M:%S.%f")
            rows.append(
                {
                    "Datetime": dt + timedelta(seconds=time_offset),
                    "Pressure": float(parts[2].replace(",", ".")) + pressure_offset,
                }
            )
        except ValueError:
            print(f"Kon Diver-regel niet parsen: {line}")
    return pd.DataFrame(rows)


def load_reference_sensor_data(path, base_dt=None, pressure_offset=0.0):
    encoding = detect_file_encoding(path)
    with open(path, "r", encoding=encoding) as handle:
        first_line = handle.readline().strip()

    match = re.match(r"^DDR_(\d{8})_(\d{6})", first_line)
    if match:
        start_dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        df = pd.read_csv(path, skiprows=1)
        if df.shape[1] < 2:
            return pd.DataFrame(columns=["Datetime", "Pressure"])
        time_col, pressure_col = df.columns[:2]
        ref = pd.DataFrame(
            {
                "Datetime": start_dt + pd.to_timedelta(pd.to_numeric(df[time_col], errors="coerce"), unit="s"),
                "Pressure": pd.to_numeric(df[pressure_col], errors="coerce") + pressure_offset,
            }
        )
        return ref.dropna().reset_index(drop=True)

    df = pd.read_csv(path)
    if df.shape[1] < 2:
        return pd.DataFrame(columns=["Datetime", "Pressure"])

    lower_cols = [str(col).lower() for col in df.columns]
    time_col = next((df.columns[i] for i, name in enumerate(lower_cols) if name in {"time", "tijd", "datum", "datetime"}), df.columns[0])
    pressure_col = next((df.columns[i] for i, name in enumerate(lower_cols) if name in {"press", "pressure", "druk", "data"}), df.columns[1])

    ref = pd.DataFrame()
    try:
        ref["Datetime"] = pd.to_datetime(df[time_col], errors="raise")
    except Exception:
        if base_dt is None:
            base_dt = datetime.now()
        ref["Datetime"] = base_dt + pd.to_timedelta(pd.to_numeric(df[time_col], errors="coerce"), unit="s")
    ref["Pressure"] = pd.to_numeric(df[pressure_col], errors="coerce") + pressure_offset
    return ref.dropna().reset_index(drop=True)


def ensure_sensor_frame(df):
    if df.empty:
        return pd.DataFrame(columns=["Datetime", "Pressure"])
    clean = df.copy()
    clean["Datetime"] = pd.to_datetime(clean["Datetime"], errors="coerce")
    clean["Pressure"] = pd.to_numeric(clean["Pressure"], errors="coerce")
    clean = clean.dropna(subset=["Datetime", "Pressure"]).sort_values("Datetime").reset_index(drop=True)
    return clean


def clean_reference_sensor_data(df_reference, pressure_min=800, pressure_max=2200, max_step=80):
    ref = ensure_sensor_frame(df_reference)
    if ref.empty:
        return ref

    ref = ref.drop_duplicates(subset="Datetime", keep="last").reset_index(drop=True)
    ref["Pressure_raw"] = ref["Pressure"]

    within_bounds = ref["Pressure"].between(pressure_min, pressure_max)
    local_median = ref["Pressure"].rolling(window=5, center=True, min_periods=1).median()
    residual = (ref["Pressure"] - local_median).abs()
    local_mad = residual.rolling(window=5, center=True, min_periods=1).median().fillna(0.0)
    local_limit = np.maximum(25.0, 6.0 * local_mad.to_numpy())

    prev_step = ref["Pressure"].diff().abs()
    next_step = ref["Pressure"].diff(-1).abs()
    isolated_spike = (prev_step > max_step) & (next_step > max_step)
    drift_spike = residual > local_limit

    bad_mask = (~within_bounds) | isolated_spike.fillna(False) | drift_spike.fillna(False)
    ref["Flag"] = np.where(bad_mask, "filtered", "ok")
    ref.loc[bad_mask, "Pressure"] = np.nan
    ref["Pressure"] = ref["Pressure"].interpolate(limit_direction="both")
    return ref


def build_reference_alignment(sensor_df, reference_df, tolerance_seconds=2.0):
    sensor = ensure_sensor_frame(sensor_df)
    reference = ensure_sensor_frame(reference_df)
    if sensor.empty or reference.empty:
        return pd.DataFrame()

    merged = pd.merge_asof(
        sensor.sort_values("Datetime"),
        reference[["Datetime", "Pressure"]].rename(columns={"Pressure": "ReferencePressure"}).sort_values("Datetime"),
        on="Datetime",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
    )
    merged = merged.dropna(subset=["Pressure", "ReferencePressure"])
    merged = merged[
        merged["Pressure"].between(pressure_floor(), PRESSURE_CAP)
        & merged["ReferencePressure"].between(pressure_floor(), PRESSURE_CAP)
    ]
    return merged


def pressure_floor():
    return 0.0


def calibrate_sensor_to_reference(sensor_df, reference_df, sensor_name, tolerance_seconds=2.0):
    sensor = ensure_sensor_frame(sensor_df)
    report = {
        "sensor": sensor_name,
        "applied": False,
        "points": 0,
        "slope": np.nan,
        "intercept": np.nan,
        "r_squared": np.nan,
        "note": "Geen kalibratie uitgevoerd.",
    }
    if sensor.empty or reference_df.empty:
        report["note"] = "Ontbrekende sensor- of referentiedata."
        return sensor, report

    merged = build_reference_alignment(sensor, reference_df, tolerance_seconds=tolerance_seconds)
    report["points"] = int(len(merged))
    if len(merged) < 3:
        report["note"] = "Te weinig overlappende punten voor kalibratie."
        return sensor, report

    x = merged["ReferencePressure"].to_numpy()
    y = merged["Pressure"].to_numpy()
    slope, intercept, r_value, _, _ = linregress(x, y)
    if np.isclose(slope, 0.0):
        report["note"] = "Kalibratie afgebroken: regressiehelling is 0."
        return sensor, report

    calibrated = sensor.copy()
    calibrated["Pressure_raw"] = calibrated["Pressure"]
    calibrated["Pressure"] = (calibrated["Pressure"] - intercept) / slope
    calibrated["CalibrationSource"] = sensor_name
    report.update(
        {
            "applied": True,
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "note": f"{sensor_name} gekalibreerd t.o.v. referentie.",
        }
    )
    return calibrated, report


def apply_calibration_mode(df_wave, df_mon, df_reference, calibration_mode):
    reports = []
    wave = ensure_sensor_frame(df_wave)
    mon = ensure_sensor_frame(df_mon)
    reference = clean_reference_sensor_data(df_reference)

    if calibration_mode == "no_calibration":
        reports.append(
            {
                "sensor": "Mode",
                "applied": False,
                "points": 0,
                "slope": np.nan,
                "intercept": np.nan,
                "r_squared": np.nan,
                "note": CALIBRATION_MODES[calibration_mode],
            }
        )
        return wave, mon, reference, reports

    if reference.empty:
        reports.append(
            {
                "sensor": "Mode",
                "applied": False,
                "points": 0,
                "slope": np.nan,
                "intercept": np.nan,
                "r_squared": np.nan,
                "note": "Geen referentiedata beschikbaar; kalibratie overgeslagen.",
            }
        )
        return wave, mon, reference, reports

    if calibration_mode in {"diver_only", "diver_and_hf"} and not mon.empty:
        mon, report = calibrate_sensor_to_reference(mon, reference, "Diver", tolerance_seconds=2.0)
        reports.append(report)

    if calibration_mode == "diver_and_hf" and not wave.empty:
        wave, report = calibrate_sensor_to_reference(wave, reference, "OSSI HF", tolerance_seconds=1.0)
        reports.append(report)

    if not reports:
        reports.append(
            {
                "sensor": "Mode",
                "applied": False,
                "points": 0,
                "slope": np.nan,
                "intercept": np.nan,
                "r_squared": np.nan,
                "note": "Geen sensoren geselecteerd voor kalibratie.",
            }
        )
    return wave, mon, reference, reports


def convert_pressure_units(pressure_pa):
    return pressure_pa, pressure_pa / 9.80665, pressure_pa / 133.322, pressure_pa / 100000


def display_pressure_summary(df_wave, df_mon, df_reference):
    rows = []
    if not df_wave.empty:
        rows += [("HF start", df_wave["Pressure"].iloc[0]), ("HF eind", df_wave["Pressure"].iloc[-1]), ("HF max", df_wave["Pressure"].max())]
    if not df_mon.empty:
        rows += [("Diver start", df_mon["Pressure"].iloc[0]), ("Diver eind", df_mon["Pressure"].iloc[-1]), ("Diver max", df_mon["Pressure"].max())]
    if not df_reference.empty:
        rows += [("Ref start", df_reference["Pressure"].iloc[0]), ("Ref eind", df_reference["Pressure"].iloc[-1]), ("Ref max", df_reference["Pressure"].max())]
    if not rows:
        messagebox.showinfo("Geen data", "Niets om samen te vatten.")
        return

    columns = ["Soort", "Pa", "mmH2O", "mmHg", "bar"]
    summary = pd.DataFrame(columns=columns)
    for label, pressure in rows:
        pa, h2o, hg, bar = convert_pressure_units(pressure)
        summary.loc[len(summary)] = [label, pa, h2o, hg, bar]

    window = tk.Toplevel()
    window.title("Druksamenvatting")
    tree = ttk.Treeview(window, columns=columns, show="headings")
    for column in columns:
        tree.heading(column, text=column)
    for _, row in summary.iterrows():
        tree.insert("", "end", values=[f"{value:.2f}" if isinstance(value, float) else value for value in row])
    tree.pack(expand=True, fill="both")
    ttk.Button(window, text="OK", command=window.destroy).pack(pady=6)


def show_calibration_report(reports):
    if not reports:
        return
    window = tk.Toplevel()
    window.title("Kalibratierapport")
    columns = ["Sensor", "Toegepast", "Punten", "Slope", "Intercept", "R2", "Opmerking"]
    tree = ttk.Treeview(window, columns=columns, show="headings")
    for column in columns:
        tree.heading(column, text=column)

    for report in reports:
        tree.insert(
            "",
            "end",
            values=(
                report["sensor"],
                "ja" if report["applied"] else "nee",
                report["points"],
                "" if pd.isna(report["slope"]) else f'{report["slope"]:.5f}',
                "" if pd.isna(report["intercept"]) else f'{report["intercept"]:.3f}',
                "" if pd.isna(report["r_squared"]) else f'{report["r_squared"]:.4f}',
                report["note"],
            ),
        )
    tree.pack(expand=True, fill="both")
    ttk.Button(window, text="Sluiten", command=window.destroy).pack(pady=6)


def create_time_selection_gui(df_wave, df_mon, df_reference, selected_directory, update_callback, export_callback, show_wave=True, show_mon=True, show_ref=True):
    root = tk.Tk()
    root.title("Tijdselectie en export")

    all_dates = []
    for frame in (df_wave, df_mon, df_reference):
        if not frame.empty:
            all_dates.extend([frame["Datetime"].min(), frame["Datetime"].max()])
    start_default, end_default = min(all_dates), max(all_dates)

    ttk.Label(root, text="Begin (YYYY-MM-DD HH:MM:SS)").grid(row=0, column=0, padx=6, pady=6, sticky="e")
    start_entry = ttk.Entry(root, width=22)
    start_entry.insert(0, start_default.strftime("%Y-%m-%d %H:%M:%S"))
    start_entry.grid(row=0, column=1, padx=6, pady=6)

    ttk.Label(root, text="Einde (YYYY-MM-DD HH:MM:SS)").grid(row=1, column=0, padx=6, pady=6, sticky="e")
    end_entry = ttk.Entry(root, width=22)
    end_entry.insert(0, end_default.strftime("%Y-%m-%d %H:%M:%S"))
    end_entry.grid(row=1, column=1, padx=6, pady=6)

    wave_var = tk.BooleanVar(value=show_wave and not df_wave.empty)
    mon_var = tk.BooleanVar(value=show_mon and not df_mon.empty)
    ref_var = tk.BooleanVar(value=show_ref and not df_reference.empty)

    datasets = ttk.LabelFrame(root, text="Toon/exporteer")
    datasets.grid(row=2, column=0, columnspan=2, padx=6, pady=6, sticky="ew")
    ttk.Checkbutton(datasets, text="HF", variable=wave_var).pack(anchor="w", padx=8, pady=2)
    ttk.Checkbutton(datasets, text="Diver", variable=mon_var).pack(anchor="w", padx=8, pady=2)
    ttk.Checkbutton(datasets, text="Referentie", variable=ref_var).pack(anchor="w", padx=8, pady=2)

    ttk.Label(root, text=f"Exportmap: {selected_directory or '(nog te kiezen)'}").grid(
        row=3, column=0, columnspan=2, padx=6, pady=(0, 6), sticky="w"
    )

    def parse_window():
        start = datetime.strptime(start_entry.get(), "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(end_entry.get(), "%Y-%m-%d %H:%M:%S")
        if start >= end:
            raise ValueError("Begin moet voor einde liggen.")
        if not any((wave_var.get(), mon_var.get(), ref_var.get())):
            raise ValueError("Selecteer minstens een dataset.")
        return start, end, wave_var.get(), mon_var.get(), ref_var.get()

    def on_update():
        try:
            update_callback(*parse_window())
        except Exception as exc:
            messagebox.showerror("Ongeldige selectie", str(exc))

    def on_export():
        try:
            export_callback(*parse_window())
            root.destroy()
        except Exception as exc:
            messagebox.showerror("Exportfout", str(exc))

    ttk.Button(root, text="Update grafiek", command=on_update).grid(row=4, column=0, padx=6, pady=10, sticky="e")
    ttk.Button(root, text="Export en sluit", command=on_export).grid(row=4, column=1, padx=6, pady=10, sticky="w")
    root.mainloop()


def plot_time_series(start, end, show_wave, show_mon, show_ref):
    global df_wave_global, df_mon_global, df_reference_global

    shift = {"value": 0.0}
    window = tk.Toplevel()
    window.title("Tijdreeks")
    fig, ax = plt.subplots(figsize=(10, 5))
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    slider_ax = fig.add_axes([0.15, 0.02, 0.7, 0.04])
    slider = Slider(slider_ax, "Shift Diver (s)", -60, 60, valinit=0, valstep=0.5)

    def draw():
        ax.clear()
        if show_wave and not df_wave_global.empty:
            wave = df_wave_global[df_wave_global["Datetime"].between(start, end)]
            wave = wave[wave["Pressure"].between(pressure_floor(), PRESSURE_CAP)]
            ax.plot(wave["Datetime"], wave["Pressure"], label="HF", color="red", marker="o", ms=2, linewidth=0.5)
        if show_mon and not df_mon_global.empty:
            mon = df_mon_global.copy()
            mon["Datetime"] = mon["Datetime"] + timedelta(seconds=shift["value"])
            mon = mon[mon["Datetime"].between(start, end)]
            mon = mon[mon["Pressure"].between(pressure_floor(), PRESSURE_CAP)]
            ax.plot(mon["Datetime"], mon["Pressure"], label="Diver", color="blue", marker="x", ms=2, linewidth=0.5)
        if show_ref and not df_reference_global.empty:
            ref = df_reference_global[df_reference_global["Datetime"].between(start, end)]
            ref = ref[ref["Pressure"].between(pressure_floor(), PRESSURE_CAP)]
            ax.plot(ref["Datetime"], ref["Pressure"], label="Referentie", color="green", linestyle="--", marker="s", ms=4)
        ax.set_xlim(start, end)
        ax.set_title("Druk over tijd")
        ax.set_xlabel("Tijd")
        ax.set_ylabel("Druk (Pa)")
        ax.grid(True)
        ax.legend()
        canvas.draw_idle()

    def on_select(eclick, erelease):
        if None in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata):
            return
        xmin, xmax = sorted((eclick.xdata, erelease.xdata))
        ymin, ymax = sorted((eclick.ydata, erelease.ydata))
        if show_ref and not df_reference_global.empty:
            mask = df_reference_global["Datetime"].between(xmin, xmax) & df_reference_global["Pressure"].between(ymin, ymax)
            df_reference_global.loc[mask, "Pressure"] = np.nan
            df_reference_global["Pressure"] = df_reference_global["Pressure"].interpolate(limit_direction="both")
        draw()

    window.selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="data",
        interactive=True,
    )
    slider.on_changed(lambda value: (shift.update({"value": value}), draw()))
    draw()
    window.lift()
    plt.show(block=False)


def plot_regressions(start, end, show_wave, show_mon, show_ref):
    if show_wave and show_mon and not df_wave_global.empty and not df_mon_global.empty:
        merged = pd.merge_asof(
            df_wave_global[df_wave_global["Datetime"].between(start, end)].sort_values("Datetime"),
            df_mon_global[df_mon_global["Datetime"].between(start, end)].sort_values("Datetime"),
            on="Datetime",
            suffixes=("_hf", "_diver"),
        )
        merged = merged[
            merged["Pressure_hf"].between(pressure_floor(), PRESSURE_CAP)
            & merged["Pressure_diver"].between(pressure_floor(), PRESSURE_CAP)
        ]
        if not merged.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            x = merged["Pressure_hf"]
            y = merged["Pressure_diver"]
            ax.scatter(x, y, color="blue", label="HF vs Diver")
            slope, intercept, r_value, _, _ = linregress(x, y)
            ax.plot(x, slope * x + intercept, color="red", label=f"y={slope:.2f}x+{intercept:.2f}, R2={r_value**2:.3f}")
            ax.set_xlabel("HF (Pa)")
            ax.set_ylabel("Diver (Pa)")
            ax.set_title("HF vs Diver")
            ax.grid(True)
            ax.legend()
            plt.show(block=False)

    if show_wave and show_ref and not df_wave_global.empty and not df_reference_global.empty:
        merged = build_reference_alignment(
            df_wave_global[df_wave_global["Datetime"].between(start, end)],
            df_reference_global[df_reference_global["Datetime"].between(start, end)],
            tolerance_seconds=1.0,
        )
        if not merged.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            x = merged["ReferencePressure"]
            y = merged["Pressure"]
            ax.scatter(x, y, color="purple", label="HF vs Referentie")
            slope, intercept, r_value, _, _ = linregress(x, y)
            ax.plot(x, slope * x + intercept, color="magenta", label=f"y={slope:.2f}x+{intercept:.2f}, R2={r_value**2:.3f}")
            ax.set_xlabel("Referentie (Pa)")
            ax.set_ylabel("HF (Pa)")
            ax.set_title("HF vs Referentie")
            ax.grid(True)
            ax.legend()
            plt.show(block=False)

    if show_mon and show_ref and not df_mon_global.empty and not df_reference_global.empty:
        merged = build_reference_alignment(
            df_mon_global[df_mon_global["Datetime"].between(start, end)],
            df_reference_global[df_reference_global["Datetime"].between(start, end)],
            tolerance_seconds=2.0,
        )
        if not merged.empty:
            fig, ax = plt.subplots(figsize=(6, 5))
            x = merged["ReferencePressure"]
            y = merged["Pressure"]
            ax.scatter(x, y, color="orange", label="Diver vs Referentie")
            slope, intercept, r_value, _, _ = linregress(x, y)
            ax.plot(x, slope * x + intercept, color="darkorange", label=f"y={slope:.2f}x+{intercept:.2f}, R2={r_value**2:.3f}")
            ax.set_xlabel("Referentie (Pa)")
            ax.set_ylabel("Diver (Pa)")
            ax.set_title("Diver vs Referentie")
            ax.grid(True)
            ax.legend()
            plt.show(block=False)


def export_data(selected_directory, start, end, show_wave, show_mon, show_ref):
    global calibration_reports_global

    outdir = selected_directory if os.path.isdir(selected_directory) else select_directory("Kies exportmap")
    if not outdir:
        raise ValueError("Geen exportmap gekozen.")

    exported = []
    combined = None

    if show_wave and not df_wave_global.empty:
        wave = df_wave_global[df_wave_global["Datetime"].between(start, end)].copy()
        wave = wave[wave["Pressure"].between(pressure_floor(), PRESSURE_CAP)]
        wave.to_csv(os.path.join(outdir, "hf_data.csv"), index=False)
        exported.append("hf_data.csv")
        combined = wave.sort_values("Datetime") if combined is None else pd.merge_asof(combined.sort_values("Datetime"), wave.sort_values("Datetime"), on="Datetime")

    if show_mon and not df_mon_global.empty:
        mon = df_mon_global[df_mon_global["Datetime"].between(start, end)].copy()
        mon = mon[mon["Pressure"].between(pressure_floor(), PRESSURE_CAP)]
        mon.to_csv(os.path.join(outdir, "diver_data.csv"), index=False)
        exported.append("diver_data.csv")
        combined = mon.sort_values("Datetime") if combined is None else pd.merge_asof(combined.sort_values("Datetime"), mon.sort_values("Datetime"), on="Datetime", direction="nearest")

    if show_ref and not df_reference_global.empty:
        ref = df_reference_global[df_reference_global["Datetime"].between(start, end)].copy()
        ref = ref[ref["Pressure"].between(pressure_floor(), PRESSURE_CAP)]
        ref.to_csv(os.path.join(outdir, "ref_data.csv"), index=False)
        exported.append("ref_data.csv")
        combined = ref.sort_values("Datetime") if combined is None else pd.merge_asof(combined.sort_values("Datetime"), ref.sort_values("Datetime"), on="Datetime", direction="nearest")

    if combined is not None and not combined.empty:
        combined.to_csv(os.path.join(outdir, "combined.csv"), index=False)
        exported.append("combined.csv")

    if calibration_reports_global:
        pd.DataFrame(calibration_reports_global).to_csv(os.path.join(outdir, "calibration_report.csv"), index=False)
        exported.append("calibration_report.csv")

    messagebox.showinfo("Klaar", "Geexporteerd naar:\n" + "\n".join(exported))


def create_time_selection_and_run(selected_directory, show_wave, show_mon, show_ref):
    all_dates = []
    for frame in (df_wave_global, df_mon_global, df_reference_global):
        if not frame.empty:
            all_dates.extend([frame["Datetime"].min(), frame["Datetime"].max()])
    if all_dates:
        start, end = min(all_dates), max(all_dates)
        plot_time_series(start, end, show_wave, show_mon, show_ref)
        plot_regressions(start, end, show_wave, show_mon, show_ref)

    def update_callback(start, end, wave, mon, ref):
        plot_time_series(start, end, wave, mon, ref)
        plot_regressions(start, end, wave, mon, ref)

    def export_callback(start, end, wave, mon, ref):
        export_data(selected_directory, start, end, wave, mon, ref)

    create_time_selection_gui(
        df_wave_global,
        df_mon_global,
        df_reference_global,
        selected_directory,
        update_callback,
        export_callback,
        show_wave=show_wave,
        show_mon=show_mon,
        show_ref=show_ref,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Vergelijk en kalibreer Diver/HF tegen referentiedruk.")
    parser.add_argument("--calibration-mode", choices=sorted(CALIBRATION_MODES.keys()), help="Kalibratiemodus die vooraf wordt gekozen.")
    return parser.parse_args()


def main(calibration_mode=None):
    global df_wave_global, df_mon_global, df_reference_global, calibration_reports_global

    if calibration_mode is None:
        use_wave, use_mon, use_ref, calibration_mode = dataset_selection_gui()
    else:
        use_wave, use_mon, use_ref = True, True, True

    if not any((use_wave, use_mon, use_ref)):
        messagebox.showwarning("Afgebroken", "Geen datasets geselecteerd.")
        return

    selected_directory = ""
    df_wave = pd.DataFrame(columns=["Datetime", "Pressure"])
    df_mon = pd.DataFrame(columns=["Datetime", "Pressure"])
    df_reference = pd.DataFrame(columns=["Datetime", "Pressure"])

    if use_wave:
        selected_directory = select_directory("Selecteer map met HF CSV-bestanden")
        if selected_directory:
            hf_offset = get_offset_input("HF offset (Pa)")
            frames = []
            for name in sorted(os.listdir(selected_directory)):
                if name.lower().endswith(".csv"):
                    frame = process_wave_file(os.path.join(selected_directory, name), hf_offset)
                    if not frame.empty:
                        frames.append(frame)
            if frames:
                df_wave = pd.concat(frames, ignore_index=True)
        else:
            use_wave = False

    if use_mon:
        mon_path = filedialog.askopenfilename(title="Selecteer Diver .mon bestand", filetypes=[("MON", "*.mon")])
        if mon_path:
            mon_offset = get_offset_input("Diver offset (Pa)")
            time_offset = get_offset_input("Diver tijdoffset (s)")
            df_mon = parse_mon_data(read_mon_file(mon_path), mon_offset, time_offset)
        else:
            use_mon = False

    if use_ref:
        ref_path = filedialog.askopenfilename(title="Selecteer referentie CSV", filetypes=[("CSV", "*.csv")])
        if ref_path:
            ref_offset = get_offset_input("Referentie offset (Pa)")
            base_dt = None
            if not df_wave.empty:
                base_dt = df_wave["Datetime"].min()
            elif not df_mon.empty:
                base_dt = df_mon["Datetime"].min()
            df_reference = load_reference_sensor_data(ref_path, base_dt=base_dt, pressure_offset=ref_offset)
        else:
            use_ref = False

    if df_wave.empty and df_mon.empty and df_reference.empty:
        messagebox.showerror("Fout", "Kon geen data inladen.")
        return

    df_wave, df_mon, df_reference, calibration_reports = apply_calibration_mode(df_wave, df_mon, df_reference, calibration_mode)

    df_wave_global = df_wave.copy()
    df_mon_global = df_mon.copy()
    df_reference_global = df_reference.copy()
    calibration_reports_global = calibration_reports

    display_pressure_summary(df_wave_global, df_mon_global, df_reference_global)
    show_calibration_report(calibration_reports_global)
    create_time_selection_and_run(selected_directory, use_wave, use_mon, use_ref)


if __name__ == "__main__":
    args = parse_args()
    main(calibration_mode=args.calibration_mode)
