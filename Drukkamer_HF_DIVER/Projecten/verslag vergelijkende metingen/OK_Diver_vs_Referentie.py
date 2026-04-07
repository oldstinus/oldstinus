
import os
import re
from io import StringIO
from datetime import datetime, timedelta

import chardet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from matplotlib.dates import num2date
from matplotlib.widgets import Button, RectangleSelector, Slider
from scipy.stats import linregress
from tkinter import filedialog, messagebox, simpledialog, ttk

try:
    from tkcalendar import DateEntry
except ImportError:  # pragma: no cover - fallback only if tkcalendar ontbreekt
    DateEntry = None

plt.ion()

# ---------- Helpers ----------

def select_directory():
    root = tk.Tk(); root.withdraw()
    return filedialog.askdirectory(title="Selecteer Directory voor export")

def get_offset_input(label):
    root = tk.Tk(); root.withdraw()
    try:
        val = simpledialog.askfloat(f"Voer {label} offset in", f"Voer de {label} offset in:")
        return 0.0 if val is None else float(val)
    except Exception:
        messagebox.showerror("Ongeldige Invoer", "Voer een geldig nummer in.")
        return 0.0

def detect_file_encoding(path):
    try:
        with open(path, 'rb') as f:
            raw = f.read()
        return chardet.detect(raw)['encoding']
    except Exception:
        return 'utf-8'

# ---------- Datum/tijd parsing uit bestandsnaam en eerste regel ----------

DATETIME_PATTERNS = [
    r'(?P<Y>\d{4})[-_/\.]?(?P<m>\d{2})[-_/\.]?(?P<d>\d{2})[ T_-]?(?P<H>\d{2})[:\-]?(?P<M>\d{2})(?:[:\-]?(?P<S>\d{2}))?',
    r'(?P<d>\d{2})[-_/\.](?P<m>\d{2})[-_/\.](?P<Y>\d{4})[ T_-](?P<H>\d{2})[:\-](?P<M>\d{2})(?:[:\-](?P<S>\d{2}))?',
    r'(?P<d>\d{2})(?P<m>\d{2})(?P<Y>\d{4})[ T_-]?(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})',
]

def detect_delimiter_from_text(text):
    sample = "\n".join(text.splitlines()[:10])
    counts = [(";", sample.count(";")), (",", sample.count(",")), ("\t", sample.count("\t"))]
    counts.sort(key=lambda item: item[1], reverse=True)
    return counts[0][0] if counts and counts[0][1] > 0 else ","

def parse_filename_stamp(date_part, time_part):
    try:
        if len(date_part) == 8:
            year = int(date_part[:4])
            month = int(date_part[4:6])
            day = int(date_part[6:8])
        elif len(date_part) == 6:
            year_short = int(date_part[:2])
            year = 1900 + year_short if year_short >= 70 else 2000 + year_short
            month = int(date_part[2:4])
            day = int(date_part[4:6])
        else:
            return None
        hour = int(time_part[:2])
        minute = int(time_part[2:4])
        second = int(time_part[4:6])
        return datetime(year, month, day, hour, minute, second)
    except Exception:
        return None

def try_parse_dt_dict(parts):
    try:
        return datetime(
            int(parts["Y"]),
            int(parts["m"]),
            int(parts["d"]),
            int(parts["H"]),
            int(parts["M"]),
            int(parts.get("S") or 0),
        )
    except Exception:
        return None

def extract_start_datetime_from_text(text):
    if not isinstance(text, str) or not text:
        return None

    compact_match = re.search(
        r"(?:^|[^0-9])(?:DDR[_-]?)?((?:19|20)\d{6}|\d{6})[_-](\d{6})(?:[^0-9]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if compact_match:
        dt = parse_filename_stamp(compact_match.group(1), compact_match.group(2))
        if dt is not None:
            return dt

    for pattern in DATETIME_PATTERNS:
        match = re.search(pattern, text)
        if match:
            dt = try_parse_dt_dict(match.groupdict())
            if dt is not None:
                return dt

    candidates = re.findall(r'[\d/\-:\._ T]{8,}', text)
    for candidate in candidates:
        cleaned = candidate.strip().replace('_', ' ')
        for dayfirst in (True, False):
            try:
                dt = pd.to_datetime(cleaned, dayfirst=dayfirst, errors='raise')
            except Exception:
                continue
            if isinstance(dt, pd.Timestamp):
                return dt.to_pydatetime()
    return None

def read_text_lines(path, max_lines=None):
    encoding = detect_file_encoding(path)
    with open(path, 'r', encoding=encoding, errors='ignore') as handle:
        lines = handle.read().splitlines()
    return lines[:max_lines] if max_lines else lines

def extract_reference_start_candidates(file_path):
    filename_dt = extract_start_datetime_from_text(os.path.basename(file_path))
    file_dt = None
    for line in read_text_lines(file_path, max_lines=5):
        file_dt = extract_start_datetime_from_text(line)
        if file_dt is not None:
            break
    return filename_dt, file_dt

def format_candidate(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S") if isinstance(dt, datetime) else "niet gedetecteerd"

def parse_manual_datetime(date_value, time_value):
    if not date_value:
        raise ValueError("Kies een datum.")
    time_text = (time_value or "").strip() or "00:00:00"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(f"{date_value} {time_text}", fmt)
        except ValueError:
            continue
    raise ValueError("Gebruik een geldige tijd in formaat HH:MM of HH:MM:SS.")

def ask_datetime_source_dialog(title, option_specs, default_manual=None, default_mode=None):
    result = {"mode": None, "manual": None}
    manual_default = default_manual or datetime.now()
    selected_default = default_mode or option_specs[0]["key"]

    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)

    ttk.Label(root, text=title, font=("", 10, "bold")).grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 4), sticky="w")
    ttk.Label(root, text="Kies hoe de startdatum en starttijd moeten worden bepaald.").grid(
        row=1, column=0, columnspan=2, padx=10, pady=(0, 8), sticky="w"
    )

    mode_var = tk.StringVar(value=selected_default)
    row = 2
    for spec in option_specs:
        label = spec["label"]
        if spec.get("preview"):
            label = f"{label} ({spec['preview']})"
        ttk.Radiobutton(root, text=label, value=spec["key"], variable=mode_var).grid(
            row=row, column=0, columnspan=2, padx=10, pady=2, sticky="w"
        )
        row += 1

    ttk.Separator(root, orient="horizontal").grid(row=row, column=0, columnspan=2, padx=10, pady=(8, 6), sticky="ew")
    row += 1
    ttk.Label(root, text="Handmatige datum").grid(row=row, column=0, padx=10, pady=4, sticky="e")
    if DateEntry is not None:
        date_widget = DateEntry(root, date_pattern="yyyy-mm-dd")
        date_widget.set_date(manual_default.date())
    else:  # pragma: no cover - fallback only if tkcalendar ontbreekt
        date_widget = ttk.Entry(root, width=14)
        date_widget.insert(0, manual_default.strftime("%Y-%m-%d"))
    date_widget.grid(row=row, column=1, padx=10, pady=4, sticky="w")
    row += 1

    ttk.Label(root, text="Handmatige tijd").grid(row=row, column=0, padx=10, pady=4, sticky="e")
    time_var = tk.StringVar(value=manual_default.strftime("%H:%M:%S"))
    time_widget = ttk.Entry(root, width=14, textvariable=time_var)
    time_widget.grid(row=row, column=1, padx=10, pady=4, sticky="w")
    row += 1

    def update_manual_state(*_args):
        state = "normal" if mode_var.get() == "manual" else "disabled"
        for widget in (date_widget, time_widget):
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    def submit():
        selected_mode = mode_var.get()
        try:
            if selected_mode == "manual":
                if DateEntry is not None:
                    date_text = date_widget.get_date().strftime("%Y-%m-%d")
                else:  # pragma: no cover - fallback only if tkcalendar ontbreekt
                    date_text = date_widget.get()
                result["manual"] = parse_manual_datetime(date_text, time_var.get())
            result["mode"] = selected_mode
            root.destroy()
        except ValueError as exc:
            messagebox.showerror("Ongeldige datum/tijd", str(exc), parent=root)

    ttk.Button(root, text="OK", command=submit).grid(row=row, column=0, padx=10, pady=10, sticky="e")
    ttk.Button(root, text="Annuleren", command=root.destroy).grid(row=row, column=1, padx=10, pady=10, sticky="w")

    update_manual_state()
    mode_var.trace_add("write", update_manual_state)
    root.mainloop()
    return result["mode"], result["manual"]

def parse_absolute_datetime_series(series):
    text = series.astype(str).str.strip()
    mask = text.str.contains(r"[-/:T ]", regex=True, na=False)
    best = None
    best_count = 0
    for dayfirst in (True, False):
        parsed = pd.to_datetime(text.where(mask), errors="coerce", dayfirst=dayfirst)
        count = int(parsed.notna().sum())
        if count > best_count:
            best = parsed
            best_count = count
    return best if best_count >= 2 else None

def numeric_series(series):
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(r"[^0-9,\.\-\+]", "", regex=True)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")

def find_reference_columns(df):
    lower_columns = [str(column).strip().lower() for column in df.columns]
    time_index = next(
        (
            index
            for index, name in enumerate(lower_columns)
            if any(token in name for token in ("time", "tijd", "datum", "date"))
        ),
        0,
    )
    pressure_index = next(
        (
            index
            for index, name in enumerate(lower_columns)
            if index != time_index and any(token in name for token in ("pressure", "druk", "data", "waarde", "value"))
        ),
        1 if df.shape[1] > 1 else 0,
    )
    return df.columns[time_index], df.columns[pressure_index]

def select_base_datetime(time_mode, manual_start=None, filename_dt=None, file_dt=None, reference_dt=None):
    if time_mode == "manual":
        return manual_start
    if time_mode == "filename":
        return filename_dt or file_dt or reference_dt
    if time_mode == "file":
        return file_dt or filename_dt or reference_dt
    if time_mode == "reference":
        return reference_dt or file_dt or filename_dt
    return filename_dt or file_dt or reference_dt or manual_start

# ---------- Diver .mon ----------

def read_mon_file(path):
    enc = detect_file_encoding(path)
    lines = []
    try:
        with open(path, 'r', encoding=enc, errors='ignore') as f:
            for i, line in enumerate(f, 1):
                s = line.strip()
                if 'END OF DATA' in s:
                    break
                if i >= 54:
                    lines.append(s)
    except Exception as e:
        print(f"Fout .mon lezen: {e}")
    return lines

def parse_mon_data(lines, pressure_offset=0.0, time_offset=0.0):
    data = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 3 and "/" in parts[0] and ":" in parts[1]:
            date_str = parts[0] + " " + parts[1]
            try:
                try:
                    dt = datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S.%f")
                except ValueError:
                    dt = datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S")
                dt = dt + timedelta(seconds=time_offset)
                p = float(parts[2].replace(',', '.')) + pressure_offset
                data.append({'Datetime': dt, 'Pressure': p})
            except Exception as e:
                print(f"Parse fout: {ln} -> {e}")
    df = pd.DataFrame(data)
    if not df.empty:
        df = df.sort_values('Datetime').reset_index(drop=True)
    return df

# ---------- Referentie ----------

def iter_reference_tables(path, encoding):
    raw_text = "\n".join(read_text_lines(path, max_lines=10))
    delimiter = detect_delimiter_from_text(raw_text)
    seen = set()
    for skiprows in (0, 1, 2):
        for header in ("infer", None):
            key = (skiprows, header)
            if key in seen:
                continue
            seen.add(key)
            kwargs = {
                "sep": delimiter,
                "engine": "python",
                "encoding": encoding,
                "skiprows": skiprows,
                "on_bad_lines": "skip",
            }
            if header is None:
                kwargs["header"] = None
            try:
                df = pd.read_csv(path, **kwargs)
            except Exception:
                continue
            if isinstance(df, pd.DataFrame) and not df.empty and df.shape[1] >= 2:
                yield df

def try_load_ddr_reference(path, encoding, base_dt):
    lines = read_text_lines(path)
    if not lines or not lines[0].strip().upper().startswith("DDR_"):
        return pd.DataFrame(columns=["Datetime", "Pressure"])
    if len(lines) < 3 or base_dt is None:
        return pd.DataFrame(columns=["Datetime", "Pressure"])

    columns = [item.strip() for item in lines[1].split(",")]
    data_text = "\n".join(lines[2:])
    try:
        df_raw = pd.read_csv(StringIO(data_text), sep=",", engine="python", header=None, names=columns)
    except Exception:
        return pd.DataFrame(columns=["Datetime", "Pressure"])

    time_column, pressure_column = find_reference_columns(df_raw)
    sec = numeric_series(df_raw[time_column])
    pressure = numeric_series(df_raw[pressure_column])
    mask = sec.notna() & pressure.notna()
    if not mask.any():
        return pd.DataFrame(columns=["Datetime", "Pressure"])

    timestamps = [base_dt + timedelta(seconds=float(value)) for value in sec[mask]]
    return pd.DataFrame({"Datetime": timestamps, "Pressure": pressure[mask].to_numpy()})

def load_reference_sensor_data(path, time_mode="filename", manual_start=None, pressure_offset=0.0, reference_dt=None):
    encoding = detect_file_encoding(path)
    filename_dt, file_dt = extract_reference_start_candidates(path)
    base_dt = select_base_datetime(
        time_mode,
        manual_start=manual_start,
        filename_dt=filename_dt,
        file_dt=file_dt,
        reference_dt=reference_dt,
    )

    ddr_frame = try_load_ddr_reference(path, encoding, base_dt)
    if not ddr_frame.empty:
        ddr_frame["Pressure"] = pd.to_numeric(ddr_frame["Pressure"], errors="coerce") + pressure_offset
        return ddr_frame.dropna().sort_values("Datetime").reset_index(drop=True)

    for df_raw in iter_reference_tables(path, encoding):
        time_column, pressure_column = find_reference_columns(df_raw)
        pressure = numeric_series(df_raw[pressure_column]) + pressure_offset
        absolute_times = parse_absolute_datetime_series(df_raw[time_column])

        if absolute_times is not None:
            valid_mask = absolute_times.notna() & pressure.notna()
            if valid_mask.any():
                dt_series = absolute_times
                if time_mode != "file":
                    if base_dt is None:
                        raise ValueError("Geen startdatum beschikbaar voor de gekozen tijdbron van het referentiebestand.")
                    first_absolute = dt_series[valid_mask].iloc[0]
                    offsets = (dt_series - first_absolute).dt.total_seconds()
                    dt_series = pd.Series(
                        [
                            base_dt + timedelta(seconds=float(value)) if pd.notna(value) else pd.NaT
                            for value in offsets
                        ],
                        index=dt_series.index,
                    )

                frame = pd.DataFrame({"Datetime": pd.to_datetime(dt_series[valid_mask]), "Pressure": pressure[valid_mask]})
                if not frame.empty:
                    return frame.sort_values("Datetime").reset_index(drop=True)

        seconds = numeric_series(df_raw[time_column])
        valid_mask = seconds.notna() & pressure.notna()
        if valid_mask.any():
            if base_dt is None:
                raise ValueError("Kon geen startdatum bepalen voor relatieve referentietijden. Kies filenaam, bestand of handmatig.")
            timestamps = [base_dt + timedelta(seconds=float(value)) for value in seconds[valid_mask]]
            frame = pd.DataFrame({"Datetime": timestamps, "Pressure": pressure[valid_mask]})
            if not frame.empty:
                return frame.sort_values("Datetime").reset_index(drop=True)

    raise ValueError("Kon geen bruikbare referentiegegevens uit het bestand inlezen.")

# ---------- Samenvatting ----------

def convert_pressure_units(pressure_pa):
    pressure_mmH2O = pressure_pa / 9.80665
    pressure_mmHg = pressure_pa / 133.322
    pressure_bar = pressure_pa / 100000
    return pressure_pa, pressure_mmH2O, pressure_mmHg, pressure_bar

def display_pressure_summary(df_diver, df_reference):
    try:
        div_start = df_diver['Pressure'].iloc[0]
        div_end   = df_diver['Pressure'].iloc[-1]
        div_max   = df_diver['Pressure'].max()

        if not df_reference.empty:
            ref_start = df_reference['Pressure'].iloc[0]
            ref_end   = df_reference['Pressure'].iloc[-1]
            ref_max   = df_reference['Pressure'].max()
        else:
            ref_start = ref_end = ref_max = np.nan

        summary_data = {
            'Druksoort': ['Diver Start', 'Diver Eind', 'Diver Max'],
            'Druk (Pa)': [div_start, div_end, div_max],
            'Druk (mmH₂O)': [convert_pressure_units(div_start)[1],
                              convert_pressure_units(div_end)[1],
                              convert_pressure_units(div_max)[1]],
            'Druk (mmHg)': [convert_pressure_units(div_start)[2],
                            convert_pressure_units(div_end)[2],
                            convert_pressure_units(div_max)[2]],
            'Druk (bar)': [convert_pressure_units(div_start)[3],
                           convert_pressure_units(div_end)[3],
                           convert_pressure_units(div_max)[3]]
        }

        if not df_reference.empty:
            summary_data['Druksoort'].extend(['Referentie Start', 'Referentie Eind', 'Referentie Max'])
            summary_data['Druk (Pa)'].extend([ref_start, ref_end, ref_max])
            summary_data['Druk (mmH₂O)'].extend([convert_pressure_units(ref_start)[1],
                                                 convert_pressure_units(ref_end)[1],
                                                 convert_pressure_units(ref_max)[1]])
            summary_data['Druk (mmHg)'].extend([convert_pressure_units(ref_start)[2],
                                                convert_pressure_units(ref_end)[2],
                                                convert_pressure_units(ref_max)[2]])
            summary_data['Druk (bar)'].extend([convert_pressure_units(ref_start)[3],
                                               convert_pressure_units(ref_end)[3],
                                               convert_pressure_units(ref_max)[3]])

        df_summary = pd.DataFrame(summary_data)

        summary_window = tk.Toplevel()
        summary_window.title("Druk Samenvatting (Diver & Referentie)")

        cols = ("Druksoort", "Druk (Pa)", "Druk (mmH₂O)", "Druk (mmHg)", "Druk (bar)")
        tree = ttk.Treeview(summary_window, columns=cols, show='headings')
        for c in cols:
            tree.heading(c, text=c)
        for _, row in df_summary.iterrows():
            tree.insert("", "end", values=(
                row['Druksoort'],
                f"{row['Druk (Pa)']:.2f}",
                f"{row['Druk (mmH₂O)']:.2f}",
                f"{row['Druk (mmHg)']:.2f}",
                f"{row['Druk (bar)']:.5f}"
            ))
        tree.pack(expand=True, fill='both')
        ttk.Button(summary_window, text="Sluiten", command=summary_window.destroy).pack(pady=10)
    except Exception as e:
        print(f"Fout bij samenvatting: {e}")

# ---------- Tijdselectie GUI ----------

def create_time_selection_gui(df_diver, df_reference, selected_directory,
                              update_plot_callback, export_and_close_callback):
    root = tk.Tk()
    root.title("Tijdselectie & Interactieve filtering (Diver vs Referentie)")

    min_time = min(df_diver['Datetime'].min(), df_reference['Datetime'].min())
    max_time = max(df_diver['Datetime'].max(), df_reference['Datetime'].max())

    ttk.Label(root, text="Begin tijd (YYYY-MM-DD HH:MM:SS):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
    start_time_entry = ttk.Entry(root, width=25); start_time_entry.insert(0, min_time.strftime("%Y-%m-%d %H:%M:%S"))
    start_time_entry.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(root, text="Eind tijd (YYYY-MM-DD HH:MM:SS):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
    end_time_entry = ttk.Entry(root, width=25); end_time_entry.insert(0, max_time.strftime("%Y-%m-%d %H:%M:%S"))
    end_time_entry.grid(row=1, column=1, padx=5, pady=5)

    info = tk.Label(root, text="Tip: in de tijdreeks-grafiek kan je een kader slepen om punten te selecteren.\nGebruik de knoppen onderaan de grafiek om die punten weg te filteren (Diver/Ref/Beide) of te resetten.", fg="blue")
    info.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

    def update_plot():
        try:
            start_time = datetime.strptime(start_time_entry.get(), "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time_entry.get(), "%Y-%m-%d %H:%M:%S")
            if start_time >= end_time:
                messagebox.showerror("Ongeldige Tijd", "Begin tijd moet voor eind tijd zijn."); return
            update_plot_callback(start_time, end_time)
        except ValueError:
            messagebox.showerror("Ongeldige Invoer", "Gebruik formaat: YYYY-MM-DD HH:MM:SS")

    def export_and_close():
        try:
            start_time = datetime.strptime(start_time_entry.get(), "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time_entry.get(), "%Y-%m-%d %H:%M:%S")
            if start_time >= end_time:
                messagebox.showerror("Ongeldige Tijd", "Begin tijd moet voor eind tijd zijn."); return
            export_and_close_callback(start_time, end_time)
            root.destroy()
        except ValueError:
            messagebox.showerror("Ongeldige Invoer", "Gebruik formaat: YYYY-MM-DD HH:MM:SS")

    ttk.Button(root, text="Update Grafiek", command=update_plot).grid(row=3, column=0, padx=5, pady=10, sticky='e')
    ttk.Button(root, text="Export Data en Sluiten", command=export_and_close).grid(row=3, column=1, padx=5, pady=10, sticky='w')
    root.mainloop()

# ---------- Interactieve selectie & plotten ----------

class InteractiveTimeSeries:
    def __init__(self, df_diver, df_reference, start_time=None, end_time=None, pressure_cap=2100):
        self.df_diver_original = df_diver.copy()
        self.df_ref_original   = df_reference.copy()
        self.df_diver = df_diver.copy()
        self.df_ref   = df_reference.copy()
        self.start_time = start_time
        self.end_time   = end_time
        self.pressure_cap = pressure_cap

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.2)  # ruimte voor knoppen

        # RectangleSelector - nieuw API (geen drawtype), x-data kunnen floats (matplotlib datenums) zijn
        self.selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            button=[1],  # linker muis
            minspanx=0, minspany=0,
            spancoords='data'
        )
        self.selection = None  # (xmin_dt, xmax_dt, ymin, ymax) met xmin/xmax als pandas Timestamps

        # Knoppen
        ax_div = plt.axes([0.10, 0.05, 0.15, 0.06])
        ax_ref = plt.axes([0.28, 0.05, 0.15, 0.06])
        ax_both= plt.axes([0.46, 0.05, 0.18, 0.06])
        ax_rst = plt.axes([0.67, 0.05, 0.15, 0.06])

        self.btn_filter_div = Button(ax_div,  'Filter Diver')
        self.btn_filter_ref = Button(ax_ref,  'Filter Ref')
        self.btn_filter_both= Button(ax_both, 'Filter Beide')
        self.btn_reset      = Button(ax_rst,  'Reset')

        self.btn_filter_div.on_clicked(lambda evt: self.apply_filter(target='diver'))
        self.btn_filter_ref.on_clicked(lambda evt: self.apply_filter(target='ref'))
        self.btn_filter_both.on_clicked(lambda evt: self.apply_filter(target='both'))
        self.btn_reset.on_clicked(self.reset_data)

        self.update_plot()

    def _to_timestamp(self, x):
        """Converteer x (float datenumber of datetime) naar pandas.Timestamp."""
        if x is None:
            return None
        if isinstance(x, (float, np.floating)):
            try:
                return pd.to_datetime(num2date(x)).tz_localize(None)
            except Exception:
                # soms geeft num2date datetime met tzinfo; verwijder tz
                return pd.to_datetime(num2date(x)).tz_localize(None)
        if isinstance(x, datetime):
            return pd.to_datetime(x)
        # al pandas Timestamp?
        try:
            return pd.to_datetime(x)
        except Exception:
            return None

    def on_select(self, eclick, erelease):
        xmin = self._to_timestamp(eclick.xdata)
        xmax = self._to_timestamp(erelease.xdata)
        if xmin is None or xmax is None:
            self.selection = None
            return
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        ymin, ymax = sorted([float(eclick.ydata), float(erelease.ydata)])
        self.selection = (xmin, xmax, ymin, ymax)

    def current_window(self, df):
        if self.start_time and self.end_time:
            m = (df['Datetime'] >= self.start_time) & (df['Datetime'] <= self.end_time)
            return df[m]
        return df

    def update_plot(self):
        self.ax.clear()
        df_div = self.current_window(self.df_diver)
        df_ref = self.current_window(self.df_ref)
        if self.pressure_cap is not None:
            df_div = df_div[df_div['Pressure'] <= self.pressure_cap]
            df_ref = df_ref[df_ref['Pressure'] <= self.pressure_cap]

        self.ax.plot(df_div['Datetime'], df_div['Pressure'], label='Diver druk', marker='x', markersize=2, linewidth=0.5)
        self.ax.plot(df_ref['Datetime'], df_ref['Pressure'], label='Referentie sensor', marker='s', markersize=4, linestyle='--')
        self.ax.set_title('Diver druk vs Referentie sensor over Tijd (kader slepen om te selecteren)')
        self.ax.set_xlabel('Tijd'); self.ax.set_ylabel('Druk (Pa)'); self.ax.legend(); self.ax.grid(True)
        plt.gcf().autofmt_xdate()
        self.fig.canvas.draw_idle()

    def apply_filter(self, target='both'):
        if not self.selection:
            messagebox.showwarning("Geen selectie", "Sleep eerst een kader over de punten die je wil verwijderen.")
            return
        xmin, xmax, ymin, ymax = self.selection

        def filter_df(df):
            m_time = (df['Datetime'] >= xmin) & (df['Datetime'] <= xmax)
            m_val  = (df['Pressure'] >= ymin) & (df['Pressure'] <= ymax)
            return df[~(m_time & m_val)].reset_index(drop=True)

        if target in ('diver', 'both'):
            self.df_diver = filter_df(self.df_diver)
        if target in ('ref', 'both'):
            self.df_ref = filter_df(self.df_ref)

        self.update_plot()

    def reset_data(self, event=None):
        self.df_diver = self.df_diver_original.copy()
        self.df_ref   = self.df_ref_original.copy()
        self.selection = None
        self.update_plot()

# XY-regressie (gereflecteerd na filtering)
def plot_xy_regression_with_slider(df_diver, df_reference, start_time=None, end_time=None, pressure_cap=2100):
    try:
        fig, ax = plt.subplots(figsize=(8, 6)); plt.subplots_adjust(bottom=0.25)

        def filtered_window(df):
            if start_time and end_time:
                m = (df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)
                df = df[m]
            if pressure_cap is not None:
                df = df[df['Pressure'] <= pressure_cap]
            return df

        df_div_f = filtered_window(df_diver)
        df_ref_f = filtered_window(df_reference)

        merged_df = pd.merge_asof(df_ref_f.sort_values('Datetime'),
                                  df_div_f.sort_values('Datetime'),
                                  on='Datetime', suffixes=('_ref', '_div'))
        merged_df = merged_df.dropna(subset=['Pressure_ref','Pressure_div'])
        if pressure_cap is not None:
            merged_df = merged_df[(merged_df['Pressure_ref'] <= pressure_cap) & (merged_df['Pressure_div'] <= pressure_cap)]

        x, y = merged_df['Pressure_ref'], merged_df['Pressure_div']
        ax.scatter(x, y, label='Data')

        if len(x) > 1:
            slope, intercept, r, _, _ = linregress(x, y)
            ax.plot(x, slope*x + intercept, label=f'Lineaire Fit: y={slope:.2f}x+{intercept:.2f}\n$R^2$={r**2:.4f}')

        ax.set_xlabel('Referentie druk (Pa)'); ax.set_ylabel('Diver druk (Pa)')
        ax.set_title('X-Y Plot: Referentie (x) vs Diver (y)'); ax.legend(); ax.grid(True)

        ax_shift = plt.axes([0.25, 0.1, 0.65, 0.03])
        slider_shift = Slider(ax_shift, 'Tijd Shift Diver (sec)', -60, 60, valinit=0, valstep=0.5)

        def update_regression(val):
            try:
                shift = slider_shift.val
                shifted = df_diver.copy(); shifted['Datetime'] = shifted['Datetime'] + timedelta(seconds=shift)
                df_div_s = filtered_window(shifted)
                df_ref_s = filtered_window(df_reference)

                merged = pd.merge_asof(df_ref_s.sort_values('Datetime'),
                                       df_div_s.sort_values('Datetime'),
                                       on='Datetime', suffixes=('_ref','_div')).dropna(subset=['Pressure_ref','Pressure_div'])
                if pressure_cap is not None:
                    merged = merged[(merged['Pressure_ref'] <= pressure_cap) & (merged['Pressure_div'] <= pressure_cap)]
                x_new, y_new = merged['Pressure_ref'], merged['Pressure_div']

                ax.clear()
                if len(x_new) > 1:
                    slope, intercept, r, _, _ = linregress(x_new, y_new)
                    ax.scatter(x_new, y_new, label='Data')
                    ax.plot(x_new, slope*x_new + intercept, label=f'Lineaire Fit: y={slope:.2f}x+{intercept:.2f}\n$R^2$={r**2:.4f}')
                else:
                    ax.scatter(x_new, y_new, label='Data')

                ax.set_xlabel('Referentie druk (Pa)'); ax.set_ylabel('Diver druk (Pa)')
                ax.set_title('X-Y Plot: Referentie (x) vs Diver (y)'); ax.legend(); ax.grid(True)
                fig.canvas.draw_idle()
            except Exception as e:
                print(f"Fout in update_regression: {e}")

        slider_shift.on_changed(update_regression)
        plt.show(block=False); plt.pause(0.001)
    except Exception as e:
        print(f"Fout in plot_xy_regression_with_slider: {e}")

# ---------- Export ----------

def export_data(df_diver, df_reference, outdir, start_time, end_time, pressure_cap=2100):
    try:
        def window(df):
            m = (df['Datetime'] >= start_time) & (df['Datetime'] <= end_time)
            df = df[m]
            if pressure_cap is not None:
                df = df[df['Pressure'] <= pressure_cap]
            return df

        df_div_f = window(df_diver)
        df_ref_f = window(df_reference)

        merged_df = pd.merge_asof(df_ref_f.sort_values('Datetime'),
                                  df_div_f.sort_values('Datetime'),
                                  on='Datetime', suffixes=('_ref', '_div'))

        f1 = os.path.join(outdir, 'diver_druk_data_filtered.csv')
        f2 = os.path.join(outdir, 'reference_druk_data_filtered.csv')
        f3 = os.path.join(outdir, 'combined_ref_diver_filtered.csv')
        df_div_f.to_csv(f1, index=False); df_ref_f.to_csv(f2, index=False); merged_df.to_csv(f3, index=False)

        messagebox.showinfo("Export Succesvol", f"Data geëxporteerd naar:\n{f1}\n{f2}\n{f3}")
    except Exception as e:
        messagebox.showerror("Export Fout", f"Er is een fout opgetreden bij het exporteren van de data:\n{e}")

# ---------- Main ----------

def main():
    outdir = select_directory()
    if not outdir:
        messagebox.showwarning("Geen Directory Geselecteerd", "Er is geen directory geselecteerd."); return

    mon_file = filedialog.askopenfilename(title="Selecteer een Diver .mon bestand", filetypes=[("MON bestanden", "*.mon")])
    if not mon_file:
        messagebox.showwarning("Geen Bestand Geselecteerd", "Er is geen .mon bestand geselecteerd."); return
    print(f".mon: {mon_file}")

    mon_pressure_offset = get_offset_input('Diver druk (Pa, + = optellen)')
    mon_time_offset = get_offset_input('Diver tijd (in seconden, + = later)')

    lines = read_mon_file(mon_file)
    if not lines:
        messagebox.showerror("Geen Data", "Geen data in .mon bestand."); return
    df_diver = parse_mon_data(lines, mon_pressure_offset, mon_time_offset)
    if df_diver.empty:
        messagebox.showerror("Geen Data", "Geen geldige .mon data."); return

    ref_file = filedialog.askopenfilename(
        title="Selecteer referentie CSV/TXT",
        filetypes=[("CSV", "*.csv"), ("TXT", "*.txt"), ("Alle bestanden", "*.*")]
    )
    if not ref_file:
        messagebox.showwarning("Geen Bestand Geselecteerd", "Er is geen referentie-bestand geselecteerd."); return
    print(f"Referentie: {ref_file}")

    filename_dt, file_dt = extract_reference_start_candidates(ref_file)
    ref_mode, ref_manual_dt = ask_datetime_source_dialog(
        "Starttijd referentiedruksensor",
        option_specs=[
            {"key": "filename", "label": "Uit filenaam / eerste regel", "preview": format_candidate(filename_dt or file_dt)},
            {"key": "file", "label": "Uit referentiebestand (absolute tijd of header)", "preview": format_candidate(file_dt)},
            {"key": "manual", "label": "Handmatig via kalender + tijd"},
        ],
        default_manual=file_dt or filename_dt or df_diver["Datetime"].min(),
        default_mode="filename" if (filename_dt or file_dt) else "manual",
    )
    if ref_mode is None:
        messagebox.showwarning("Afgebroken", "Geen starttijdmethode gekozen voor het referentiebestand."); return

    try:
        df_reference = load_reference_sensor_data(
            ref_file,
            time_mode=ref_mode,
            manual_start=ref_manual_dt,
        )
    except Exception as e:
        messagebox.showerror("Referentie Leesfout", str(e)); return
    if df_reference.empty:
        messagebox.showerror("Geen Data", "Geen geldige referentie-data."); return

    display_pressure_summary(df_diver, df_reference)

    # 1) Interactieve tijdreeks met kader-selectie en filterknoppen
    its = InteractiveTimeSeries(df_diver, df_reference)

    # 2) XY-regressie die de (eventueel gefilterde) data gebruikt
    def update_plot_with_time_range(start_time, end_time):
        its.start_time = start_time
        its.end_time   = end_time
        its.update_plot()
        plot_xy_regression_with_slider(its.df_diver, its.df_ref, start_time, end_time)

    def export_and_close(start_time, end_time):
        plot_xy_regression_with_slider(its.df_diver, its.df_ref, start_time, end_time)
        export_data(its.df_diver, its.df_ref, outdir, start_time, end_time)

    create_time_selection_gui(its.df_diver, its.df_ref, outdir, update_plot_with_time_range, export_and_close)

if __name__ == "__main__":
    main()
