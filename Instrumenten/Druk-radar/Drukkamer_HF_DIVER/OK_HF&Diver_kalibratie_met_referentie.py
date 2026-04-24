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

# Schakel interactieve modus in
plt.ion()

# Functie om een directory te selecteren
def select_directory():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(title="Selecteer Directory")
    return folder_selected

# Functie om een numerieke offset in te voeren
def get_offset_input(label):
    root = tk.Tk()
    root.withdraw()
    try:
        offset_value = simpledialog.askfloat(f"Voer {label} offset in", f"Voer de {label} offset in:")
        if offset_value is not None:
            return offset_value
        else:
            return 0.0
    except ValueError:
        messagebox.showerror("Ongeldige Invoer", "Voer een geldig nummer in.")
        return 0.0

# Functies om HF CSV-bestanden te verwerken
def extract_wave_file_info(file_path):
    encoding = detect_file_encoding(file_path)
    sampling_frequency = 8.0
    start_datetime = None
    try:
        with open(file_path, 'r', encoding=encoding, errors='ignore') as handle:
            lines = handle.readlines()
        if len(lines) > 4:
            try:
                sampling_frequency = float(lines[4].strip().split(",")[1].strip())
            except Exception:
                sampling_frequency = 8.0
        if len(lines) > 10:
            try:
                start_date = lines[9].strip().split(",")[2].strip()
                start_time = lines[10].strip().split(",")[2].strip()
                start_datetime = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M:%S")
            except Exception:
                start_datetime = None
    except Exception as e:
        print(f"Fout bij het uitlezen van HF-header voor {file_path}: {e}")
    return sampling_frequency, start_datetime

def process_wave_file(file_path, pressure_offset=0.0):
    print(f"Bestand inlezen: {file_path}")
    try:
        encoding = detect_file_encoding(file_path)
        sampling_frequency, header_start = extract_wave_file_info(file_path)
        data_numeric = pd.read_csv(file_path, encoding=encoding, skiprows=11, header=None)
        filtered_data = data_numeric[data_numeric[1] == 'C1']
        measurements = pd.to_numeric(filtered_data[2], errors='coerce').dropna().reset_index(drop=True)
        measurements = (measurements * 1000) + pressure_offset
        print(f"Meetwaarden gevonden: {len(measurements)}")
        return measurements, sampling_frequency, header_start
    except Exception as e:
        print(f"Fout bij het verwerken van {file_path}: {e}")
        return None, None, None

# Functie om de bestandsencoding te detecteren
def detect_file_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            rawdata = f.read()
            result = chardet.detect(rawdata)
            return result['encoding']
    except Exception as e:
        print(f"Fout bij het detecteren van encoding voor {file_path}: {e}")
        return 'utf-8'  # Standaard encoding

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
            year = int(date_part[:4]); month = int(date_part[4:6]); day = int(date_part[6:8])
        elif len(date_part) == 6:
            year_short = int(date_part[:2])
            year = 1900 + year_short if year_short >= 70 else 2000 + year_short
            month = int(date_part[2:4]); day = int(date_part[4:6])
        else:
            return None
        return datetime(year, month, day, int(time_part[:2]), int(time_part[2:4]), int(time_part[4:6]))
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

def extract_reference_start_candidates(path):
    filename_dt = extract_start_datetime_from_text(os.path.basename(path))
    file_dt = None
    for line in read_text_lines(path, max_lines=5):
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
                else:  # pragma: no cover
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

def build_wave_dataset(file_paths, pressure_offset=0.0, time_mode="filename", manual_start=None, reference_start=None):
    frames = []
    sequential_start = manual_start if time_mode == "manual" else reference_start if time_mode == "reference" else None

    for file_path in sorted(file_paths):
        measurements, sampling_frequency, header_start = process_wave_file(file_path, pressure_offset)
        if measurements is None or measurements.empty:
            continue

        filename_start = extract_start_datetime_from_text(os.path.basename(file_path))
        dt_step = 1.0 / (sampling_frequency or 8.0)

        if time_mode == "manual":
            start_datetime = sequential_start
        elif time_mode == "reference":
            start_datetime = sequential_start or reference_start
        else:
            start_datetime = filename_start or header_start or sequential_start

        if start_datetime is None:
            raise ValueError(
                f"Kon geen starttijd bepalen voor HF-bestand {os.path.basename(file_path)}. "
                "Kies handmatig of gebruik de referentiestart."
            )

        timestamps = [start_datetime + timedelta(seconds=index * dt_step) for index in range(len(measurements))]
        frames.append(pd.DataFrame({"Datetime": timestamps, "Pressure": measurements.to_numpy()}))
        sequential_start = timestamps[-1] + timedelta(seconds=dt_step)

    if not frames:
        return pd.DataFrame(columns=["Datetime", "Pressure"])

    return pd.concat(frames, ignore_index=True).sort_values("Datetime").reset_index(drop=True)

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

# Functie om een .mon-bestand te lezen
def read_mon_file(file_path):
    encoding = detect_file_encoding(file_path)
    data_lines = []
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if 'END OF DATA' in line:
                    print("Einde van gegevensbestand bereikt.")
                    break
                elif line_num >= 54:
                    data_lines.append(line)
    except Exception as e:
        print(f"Fout bij het lezen van het .mon bestand: {e}")
    return data_lines

# Functie om .mon-gegevens te parseren
def parse_mon_data(data_lines, pressure_offset=0, time_offset=0):
    data = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 3 and "/" in parts[0] and ":" in parts[1]:
            try:
                date_str = parts[0] + " " + parts[1]
                try:
                    datetime_obj = datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S.%f")
                except ValueError:
                    datetime_obj = datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S")
                datetime_obj = datetime_obj + timedelta(seconds=time_offset)
                pressure = float(parts[2].replace(',', '.')) + pressure_offset
                data.append({'Datetime': datetime_obj, 'Pressure': pressure})
            except ValueError as ve:
                print(f"Fout bij het parsen van regel: {line}. Error: {ve}")
    df = pd.DataFrame(data)
    return df.sort_values('Datetime').reset_index(drop=True) if not df.empty else df

# Functie om druk om te rekenen naar verschillende eenheden
def convert_pressure_units(pressure_pa):
    pressure_mmH2O = pressure_pa / 9.80665  # Omrekening van Pa naar mmH2O
    pressure_mmHg = pressure_pa / 133.322   # Omrekening van Pa naar mmHg
    pressure_bar = pressure_pa / 100000     # Omrekening van Pa naar bar
    return pressure_pa, pressure_mmH2O, pressure_mmHg, pressure_bar

# Functie om de maximum-, begin- en einddrukken weer te geven in verschillende eenheden
def display_pressure_summary(df_wave, df_mon, df_reference):
    try:
        # Verkrijg de begin-, eind- en maximumdruk in Pa voor HF druksonde
        wave_start_pressure = df_wave['Pressure'].iloc[0]
        wave_end_pressure = df_wave['Pressure'].iloc[-1]
        wave_max_pressure = df_wave['Pressure'].max()

        # Verkrijg de begin-, eind- en maximumdruk in Pa voor Diver druk
        mon_start_pressure = df_mon['Pressure'].iloc[0]
        mon_end_pressure = df_mon['Pressure'].iloc[-1]
        mon_max_pressure = df_mon['Pressure'].max()

        # Verkrijg de begin-, eind- en maximumdruk in Pa voor Referentie sensor
        if not df_reference.empty:
            ref_start_pressure = df_reference['Pressure'].iloc[0]
            ref_end_pressure = df_reference['Pressure'].iloc[-1]
            ref_max_pressure = df_reference['Pressure'].max()
        else:
            ref_start_pressure = ref_end_pressure = ref_max_pressure = np.nan

        # Zet drukken om naar verschillende eenheden
        summary_data = {
            'Druksoort': [
                'HF Druksonde Start', 'HF Druksonde Eind', 'HF Druksonde Max',
                'Diver Druk Start', 'Diver Druk Eind', 'Diver Druk Max'
            ],
            'Druk (Pa)': [
                wave_start_pressure, wave_end_pressure, wave_max_pressure,
                mon_start_pressure, mon_end_pressure, mon_max_pressure
            ],
            'Druk (mmH₂O)': [
                convert_pressure_units(wave_start_pressure)[1], 
                convert_pressure_units(wave_end_pressure)[1],
                convert_pressure_units(wave_max_pressure)[1], 
                convert_pressure_units(mon_start_pressure)[1],
                convert_pressure_units(mon_end_pressure)[1], 
                convert_pressure_units(mon_max_pressure)[1]
            ],
            'Druk (mmHg)': [
                convert_pressure_units(wave_start_pressure)[2], 
                convert_pressure_units(wave_end_pressure)[2],
                convert_pressure_units(wave_max_pressure)[2], 
                convert_pressure_units(mon_start_pressure)[2],
                convert_pressure_units(mon_end_pressure)[2], 
                convert_pressure_units(mon_max_pressure)[2]
            ],
            'Druk (bar)': [
                convert_pressure_units(wave_start_pressure)[3], 
                convert_pressure_units(wave_end_pressure)[3],
                convert_pressure_units(wave_max_pressure)[3], 
                convert_pressure_units(mon_start_pressure)[3],
                convert_pressure_units(mon_end_pressure)[3], 
                convert_pressure_units(mon_max_pressure)[3]
            ]
        }

        if not df_reference.empty:
            # Voeg referentie sensor gegevens toe aan de samenvatting
            summary_data['Druksoort'].extend([
                'Referentie Sensor Start', 'Referentie Sensor Eind', 'Referentie Sensor Max'
            ])
            summary_data['Druk (Pa)'].extend([
                ref_start_pressure, ref_end_pressure, ref_max_pressure
            ])
            summary_data['Druk (mmH₂O)'].extend([
                convert_pressure_units(ref_start_pressure)[1], 
                convert_pressure_units(ref_end_pressure)[1],
                convert_pressure_units(ref_max_pressure)[1]
            ])
            summary_data['Druk (mmHg)'].extend([
                convert_pressure_units(ref_start_pressure)[2], 
                convert_pressure_units(ref_end_pressure)[2],
                convert_pressure_units(ref_max_pressure)[2]
            ])
            summary_data['Druk (bar)'].extend([
                convert_pressure_units(ref_start_pressure)[3], 
                convert_pressure_units(ref_end_pressure)[3],
                convert_pressure_units(ref_max_pressure)[3]
            ])

        # Maak de samenvatting als DataFrame
        df_summary = pd.DataFrame(summary_data)

        # Toon de samenvatting in een nieuw Tkinter-venster
        summary_window = tk.Toplevel()
        summary_window.title("Druk Samenvatting")

        tree = ttk.Treeview(summary_window, columns=("Druksoort", "Druk (Pa)", "Druk (mmH₂O)", "Druk (mmHg)", "Druk (bar)"), show='headings')
        tree.heading("Druksoort", text="Druksoort")
        tree.heading("Druk (Pa)", text="Druk (Pa)")
        tree.heading("Druk (mmH₂O)", text="Druk (mmH₂O)")
        tree.heading("Druk (mmHg)", text="Druk (mmHg)")
        tree.heading("Druk (bar)", text="Druk (bar)")

        for index, row in df_summary.iterrows():
            tree.insert("", "end", values=(
                row['Druksoort'],
                f"{row['Druk (Pa)']:.2f}",
                f"{row['Druk (mmH₂O)']:.2f}",
                f"{row['Druk (mmHg)']:.2f}",
                f"{row['Druk (bar)']:.5f}"
            ))

        tree.pack(expand=True, fill='both')

        # Voeg een sluitknop toe
        close_button = ttk.Button(summary_window, text="Sluiten", command=summary_window.destroy)
        close_button.pack(pady=10)
    except Exception as e:
        print(f"Fout bij het tonen van druk samenvatting: {e}")

def normalize_sensor_frame(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=["Datetime", "Pressure"])
    if "Datetime" not in df.columns or "Pressure" not in df.columns:
        return pd.DataFrame(columns=["Datetime", "Pressure"])

    normalized = df[["Datetime", "Pressure"]].copy()
    normalized["Datetime"] = pd.to_datetime(normalized["Datetime"], errors="coerce")
    normalized["Pressure"] = pd.to_numeric(normalized["Pressure"], errors="coerce")
    normalized = normalized.dropna(subset=["Datetime"]).sort_values("Datetime").reset_index(drop=True)
    return normalized

def build_sensor_frames(df_wave, df_mon, df_reference):
    sensor_frames = {}
    if not df_wave.empty:
        sensor_frames["HF druksonde"] = normalize_sensor_frame(df_wave)
    if not df_mon.empty:
        sensor_frames["Diver"] = normalize_sensor_frame(df_mon)
    if not df_reference.empty:
        sensor_frames["Referentie"] = normalize_sensor_frame(df_reference)
    return sensor_frames

def clone_sensor_frames(sensor_frames):
    return {name: normalize_sensor_frame(frame) for name, frame in sensor_frames.items()}

def filter_sensor_frame(df, start_time=None, end_time=None, selection=None, pressure_cap=2100):
    filtered = normalize_sensor_frame(df)
    if start_time is not None and end_time is not None:
        filtered = filtered[(filtered["Datetime"] >= start_time) & (filtered["Datetime"] <= end_time)]
    if pressure_cap is not None:
        filtered = filtered[filtered["Pressure"].isna() | (filtered["Pressure"] <= pressure_cap)]
    if selection is not None:
        xmin, xmax, ymin, ymax = selection
        filtered = filtered[
            (filtered["Datetime"] >= xmin)
            & (filtered["Datetime"] <= xmax)
            & (filtered["Pressure"] >= ymin)
            & (filtered["Pressure"] <= ymax)
        ]
    return filtered.sort_values("Datetime").reset_index(drop=True)

def merge_sensor_frames(df_x, df_y, tolerance_seconds=2.0):
    if df_x.empty or df_y.empty:
        return pd.DataFrame(columns=["Datetime", "Pressure_x", "Pressure_y"])
    merged = pd.merge_asof(
        df_x.sort_values("Datetime"),
        df_y.sort_values("Datetime"),
        on="Datetime",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
        suffixes=("_x", "_y"),
    )
    return merged.dropna(subset=["Pressure_x", "Pressure_y"]).reset_index(drop=True)

def calculate_relationship(sensor_frames, x_sensor, y_sensor, start_time=None, end_time=None, selection=None, shift_seconds=0.0, pressure_cap=2100, tolerance_seconds=2.0):
    df_x = filter_sensor_frame(sensor_frames.get(x_sensor, pd.DataFrame(columns=["Datetime", "Pressure"])), start_time, end_time, selection, pressure_cap)
    df_y = filter_sensor_frame(sensor_frames.get(y_sensor, pd.DataFrame(columns=["Datetime", "Pressure"])), start_time, end_time, selection, pressure_cap)
    if shift_seconds:
        df_y = df_y.copy()
        df_y["Datetime"] = df_y["Datetime"] + timedelta(seconds=shift_seconds)

    merged = merge_sensor_frames(df_x, df_y, tolerance_seconds=tolerance_seconds)
    if merged.empty:
        return {
            "merged": merged,
            "x": pd.Series(dtype=float),
            "y": pd.Series(dtype=float),
            "slope": np.nan,
            "intercept": np.nan,
            "r_squared": np.nan,
            "count": 0,
        }

    x = pd.to_numeric(merged["Pressure_x"], errors="coerce")
    y = pd.to_numeric(merged["Pressure_y"], errors="coerce")
    valid_mask = (~np.isnan(x)) & (~np.isnan(y))
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    result = {
        "merged": merged.loc[valid_mask].reset_index(drop=True),
        "x": x_valid,
        "y": y_valid,
        "slope": np.nan,
        "intercept": np.nan,
        "r_squared": np.nan,
        "count": int(len(x_valid)),
    }
    if len(x_valid) >= 2:
        slope, intercept, r_value, _, _ = linregress(x_valid, y_valid)
        result["slope"] = slope
        result["intercept"] = intercept
        result["r_squared"] = r_value ** 2
    return result

def selection_to_text(selection):
    if selection is None:
        return "volledig huidig venster"
    xmin, xmax, ymin, ymax = selection
    return (
        f"{xmin.strftime('%Y-%m-%d %H:%M:%S')} tot {xmax.strftime('%Y-%m-%d %H:%M:%S')}, "
        f"druk {ymin:.1f} tot {ymax:.1f} Pa"
    )

def mark_selection_as_nan(df, selection):
    if selection is None or df.empty:
        return df.copy(), 0

    xmin, xmax, ymin, ymax = selection
    updated = df.copy()
    pressure = pd.to_numeric(updated["Pressure"], errors="coerce")
    mask = (
        (updated["Datetime"] >= xmin)
        & (updated["Datetime"] <= xmax)
        & pressure.notna()
        & (pressure >= ymin)
        & (pressure <= ymax)
    )
    changed = int(mask.sum())
    if changed > 0:
        updated.loc[mask, "Pressure"] = np.nan
    return updated, changed

def sensor_export_filename(sensor_name):
    mapping = {
        "HF druksonde": "hf_druksonde_data_filtered.csv",
        "Diver": "diver_druk_data_filtered.csv",
        "Referentie": "referentie_druk_data_filtered.csv",
    }
    if sensor_name in mapping:
        return mapping[sensor_name]
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", sensor_name).strip("_").lower() or "sensor"
    return f"{safe_name}_data_filtered.csv"

# Functie om de tijdselectie GUI te creëren
def create_time_selection_gui(df_wave, df_mon, df_reference, selected_directory, update_plot_callback, export_and_close_callback, available_sensors):
    root = tk.Tk()
    root.title("Tijdselectie voor Grafiek")

    # Bepaal de minimale en maximale tijd inclusief referentiesensor
    min_time = min(df_wave['Datetime'].min(), df_mon['Datetime'].min())
    if not df_reference.empty:
        min_time = min(min_time, df_reference['Datetime'].min())
    max_time = max(df_wave['Datetime'].max(), df_mon['Datetime'].max())
    if not df_reference.empty:
        max_time = max(max_time, df_reference['Datetime'].max())

    # Labels en invoervelden voor begin- en eindtijd
    ttk.Label(root, text="Begin tijd (YYYY-MM-DD HH:MM:SS):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
    start_time_entry = ttk.Entry(root, width=25)
    start_time_entry.insert(0, min_time.strftime("%Y-%m-%d %H:%M:%S"))
    start_time_entry.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(root, text="Eind tijd (YYYY-MM-DD HH:MM:SS):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
    end_time_entry = ttk.Entry(root, width=25)
    end_time_entry.insert(0, max_time.strftime("%Y-%m-%d %H:%M:%S"))
    end_time_entry.grid(row=1, column=1, padx=5, pady=5)

    default_x = "HF druksonde" if "HF druksonde" in available_sensors else available_sensors[0]
    default_y_candidates = [name for name in ("Diver", "Referentie") if name in available_sensors and name != default_x]
    default_y = default_y_candidates[0] if default_y_candidates else next((name for name in available_sensors if name != default_x), default_x)

    ttk.Label(root, text="X-sensor voor XY:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
    x_sensor_var = tk.StringVar(value=default_x)
    x_sensor_combo = ttk.Combobox(root, textvariable=x_sensor_var, values=available_sensors, state="readonly", width=22)
    x_sensor_combo.grid(row=2, column=1, padx=5, pady=5)

    ttk.Label(root, text="Y-sensor voor XY:").grid(row=3, column=0, padx=5, pady=5, sticky='e')
    y_sensor_var = tk.StringVar(value=default_y)
    y_sensor_combo = ttk.Combobox(root, textvariable=y_sensor_var, values=available_sensors, state="readonly", width=22)
    y_sensor_combo.grid(row=3, column=1, padx=5, pady=5)

    ttk.Label(
        root,
        text="Gebruik in de tijdgrafiek een kader. Klik op 'Gebruik gebied' om alleen dat gebied te vergelijken, op 'Update data' na een optie-wijziging en gebruik de NaN-knoppen om pieken te negeren.",
        foreground="blue",
        wraplength=420,
    ).grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky='w')

    def parse_selection():
        start_time = datetime.strptime(start_time_entry.get(), "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(end_time_entry.get(), "%Y-%m-%d %H:%M:%S")
        x_sensor = x_sensor_var.get()
        y_sensor = y_sensor_var.get()
        if start_time >= end_time:
            raise ValueError("Begin tijd moet voor eind tijd zijn.")
        if x_sensor == y_sensor:
            raise ValueError("Kies twee verschillende druksensoren voor de XY-grafiek.")
        return start_time, end_time, x_sensor, y_sensor

    # Functie om de plot bij te werken bij het klikken op de knop
    def update_plot():
        try:
            update_plot_callback(*parse_selection())
        except ValueError as exc:
            messagebox.showerror("Ongeldige Invoer", str(exc))

    # Functie om te exporteren en sluiten
    def export_and_close():
        try:
            export_and_close_callback(*parse_selection())
            root.destroy()  # Sluit de GUI na export
        except ValueError as exc:
            messagebox.showerror("Ongeldige Invoer", str(exc))

    # Knop om de grafiek te updaten
    update_button = ttk.Button(root, text="Update data", command=update_plot)
    update_button.grid(row=5, column=0, padx=5, pady=10, sticky='e')

    # Knop om de data te exporteren en de GUI te sluiten
    export_button = ttk.Button(root, text="Export Data en Sluiten", command=export_and_close)
    export_button.grid(row=5, column=1, padx=5, pady=10, sticky='w')

    for widget in (start_time_entry, end_time_entry, x_sensor_combo, y_sensor_combo):
        widget.bind("<Return>", lambda _event: update_plot())

    root.mainloop()

# Functie om de gecombineerde grafiek te plotten
def _legacy_plot_combined_graph(df_wave, df_mon, df_reference, start_time=None, end_time=None):
    try:
        print("Start plot_combined_graph")
        fig, ax = plt.subplots(figsize=(12, 6))

        if start_time and end_time:
            mask_wave = (df_wave['Datetime'] >= start_time) & (df_wave['Datetime'] <= end_time)
            mask_mon = (df_mon['Datetime'] >= start_time) & (df_mon['Datetime'] <= end_time)
            mask_ref = (df_reference['Datetime'] >= start_time) & (df_reference['Datetime'] <= end_time)
            df_wave_plot = df_wave[mask_wave]
            df_mon_plot = df_mon[mask_mon]
            df_ref_plot = df_reference[mask_ref]
        else:
            df_wave_plot = df_wave
            df_mon_plot = df_mon
            df_ref_plot = df_reference

        # Filteren van drukwaarden <=2100
        df_wave_plot = df_wave_plot[df_wave_plot['Pressure'] <= 2100]
        df_mon_plot = df_mon_plot[df_mon_plot['Pressure'] <= 2100]
        df_ref_plot = df_ref_plot[df_ref_plot['Pressure'] <= 2100]

        # Plot HF druksonde druk
        ax.plot(df_wave_plot['Datetime'], df_wave_plot['Pressure'], label='HF druksonde druk', color='red', marker='o', markersize=2, linewidth=0.5)

        # Plot Diver druk
        ax.plot(df_mon_plot['Datetime'], df_mon_plot['Pressure'], label='Diver druk', color='blue', marker='x', markersize=2, linewidth=0.5)

        # Plot Referentie sensor
        if not df_ref_plot.empty:
            ax.plot(df_ref_plot['Datetime'], df_ref_plot['Pressure'], label='Referentie sensor', color='green', marker='s', markersize=6, linestyle='--')

        ax.set_title('HF druksonde druk vs Diver druk over Tijd')
        ax.set_xlabel('Tijd')
        ax.set_ylabel('Druk (Pa)')
        ax.legend()
        ax.grid(True)
        plt.gcf().autofmt_xdate()
        plt.show(block=False)  # Niet blokkerend
        plt.pause(0.001)  # Verwerk GUI events
        print("Einde plot_combined_graph")
    except Exception as e:
        print(f"Fout in plot_combined_graph: {e}")

# Functie om een X-Y grafiek te maken met lineaire regressie en tijdshift-schuifregelaar
def _legacy_plot_xy_regression_with_slider(df_wave, df_mon, start_time=None, end_time=None):
    try:
        print("Start plot_xy_regression_with_slider")
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.25)

        if start_time and end_time:
            mask_wave = (df_wave['Datetime'] >= start_time) & (df_wave['Datetime'] <= end_time)
            mask_mon = (df_mon['Datetime'] >= start_time) & (df_mon['Datetime'] <= end_time)
            df_wave_filtered = df_wave[mask_wave]
            df_mon_filtered = df_mon[mask_mon]
        else:
            df_wave_filtered = df_wave
            df_mon_filtered = df_mon

        # Initial merge zonder verschuiving
        merged_df = pd.merge_asof(df_wave_filtered.sort_values('Datetime'), df_mon_filtered.sort_values('Datetime'),
                                  on='Datetime', suffixes=('_hf', '_diver'))

        # Filteren van drukwaarden <=2100
        merged_df = merged_df[(merged_df['Pressure_hf'] <= 2100) & (merged_df['Pressure_diver'] <= 2100)]

        x = merged_df['Pressure_hf']
        y = merged_df['Pressure_diver']

        scatter = ax.scatter(x, y, color='blue', label='Data')

        # Initial regression
        if len(x) > 0 and len(y) > 0:
            slope, intercept, r_value, _, _ = linregress(x, y)
            line = slope * x + intercept
            r_squared = r_value**2
            regression_line, = ax.plot(x, line, color='red', label=f'Lineaire Fit: y={slope:.2f}x+{intercept:.2f}\n$R^2$={r_squared:.4f}')
        else:
            print("Geen geldige data voor initiële regressie.")

        ax.set_xlabel('HF druksonde druk (Pa)')
        ax.set_ylabel('Diver druk (Pa)')
        ax.set_title('X-Y Plot van HF druksonde druk tegen Diver druk')
        ax.legend()
        ax.grid(True)

        # Schuifregelaar voor tijdverschuiving
        ax_shift = plt.axes([0.25, 0.1, 0.65, 0.03])
        slider_shift = Slider(ax_shift, 'Tijd Shift (sec)', -60, 60, valinit=0, valstep=0.5)

        def update_regression(val):
            try:
                shift = slider_shift.val
                shifted_datetimes = df_mon['Datetime'] + timedelta(seconds=shift)
                shifted_df_mon = df_mon.copy()
                shifted_df_mon['Datetime'] = shifted_datetimes

                if start_time and end_time:
                    mask_wave = (df_wave['Datetime'] >= start_time) & (df_wave['Datetime'] <= end_time)
                    mask_mon = (shifted_df_mon['Datetime'] >= start_time) & (shifted_df_mon['Datetime'] <= end_time)
                    df_wave_shifted = df_wave[mask_wave]
                    df_mon_shifted = shifted_df_mon[mask_mon]
                else:
                    df_wave_shifted = df_wave
                    df_mon_shifted = shifted_df_mon

                # Initial merge met verschuiving
                merged = pd.merge_asof(df_wave_shifted.sort_values('Datetime'), df_mon_shifted.sort_values('Datetime'),
                                       on='Datetime', suffixes=('_hf', '_diver'))

                # Filteren van drukwaarden <=2100
                merged = merged[(merged['Pressure_hf'] <= 2100) & (merged['Pressure_diver'] <= 2100)]

                x_new = merged['Pressure_hf']
                y_new = merged['Pressure_diver']

                # Verwijder NaN waarden
                valid_mask = (~np.isnan(x_new)) & (~np.isnan(y_new))
                x_valid = x_new[valid_mask]
                y_valid = y_new[valid_mask]

                if len(x_valid) > 0 and len(y_valid) > 0:
                    slope, intercept, r_value, _, _ = linregress(x_valid, y_valid)
                    line = slope * x_valid + intercept
                    r_squared = r_value**2

                    ax.clear()
                    ax.scatter(x_valid, y_valid, color='blue', label='Data')
                    ax.plot(x_valid, line, color='red', label=f'Lineaire Fit: y={slope:.2f}x+{intercept:.2f}\n$R^2$={r_squared:.4f}')
                    ax.set_xlabel('HF druksonde druk (Pa)')
                    ax.set_ylabel('Diver druk (Pa)')
                    ax.set_title('X-Y Plot van HF druksonde druk tegen Diver druk')
                    ax.legend()
                    ax.grid(True)
                    fig.canvas.draw_idle()
                else:
                    print("Geen geldige data voor regressie.")
            except Exception as e:
                print(f"Fout in update_regression: {e}")

        slider_shift.on_changed(update_regression)

        plt.show(block=False)  # Niet blokkerend
        plt.pause(0.001)  # Verwerk GUI events
        print("Einde plot_xy_regression_with_slider")
    except Exception as e:
        print(f"Fout in plot_xy_regression_with_slider: {e}")

class InteractiveComparisonPlot:
    def __init__(self, sensor_frames, start_time=None, end_time=None, x_sensor="HF druksonde", y_sensor="Diver", pressure_cap=2100):
        self.original_sensor_frames = clone_sensor_frames(sensor_frames)
        self.sensor_frames = clone_sensor_frames(sensor_frames)
        self.start_time = start_time
        self.end_time = end_time
        self.x_sensor = x_sensor
        self.y_sensor = y_sensor
        self.pressure_cap = pressure_cap
        self.selection = None
        self.analysis_region = None
        self.default_limits = None
        self.nan_edits = {name: 0 for name in self.sensor_frames}

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.40)

        self.selector = RectangleSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],
            minspanx=0,
            minspany=0,
            spancoords='data',
        )

        ax_zoom = plt.axes([0.05, 0.14, 0.15, 0.06])
        ax_reset_zoom = plt.axes([0.22, 0.14, 0.15, 0.06])
        ax_use_region = plt.axes([0.39, 0.14, 0.15, 0.06])
        ax_clear = plt.axes([0.56, 0.14, 0.15, 0.06])
        ax_refresh = plt.axes([0.73, 0.14, 0.22, 0.06])
        ax_nan_x = plt.axes([0.05, 0.06, 0.15, 0.06])
        ax_nan_y = plt.axes([0.22, 0.06, 0.15, 0.06])
        ax_nan_compare = plt.axes([0.39, 0.06, 0.15, 0.06])
        ax_restore = plt.axes([0.56, 0.06, 0.15, 0.06])
        ax_recalc = plt.axes([0.73, 0.06, 0.22, 0.06])

        self.btn_zoom = Button(ax_zoom, 'Zoom selectie')
        self.btn_reset_zoom = Button(ax_reset_zoom, 'Reset zoom')
        self.btn_use_region = Button(ax_use_region, 'Gebruik gebied')
        self.btn_clear = Button(ax_clear, 'Wis selectie/gebied')
        self.btn_refresh = Button(ax_refresh, 'Update data')
        self.btn_nan_x = Button(ax_nan_x, 'NaN X')
        self.btn_nan_y = Button(ax_nan_y, 'NaN Y')
        self.btn_nan_compare = Button(ax_nan_compare, 'NaN Verg.')
        self.btn_restore = Button(ax_restore, 'Herstel alles')
        self.btn_recalc = Button(ax_recalc, 'Herbereken relatie')

        self.btn_zoom.on_clicked(self.zoom_selection)
        self.btn_reset_zoom.on_clicked(self.reset_zoom)
        self.btn_use_region.on_clicked(self.use_selection_as_region)
        self.btn_clear.on_clicked(self.clear_selection)
        self.btn_refresh.on_clicked(self.refresh_current_view)
        self.btn_nan_x.on_clicked(self.mark_x_selection_as_nan)
        self.btn_nan_y.on_clicked(self.mark_y_selection_as_nan)
        self.btn_nan_compare.on_clicked(self.mark_comparison_selection_as_nan)
        self.btn_restore.on_clicked(self.restore_all)
        self.btn_recalc.on_clicked(self.recalculate_relationship)

        self.update_plot()

    def _to_timestamp(self, value):
        if value is None:
            return None
        if isinstance(value, (float, np.floating)):
            return pd.to_datetime(num2date(value)).tz_localize(None)
        if isinstance(value, datetime):
            return pd.to_datetime(value)
        try:
            return pd.to_datetime(value)
        except Exception:
            return None

    def set_window(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.selection = None
        self.analysis_region = None
        self.refresh_current_view()

    def set_sensor_pair(self, x_sensor, y_sensor):
        self.x_sensor = x_sensor
        self.y_sensor = y_sensor
        self.selection = None
        self.refresh_current_view()

    def apply_view_options(self, start_time, end_time, x_sensor, y_sensor):
        self.start_time = start_time
        self.end_time = end_time
        self.x_sensor = x_sensor
        self.y_sensor = y_sensor
        self.selection = None
        self.analysis_region = None
        self.refresh_current_view()

    def refresh_current_view(self, _event=None):
        self.sensor_frames = clone_sensor_frames(self.sensor_frames)
        self.update_plot()
        self.reset_zoom()

    def current_frame(self, sensor_name):
        df = self.sensor_frames.get(sensor_name, pd.DataFrame(columns=["Datetime", "Pressure"]))
        return filter_sensor_frame(df, self.start_time, self.end_time, selection=None, pressure_cap=self.pressure_cap)

    def current_analysis_region(self):
        return self.analysis_region if self.analysis_region is not None else self.selection

    def _build_title(self):
        title = f"Druk over tijd | XY: {self.x_sensor} vs {self.y_sensor}"
        title += f"\nAnalysegebied: {selection_to_text(self.analysis_region)}"
        title += f"\nActieselectie: {selection_to_text(self.selection) if self.selection is not None else 'geen'}"
        edited = [f"{name}: {count}" for name, count in self.nan_edits.items() if count > 0]
        if edited:
            title += "\nPieken genegeerd: " + ", ".join(edited)
        return title

    def on_select(self, eclick, erelease):
        xmin = self._to_timestamp(eclick.xdata)
        xmax = self._to_timestamp(erelease.xdata)
        if xmin is None or xmax is None or eclick.ydata is None or erelease.ydata is None:
            self.selection = None
            return
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        ymin, ymax = sorted([float(eclick.ydata), float(erelease.ydata)])
        self.selection = (xmin, xmax, ymin, ymax)
        self.ax.set_title(self._build_title())
        self.fig.canvas.draw_idle()

    def zoom_selection(self, _event=None):
        if not self.selection:
            messagebox.showwarning("Geen selectie", "Sleep eerst een kader over het te vergelijken gebied.")
            return
        xmin, xmax, ymin, ymax = self.selection
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.fig.canvas.draw_idle()

    def reset_zoom(self, _event=None):
        if self.default_limits is None:
            return
        xlim, ylim = self.default_limits
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.fig.canvas.draw_idle()

    def clear_selection(self, _event=None):
        if self.selection is not None:
            self.selection = None
        else:
            self.analysis_region = None
        self.update_plot()

    def use_selection_as_region(self, _event=None):
        if not self.selection:
            messagebox.showwarning("Geen selectie", "Sleep eerst een kader over het analysegebied.")
            return
        self.analysis_region = self.selection
        self.update_plot()

    def _short_sensor_label(self, sensor_name):
        short_map = {
            "HF druksonde": "HF",
            "Diver": "Diver",
            "OSSI": "OSSI",
            "Referentie": "Ref",
        }
        return short_map.get(sensor_name, sensor_name[:8])

    def _comparison_nan_targets(self):
        targets = []
        for sensor_name in ("Diver", "OSSI", "Referentie"):
            if sensor_name in self.sensor_frames and sensor_name not in targets:
                targets.append(sensor_name)
        if targets:
            return targets

        for sensor_name in (self.x_sensor, self.y_sensor):
            if sensor_name in self.sensor_frames and sensor_name not in targets:
                targets.append(sensor_name)
        return targets

    def _sync_button_labels(self):
        self.btn_nan_x.label.set_text(f"NaN {self._short_sensor_label(self.x_sensor)}")
        self.btn_nan_y.label.set_text(f"NaN {self._short_sensor_label(self.y_sensor)}")
        compare_targets = self._comparison_nan_targets()
        compare_label = "+".join(self._short_sensor_label(name) for name in compare_targets) if compare_targets else "Verg."
        self.btn_nan_compare.label.set_text(f"NaN {compare_label}")

    def _mark_selection_as_nan_for_sensors(self, sensor_names):
        if not self.selection:
            messagebox.showwarning("Geen selectie", "Sleep eerst een kader rond de pieken die genegeerd moeten worden.")
            return

        changed_per_sensor = []
        for sensor_name in sensor_names:
            updated_frame, changed = mark_selection_as_nan(
                self.sensor_frames.get(sensor_name, pd.DataFrame(columns=["Datetime", "Pressure"])),
                self.selection,
            )
            if changed > 0:
                self.sensor_frames[sensor_name] = updated_frame
                self.nan_edits[sensor_name] += changed
                changed_per_sensor.append((sensor_name, changed))

        if not changed_per_sensor:
            messagebox.showwarning(
                "Geen pieken gevonden",
                "Geen punten gevonden in het geselecteerde kader voor de gekozen sensoren.",
            )
            return

        self.update_plot()
        details = "\n".join(f"{sensor_name}: {changed} punten" for sensor_name, changed in changed_per_sensor)
        messagebox.showinfo(
            "Pieken op NaN gezet",
            f"De selectie is genegeerd voor:\n{details}",
        )

    def mark_x_selection_as_nan(self, _event=None):
        self._mark_selection_as_nan_for_sensors([self.x_sensor])

    def mark_y_selection_as_nan(self, _event=None):
        self._mark_selection_as_nan_for_sensors([self.y_sensor])

    def mark_comparison_selection_as_nan(self, _event=None):
        self._mark_selection_as_nan_for_sensors(self._comparison_nan_targets())

    def restore_all(self, _event=None):
        self.sensor_frames = clone_sensor_frames(self.original_sensor_frames)
        self.nan_edits = {name: 0 for name in self.sensor_frames}
        self.selection = None
        self.analysis_region = None
        self.update_plot()
        self.reset_zoom()

    def recalculate_relationship(self, _event=None):
        region = self.current_analysis_region()
        result = plot_xy_regression_with_slider(
            self.sensor_frames,
            self.x_sensor,
            self.y_sensor,
            start_time=self.start_time,
            end_time=self.end_time,
            selection=region,
            pressure_cap=self.pressure_cap,
        )
        if result["count"] < 2 or not np.isfinite(result["slope"]):
            messagebox.showwarning(
                "Te weinig data",
                f"Geen geldige relatie voor {self.x_sensor} tegen {self.y_sensor}\n"
                f"op gebied: {selection_to_text(region)}",
            )
            return
        messagebox.showinfo(
            "Herberekende relatie",
            f"{self.y_sensor} t.o.v. {self.x_sensor}\n"
            f"Gebied: {selection_to_text(region)}\n"
            f"Punten: {result['count']}\n"
            f"Slope: {result['slope']:.6f}\n"
            f"Intercept: {result['intercept']:.3f}\n"
            f"R²: {result['r_squared']:.4f}",
        )

    def update_plot(self):
        self.ax.clear()
        self._sync_button_labels()
        plotted_frames = []
        style_map = {
            "HF druksonde": {"color": "red", "marker": "o", "markersize": 2, "linewidth": 0.5},
            "Diver": {"color": "blue", "marker": "x", "markersize": 2, "linewidth": 0.5},
            "Referentie": {"color": "green", "marker": "s", "markersize": 4, "linewidth": 0.8, "linestyle": "--"},
        }

        for sensor_name in ("HF druksonde", "Diver", "Referentie"):
            frame = self.current_frame(sensor_name)
            if frame.empty:
                continue
            plotted_frames.append(frame)
            self.ax.plot(frame["Datetime"], frame["Pressure"], label=sensor_name, **style_map[sensor_name])

        if plotted_frames:
            all_datetimes = pd.concat([frame["Datetime"] for frame in plotted_frames], ignore_index=True)
            all_pressures = pd.to_numeric(
                pd.concat([frame["Pressure"] for frame in plotted_frames], ignore_index=True),
                errors="coerce",
            )
            valid_pressures = all_pressures.dropna()
            y_limits = (0.0, 1.0) if valid_pressures.empty else (float(valid_pressures.min()), float(valid_pressures.max()))
            self.default_limits = (
                (all_datetimes.min(), all_datetimes.max()),
                y_limits,
            )
            self.ax.set_xlim(self.default_limits[0])
            self.ax.set_ylim(self.default_limits[1])
            if self.analysis_region is not None:
                xmin, xmax, ymin, ymax = self.analysis_region
                self.ax.axvspan(xmin, xmax, color='gold', alpha=0.10)
                self.ax.hlines([ymin, ymax], xmin, xmax, colors='goldenrod', linestyles='--', linewidth=1.0)

        self.ax.set_title(self._build_title())
        self.ax.set_xlabel('Tijd')
        self.ax.set_ylabel('Druk (Pa)')
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(handles, labels)
        self.ax.grid(True)
        self.fig.autofmt_xdate()
        self.fig.canvas.draw_idle()

def plot_xy_regression_with_slider(sensor_frames, x_sensor, y_sensor, start_time=None, end_time=None, selection=None, pressure_cap=2100):
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.25)

        ax_shift = plt.axes([0.25, 0.1, 0.65, 0.03])
        slider_shift = Slider(ax_shift, f'Tijd Shift {y_sensor} (sec)', -60, 60, valinit=0, valstep=0.5)

        def draw_relationship(shift_seconds):
            result = calculate_relationship(
                sensor_frames,
                x_sensor,
                y_sensor,
                start_time=start_time,
                end_time=end_time,
                selection=selection,
                shift_seconds=shift_seconds,
                pressure_cap=pressure_cap,
            )
            ax.clear()
            if result["count"] >= 1:
                ax.scatter(result["x"], result["y"], color='blue', label='Data')
            if result["count"] >= 2 and np.isfinite(result["slope"]):
                line_df = pd.DataFrame({
                    "x": result["x"],
                    "line": result["slope"] * result["x"] + result["intercept"],
                }).sort_values("x")
                ax.plot(
                    line_df["x"],
                    line_df["line"],
                    color='red',
                    label=f"Lineaire Fit: y={result['slope']:.4f}x+{result['intercept']:.3f}\n$R^2$={result['r_squared']:.4f}",
                )
            ax.set_title(f"X-Y Plot: {x_sensor} (x) vs {y_sensor} (y)\nGebied: {selection_to_text(selection)}")
            ax.set_xlabel(f'{x_sensor} druk (Pa)')
            ax.set_ylabel(f'{y_sensor} druk (Pa)')
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels)
            ax.grid(True)
            fig.canvas.draw_idle()
            return result

        result = draw_relationship(0.0)

        def update_regression(_val):
            draw_relationship(slider_shift.val)

        slider_shift.on_changed(update_regression)

        plt.show(block=False)
        plt.pause(0.001)
        return result
    except Exception as e:
        print(f"Fout in plot_xy_regression_with_slider: {e}")
        return {
            "merged": pd.DataFrame(),
            "x": pd.Series(dtype=float),
            "y": pd.Series(dtype=float),
            "slope": np.nan,
            "intercept": np.nan,
            "r_squared": np.nan,
            "count": 0,
        }

# Functie om data te exporteren naar aparte CSV-bestanden
def _legacy_export_data(df_wave, df_mon, selected_directory, start_time, end_time):
    try:
        # Filter de data op de geselecteerde tijdsperiode
        mask_wave = (df_wave['Datetime'] >= start_time) & (df_wave['Datetime'] <= end_time)
        mask_mon = (df_mon['Datetime'] >= start_time) & (df_mon['Datetime'] <= end_time)

        df_wave_filtered = df_wave[mask_wave]
        df_mon_filtered = df_mon[mask_mon]

        # Filter drukwaarden <=2100
        df_wave_filtered = df_wave_filtered[df_wave_filtered['Pressure'] <= 2100]
        df_mon_filtered = df_mon_filtered[df_mon_filtered['Pressure'] <= 2100]

        # Combineer de data voor de gecombineerde CSV
        merged_df = pd.merge_asof(df_wave_filtered.sort_values('Datetime'), df_mon_filtered.sort_values('Datetime'),
                                  on='Datetime', suffixes=('_hf', '_diver'))

        # Defineer bestandsnamen
        hf_filename = os.path.join(selected_directory, 'hf_druksonde_data_filtered.csv')
        diver_filename = os.path.join(selected_directory, 'diver_druk_data_filtered.csv')
        combined_filename = os.path.join(selected_directory, 'combined_data_filtered.csv')

        # Sla de gefilterde HF druksonde data op
        df_wave_filtered.to_csv(hf_filename, index=False)
        print(f"HF druksonde data geëxporteerd naar {hf_filename}")

        # Sla de gefilterde Diver druk data op
        df_mon_filtered.to_csv(diver_filename, index=False)
        print(f"Diver druk data geëxporteerd naar {diver_filename}")

        # Sla de gecombineerde data op
        merged_df.to_csv(combined_filename, index=False)
        print(f"Gecombineerde data geëxporteerd naar {combined_filename}")

        # Informeer de gebruiker
        messagebox.showinfo("Export Succesvol", f"Data succesvol geëxporteerd naar:\n{hf_filename}\n{diver_filename}\n{combined_filename}")
    except Exception as e:
        print(f"Fout bij het exporteren van data: {e}")
        messagebox.showerror("Export Fout", f"Er is een fout opgetreden bij het exporteren van de data:\n{e}")

def export_data(sensor_frames, selected_directory, start_time, end_time, x_sensor, y_sensor, selection=None, pressure_cap=2100):
    try:
        exported_files = []

        for sensor_name, df in sensor_frames.items():
            filtered_df = filter_sensor_frame(
                df,
                start_time=start_time,
                end_time=end_time,
                selection=selection,
                pressure_cap=pressure_cap,
            )
            output_path = os.path.join(selected_directory, sensor_export_filename(sensor_name))
            filtered_df.to_csv(output_path, index=False)
            exported_files.append(output_path)
            print(f"{sensor_name} data geexporteerd naar {output_path}")

        relationship = calculate_relationship(
            sensor_frames,
            x_sensor,
            y_sensor,
            start_time=start_time,
            end_time=end_time,
            selection=selection,
            pressure_cap=pressure_cap,
        )
        combined_filename = os.path.join(selected_directory, 'combined_data_filtered.csv')
        relationship["merged"].to_csv(combined_filename, index=False)
        exported_files.append(combined_filename)
        print(f"Gecombineerde data geexporteerd naar {combined_filename}")

        exported_text = "\n".join(exported_files)
        messagebox.showinfo(
            "Export Succesvol",
            f"Data succesvol geexporteerd voor gebied:\n{selection_to_text(selection)}\n\n{exported_text}",
        )
    except Exception as e:
        print(f"Fout bij het exporteren van data: {e}")
        messagebox.showerror("Export Fout", f"Er is een fout opgetreden bij het exporteren van de data:\n{e}")

# Referentiesensor inlezen gebeurt hoger via load_reference_sensor_data(path, ...)

# Hoofdprogramma
def main():
    selected_directory = select_directory()
    if not selected_directory:
        messagebox.showwarning("Geen Directory Geselecteerd", "Er is geen directory geselecteerd.")
        return

    mon_file = filedialog.askopenfilename(title="Selecteer een .mon bestand", filetypes=[("MON bestanden", "*.mon")])
    if not mon_file:
        messagebox.showwarning("Geen Bestand Geselecteerd", "Er is geen .mon bestand geselecteerd.")
        return
    print(f"Verwerken van .mon bestand: {mon_file}")

    ref_file = filedialog.askopenfilename(
        title="Selecteer referentie CSV/TXT",
        filetypes=[("CSV", "*.csv"), ("TXT", "*.txt"), ("Alle bestanden", "*.*")],
    )
    if not ref_file:
        messagebox.showwarning("Geen Bestand Geselecteerd", "Er is geen referentie-bestand geselecteerd.")
        return
    print(f"Referentie: {ref_file}")

    files = [
        file_name
        for file_name in sorted(os.listdir(selected_directory))
        if file_name.lower().endswith('.csv')
        and not file_name.endswith('_filtered.csv')
        and file_name != 'combined_data_filtered.csv'
        and file_name != os.path.basename(ref_file)
    ]
    if not files:
        messagebox.showwarning("Geen CSV-bestanden", "Geen ongefilterde HF CSV-bestanden gevonden in de geselecteerde directory.")
        return

    wave_pressure_offset = get_offset_input('HF druksonde druk')
    mon_pressure_offset = get_offset_input('Diver druk')
    mon_time_offset = get_offset_input('Diver tijd (in seconden)')
    ref_offset = get_offset_input('Referentie druk')

    data_lines = read_mon_file(mon_file)
    if not data_lines:
        messagebox.showerror("Geen Data", "Geen data gevonden in het .mon bestand.")
        return

    df_mon = parse_mon_data(data_lines, mon_pressure_offset, mon_time_offset)
    if df_mon.empty:
        messagebox.showerror("Geen Data", "Geen geldige data geparsed uit het .mon bestand.")
        return

    filename_dt, file_dt = extract_reference_start_candidates(ref_file)
    ref_mode, ref_manual_dt = ask_datetime_source_dialog(
        "Starttijd referentiedruksensor",
        option_specs=[
            {"key": "filename", "label": "Uit filenaam / eerste regel", "preview": format_candidate(filename_dt or file_dt)},
            {"key": "file", "label": "Uit referentiebestand (absolute tijd of header)", "preview": format_candidate(file_dt)},
            {"key": "manual", "label": "Handmatig via kalender + tijd"},
        ],
        default_manual=file_dt or filename_dt or df_mon["Datetime"].min(),
        default_mode="filename" if (filename_dt or file_dt) else "manual",
    )
    if ref_mode is None:
        messagebox.showwarning("Afgebroken", "Geen starttijdmethode gekozen voor het referentiebestand.")
        return

    try:
        df_reference = load_reference_sensor_data(
            ref_file,
            time_mode=ref_mode,
            manual_start=ref_manual_dt,
            pressure_offset=ref_offset,
        )
    except Exception as e:
        messagebox.showerror("Referentie Leesfout", str(e))
        return

    if df_reference.empty:
        messagebox.showerror("Geen Data", "Geen geldige referentie-data.")
        return

    first_wave_file = os.path.join(selected_directory, files[0])
    first_wave_filename_dt = extract_start_datetime_from_text(files[0])
    _, first_wave_header_dt = extract_wave_file_info(first_wave_file)
    reference_start = df_reference["Datetime"].min()
    wave_mode, wave_manual_dt = ask_datetime_source_dialog(
        "Starttijd HF druksonde",
        option_specs=[
            {"key": "filename", "label": "Uit filenaam / HF-header", "preview": format_candidate(first_wave_filename_dt or first_wave_header_dt)},
            {"key": "reference", "label": "Overnemen van referentiedruksensor", "preview": format_candidate(reference_start)},
            {"key": "manual", "label": "Handmatig via kalender + tijd"},
        ],
        default_manual=first_wave_filename_dt or first_wave_header_dt or reference_start or df_mon["Datetime"].min(),
        default_mode="filename" if (first_wave_filename_dt or first_wave_header_dt) else "reference",
    )
    if wave_mode is None:
        messagebox.showwarning("Afgebroken", "Geen starttijdmethode gekozen voor de HF-bestanden.")
        return

    try:
        hf_paths = [os.path.join(selected_directory, file_name) for file_name in files]
        df_wave = build_wave_dataset(
            hf_paths,
            pressure_offset=wave_pressure_offset,
            time_mode=wave_mode,
            manual_start=wave_manual_dt,
            reference_start=reference_start,
        )
    except Exception as e:
        messagebox.showerror("HF Leesfout", str(e))
        return

    if df_wave.empty:
        messagebox.showerror("Geen Data", "Geen geldige HF-data gevonden.")
        return

    display_pressure_summary(df_wave, df_mon, df_reference)
    sensor_frames = build_sensor_frames(df_wave, df_mon, df_reference)
    available_sensors = list(sensor_frames.keys())
    if len(available_sensors) < 2:
        messagebox.showerror("Te weinig sensoren", "Er zijn minstens twee druksensoren nodig voor een XY-vergelijking.")
        return

    all_time_values = []
    for frame in sensor_frames.values():
        all_time_values.extend([frame["Datetime"].min(), frame["Datetime"].max()])
    initial_start = min(all_time_values)
    initial_end = max(all_time_values)

    default_x = "HF druksonde" if "HF druksonde" in available_sensors else available_sensors[0]
    default_y_candidates = [name for name in ("Diver", "Referentie") if name in available_sensors and name != default_x]
    default_y = default_y_candidates[0] if default_y_candidates else next((name for name in available_sensors if name != default_x), available_sensors[0])

    comparison_plot = InteractiveComparisonPlot(
        sensor_frames,
        start_time=initial_start,
        end_time=initial_end,
        x_sensor=default_x,
        y_sensor=default_y,
        pressure_cap=2100,
    )

    def update_plot_with_time_range(start_time, end_time, x_sensor, y_sensor):
        comparison_plot.apply_view_options(start_time, end_time, x_sensor, y_sensor)

    def export_and_close(start_time, end_time, x_sensor, y_sensor):
        if comparison_plot.start_time != start_time or comparison_plot.end_time != end_time:
            comparison_plot.apply_view_options(start_time, end_time, x_sensor, y_sensor)
        elif comparison_plot.x_sensor != x_sensor or comparison_plot.y_sensor != y_sensor:
            comparison_plot.apply_view_options(start_time, end_time, x_sensor, y_sensor)
        export_data(
            comparison_plot.sensor_frames,
            selected_directory,
            start_time,
            end_time,
            x_sensor,
            y_sensor,
            selection=comparison_plot.current_analysis_region(),
            pressure_cap=comparison_plot.pressure_cap,
        )

    create_time_selection_gui(
        df_wave,
        df_mon,
        df_reference,
        selected_directory,
        update_plot_with_time_range,
        export_and_close,
        available_sensors,
    )

if __name__ == "__main__":
    main()
