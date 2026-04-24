import os
import struct
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import logging
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkcalendar import DateEntry
import re
from PIL import Image, ImageTk, __version__ as PILLOW_VERSION
import base64
from io import BytesIO
import plotly.graph_objects as go
import csv

# Stel logging in
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------- Parser Functionaliteit -------------------

class TimeConverter:
    @staticmethod
    def convert_time(seconds_since_base):
        base_date = datetime(1984, 3, 1)
        try:
            vectorized_convert = np.vectorize(
                lambda sec: (base_date + timedelta(seconds=sec)).strftime("%d/%m/%y %H:%M:%S")
            )
            return vectorized_convert(seconds_since_base)
        except Exception as e:
            logging.error(f"Time conversion error: {e}")
            if isinstance(seconds_since_base, np.ndarray):
                return np.full_like(seconds_since_base, "Invalid Time", dtype=object)
            else:
                return "Invalid Time"

def YSI6SeriesParse(filename, mode):
    """
    Parser voor YSI 6 Series bestanden.
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be a string.")
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"{filename} bestaat niet.")
    try:
        with open(filename, 'rb') as f:
            data = f.read()
    except Exception as e:
        raise IOError(f"Fout bij het lezen van bestand {filename}: {e}")

    header = readHeader(data)
    records = readRecords(header, data)

    sample_data = {
        'toolbox_input_file': filename,
        'meta': {
            'instrument_make': 'YSI',
            'instrument_model': '6 Series',
            'instrument_serial_no': '',
            'instrument_sample_interval': np.median(np.diff(records['time']) * 24 * 3600),
            'featureType': mode
        },
        'dimensions': [],
        'variables': []
    }

    # Definieer dimensie TIME
    time_type = netcdf3ToMatlabType(imosParameters('TIME', 'type'))
    sample_data['dimensions'].append({
        'name': 'TIME',
        'typeCastFunc': float,
        'data': records['time'].astype(float)
    })

    # Definieer een aantal initiële variabelen
    variables_info = [
        {'name': 'TIMESERIES', 'data': 1, 'dimensions': []},
        {'name': 'LATITUDE', 'data': np.nan, 'dimensions': []},
        {'name': 'LONGITUDE', 'data': np.nan, 'dimensions': []},
        {'name': 'NOMINAL_DEPTH', 'data': np.nan, 'dimensions': []}
    ]

    for var in variables_info:
        var_type = netcdf3ToMatlabType(imosParameters(var['name'], 'type'))
        var_entry = {
            'name': var['name'],
            'typeCastFunc': eval(var_type),
            'data': eval(var_type)(var['data']),
            'dimensions': var['dimensions']
        }
        sample_data['variables'].append(var_entry)

    # Verwerk overige velden
    fields = [key for key in records.keys() if key != 'time']
    coordinates = 'TIME LATITUDE LONGITUDE NOMINAL_DEPTH'

    for field in fields:
        field_data = records[field]
        if field.lower() in ['latitude', 'longitude']:
            dimensions = []
        else:
            dimensions = [0]  # Tijd is de eerste dimensie
        var_entry = {}
        var_entry['dimensions'] = dimensions
        if field == 'temperature':
            var_entry['name'] = 'Temperatuur (°C)'
            var_entry['data'] = imos_cast(field_data, 'Temperatuur (°C)')
        elif field == 'cond':
            var_entry['name'] = 'Conductiviteit (`µS)'
            var_entry['data'] = imos_cast(field_data * 1000.0, 'Conductiviteit')
        elif field == 'spcond':
            var_entry['name'] = 'SPEC_CNDC'
            var_entry['data'] = imos_cast(field_data / 10.0, 'SPEC_CNDC')
        elif field == 'tds':
            var_entry['name'] = 'TDS'
            var_entry['data'] = imos_cast(field_data, 'TDS')
        elif field == 'salinity':
            var_entry['name'] = 'PSAL'
            var_entry['data'] = imos_cast(field_data, 'PSAL')
        elif field == 'ph':
            var_entry['name'] = 'pH'
            var_entry['data'] = imos_cast(field_data, 'pH')
        elif field == 'orp':
            var_entry['name'] = 'ORP'
            var_entry['data'] = imos_cast(field_data, 'ORP')
        elif field == 'depth':
            var_entry['name'] = 'Diepte (m)'
            var_entry['data'] = imos_cast(field_data, 'Diepte (m)')
        elif field == 'bp':
            var_entry['name'] = 'PRES'
            var_entry['data'] = imos_cast(field_data / 1.45037738, 'PRES')
        elif field == 'battery':
            var_entry['name'] = 'BAT_VOLT'
            var_entry['data'] = imos_cast(field_data, 'BAT_VOLT')
        elif field == 'chlorophyll':
            var_entry['name'] = 'CPHL'
            var_entry['data'] = imos_cast(field_data, 'CPHL')
            var_entry['comment'] = getCPHLcomment('unknown', '470nm', 'above 630nm')
        elif field == 'latitude':
            sample_data['variables'][1]['data'] = imos_cast(field_data, 'LATITUDE')
            continue
        elif field == 'longitude':
            sample_data['variables'][2]['data'] = imos_cast(field_data, 'LONGITUDE')
            continue
        elif field == 'turbidity':
            var_entry['name'] = 'Turbiditeit (NTU)'
            var_entry['data'] = imos_cast(field_data, 'Turbiditeit (NTU)')
            var_entry['comment'] = 'Turbidity from 6136 sensor.'
        elif field == 'odo':
            var_entry['name'] = 'DOXS'
            var_entry['data'] = imos_cast(field_data, 'DOXS')
            var_entry['comment'] = 'Dissolved oxygen saturation from ROX optical sensor.'
        elif field == 'odo2':
            var_entry['name'] = 'DOXY'
            var_entry['data'] = imos_cast(field_data, 'DOXY')
            var_entry['comment'] = 'Dissolved oxygen from ROX optical sensor.'
        else:
            continue

        var_entry['coordinates'] = coordinates
        var_entry['typeCastFunc'] = netcdf3ToMatlabType(imosParameters(var_entry['name'], 'type'))
        sample_data['variables'].append(var_entry)

    return sample_data

def readHeader(data):
    header = {
        'recordFmt': []
    }
    try:
        idx = data.index(66)  # ASCII-code voor 'B'
    except ValueError:
        raise ValueError("Sync byte 0x42 niet gevonden in data.")
    while True:
        entry = data[idx:idx+15]
        if len(entry) < 15:
            break
        if entry[0] != 66:
            break
        header['recordFmt'].append(entry[3])
        idx += 15
    header['recordStart'] = idx
    header['recordLength'] = 1 + (len(header['recordFmt']) + 1) * 4 # type: ignore
    return header

def readRecords(header, data):
    records = {}
    record_length = header['recordLength']
    record_start = header['recordStart']
    record_fmt = header['recordFmt']
    rNum = 0
    cpu_endianness = sys.byteorder  # 'little' of 'big'
    data = data[record_start:]
    while len(data) >= record_length:
        record = data[:record_length]
        data = data[record_length:]
        rNum += 1
        if record[0] != 68:  # 0x44
            try:
                next_sync = record.index(68, 1)
                data = record[next_sync:] + data
            except ValueError:
                continue
            continue
        time_bytes = record[1:5]
        time_val = byte_cast(time_bytes, 'L', 'I', cpu_endianness)
        records.setdefault('time', []).append(time_val)
        num_floats = (len(record) - 5) // 4
        vals = byte_cast(record[5:], 'L', 'f', cpu_endianness, count=num_floats)
        for fmt, val in zip(record_fmt, vals):
            if fmt == 1:
                records.setdefault('temperature', []).append(val)
            elif fmt == 4:
                records.setdefault('cond', []).append(val)
            elif fmt == 6:
                records.setdefault('spcond', []).append(val)
            elif fmt == 10:
                records.setdefault('tds', []).append(val)
            elif fmt == 12:
                records.setdefault('salinity', []).append(val)
            elif fmt == 18:
                records.setdefault('ph', []).append(val)
            elif fmt == 19:
                records.setdefault('orp', []).append(val)
            elif fmt == 22:
                records.setdefault('depth', []).append(val)
            elif fmt == 24:
                records.setdefault('bp', []).append(val)
            elif fmt == 28:
                records.setdefault('battery', []).append(val)
            elif fmt == 193:
                records.setdefault('chlorophyll', []).append(val)
            elif fmt == 196:
                records.setdefault('latitude', []).append(val)
            elif fmt == 197:
                records.setdefault('longitude', []).append(val)
            elif fmt == 203:
                records.setdefault('turbidity', []).append(val)
            elif fmt == 211:
                records.setdefault('odo', []).append(val)
            elif fmt == 212:
                records.setdefault('odo2', []).append(val)
        # Einde record verwerking
    for key in records:
        records[key] = np.array(records[key])
    return records

def byte_cast(byte_data, endian_indicator, data_type, cpu_endianness, count=1):
    if endian_indicator == 'L':
        endian = '<'
    else:
        endian = '>'
    format_str = endian + data_type * count
    try:
        unpacked = struct.unpack(format_str, byte_data)
    except struct.error as e:
        raise ValueError(f"Fout bij het unpacken van data: {e}")
    if count == 1:
        return unpacked[0]
    return unpacked

def netcdf3ToMatlabType(netcdf_type):
    type_mapping = {
        'float': 'float',
        'double': 'float',
        'int': 'int',
        'short': 'int',
        'byte': 'int',
        'char': 'str',
    }
    return type_mapping.get(netcdf_type.lower(), 'float')

def imosParameters(variable_name, parameter):
    imos_params = {
        'TIME': {'type': 'float'},
        'TIMESERIES': {'type': 'float'},
        'LATITUDE': {'type': 'float'},
        'LONGITUDE': {'type': 'float'},
        'NOMINAL_DEPTH': {'type': 'float'},
        'Temperatuur (°C)': {'type': 'float'},
        'CONDUCTIVITEIT (µS)': {'type': 'float'},
        'SPEC_CNDC': {'type': 'float'},
        'TDS': {'type': 'float'},
        'PSAL': {'type': 'float'},
        'PH': {'type': 'float'},
        'ORP': {'type': 'float'},
        'DIEPTE (m)': {'type': 'float'},
        'PRES': {'type': 'float'},
        'BAT_VOLT': {'type': 'float'},
        'CPHL': {'type': 'float'},
        'TURBIDITEIT (NTU)': {'type': 'float'},
        'DOXS': {'type': 'float'},
        'DOXY': {'type': 'float'},
    }
    return imos_params.get(variable_name.upper(), {}).get(parameter.lower(), 'float')

def getCPHLcomment(param1, param2, param3):
    return f"Chlorophyll measurement parameters: {param1}, {param2}, {param3}."

def imos_cast(data, variable_name):
    cast_type = netcdf3ToMatlabType(imosParameters(variable_name, 'type'))
    if cast_type == 'float':
        return data.astype(float)
    elif cast_type == 'int':
        return data.astype(int)
    elif cast_type == 'str':
        return data.astype(str)
    else:
        return data

# ------------------- Plotter Functionaliteit -------------------

def pillow_version_at_least(major, minor, patch=0):
    version = tuple(map(int, PILLOW_VERSION.split('.')[:3]))
    return version >= (major, minor, patch)

# ------------------- Geïntegreerde GUI Implementatie -------------------

class IntegratedGUI:
    def __init__(self, master):
        self.master = master
        master.title("YSI 6 Series Parser en CSV Plotter")
        master.columnconfigure(1, weight=1)

        # Titelkader
        self.title_frame = tk.Frame(master, bd=2, relief=tk.RIDGE, padx=10, pady=10)
        self.title_frame.grid(row=0, column=0, columnspan=3, pady=10, sticky="ew")
        project_frame = tk.Frame(self.title_frame)
        project_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(project_frame, text="Projectnummer:").pack(anchor='w')
        self.project_entry = tk.Entry(project_frame, textvariable=tk.StringVar(value='24_059'), width=20)
        self.project_entry.pack(anchor='w')
        info_frame = tk.Frame(self.title_frame)
        info_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(info_frame, text="Naam maker:").grid(row=0, column=0, sticky='w')
        self.creator_entry = tk.Entry(info_frame, textvariable=tk.StringVar(), width=20)
        self.creator_entry.grid(row=0, column=1, sticky='w')
        tk.Label(info_frame, text="Locatie:").grid(row=1, column=0, sticky='w')
        self.location_entry = tk.Entry(info_frame, textvariable=tk.StringVar(), width=20)
        self.location_entry.grid(row=1, column=1, sticky='w')
        tk.Label(info_frame, text="Datum aanmaak:").grid(row=2, column=0, sticky='w')
        self.creation_date_entry = tk.Entry(info_frame, textvariable=tk.StringVar(value=datetime.today().strftime('%d/%m/%Y')), width=12)
        self.creation_date_entry.grid(row=2, column=1, sticky='w')

        # Parser Sectie
        self.file_label = ttk.Label(master, text="Selecteer Bestand:")
        self.file_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.file_path = tk.StringVar()
        self.file_entry = ttk.Entry(master, textvariable=self.file_path, width=50, state='readonly')
        self.file_entry.grid(row=1, column=1, padx=10, pady=10, sticky=tk.EW)
        self.browse_button = ttk.Button(master, text="Bladeren...", command=self.browse_file)
        self.browse_button.grid(row=1, column=2, padx=10, pady=10)
        self.type_label = ttk.Label(master, text="Bestandstype:")
        self.type_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        self.type_var = tk.StringVar()
        self.type_combobox = ttk.Combobox(master, textvariable=self.type_var, state='readonly')
        self.type_combobox['values'] = ('DAT', 'CSV')
        self.type_combobox.current(0)
        self.type_combobox.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W)
        self.parse_button = ttk.Button(master, text="Converteer Data", command=self.parse_and_load)
        self.parse_button.grid(row=2, column=2, padx=10, pady=20)
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(master, textvariable=self.status_var, foreground="blue")
        self.status_label.grid(row=3, column=0, columnspan=3, padx=10, pady=10, sticky=tk.W)

        # Plotter Sectie
        delimiter_frame = tk.Frame(master)
        delimiter_frame.grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
        tk.Label(delimiter_frame, text="Kies delimiter:").pack(side=tk.LEFT)
        delimiter_options = ['.', ',', ';', '\t', '|', ' ']
        self.delimiter_var = tk.StringVar(value='.')
        delimiter_menu = tk.OptionMenu(delimiter_frame, self.delimiter_var, *delimiter_options)
        delimiter_menu.pack(side=tk.LEFT)
        decimal_frame = tk.Frame(master)
        decimal_frame.grid(row=4, column=1, padx=10, pady=5, sticky=tk.W)
        tk.Label(decimal_frame, text="Kies decimaal teken:").pack(side=tk.LEFT)
        decimal_options = [',', '.']
        self.decimal_var = tk.StringVar(value=',')
        decimal_menu = tk.OptionMenu(decimal_frame, self.decimal_var, *decimal_options)
        decimal_menu.pack(side=tk.LEFT)
        time_column_frame = tk.Frame(master)
        time_column_frame.grid(row=5, column=0, padx=10, pady=5, sticky=tk.W)
        tk.Label(time_column_frame, text="Selecteer tijdkolom:").pack(side=tk.LEFT)
        self.time_column_var = tk.StringVar()
        self.time_menu = tk.OptionMenu(time_column_frame, self.time_column_var, '')
        self.time_menu.pack(side=tk.LEFT)
        start_datetime_frame = tk.Frame(master)
        start_datetime_frame.grid(row=5, column=1, padx=10, pady=5, sticky=tk.W)
        tk.Label(start_datetime_frame, text="Startdatum:").pack(side=tk.LEFT)
        self.start_date_entry = DateEntry(start_datetime_frame, date_pattern='dd/mm/yyyy')
        self.start_date_entry.pack(side=tk.LEFT)
        tk.Label(start_datetime_frame, text="Tijd (HH:MM:SS):").pack(side=tk.LEFT)
        self.start_hour_spinbox = tk.Spinbox(start_datetime_frame, from_=0, to=23, width=2, format="%02.0f")
        self.start_hour_spinbox.pack(side=tk.LEFT)
        tk.Label(start_datetime_frame, text=":").pack(side=tk.LEFT)
        self.start_minute_spinbox = tk.Spinbox(start_datetime_frame, from_=0, to=59, width=2, format="%02.0f")
        self.start_minute_spinbox.pack(side=tk.LEFT)
        tk.Label(start_datetime_frame, text=":").pack(side=tk.LEFT)
        self.start_second_spinbox = tk.Spinbox(start_datetime_frame, from_=0, to=59, width=2, format="%02.0f")
        self.start_second_spinbox.pack(side=tk.LEFT)
        end_datetime_frame = tk.Frame(master)
        end_datetime_frame.grid(row=6, column=0, padx=10, pady=5, sticky=tk.W)
        tk.Label(end_datetime_frame, text="Einddatum:").pack(side=tk.LEFT)
        self.end_date_entry = DateEntry(end_datetime_frame, date_pattern='dd/mm/yyyy')
        self.end_date_entry.pack(side=tk.LEFT)
        tk.Label(end_datetime_frame, text="Tijd (HH:MM:SS):").pack(side=tk.LEFT)
        self.end_hour_spinbox = tk.Spinbox(end_datetime_frame, from_=0, to=23, width=2, format="%02.0f")
        self.end_hour_spinbox.pack(side=tk.LEFT)
        tk.Label(end_datetime_frame, text=":").pack(side=tk.LEFT)
        self.end_minute_spinbox = tk.Spinbox(end_datetime_frame, from_=0, to=59, width=2, format="%02.0f")
        self.end_minute_spinbox.pack(side=tk.LEFT)
        tk.Label(end_datetime_frame, text=":").pack(side=tk.LEFT)
        self.end_second_spinbox = tk.Spinbox(end_datetime_frame, from_=0, to=59, width=2, format="%02.0f")
        self.end_second_spinbox.pack(side=tk.LEFT)
        logo_frame = tk.Frame(master)
        logo_frame.grid(row=6, column=1, padx=10, pady=5, sticky=tk.W)
        tk.Label(logo_frame, text="Logo-bestand:").pack(side=tk.LEFT)
        self.logo_path_var = tk.StringVar()
        self.logo_entry = tk.Entry(logo_frame, textvariable=self.logo_path_var, width=40)
        self.logo_entry.pack(side=tk.LEFT)
        self.logo_browse_button = tk.Button(logo_frame, text="Bladeren...", command=self.select_logo)
        self.logo_browse_button.pack(side=tk.LEFT)
        self.column_frame = tk.LabelFrame(master, text="Selecteer kolommen en assen:")
        self.column_frame.grid(row=7, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W+tk.E)
        self.axis_selection_vars = {}  # Key: kolomnaam, Value: StringVar
        self.preview_frame = tk.LabelFrame(master, text="Data Preview (Eerste 10 Rijen):")
        self.preview_frame.grid(row=8, column=0, columnspan=2, padx=10, pady=5, sticky=tk.W+tk.E)
        self.plot_button = tk.Button(master, text="Genereer Grafiek", command=self.generate_plot)
        self.plot_button.grid(row=9, column=0, columnspan=2, pady=10)

    def browse_file(self):
        filetypes = (("DAT bestanden", "*.dat"), ("CSV bestanden", "*.csv"), ("Alle bestanden", "*.*"))
        filename = filedialog.askopenfilename(title="Open Bestand", filetypes=filetypes)
        if filename:
            self.file_path.set(filename)
            selected_type = self.type_var.get()
            if selected_type == 'DAT':
                self.status_var.set("DAT bestand geselecteerd. Klik op 'Converteer Data'.")
            else:
                self.status_var.set("CSV bestand geselecteerd. Laad kolommen...")

    def parse_and_load(self):
        filename = self.file_path.get()
        selected_type = self.type_var.get()
        if not filename:
            messagebox.showerror("Fout", "Selecteer een bestand om te converteren.")
            return
        try:
            if selected_type == 'DAT':
                self.status_var.set("Converteer van DAT bestand...")
                self.master.update_idletasks()
                sample_data = YSI6SeriesParse(filename, 'Mode1')
                df = self.sample_data_to_dataframe(sample_data)
                self.status_var.set("DAT bestand geconverteerd en data geladen.")
            elif selected_type == 'CSV':
                self.status_var.set("CSV bestand laden...")
                self.master.update_idletasks()
                delimiter = self.delimiter_var.get()
                decimal_sign = self.decimal_var.get()
                df = pd.read_csv(filename, delimiter=delimiter, decimal=decimal_sign)
                self.status_var.set("CSV bestand geladen.")
            else:
                messagebox.showerror("Fout", "Ongeldig bestandstype geselecteerd.")
                return
            self.loaded_data = df
            self.load_columns(df)
        except Exception as e:
            logging.error(f"Fout bij het converteren of laden van data: {e}")
            messagebox.showerror("Fout", f"Er is een fout opgetreden: {e}")
            self.status_var.set("Fout bij het converteren of laden van data.")

    def load_columns(self, df):
        try:
            combined_names = df.columns.tolist()
            if combined_names:
                self.time_column_var.set(combined_names[0])
            else:
                messagebox.showerror("Fout", "Geen kolommen gevonden in het bestand.")
                return
            self.time_menu['menu'].delete(0, 'end')
            for col in combined_names:
                self.time_menu['menu'].add_command(label=col, command=lambda c=col: self.update_time_column(c))
            for widget in self.column_frame.winfo_children():
                widget.destroy()
            self.axis_selection_vars.clear()
            for col in combined_names:
                if col != self.time_column_var.get():
                    frame = tk.Frame(self.column_frame)
                    frame.pack(anchor='w', pady=2)
                    tk.Label(frame, text=col).pack(side=tk.LEFT)
                    axis_var = tk.StringVar(value='None')
                    self.axis_selection_vars[col] = axis_var
                    axis_menu = tk.OptionMenu(frame, axis_var, 'None', 'Primair', 'Secundair')
                    axis_menu.pack(side=tk.LEFT)
            self.show_data_preview(df)
            self.update_datetime_widgets(df)
            logging.info("Kolommen succesvol geladen en verwerkt.")
        except Exception as e:
            logging.error(f"Fout bij het laden van kolommen: {e}")
            messagebox.showerror("Fout", f"Kan kolommen niet laden: {e}")

    def show_data_preview(self, df):
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        try:
            preview_data = df.head(10).to_string()
            preview_text = tk.Text(self.preview_frame, height=10, width=100)
            preview_text.pack()
            preview_text.insert(tk.END, preview_data)
            preview_text.config(state='disabled')
        except Exception as e:
            logging.warning(f"Kan data preview niet tonen: {e}")

    def update_time_column(self, col):
        self.time_column_var.set(col)
        self.update_datetime_widgets(self.loaded_data)

    def update_datetime_widgets(self, df):
        try:
            time_column = self.time_column_var.get()
            if time_column not in df.columns:
                messagebox.showerror("Fout", f"Tijdkolom '{time_column}' bestaat niet in de data.")
                return
            df[time_column] = pd.to_datetime(df[time_column], dayfirst=True, errors='coerce')
            df = df.dropna(subset=[time_column])
            if df.empty:
                messagebox.showwarning("Waarschuwing", "De tijdkolom bevat geen valide datums.")
                return
            min_datetime = df[time_column].min()
            max_datetime = df[time_column].max()
            self.start_date_entry.set_date(min_datetime.date())
            self.start_hour_spinbox.delete(0, tk.END)
            self.start_hour_spinbox.insert(0, f"{min_datetime.hour:02}")
            self.start_minute_spinbox.delete(0, tk.END)
            self.start_minute_spinbox.insert(0, f"{min_datetime.minute:02}")
            self.start_second_spinbox.delete(0, tk.END)
            self.start_second_spinbox.insert(0, f"{min_datetime.second:02}")
            self.end_date_entry.set_date(max_datetime.date())
            self.end_hour_spinbox.delete(0, tk.END)
            self.end_hour_spinbox.insert(0, f"{max_datetime.hour:02}")
            self.end_minute_spinbox.delete(0, tk.END)
            self.end_minute_spinbox.insert(0, f"{max_datetime.minute:02}")
            self.end_second_spinbox.delete(0, tk.END)
            self.end_second_spinbox.insert(0, f"{max_datetime.second:02}")
        except Exception as e:
            logging.error(f"Fout bij het instellen van start- en einddatum: {e}")
            messagebox.showerror("Fout", f"Kan start- en einddatum niet instellen: {e}")

    def select_logo(self):
        logo_path = filedialog.askopenfilename(
            title="Selecteer het logo-bestand",
            filetypes=(("Afbeeldingsbestanden", "*.png;*.jpg;*.jpeg;*.gif"), ("Alle bestanden", "*.*"))
        )
        if logo_path:
            self.logo_entry.delete(0, tk.END)
            self.logo_entry.insert(0, logo_path)
            self.logo_path_var.set(logo_path)

    def generate_plot(self):
        """
        Genereert de interactieve Plotly-grafiek en slaat deze op als HTML, en exporteert daarnaast
        de gefilterde data (voor de geselecteerde periode en parameters) als CSV. Hierbij wordt:
          - Temperatuur en pH afgerond op 2 decimalen,
          - Diepte op 3 decimalen,
          - Conductiviteit en Turbiditeit afgerond op 0 decimalen.
          
        De waarden die bij mouse-over getoond worden (hover text) worden eveneens afgerond volgens deze specificaties.
        """
        if not hasattr(self, 'loaded_data'):
            messagebox.showwarning("Waarschuwing", "Laad eerst data door een bestand te converteren.")
            return

        df = self.loaded_data
        delimiter = self.delimiter_var.get()
        decimal_sign = self.decimal_var.get()
        time_column = self.time_column_var.get()

        selected_columns_primary = [col for col, var in self.axis_selection_vars.items() if var.get() == 'Primair']
        selected_columns_secondary = [col for col, var in self.axis_selection_vars.items() if var.get() == 'Secundair']
        if not selected_columns_primary and not selected_columns_secondary:
            messagebox.showwarning("Waarschuwing", "Selecteer minstens één kolom om te plotten.")
            return

        try:
            # Converteer tijdkolom
            try:
                df[time_column] = pd.to_datetime(df[time_column], dayfirst=True, errors='coerce')
                if df[time_column].isnull().all():
                    messagebox.showerror("Fout", "Kan de tijdkolom niet converteren naar datetime-formaat. Controleer het datum-tijdformaat.")
                    return
            except Exception as e:
                messagebox.showerror("Fout", f"Er is een fout opgetreden bij het parsen van de tijdkolom: {e}")
                return

            # Bepaal start- en einddatetime
            start_date = self.start_date_entry.get_date()
            start_hour = int(self.start_hour_spinbox.get())
            start_minute = int(self.start_minute_spinbox.get())
            start_second = int(self.start_second_spinbox.get())
            start_datetime = datetime.combine(start_date, datetime.min.time()).replace(
                hour=start_hour, minute=start_minute, second=start_second
            )
            end_date = self.end_date_entry.get_date()
            end_hour = int(self.end_hour_spinbox.get())
            end_minute = int(self.end_minute_spinbox.get())
            end_second = int(self.end_second_spinbox.get())
            end_datetime = datetime.combine(end_date, datetime.min.time()).replace(
                hour=end_hour, minute=end_minute, second=end_second
            )

            # Filter data op de geselecteerde periode
            df_filtered = df[(df[time_column] >= start_datetime) & (df[time_column] <= end_datetime)]
            if df_filtered.empty:
                messagebox.showwarning("Waarschuwing", "Geen data beschikbaar binnen de geselecteerde periode.")
                return

            # Definieer een mapping voor hover-afronding:
            hover_rounding = {
                "Temperatuur (°C)": 2,
                "pH": 2,
                "Diepte (m)": 3,
                "SPEC_CNDC": 0,
                "Conductiviteit (`µS)": 0,
                "Turbiditeit (NTU)": 0
            }

            # Bouw de Plotly-grafiek
            fig = go.Figure()
            primary_color = 'blue'
            secondary_color = 'green'

            for param in selected_columns_primary:
                try:
                    y_values = df_filtered[param]
                    round_spec = hover_rounding.get(param, None)
                    if round_spec is not None:
                        hovertemplate = "%{x}<br>" + f"{param}: %{{y:.{round_spec}f}}<extra></extra>"
                    else:
                        hovertemplate = "%{x}<br>" + f"{param}: %{{y}}<extra></extra>"
                    trace_name = f'{param} (Primair)'
                    fig.add_trace(go.Scatter(
                        x=df_filtered[time_column],
                        y=y_values,
                        mode='lines',
                        name=trace_name,
                        yaxis='y1',
                        line=dict(),
                        hovertemplate=hovertemplate
                    ))
                except Exception as e:
                    logging.warning(f"Kan kolom '{param}' niet plotten: {e}")
                    messagebox.showwarning("Waarschuwing", f"Kan kolom '{param}' niet plotten: {e}")
                    continue

            for param in selected_columns_secondary:
                try:
                    y_values = df_filtered[param]
                    round_spec = hover_rounding.get(param, None)
                    if round_spec is not None:
                        hovertemplate = "%{x}<br>" + f"{param}: %{{y:.{round_spec}f}}<extra></extra>"
                    else:
                        hovertemplate = "%{x}<br>" + f"{param}: %{{y}}<extra></extra>"
                    trace_name = f'{param} (Secundair)'
                    fig.add_trace(go.Scatter(
                        x=df_filtered[time_column],
                        y=y_values,
                        mode='lines',
                        name=trace_name,
                        yaxis='y2',
                        line=dict(),
                        hovertemplate=hovertemplate
                    ))
                except Exception as e:
                    logging.warning(f"Kan kolom '{param}' niet plotten: {e}")
                    messagebox.showwarning("Waarschuwing", f"Kan kolom '{param}' niet plotten: {e}")
                    continue

            y1_title = "Waarden (Primair)"
            y2_title = "Waarden (Secundair)"
            fig.update_layout(
                xaxis_title="Tijd",
                yaxis_title=y1_title,
                yaxis=dict(
                    title=dict(text=y1_title, font=dict(family="Arial Black, sans-serif", size=14, color=primary_color)),
                    side="left",
                    tickfont=dict(family="Arial", size=12, color=primary_color),
                    autorange=True
                ),
                yaxis2=dict(
                    title=dict(text=y2_title, font=dict(family="Arial Black, sans-serif", size=14, color=secondary_color)),
                    overlaying='y',
                    side='right',
                    tickfont=dict(family="Arial", size=12, color=secondary_color),
                    autorange=True
                ),
                legend_title=dict(text="Parameters", font=dict(family="Arial Black, sans-serif", size=14, color="Black")),
                legend=dict(itemsizing='constant'),
                # Belangrijk: zet hovermode op "closest" zodat de individuele hovertemplate wordt toegepast
                hovermode="closest",
                hoverlabel=dict(font_size=14, font_family="Arial")
            )

            annotations = []
            info_text = (f"Projectnummer: {self.get_project_number()}<br>"
                         f"Naam: {self.get_creator_name()}<br>"
                         f"Datum: {self.get_creation_date()}<br>"
                         f"Locatie: {self.get_location_name()}<br>"
                         f"Periode: {start_datetime.strftime('%d/%m/%Y %H:%M:%S')} - {end_datetime.strftime('%d/%m/%Y %H:%M:%S')}")
            annotations.append(dict(
                x=0.5,
                y=1.15,
                xref='paper',
                yref='paper',
                text=info_text,
                showarrow=False,
                font=dict(size=16, color='black'),
                xanchor='center',
                yanchor='top'
            ))

            logo_path = self.logo_path_var.get()
            if logo_path and os.path.exists(logo_path):
                try:
                    with Image.open(logo_path) as img:
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        encoded_image = base64.b64encode(buffered.getvalue()).decode()
                    fig.add_layout_image(
                        dict(
                            source=f'data:image/png;base64,{encoded_image}',
                            xref="paper", yref="paper",
                            x=0.99, y=1.15,
                            sizex=0.3, sizey=0.3,
                            xanchor="right", yanchor="top"
                        )
                    )
                except Exception as e:
                    logging.warning(f"Kan logo niet toevoegen aan de grafiek: {e}")
                    messagebox.showwarning("Waarschuwing", f"Kan logo niet toevoegen aan de grafiek: {e}")

            fig.update_layout(annotations=annotations)
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )

            start_datetime_str = start_datetime.strftime('%Y%m%d_%H%M%S')
            end_datetime_str = end_datetime.strftime('%Y%m%d_%H%M%S')
            safe_location_name = re.sub(r'[^\w\-_\. ]', '_', self.get_location_name().strip())[:50]
            html_filename = f"{start_datetime_str}_{end_datetime_str}_{safe_location_name}.html"
            output_dir = os.path.dirname(self.file_path.get())
            output_html_path = os.path.join(output_dir, html_filename)
            fig.write_html(output_html_path)
            logging.info(f"Grafiek succesvol opgeslagen als {output_html_path}")

            # CSV export met afronding
            export_cols = [col for col in df_filtered.columns if col == time_column or col in (selected_columns_primary + selected_columns_secondary)]
            export_map = {
                time_column: ("Date Time", "D/M/Y HH:MM:SS", None),
                "Temperatuur (°C)": ("Temp", "C", 2),
                "SPEC_CNDC": ("SpCond", "uS", 0),
                "Conductiviteit (`µS)": ("SpCond", "uS", 0),
                "PSAL": ("Sal", "ppt", None),
                "Diepte (m)": ("Depth", "meters", 3),
                "pH": ("pH", "", 2),
                "ORP": ("pH", "mV", None),
                "Turbiditeit (NTU)": ("Turbid+", "NTU", 0),
                "BAT_VOLT": ("Battery", "volts", None)
            }
            header_row = []
            unit_row = []
            for col in export_cols:
                if col in export_map:
                    header_name, unit, _ = export_map[col]
                else:
                    header_name, unit = col, ""
                header_row.append(header_name)
                unit_row.append(unit)
            
            export_df = df_filtered[export_cols].copy()
            for col in export_df.columns:
                if col in export_map:
                    rounding_spec = export_map[col][2]
                    if rounding_spec is not None and pd.api.types.is_numeric_dtype(export_df[col]):
                        export_df[col] = export_df[col].round(rounding_spec)
                        if rounding_spec == 0:
                            export_df[col] = export_df[col].astype('Int64')

            output_csv_path = os.path.join(output_dir, os.path.splitext(html_filename)[0] + ".csv")
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                writer.writerow(header_row)
                writer.writerow(unit_row)
                for _, row in export_df.iterrows():
                    writer.writerow(row.tolist())

            messagebox.showinfo("Succes", f"De interactieve grafiek is opgeslagen als:\n{output_html_path}\n\nEn de geëxporteerde CSV als:\n{output_csv_path}")
            logging.info(f"CSV export succesvol opgeslagen als {output_csv_path}")

        except Exception as e:
            logging.error(f"Fout bij het genereren van de grafiek of export: {e}")
            messagebox.showerror("Fout", f"Er is een fout opgetreden: {e}")

    def sample_data_to_dataframe(self, sample_data):
        data_dict = {}
        if sample_data['dimensions']:
            num_records = len(sample_data['dimensions'][0]['data'])
        else:
            num_records = 0
        logging.info(f"Aantal records om te schrijven: {num_records}")
        for dim in sample_data['dimensions']:
            if dim['name'].upper() == 'TIME':
                time_seconds = dim['data']
                data_dict['TIME'] = TimeConverter.convert_time(time_seconds)
                logging.debug(f"TIME geconverteerd: {data_dict['TIME'][:5]}")
        for var in sample_data['variables']:
            var_name = var['name']
            var_data = var['data']
            if isinstance(var_data, np.ndarray):
                data_dict[var_name] = var_data
                logging.debug(f"Variabele '{var_name}' toegevoegd met vorm {var_data.shape}")
            else:
                data_length = len(data_dict['TIME']) if 'TIME' in data_dict else 1
                data_dict[var_name] = [var_data] * data_length
                logging.debug(f"Variabele '{var_name}' toegevoegd als scalar gerepliceerd naar lengte {data_length}")
        df = pd.DataFrame(data_dict)
        logging.debug(f"DataFrame gecreëerd met kolommen: {df.columns}")
        return df

    def get_project_number(self):
        return self.project_entry.get()

    def get_creator_name(self):
        return self.creator_entry.get()

    def get_creation_date(self):
        return self.creation_date_entry.get()

    def get_location_name(self):
        return self.location_entry.get()

# ------------------- Main Execution -------------------

def main():
    if not pillow_version_at_least(8, 0, 0):
        messagebox.showerror("Dependency Error", "Pillow versie 8.0.0 of hoger is vereist.")
        return
    root = tk.Tk()
    app = IntegratedGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
