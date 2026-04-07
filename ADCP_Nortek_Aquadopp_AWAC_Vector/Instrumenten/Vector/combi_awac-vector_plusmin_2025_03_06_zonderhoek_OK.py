import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkcalendar import DateEntry
import os
from datetime import datetime, timedelta
import re

# Globale opslag zodat de GUI data behoudt
global_data = {}

# ---------------------------------------
# FUNCTIES VOOR VECTOR SENSOR VERWERKING
# ---------------------------------------

def get_transformation_matrix_from_hdr(hdr_file_path):
    try:
        with open(hdr_file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het lezen van het HDR bestand: {e}")
        raise
    for i, line in enumerate(lines):
        if "Transformation matrix" in line:
            parts = line.split("Transformation matrix")[-1].strip().split()
            if parts:
                try:
                    row1 = [float(x) for x in parts]
                except Exception as e:
                    messagebox.showerror("Fout", f"Fout bij het parsen van de eerste rij: {e}")
                    raise
            else:
                row1 = []
            try:
                row2 = [float(x) for x in lines[i+1].strip().split()]
                row3 = [float(x) for x in lines[i+2].strip().split()]
            except Exception as e:
                messagebox.showerror("Fout", f"Fout bij het parsen van de transformatie matrix: {e}")
                raise
            if len(row1) < 3:
                needed = 3 - len(row1)
                row1_extra = row2[:needed]
                row1.extend(row1_extra)
                row2 = row2[needed:]
            transformation_matrix = np.array([row1, row2, row3])
            return transformation_matrix
    raise ValueError("Transformatie matrix niet gevonden in het HDR bestand.")

def angle_in_range(angle, lower, upper):
    if lower <= upper:
        return (angle >= lower) and (angle <= upper)
    else:
        return (angle >= lower) or (angle <= upper)

def load_vector_data(dat_file, sen_file):
    dat_columns = [
        'Burst_counter', 'Ensemble_counter', 'Velocity_Beam1', 'Velocity_Beam2', 'Velocity_Beam3',
        'Amplitude_Beam1', 'Amplitude_Beam2', 'Amplitude_Beam3', 'SNR_Beam1', 'SNR_Beam2', 'SNR_Beam3',
        'Correlation_Beam1', 'Correlation_Beam2', 'Correlation_Beam3', 'Pressure', 'Analog_input1',
        'Analog_input2', 'Checksum'
    ]
    sen_columns = [
        'Month', 'Day', 'Year', 'Hour', 'Minute', 'Second', 'Error_code', 'Status_code', 'Battery_voltage',
        'Soundspeed', 'Heading', 'Pitch', 'Roll', 'Temperature', 'Analog_input', 'Checksum'
    ]
    try:
        df_dat = pd.read_csv(dat_file, sep='\s+', header=None, names=dat_columns, comment='#')
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het lezen van het .dat bestand: {e}")
        raise
    try:
        df_sen = pd.read_csv(sen_file, sep='\s+', header=None, names=sen_columns, comment='#')
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het lezen van het .sen bestand: {e}")
        raise
    # Trim naar de kortste dataset
    min_len = min(len(df_dat), len(df_sen))
    df_dat = df_dat.iloc[:min_len].reset_index(drop=True)
    df_sen = df_sen.iloc[:min_len].reset_index(drop=True)
    try:
        df_sen['Datetime'] = pd.to_datetime(df_sen[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het creëren van de Datetime kolom: {e}")
        raise
    df_dat['Datetime'] = df_sen['Datetime']
    # Gebruik alleen geldige metingen (Checksum == 0)
    df_vector = df_dat[df_dat['Checksum'] == 0].copy()
    # Bereken de snelheidsgrootte en de originele richting
    df_vector['Resultant_Speed'] = np.sqrt(df_vector['Velocity_Beam1']**2 +
                                            df_vector['Velocity_Beam2']**2 +
                                            df_vector['Velocity_Beam3']**2)
    df_vector['Direction'] = (np.degrees(np.arctan2(df_vector['Velocity_Beam2'], df_vector['Velocity_Beam1'])) + 360) % 360
    return df_vector

def transform_vector_velocities(df_vector, transformation_matrix):
    V_beam = df_vector[['Velocity_Beam1', 'Velocity_Beam2', 'Velocity_Beam3']].values
    V_ENU = V_beam.dot(transformation_matrix.T)
    df_vector['Velocity_East'] = V_ENU[:, 0]
    df_vector['Velocity_North'] = V_ENU[:, 1]
    df_vector['Velocity_Up'] = V_ENU[:, 2]
    return df_vector

def calculate_vector_signed_speed(df_vector, pos_range, neg_range):
    # Gebruik de huidige Direction om te bepalen of de snelheid positief of negatief is
    def get_sign(angle):
        if angle_in_range(angle, neg_range[0], neg_range[1]):
            return -1
        elif angle_in_range(angle, pos_range[0], pos_range[1]):
            return 1
        else:
            return 1
    df_vector['Velocity_Channel'] = df_vector['Resultant_Speed'] * df_vector['Direction'].apply(get_sign)
    return df_vector

# ---------------------------------------
# FUNCTIES VOOR AWAC SENSOR VERWERKING
# ---------------------------------------

def load_awac_data(csv_file):
    try:
        df_awac = pd.read_csv(csv_file, sep=';', header=0, encoding='utf-8')
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het lezen van het AWAC CSV bestand: {e}")
        raise
    df_awac.columns = df_awac.columns.str.strip()
    if 'DateTime' not in df_awac.columns:
        messagebox.showerror("Fout", "Kolom 'DateTime' ontbreekt in het AWAC CSV bestand.")
        raise ValueError("Kolom 'DateTime' ontbreekt.")
    try:
        df_awac['DateTime'] = pd.to_datetime(df_awac['DateTime'], format='%d/%m/%Y %H:%M:%S')
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het converteren van 'DateTime' in AWAC data: {e}")
        raise
    return df_awac

def signed_speed(speed, direction, pos_min, pos_max):
    direction = direction % 360
    if pos_min <= pos_max:
        mask_positive = (direction >= pos_min) & (direction <= pos_max)
    else:
        mask_positive = (direction >= pos_min) | (direction <= pos_max)
    return np.where(mask_positive, speed, -speed)

def get_awac_cells(df_awac):
    # Zoek kolommen die beginnen met 'Speed#' en 'Dir#'
    speed_cols = [col for col in df_awac.columns if col.startswith('Speed#')]
    dir_cols = [col for col in df_awac.columns if col.startswith('Dir#')]
    speed_cols.sort()
    dir_cols.sort()
    depth_pattern = re.compile(r'\((\d+(?:\.\d+)?)m\)')
    cells = {}
    for sp_col, d_col in zip(speed_cols, dir_cols):
        match = depth_pattern.search(sp_col)
        if match:
            depth = float(match.group(1))
            key = f"{depth} m"
        else:
            key = sp_col
        cells[key] = (sp_col, d_col)
    return cells

def process_awac_data_by_cell(df_awac, pos_min, pos_max, cell_key, cells):
    if cell_key not in cells:
        raise ValueError("Geselecteerde cel niet gevonden.")
    sp_col, d_col = cells[cell_key]
    speed_data = df_awac[sp_col].astype(float)
    dir_data = df_awac[d_col].astype(float)
    df_awac_processed = df_awac.copy()
    # Bewaar de oorspronkelijke richting
    df_awac_processed['Direction'] = dir_data
    df_awac_processed['Speed'] = speed_data
    # Bereken de signed snelheid (op basis van de huidige richting)
    df_awac_processed['signed_speed'] = signed_speed(speed_data, dir_data, pos_min, pos_max)
    return df_awac_processed

# ---------------------------------------
# RESAMPLING & MOVING AVERAGE
# ---------------------------------------

def resample_and_average(df, time_col, value_col, new_index, window_size):
    df = df.set_index(time_col)
    df_resampled = df.reindex(new_index).interpolate(method='time')
    df_resampled[f'{value_col}_avg'] = df_resampled[value_col].rolling(window=window_size, center=True, min_periods=1).mean()
    return df_resampled

# ---------------------------------------
# DE GUI MET GRAFIEKOPTIES & EXPORT
# ---------------------------------------

class SensorComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vergelijking Sensor Data")
        self.global_data = {}  # hier worden de geladen data en resultaten opgeslagen

        # --- Frame: Bestanden en Instellingen ---
        self.frame_files = ttk.LabelFrame(root, text="Bestanden en Instellingen", padding="10")
        self.frame_files.grid(row=0, column=0, padx=10, pady=5, sticky="ew")

        # Vector sensor
        ttk.Label(self.frame_files, text="Vector .dat:").grid(row=0, column=0, sticky="w")
        self.vector_dat_var = tk.StringVar()
        ttk.Entry(self.frame_files, textvariable=self.vector_dat_var, width=50).grid(row=0, column=1)
        ttk.Button(self.frame_files, text="Browse", command=self.browse_vector_dat).grid(row=0, column=2)

        ttk.Label(self.frame_files, text="Vector .sen:").grid(row=1, column=0, sticky="w")
        self.vector_sen_var = tk.StringVar()
        ttk.Entry(self.frame_files, textvariable=self.vector_sen_var, width=50).grid(row=1, column=1)
        ttk.Button(self.frame_files, text="Browse", command=self.browse_vector_sen).grid(row=1, column=2)

        ttk.Label(self.frame_files, text="Vector .hdr:").grid(row=2, column=0, sticky="w")
        self.vector_hdr_var = tk.StringVar()
        ttk.Entry(self.frame_files, textvariable=self.vector_hdr_var, width=50).grid(row=2, column=1)
        ttk.Button(self.frame_files, text="Browse", command=self.browse_vector_hdr).grid(row=2, column=2)

        # AWAC sensor
        ttk.Label(self.frame_files, text="AWAC CSV:").grid(row=3, column=0, sticky="w")
        self.awac_csv_var = tk.StringVar()
        ttk.Entry(self.frame_files, textvariable=self.awac_csv_var, width=50).grid(row=3, column=1)
        ttk.Button(self.frame_files, text="Browse", command=self.browse_awac_csv).grid(row=3, column=2)

        # Tijd
        ttk.Label(self.frame_files, text="Startdatum (YYYY-MM-DD HH:MM:SS):").grid(row=4, column=0, sticky="w")
        self.start_datetime_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ttk.Entry(self.frame_files, textvariable=self.start_datetime_var, width=30).grid(row=4, column=1, sticky="w")
        ttk.Label(self.frame_files, text="Einddatum (YYYY-MM-DD HH:MM:SS):").grid(row=5, column=0, sticky="w")
        self.end_datetime_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        ttk.Entry(self.frame_files, textvariable=self.end_datetime_var, width=30).grid(row=5, column=1, sticky="w")

        # Snelheidsranges
        ttk.Label(self.frame_files, text="Positieve range (min, max):").grid(row=6, column=0, sticky="w")
        self.pos_min_var = tk.StringVar(value="246")
        self.pos_max_var = tk.StringVar(value="67")
        ttk.Entry(self.frame_files, textvariable=self.pos_min_var, width=5).grid(row=6, column=1, sticky="w")
        ttk.Entry(self.frame_files, textvariable=self.pos_max_var, width=5).grid(row=6, column=1, padx=(60,0), sticky="w")
        ttk.Label(self.frame_files, text="Negatieve range (min, max):").grid(row=7, column=0, sticky="w")
        self.neg_min_var = tk.StringVar(value="67")
        self.neg_max_var = tk.StringVar(value="246")
        ttk.Entry(self.frame_files, textvariable=self.neg_min_var, width=5).grid(row=7, column=1, sticky="w")
        ttk.Entry(self.frame_files, textvariable=self.neg_max_var, width=5).grid(row=7, column=1, padx=(60,0), sticky="w")
        
        # Nieuwe velden voor hoekoffsets
        ttk.Label(self.frame_files, text="Vector Hoekoffset (°):").grid(row=8, column=0, sticky="w")
        self.vector_offset_var = tk.StringVar(value="0")
        ttk.Entry(self.frame_files, textvariable=self.vector_offset_var, width=5).grid(row=8, column=1, sticky="w")
        
        ttk.Label(self.frame_files, text="AWAC Hoekoffset (°):").grid(row=9, column=0, sticky="w")
        self.awac_offset_var = tk.StringVar(value="0")
        ttk.Entry(self.frame_files, textvariable=self.awac_offset_var, width=5).grid(row=9, column=1, sticky="w")
        
        # Moving average windows
        ttk.Label(self.frame_files, text="Vector MA window:").grid(row=10, column=0, sticky="w")
        self.vector_ma_var = tk.StringVar(value="5")
        ttk.Entry(self.frame_files, textvariable=self.vector_ma_var, width=5).grid(row=10, column=1, sticky="w")
        ttk.Label(self.frame_files, text="AWAC MA window:").grid(row=11, column=0, sticky="w")
        self.awac_ma_var = tk.StringVar(value="5")
        ttk.Entry(self.frame_files, textvariable=self.awac_ma_var, width=5).grid(row=11, column=1, sticky="w")

        # Laad data knop
        ttk.Button(self.frame_files, text="Laad Data", command=self.load_data).grid(row=12, column=0, columnspan=3, pady=10)

        # --- Frame: Grafiek & Export ---
        self.frame_graph = ttk.LabelFrame(root, text="Grafiek & Export", padding="10")
        self.frame_graph.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        ttk.Label(self.frame_graph, text="Selecteer AWAC cel:").grid(row=0, column=0, sticky="w")
        self.awac_cell_var = tk.StringVar()
        self.awac_cell_dropdown = ttk.Combobox(self.frame_graph, textvariable=self.awac_cell_var, state="readonly", width=20)
        self.awac_cell_dropdown.grid(row=0, column=1, sticky="w")
        ttk.Button(self.frame_graph, text="Genereer Grafiek", command=self.generate_graph).grid(row=1, column=0, pady=5)
        ttk.Button(self.frame_graph, text="Exporteer Vergelijkde Data", command=self.export_data).grid(row=1, column=1, pady=5)
        ttk.Button(self.frame_graph, text="Afsluiten", command=self.root.quit).grid(row=1, column=2, pady=5)

    # --- File Browsing functies ---
    def browse_vector_dat(self):
        filename = filedialog.askopenfilename(title="Selecteer .dat bestand", filetypes=[("DAT files", "*.dat"), ("All files", "*.*")])
        if filename:
            self.vector_dat_var.set(filename)
    def browse_vector_sen(self):
        filename = filedialog.askopenfilename(title="Selecteer .sen bestand", filetypes=[("SEN files", "*.sen"), ("All files", "*.*")])
        if filename:
            self.vector_sen_var.set(filename)
    def browse_vector_hdr(self):
        filename = filedialog.askopenfilename(title="Selecteer .hdr bestand", filetypes=[("HDR files", "*.hdr"), ("All files", "*.*")])
        if filename:
            self.vector_hdr_var.set(filename)
    def browse_awac_csv(self):
        filename = filedialog.askopenfilename(title="Selecteer AWAC CSV bestand", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filename:
            self.awac_csv_var.set(filename)

    # --- Data laden ---
    def load_data(self):
        try:
            start_dt = datetime.strptime(self.start_datetime_var.get(), "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(self.end_datetime_var.get(), "%Y-%m-%d %H:%M:%S")
            if start_dt >= end_dt:
                messagebox.showerror("Fout", "Startdatum/tijd moet vóór einddatum/tijd liggen.")
                return
            pos_min = float(self.pos_min_var.get()) % 360
            pos_max = float(self.pos_max_var.get()) % 360
            neg_min = float(self.neg_min_var.get()) % 360
            neg_max = float(self.neg_max_var.get()) % 360
        except Exception as e:
            messagebox.showerror("Fout", f"Fout in invoer: {e}")
            return

        # Vector sensor data laden
        try:
            df_vector = load_vector_data(self.vector_dat_var.get(), self.vector_sen_var.get())
            df_vector['Datetime'] = pd.to_datetime(df_vector['Datetime'])
            df_vector = df_vector[(df_vector['Datetime'] >= start_dt) & (df_vector['Datetime'] <= end_dt)].copy()
            if df_vector.empty:
                messagebox.showerror("Fout", "Geen vector data binnen het gekozen tijdsinterval.")
                return
            transformation_matrix = get_transformation_matrix_from_hdr(self.vector_hdr_var.get())
            df_vector = transform_vector_velocities(df_vector, transformation_matrix)
            # Bewaar de originele richting voor dynamische offset-toepassing
            df_vector['Original_Direction'] = df_vector['Direction']
        except Exception as e:
            messagebox.showerror("Fout", f"Fout bij vector data: {e}")
            return

        # AWAC sensor data laden
        try:
            df_awac = load_awac_data(self.awac_csv_var.get())
            df_awac = df_awac[(df_awac['DateTime'] >= start_dt) & (df_awac['DateTime'] <= end_dt)].copy()
            if df_awac.empty:
                messagebox.showerror("Fout", "Geen AWAC data binnen het gekozen tijdsinterval.")
                return
            cells = get_awac_cells(df_awac)
            if not cells:
                messagebox.showerror("Fout", "Geen cellen (Speed#/Dir#) gevonden in AWAC data.")
                return
        except Exception as e:
            messagebox.showerror("Fout", f"Fout bij AWAC data: {e}")
            return

        self.global_data['df_vector'] = df_vector
        self.global_data['df_awac'] = df_awac
        self.global_data['cells'] = cells

        # Vul de dropdown voor AWAC celselectie
        cell_keys = list(cells.keys())
        self.awac_cell_dropdown['values'] = cell_keys
        self.awac_cell_var.set(cell_keys[0])
        messagebox.showinfo("Info", "Data succesvol geladen.")

    # --- Grafiek genereren ---
    def generate_graph(self):
        if not self.global_data:
            messagebox.showerror("Fout", "Laad eerst de data!")
            return
        try:
            vector_ma = int(self.vector_ma_var.get())
            awac_ma = int(self.awac_ma_var.get())
            pos_min = float(self.pos_min_var.get()) % 360
            pos_max = float(self.pos_max_var.get()) % 360
            neg_min = float(self.neg_min_var.get()) % 360
            neg_max = float(self.neg_max_var.get()) % 360
        except Exception as e:
            messagebox.showerror("Fout", f"Fout in averaging instellingen: {e}")
            return

        # Pas de vector offset dynamisch toe
        df_vector = self.global_data['df_vector'].copy()
        try:
            vector_offset = float(self.vector_offset_var.get()) % 360
        except Exception as e:
            messagebox.showerror("Fout", f"Fout bij verwerken vector hoekoffset: {e}")
            return
        df_vector['Direction'] = (df_vector['Original_Direction'] + vector_offset) % 360
        df_vector = calculate_vector_signed_speed(df_vector, (pos_min, pos_max), (neg_min, neg_max))

        # Verwerk AWAC data en pas dynamisch de AWAC offset toe
        df_awac = self.global_data['df_awac']
        cells = self.global_data['cells']
        selected_cell = self.awac_cell_var.get()

        try:
            df_awac_processed = process_awac_data_by_cell(df_awac.copy(), pos_min, pos_max, selected_cell, cells)
            try:
                awac_offset = float(self.awac_offset_var.get()) % 360
            except Exception as e:
                messagebox.showerror("Fout", f"Fout bij verwerken AWAC hoekoffset: {e}")
                return
            df_awac_processed['Direction'] = (df_awac_processed['Direction'] + awac_offset) % 360
            df_awac_processed['signed_speed'] = signed_speed(df_awac_processed['Speed'], df_awac_processed['Direction'], pos_min, pos_max)
        except Exception as e:
            messagebox.showerror("Fout", f"Fout bij verwerken AWAC data: {e}")
            return

        # Bepaal de gemeenschappelijke tijdsreeks
        start_common = max(df_vector['Datetime'].min(), df_awac_processed['DateTime'].min())
        end_common = min(df_vector['Datetime'].max(), df_awac_processed['DateTime'].max())
        common_index = pd.date_range(start=start_common, end=end_common, freq='1S')

        # Resample & Moving Averages
        df_vector_speed = resample_and_average(df_vector, 'Datetime', 'Resultant_Speed', common_index, vector_ma)
        df_awac_speed = resample_and_average(df_awac_processed, 'DateTime', 'Speed', common_index, awac_ma)
        df_vector_signed = resample_and_average(df_vector, 'Datetime', 'Velocity_Channel', common_index, vector_ma)
        df_awac_signed = resample_and_average(df_awac_processed, 'DateTime', 'signed_speed', common_index, awac_ma)
        df_vector_dir = resample_and_average(df_vector, 'Datetime', 'Direction', common_index, vector_ma)
        df_awac_dir = resample_and_average(df_awac_processed, 'DateTime', 'Direction', common_index, awac_ma)

        # --- Tijdshift-optimalisatie voor signed snelheden ---
        time_shifts = np.arange(-30, 31, 1)
        best_r2_time = -np.inf
        best_shift = 0
        for shift in time_shifts:
            shifted_index = common_index + pd.Timedelta(seconds=shift)
            awac_shifted = df_awac_signed.reindex(shifted_index).interpolate(method='time')
            valid_mask = (~df_vector_signed['Velocity_Channel_avg'].isna()) & (~awac_shifted['signed_speed_avg'].isna())
            if valid_mask.sum() > 0:
                r = np.corrcoef(df_vector_signed.loc[valid_mask, 'Velocity_Channel_avg'],
                                awac_shifted.loc[valid_mask, 'signed_speed_avg'])[0,1]
                r2 = r**2
                if r2 > best_r2_time:
                    best_r2_time = r2
                    best_shift = shift

        best_shifted_index = common_index + pd.Timedelta(seconds=best_shift)
        df_awac_signed_shifted = df_awac_signed.reindex(best_shifted_index).interpolate(method='time')

        # --- Grafieken ---
        fig, axs = plt.subplots(3, 3, figsize=(18, 20))
        fig.suptitle("Vergelijking van Sensor Data (met dynamische hoekoffset)", fontsize=16)

        # Subplot (0,0): Vector snelheid (magnitude)
        axs[0,0].plot(common_index, df_vector_speed['Resultant_Speed_avg'], color='blue')
        axs[0,0].set_title("Vector Snelheid (MA)")
        axs[0,0].set_ylabel("Snelheid (m/s)")
        axs[0,0].grid(True)

        # Subplot (0,1): AWAC snelheid (magnitude)
        axs[0,1].plot(common_index, df_awac_speed['Speed_avg'], color='red')
        axs[0,1].set_title("AWAC Snelheid (MA)")
        axs[0,1].set_ylabel("Snelheid (m/s)")
        axs[0,1].grid(True)

        # Subplot (0,2): Gecombineerde snelheden (origineel)
        axs[0,2].plot(common_index, df_vector_speed['Resultant_Speed_avg'], label="Vector Snelheid", color='blue')
        axs[0,2].plot(common_index, df_awac_speed['Speed_avg'], label="AWAC Snelheid", color='red')
        axs[0,2].set_title("Gecombineerde Snelheden (Origineel)")
        axs[0,2].set_ylabel("Snelheid (m/s)")
        axs[0,2].legend()
        axs[0,2].grid(True)

        # Subplot (1,0): Vector signed snelheden over tijd
        axs[1,0].plot(common_index, df_vector_signed['Velocity_Channel_avg'], color='blue')
        axs[1,0].set_title("Vector Signed Snelheden (MA)")
        axs[1,0].set_ylabel("Snelheid (m/s)")
        axs[1,0].grid(True)

        # Subplot (1,1): AWAC signed snelheden over tijd (origineel)
        axs[1,1].plot(common_index, df_awac_signed['signed_speed_avg'], color='red')
        axs[1,1].set_title("AWAC Signed Snelheden (MA)")
        axs[1,1].set_ylabel("Snelheid (m/s)")
        axs[1,1].grid(True)

        # Subplot (1,2): Vergelijking signed snelheden met optimale tijdshift
        axs[1,2].plot(common_index, df_vector_signed['Velocity_Channel_avg'], label="Vector Signed", color='blue')
        axs[1,2].plot(common_index, df_awac_signed_shifted['signed_speed_avg'], label=f"AWAC Signed (Shift = {best_shift} s)", color='orange')
        axs[1,2].set_title("Tijdlijn Signed Snelheden\n(Beste R² = {:.3f})".format(best_r2_time))
        axs[1,2].set_xlabel("Tijd")
        axs[1,2].set_ylabel("Snelheid (m/s)")
        axs[1,2].legend()
        axs[1,2].grid(True)

        # Subplot (2,0): Vector richting (origineel)
        axs[2,0].plot(common_index, df_vector_dir['Direction_avg'], color='blue')
        axs[2,0].set_title("Vector Richting (MA)")
        axs[2,0].set_ylabel("Richting (°)")
        axs[2,0].grid(True)

        # Subplot (2,1): AWAC richting (origineel)
        axs[2,1].plot(common_index, df_awac_dir['Direction_avg'], color='red')
        axs[2,1].set_title("AWAC Richting (MA)")
        axs[2,1].set_ylabel("Richting (°)")
        axs[2,1].grid(True)

        # Subplot (2,2): Scatter van signed snelheden
        axs[2,2].scatter(df_vector_signed['Velocity_Channel_avg'], df_awac_signed['signed_speed_avg'], color='green', alpha=0.6)
        axs[2,2].plot([df_vector_signed['Velocity_Channel_avg'].min(), df_vector_signed['Velocity_Channel_avg'].max()],
                       [df_vector_signed['Velocity_Channel_avg'].min(), df_vector_signed['Velocity_Channel_avg'].max()],
                       'k--', label="1:1 lijn")
        axs[2,2].set_title("Scatter Signed Snelheden (Plus/Min)")
        axs[2,2].set_xlabel("Vector Signed (m/s)")
        axs[2,2].set_ylabel("AWAC Signed (m/s)")
        axs[2,2].legend()
        axs[2,2].grid(True)

        # Zorg dat de tijd-as leesbaar is
        for ax in [axs[1,2]]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S'))
        fig.autofmt_xdate()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Sla de gesynchroniseerde data op voor export
        self.global_data['df_compared'] = pd.DataFrame({
            'Time': common_index,
            'Vector_Speed_MA': df_vector_speed['Resultant_Speed_avg'],
            'AWAC_Speed_MA': df_awac_speed['Speed_avg'],
            'Vector_Signed_MA': df_vector_signed['Velocity_Channel_avg'],
            'AWAC_Signed_MA': df_awac_signed['signed_speed_avg'],
            'AWAC_Signed_MA_TimeShifted': df_awac_signed_shifted['signed_speed_avg'],
            'Vector_Direction_MA': df_vector_dir['Direction_avg'],
            'AWAC_Direction_MA': df_awac_dir['Direction_avg']
        })

    # --- Data exporteren ---
    def export_data(self):
        if 'df_compared' not in self.global_data:
            messagebox.showerror("Fout", "Genereer eerst de grafiek zodat de gesynchroniseerde data beschikbaar is.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if filename:
            try:
                self.global_data['df_compared'].to_csv(filename, index=False)
                messagebox.showinfo("Info", f"Data succesvol geëxporteerd naar {filename}")
            except Exception as e:
                messagebox.showerror("Fout", f"Fout bij exporteren: {e}")

# ---------------------------------------
# MAIN: Start de GUI
# ---------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = SensorComparisonApp(root)
    root.mainloop()
