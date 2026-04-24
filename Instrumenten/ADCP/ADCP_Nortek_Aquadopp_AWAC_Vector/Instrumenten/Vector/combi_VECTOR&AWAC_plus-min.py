import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000  # Voorkom problemen bij zeer grote aantallen punten

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import os
import re
import tkinter as tk
from tkinter import ttk, messagebox
import logging
from datetime import datetime
from tkcalendar import DateEntry

# Configureer logging
logging.basicConfig(
    filename='data_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ====================== 1. Globale Transformatie Matrix voor de Vector ======================
# Vervang deze door de matrix uit jouw Nortek header of firmware.
# Dit voorbeeld komt uit "Transform.m".
T_ORG = np.array([
    [2896,  2896,    0],
    [-2896, 2896,    0],
    [-2896, -2896, 5792]
], dtype=float)

# Schaal de matrix (Nortek gebruikt vaak 1/4096).
T_ORG /= 4096.0


def apply_orientation(T, statusbit0):
    """
    Als de sensor 'omhoog' kijkt (statusbit0=1), moeten rijen 2 en 3 van T negatief.
    Zie Transform.m van Nortek.
    """
    T_mod = T.copy()
    if statusbit0 == 1:
        T_mod[1, :] = -T_mod[1, :]
        T_mod[2, :] = -T_mod[2, :]
    return T_mod


def beam_to_enu_single(beam, heading, pitch, roll, statusbit0, T_org):
    """
    Transformatie van beam-velocities naar ENU, rekening houdend met
    heading, pitch, roll, en sensor-oriëntatie (statusbit0).
    - beam: np.array([v1, v2, v3])
    - heading, pitch, roll in graden
    - T_org: basis T-matrix (3x3)
    """
    # 1) Pas oriëntatie toe
    T = apply_orientation(T_org, statusbit0)

    # 2) Heading, pitch, roll in radians
    #    Nortek hanteert vaak heading-90 => 0° = East, 90° = North
    hh = np.radians(heading - 90.0)
    pp = np.radians(pitch)
    rr = np.radians(roll)

    # 3) Heading-matrix H
    H = np.array([
        [np.cos(hh),  np.sin(hh), 0],
        [-np.sin(hh), np.cos(hh), 0],
        [         0,           0, 1]
    ])

    # 4) Tilt-matrix P
    P = np.array([
        [np.cos(pp),              -np.sin(pp)*np.sin(rr),  -np.cos(rr)*np.sin(pp)],
        [0,                        np.cos(rr),              -np.sin(rr)           ],
        [np.sin(pp),  np.sin(rr)*np.cos(pp),   np.cos(pp)*np.cos(rr)]
    ])

    # 5) Totale matrix R
    R = H @ P @ T

    # 6) ENU = R * beam
    enu = R @ beam
    return enu  # [uE, uN, uU]


def transform_velocities_correctly(data_filtered):
    """
    Voer voor elke rij in data_filtered de transformatie van beam -> ENU uit,
    rekening houdend met heading, pitch, roll, en statusbit0.
    Verwacht kolommen: Velocity_Beam1, Velocity_Beam2, Velocity_Beam3,
                       Heading, Pitch, Roll, Status_code (of iets dergelijks).
    """
    enu_east = []
    enu_north = []
    enu_up = []

    for i, row in data_filtered.iterrows():
        beam = np.array([
            row['Velocity_Beam1'],
            row['Velocity_Beam2'],
            row['Velocity_Beam3']
        ], dtype=float)

        heading = float(row['Heading'])
        pitch   = float(row['Pitch'])
        roll    = float(row['Roll'])

        # statusbit0 = 1 => instrument is in 'down orientation' (bijv. sensor kijkt omhoog)
        # Hier aannemen dat 'Status_code' bits bevat, bit0 is LSB.
        status_code = int(row['Status_code'])
        statusbit0 = status_code & 1

        enu = beam_to_enu_single(beam, heading, pitch, roll, statusbit0, T_ORG)
        enu_east.append(enu[0])
        enu_north.append(enu[1])
        enu_up.append(enu[2])

    data_filtered['Velocity_East'] = enu_east
    data_filtered['Velocity_North'] = enu_north
    data_filtered['Velocity_Up'] = enu_up

    return data_filtered

# ====================== 2. AWAC Data Processing ======================

def extract_depths(speed_cols):
    depth_pattern = re.compile(r'\((\d+(?:\.\d+)?)m\)')
    cell_positions = []
    for col in speed_cols:
        match = depth_pattern.search(col)
        if match:
            depth = float(match.group(1))
            cell_positions.append(depth)
        else:
            logging.warning(f"AWAC: Diepte-info niet gevonden in kolom '{col}'.")
            cell_positions.append(np.nan)
    return cell_positions

def load_awac_data(csv_file_path, start_time, end_time):
    try:
        df = pd.read_csv(csv_file_path, sep=';', header=0, encoding='utf-8')
        df.columns = df.columns.str.strip()
    except Exception as e:
        logging.error(f"Fout bij lezen AWAC CSV: {e}")
        raise

    if 'DateTime' not in df.columns:
        raise ValueError("Kolom 'DateTime' niet gevonden in AWAC CSV.")

    # AWAC DateTime in dd/mm/yyyy HH:MM:SS
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M:%S')
    df_filtered = df[(df['DateTime'] >= start_time) & (df['DateTime'] <= end_time)]
    if df_filtered.empty:
        raise ValueError("Geen AWAC data in de opgegeven periode.")

    df_filtered.set_index('DateTime', inplace=True)

    speed_cols = [c for c in df_filtered.columns if c.startswith('Speed#')]
    dir_cols = [c for c in df_filtered.columns if c.startswith('Dir#')]

    if len(speed_cols) != len(dir_cols):
        raise ValueError("AWAC: aantal snelheids- en richtingskolommen komt niet overeen.")

    speed_cols.sort()
    dir_cols.sort()
    cell_positions = extract_depths(speed_cols)
    cell_positions = np.array(cell_positions)

    if np.any(np.isnan(cell_positions)):
        cell_positions = [0.90 + 0.50 * i for i in range(len(speed_cols))]
        logging.warning("AWAC: niet alle diepte-info gevonden. Standaardwaarden gebruikt.")

    return df_filtered, speed_cols, dir_cols, cell_positions

# ====================== 3. Vector Data Processing ======================

def create_vector_datetime_column(data_sen):
    try:
        data_sen['Datetime'] = pd.to_datetime(
            data_sen[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']]
        )
        return data_sen
    except Exception as e:
        logging.error(f"Vector: fout bij maken Datetime kolom: {e}")
        raise

def calculate_resultant_speed_direction(data_filtered):
    # Grootte van horizontale snelheid
    data_filtered['Resultant_Speed'] = np.sqrt(
        data_filtered['Velocity_East']**2 + data_filtered['Velocity_North']**2
    )
    # Richting in graden, 0°=East, 90°=North
    data_filtered['Direction'] = (
        np.degrees(np.arctan2(data_filtered['Velocity_North'], data_filtered['Velocity_East'])) + 360
    ) % 360
    return data_filtered

def load_vector_data(dat_file_path, sen_file_path,
                     dat_column_names, sen_column_names,
                     start_datetime, end_datetime):
    try:
        data_dat = pd.read_csv(dat_file_path, sep=r'\s+', header=None,
                               names=dat_column_names, comment='#', engine='python')
        logging.info("Vector: .dat bestand ingelezen.")
    except Exception as e:
        logging.error(f"Vector: fout bij lezen .dat: {e}")
        raise

    try:
        data_sen = pd.read_csv(sen_file_path, sep=r'\s+', header=None,
                               names=sen_column_names, comment='#', engine='python')
        logging.info("Vector: .sen bestand ingelezen.")
    except Exception as e:
        logging.error(f"Vector: fout bij lezen .sen: {e}")
        raise

    # Check gelijke aantal rijen
    if len(data_dat) != len(data_sen):
        logging.warning(f"Vector: .dat rijen={len(data_dat)}, .sen rijen={len(data_sen)} -> trimmen.")
        min_len = min(len(data_dat), len(data_sen))
        data_dat = data_dat.iloc[:min_len].reset_index(drop=True)
        data_sen = data_sen.iloc[:min_len].reset_index(drop=True)

    # Maak Datetime aan
    data_sen = create_vector_datetime_column(data_sen)
    data_dat['Datetime'] = data_sen['Datetime']

    # Kopieer heading, pitch, roll, status naar data_dat
    # (pas kolomnamen aan jouw sen_column_names aan!)
    # Voorbeeld: in sen_column_names staat 'Status_code' op index 7,
    #            'Heading' op 10, 'Pitch' op 11, 'Roll' op 12, etc.
    data_dat['Heading'] = data_sen['Heading']
    data_dat['Pitch']   = data_sen['Pitch']
    data_dat['Roll']    = data_sen['Roll']
    data_dat['Status_code'] = data_sen['Status_code']

    # Filter op geldige metingen (Checksum == 0)
    data_valid = data_dat[data_dat['Checksum'] == 0].reset_index(drop=True)
    data_filtered = data_valid[
        (data_valid['Datetime'] >= start_datetime) &
        (data_valid['Datetime'] <= end_datetime)
    ].reset_index(drop=True)

    if data_filtered.empty:
        logging.warning("Vector: geen data na filtering.")
        return None, None, None

    # --- HIER GEBEURT NU DE CORRECTE TRANSFORMATIE (per ensemble) ---
    data_filtered = transform_velocities_correctly(data_filtered)

    # Bereken resulterende snelheid en richting
    data_filtered = calculate_resultant_speed_direction(data_filtered)

    # Diepte uit Pressure (vereenvoudigd, 1 dbar ~ 1 m)
    if 'Pressure' in data_filtered.columns:
        avg_p = data_filtered['Pressure'].mean()
        fixed_vector_depth = avg_p
        logging.info(f"Vector: gem. druk={avg_p:.2f} dbar -> diepte={fixed_vector_depth:.2f}m")
    else:
        raise ValueError("Vector: geen 'Pressure' kolom gevonden.")

    return data_filtered, T_ORG, fixed_vector_depth

# ====================== 4. Lopend Gemiddelde en Teken-Correctie ======================

def apply_running_average(series, window_size):
    """Past een lopend gemiddelde (center=True) toe op een pandas Series."""
    return series.rolling(window=window_size, min_periods=1, center=True).mean()

def apply_direction_sign(speed, direction, tol=10):
    """
    Pas het teken van de snelheid aan op basis van de gemeten richting.
    Alle snelheden ~67° of ~246° worden negatief (tegen de kanaalrichting),
    de rest positief. Pas dit evt. aan jouw situatie aan.
    """
    if (abs(direction - 67) <= tol) or (abs(direction - 246) <= tol):
        return -abs(speed)
    else:
        return abs(speed)

# ====================== 5. Plotten van Tijdreeksen (Snelheid & Richting) ======================

def plot_combined_data(time_awac, speed_awac, direction_awac,
                       speed_awac_smoothed, direction_awac_smoothed,
                       time_vector, speed_vector, direction_vector,
                       speed_vector_smoothed, direction_vector_smoothed,
                       depth_awac, fixed_vector_depth,
                       directory, awac_window_size, vector_window_size):
    fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True, constrained_layout=True)

    # Snelheid
    axs[0].plot(time_awac, speed_awac, label='AWAC Snelheid (gesigneerd)', color='tab:blue', alpha=0.5)
    axs[0].plot(time_awac, speed_awac_smoothed, label=f'AWAC Snelheid (lm. gem. {awac_window_size})', color='tab:blue')
    axs[0].plot(time_vector, speed_vector, label='Vector Snelheid (gesigneerd)', color='tab:orange', alpha=0.5)
    axs[0].plot(time_vector, speed_vector_smoothed, label=f'Vector Snelheid (lm. gem. {vector_window_size})', color='tab:orange')
    axs[0].set_ylabel('Snelheid (m/s)')
    axs[0].set_title(f'Snelheid op Dieptes: AWAC {depth_awac:.2f}m vs Vector {fixed_vector_depth:.2f}m')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    # Richting
    axs[1].plot(time_awac, direction_awac, label='AWAC Richting', color='tab:green', alpha=0.5)
    axs[1].plot(time_awac, direction_awac_smoothed, label=f'AWAC Richting (lm. gem. {awac_window_size})', color='tab:green')
    axs[1].plot(time_vector, direction_vector, label='Vector Richting', color='tab:red', alpha=0.5)
    axs[1].plot(time_vector, direction_vector_smoothed, label=f'Vector Richting (lm. gem. {vector_window_size})', color='tab:red')
    axs[1].set_ylabel('Richting (°)')
    axs[1].set_xlabel('Tijd')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    # Datum-as formatteren
    date_format = mdates.DateFormatter('%d/%m/%Y\n%H:%M:%S')
    axs[1].xaxis.set_major_formatter(date_format)

    # Opslaan
    save_path = os.path.join(directory, f'combined_velocity_direction_depth_{depth_awac:.2f}m_{fixed_vector_depth:.2f}m.png')
    fig.savefig(save_path, dpi=300)
    logging.info(f"Gecombineerde grafiek opgeslagen als {save_path}")

    # AWAC CSV
    awac_csv_path = os.path.join(directory, f'AWAC_selected_depth_{depth_awac:.2f}m.csv')
    awac_selected_df = pd.DataFrame({
        'Datetime': time_awac,
        'Speed_AWAC_Signed': speed_awac,
        'Direction_AWAC': direction_awac,
        'Speed_AWAC_Smoothed': speed_awac_smoothed,
        'Direction_AWAC_Smoothed': direction_awac_smoothed
    })
    awac_selected_df.to_csv(awac_csv_path, index=False)

    # Vector CSV
    vector_csv_path = os.path.join(directory, f'Vector_selected_depth_{fixed_vector_depth:.2f}m.csv')
    vector_selected_df = pd.DataFrame({
        'Datetime': time_vector,
        'Speed_Vector_Signed': speed_vector,
        'Direction_Vector': direction_vector,
        'Speed_Vector_Smoothed': speed_vector_smoothed,
        'Direction_Vector_Smoothed': direction_vector_smoothed,
        'Depth_Vector_m': fixed_vector_depth
    })
    vector_selected_df.to_csv(vector_csv_path, index=False)

    plt.show()

# ====================== 6. (Optioneel) Roosplot van Uitgemiddelde Data ======================
def plot_speed_direction_rose(time, speed, direction, label, output_dir):
    """
    Polaire scatter-plot (roos) van uitgemiddelde snelheid-richting.
    Straal = |speed|, hoek = direction (graden), kleur = snelheid.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    angles = np.radians(direction)
    r = np.abs(speed)  # negatieve straal heeft geen zin in polar coords

    sc = ax.scatter(angles, r, c=r, cmap='viridis', alpha=0.7, s=10)
    ax.set_theta_direction(-1)      # hoeken met de klok mee
    ax.set_theta_offset(np.pi/2)    # 0° bovenaan
    ax.set_title(f'{label} Snelheid-Richting Roos (uitgemiddeld)')

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Snelheid (m/s)')

    fname = os.path.join(output_dir, f'{label}_speed_direction_rose.png')
    fig.savefig(fname, dpi=300)
    plt.show()
    logging.info(f"{label} roosplot opgeslagen als {fname}")

# ====================== 7. GUI en Integratie ======================

def main():
    # Pas deze paden aan jouw situatie aan
    awac_csv_file_path = r'C:\Users\claeysst\Desktop\werkfiles\verwerken AWAC\kanne04.csv'
    vector_dat_file_path = r'C:\Users\claeysst\Desktop\werkfiles\Verwerken Vector\KANNE05.dat'
    vector_sen_file_path = r'C:\Users\claeysst\Desktop\werkfiles\Verwerken Vector\KANNE05.sen'
    
    dat_column_names = [
        'Burst_counter',
        'Ensemble_counter',
        'Velocity_Beam1',
        'Velocity_Beam2',
        'Velocity_Beam3',
        'Amplitude_Beam1',
        'Amplitude_Beam2',
        'Amplitude_Beam3',
        'SNR_Beam1',
        'SNR_Beam2',
        'SNR_Beam3',
        'Correlation_Beam1',
        'Correlation_Beam2',
        'Correlation_Beam3',
        'Pressure',
        'Analog_input1',
        'Analog_input2',
        'Checksum'
    ]
    # Zorg dat deze kolomnamen overeenkomen met wat er echt in je .sen staat
    sen_column_names = [
        'Month',
        'Day',
        'Year',
        'Hour',
        'Minute',
        'Second',
        'Error_code',
        'Status_code',   # let op: nodig voor oriëntatie
        'Battery_voltage',
        'Soundspeed',
        'Heading',       # heading in graden
        'Pitch',         # pitch in graden
        'Roll',          # roll in graden
        'Temperature',
        'Analog_input',
        'Checksum'
    ]

    # Globale variabelen voor ingelezen data
    awac_df_filtered = None
    awac_speed_cols = None
    awac_dir_cols = None
    awac_cell_positions = None
    vector_data_filtered = None
    fixed_vector_depth = None

    root = tk.Tk()
    root.title("Data Laden, Tijdselectie en Vergelijking")
    root.geometry("1000x700")

    # ----- AWAC Tijdselectie (dd/mm/yyyy) -----
    awac_label = ttk.Label(root, text="AWAC Tijdselectie", font=('Arial', 12, 'bold'))
    awac_label.grid(row=0, column=0, columnspan=4, pady=(10,0))

    ttk.Label(root, text="Begin Datum (dd/mm/yyyy):").grid(row=1, column=0, sticky='e', padx=5, pady=2)
    awac_start_date_entry = DateEntry(root, date_pattern='dd/mm/yyyy')
    awac_start_date_entry.grid(row=1, column=1, sticky='w', padx=5, pady=2)
    
    ttk.Label(root, text="Begin Tijd (HH:MM:SS):").grid(row=1, column=2, sticky='e', padx=5, pady=2)
    awac_start_time_entry = ttk.Entry(root)
    awac_start_time_entry.grid(row=1, column=3, sticky='w', padx=5, pady=2)
    awac_start_time_entry.insert(0, "00:00:00")
    
    ttk.Label(root, text="Eind Datum (dd/mm/yyyy):").grid(row=2, column=0, sticky='e', padx=5, pady=2)
    awac_end_date_entry = DateEntry(root, date_pattern='dd/mm/yyyy')
    awac_end_date_entry.grid(row=2, column=1, sticky='w', padx=5, pady=2)
    
    ttk.Label(root, text="Eind Tijd (HH:MM:SS):").grid(row=2, column=2, sticky='e', padx=5, pady=2)
    awac_end_time_entry = ttk.Entry(root)
    awac_end_time_entry.grid(row=2, column=3, sticky='w', padx=5, pady=2)
    awac_end_time_entry.insert(0, "23:59:59")
    
    # ----- Vector Tijdselectie (yyyy-mm-dd) -----
    vector_label = ttk.Label(root, text="Vector Tijdselectie", font=('Arial', 12, 'bold'))
    vector_label.grid(row=3, column=0, columnspan=4, pady=(10,0))
    
    ttk.Label(root, text="Begin Datum (yyyy-mm-dd):").grid(row=4, column=0, sticky='e', padx=5, pady=2)
    vector_start_date_entry = DateEntry(root, date_pattern='yyyy-mm-dd')
    vector_start_date_entry.grid(row=4, column=1, sticky='w', padx=5, pady=2)
    
    ttk.Label(root, text="Begin Tijd (HH:MM:SS):").grid(row=4, column=2, sticky='e', padx=5, pady=2)
    vector_start_time_entry = ttk.Entry(root)
    vector_start_time_entry.grid(row=4, column=3, sticky='w', padx=5, pady=2)
    vector_start_time_entry.insert(0, "00:00:00")
    
    ttk.Label(root, text="Eind Datum (yyyy-mm-dd):").grid(row=5, column=0, sticky='e', padx=5, pady=2)
    vector_end_date_entry = DateEntry(root, date_pattern='yyyy-mm-dd')
    vector_end_date_entry.grid(row=5, column=1, sticky='w', padx=5, pady=2)
    
    ttk.Label(root, text="Eind Tijd (HH:MM:SS):").grid(row=5, column=2, sticky='e', padx=5, pady=2)
    vector_end_time_entry = ttk.Entry(root)
    vector_end_time_entry.grid(row=5, column=3, sticky='w', padx=5, pady=2)
    vector_end_time_entry.insert(0, "23:59:59")
    
    # ----- Data Laden Knop -----
    def on_load_data():
        nonlocal awac_df_filtered, awac_speed_cols, awac_dir_cols, awac_cell_positions
        nonlocal vector_data_filtered, fixed_vector_depth
        try:
            # AWAC start/eind
            awac_start_dt_str = awac_start_date_entry.get() + " " + awac_start_time_entry.get()
            awac_end_dt_str = awac_end_date_entry.get() + " " + awac_end_time_entry.get()
            awac_start_dt = datetime.strptime(awac_start_dt_str, "%d/%m/%Y %H:%M:%S")
            awac_end_dt = datetime.strptime(awac_end_dt_str, "%d/%m/%Y %H:%M:%S")
            if awac_start_dt >= awac_end_dt:
                messagebox.showerror("Fout", "AWAC: Begin datetime moet vóór eind datetime liggen.")
                return
            
            # Vector start/eind
            vector_start_dt_str = vector_start_date_entry.get() + " " + vector_start_time_entry.get()
            vector_end_dt_str = vector_end_date_entry.get() + " " + vector_end_time_entry.get()
            vector_start_dt = datetime.strptime(vector_start_dt_str, "%Y-%m-%d %H:%M:%S")
            vector_end_dt = datetime.strptime(vector_end_dt_str, "%Y-%m-%d %H:%M:%S")
            if vector_start_dt >= vector_end_dt:
                messagebox.showerror("Fout", "Vector: Begin datetime moet vóór eind datetime liggen.")
                return
            
            # AWAC data
            awac_df_filtered, awac_speed_cols, awac_dir_cols, awac_cell_positions = load_awac_data(
                awac_csv_file_path, awac_start_dt, awac_end_dt
            )
            logging.info("AWAC data succesvol geladen.")
            
            # Dropdown bijwerken
            available_depths_awac = sorted(awac_cell_positions)
            depth_dropdown_awac['values'] = available_depths_awac
            if available_depths_awac:
                depth_var_awac.set(available_depths_awac[0])
            
            # Vector data
            vector_data_filtered, _, fixed_vector_depth = load_vector_data(
                vector_dat_file_path,
                vector_sen_file_path,
                dat_column_names,
                sen_column_names,
                vector_start_dt,
                vector_end_dt
            )
            if vector_data_filtered is None:
                messagebox.showwarning("Waarschuwing", "Geen Vector data na filtering.")
                return
            
            label_vector_depth.config(
                text=f"Vector data is op een automatisch berekende diepte van {fixed_vector_depth:.2f}m."
            )
            messagebox.showinfo("Info", "Data succesvol geladen.")
            
        except Exception as e:
            messagebox.showerror("Fout", f"Fout bij het laden van de data: {e}")
            logging.error(f"Fout in on_load_data: {e}")
    
    load_button = ttk.Button(root, text="Laad Data", command=on_load_data)
    load_button.grid(row=6, column=0, columnspan=4, pady=(10,10))
    
    # ----- Overige Instellingen -----
    ttk.Label(root, text="Selecteer de gewenste AWAC diepte boven de bodem (m):").grid(
        row=7, column=0, columnspan=2, padx=10, pady=10, sticky='w'
    )
    depth_var_awac = tk.StringVar()
    depth_dropdown_awac = ttk.Combobox(root, textvariable=depth_var_awac, state="readonly")
    depth_dropdown_awac.grid(row=7, column=2, columnspan=2, padx=10, pady=10, sticky='ew')
    
    label_vector_depth = ttk.Label(root, text="Vector data is op een automatisch berekende diepte: -")
    label_vector_depth.grid(row=8, column=0, columnspan=4, padx=10, pady=10, sticky='w')
    
    ttk.Label(root, text="Lopend gemiddelde AWAC (aantal datapunten):").grid(
        row=9, column=0, columnspan=2, padx=10, pady=5, sticky='w'
    )
    awac_window_entry = ttk.Entry(root)
    awac_window_entry.grid(row=9, column=2, columnspan=2, padx=10, pady=5, sticky='ew')
    awac_window_entry.insert(0, "1")
    
    ttk.Label(root, text="Lopend gemiddelde Vector (aantal datapunten):").grid(
        row=10, column=0, columnspan=2, padx=10, pady=5, sticky='w'
    )
    vector_window_entry = ttk.Entry(root)
    vector_window_entry.grid(row=10, column=2, columnspan=2, padx=10, pady=5, sticky='ew')
    vector_window_entry.insert(0, "1")
    
    ttk.Label(root, text="Tijdverschuiving tussen AWAC en Vector (minuten):").grid(
        row=11, column=0, columnspan=2, padx=10, pady=5, sticky='w'
    )
    time_shift_var = tk.IntVar()
    time_shift_slider = ttk.Scale(
        root, from_=-180, to=180, orient='horizontal',
        variable=time_shift_var,
        command=lambda val: time_shift_var.set(int(float(val)))
    )
    time_shift_slider.grid(row=11, column=2, columnspan=1, padx=10, pady=5, sticky='ew')
    time_shift_slider.set(0)
    time_shift_label = ttk.Label(root, textvariable=time_shift_var)
    time_shift_label.grid(row=11, column=3, padx=10, pady=5, sticky='w')
    
    # ----- Plot & Opslaan Knop -----
    def on_plot():
        try:
            if awac_df_filtered is None or vector_data_filtered is None:
                messagebox.showerror("Fout", "Laad eerst data met de juiste tijdselectie.")
                return
            
            try:
                awac_window_size = int(awac_window_entry.get())
                if awac_window_size < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Fout", "Ongeldige waarde voor lopend gemiddelde AWAC.")
                return

            try:
                vector_window_size = int(vector_window_entry.get())
                if vector_window_size < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Fout", "Ongeldige waarde voor lopend gemiddelde Vector.")
                return

            time_shift_minutes = time_shift_var.get()

            # AWAC diepte
            selected_depth_awac = depth_var_awac.get()
            if not selected_depth_awac:
                messagebox.showerror("Fout", "Selecteer een AWAC diepte.")
                return
            try:
                depth_awac = float(selected_depth_awac)
            except ValueError:
                messagebox.showerror("Fout", "Ongeldige diepte.")
                return

            # Haal kolommen
            closest_idx_awac = (np.abs(awac_cell_positions - depth_awac)).argmin()
            closest_depth_awac = awac_cell_positions[closest_idx_awac]
            speed_col_awac = awac_speed_cols[closest_idx_awac]
            dir_col_awac   = awac_dir_cols[closest_idx_awac]
            speed_awac     = awac_df_filtered[speed_col_awac].values
            direction_awac = awac_df_filtered[dir_col_awac].values
            time_awac      = awac_df_filtered.index

            # Teken-correctie AWAC
            speed_awac_signed = np.array([
                apply_direction_sign(s, d) for s, d in zip(speed_awac, direction_awac)
            ])

            # Vector data met tijdverschuiving
            shifted_vector = vector_data_filtered.copy()
            shifted_vector['Datetime'] = shifted_vector['Datetime'] + pd.to_timedelta(time_shift_minutes, unit='m')
            speed_vector = shifted_vector['Resultant_Speed'].values
            direction_vector = shifted_vector['Direction'].values
            time_vector = shifted_vector['Datetime']

            # Teken-correctie Vector
            speed_vector_signed = np.array([
                apply_direction_sign(s, d) for s, d in zip(speed_vector, direction_vector)
            ])

            # Lopend gemiddelde
            speed_awac_smoothed = apply_running_average(pd.Series(speed_awac_signed), awac_window_size).values
            direction_awac_smoothed = apply_running_average(pd.Series(direction_awac), awac_window_size).values
            speed_vector_smoothed = apply_running_average(pd.Series(speed_vector_signed), vector_window_size).values
            direction_vector_smoothed = apply_running_average(pd.Series(direction_vector), vector_window_size).values

            # Plot en sla op
            output_dir = os.path.dirname(awac_csv_file_path)
            plot_combined_data(
                time_awac, speed_awac_signed, direction_awac, speed_awac_smoothed, direction_awac_smoothed,
                time_vector, speed_vector_signed, direction_vector, speed_vector_smoothed, direction_vector_smoothed,
                closest_depth_awac, fixed_vector_depth,
                output_dir, awac_window_size, vector_window_size
            )

            # (Optioneel) roosplots van de uitgemiddelde data
            # plot_speed_direction_rose(time_awac, speed_awac_smoothed, direction_awac_smoothed,
            #                           label='AWAC', output_dir=output_dir)
            # plot_speed_direction_rose(time_vector, speed_vector_smoothed, direction_vector_smoothed,
            #                           label='Vector', output_dir=output_dir)

        except Exception as e:
            messagebox.showerror("Fout", f"Fout bij het plotten: {e}")
            logging.error(f"Fout in on_plot: {e}")
    
    plot_button = ttk.Button(root, text="Plot en Sla Op", command=on_plot)
    plot_button.grid(row=12, column=0, columnspan=4, pady=(10,5))
    
    exit_button = ttk.Button(root, text="Afsluiten", command=root.destroy)
    exit_button.grid(row=13, column=0, columnspan=4, pady=(5,10))
    
    # Kolommen laten meeschalen
    for i in range(4):
        root.grid_columnconfigure(i, weight=1)
    
    root.mainloop()

if __name__ == "__main__":
    main()
