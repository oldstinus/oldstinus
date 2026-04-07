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

# Configureer logging naar een bestand voor latere referentie
logging.basicConfig(
    filename='data_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --------------------- AWAC Data Processing Functions ---------------------

def read_awac_time_period(file_path):
    """Leest start- en einddatum/tijd uit het tekstbestand voor AWAC data."""
    try:
        with open(file_path, 'r') as f:
            start_time_str = f.readline().strip()
            end_time_str = f.readline().strip()
        start_time = pd.to_datetime(start_time_str, format='%d/%m/%Y %H:%M:%S')
        end_time = pd.to_datetime(end_time_str, format='%d/%m/%Y %H:%M:%S')
        return start_time, end_time
    except Exception as e:
        logging.error(f"Fout bij het lezen van AWAC 'begin-eind.txt': {e}")
        raise

def extract_depths(speed_cols):
    """Haal diepte-informatie uit AWAC kolomnamen."""
    depth_pattern = re.compile(r'\((\d+(?:\.\d+)?)m\)')
    cell_positions = []
    for col in speed_cols:
        match = depth_pattern.search(col)
        if match:
            depth = float(match.group(1))
            cell_positions.append(depth)
        else:
            logging.warning(f"AWAC: Diepte-informatie niet gevonden in kolom '{col}'.")
            cell_positions.append(np.nan)
    return cell_positions

def load_awac_data(csv_file_path, start_time, end_time):
    """Laadt en filtert de AWAC CSV data."""
    try:
        df = pd.read_csv(csv_file_path, sep=';', header=0, encoding='utf-8')
        df.columns = df.columns.str.strip()
    except Exception as e:
        logging.error(f"Fout bij het lezen van AWAC CSV bestand: {e}")
        raise

    if 'DateTime' not in df.columns:
        raise ValueError("De kolom 'DateTime' is niet gevonden in de AWAC DataFrame.")

    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M:%S')
    df_filtered = df[(df['DateTime'] >= start_time) & (df['DateTime'] <= end_time)]

    if df_filtered.empty:
        raise ValueError("Geen AWAC data beschikbaar binnen de opgegeven tijdsperiode.")

    df_filtered.set_index('DateTime', inplace=True)

    speed_cols = [col for col in df_filtered.columns if col.startswith('Speed#')]
    dir_cols = [col for col in df_filtered.columns if col.startswith('Dir#')]

    if len(speed_cols) != len(dir_cols):
        raise ValueError("Het aantal AWAC snelheids- en richtingskolommen komt niet overeen.")

    speed_cols.sort()
    dir_cols.sort()
    num_cells = len(speed_cols)
    logging.info(f"AWAC: Aantal cellen gedetecteerd: {num_cells}")

    cell_positions = extract_depths(speed_cols)
    if np.any(np.isnan(cell_positions)):
        cell_positions = [0.90 + 0.50 * i for i in range(num_cells)]
        logging.warning("AWAC: Niet alle diepte-informatie kon worden gevonden. Standaardwaarden worden gebruikt.")

    cell_positions = np.array(cell_positions)
    return df_filtered, speed_cols, dir_cols, cell_positions

# --------------------- Vector Data Processing Functions ---------------------

def read_vector_time_bounds(txt_file_path):
    """Leest start- en einddatum/tijd uit het tekstbestand voor Vector data."""
    try:
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
            if len(lines) < 2:
                raise ValueError("Het tekstbestand moet minstens twee regels bevatten: startdatum/tijd en einddatum/tijd.")
            start_str = lines[0].strip()
            end_str = lines[1].strip()
            start_datetime = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            end_datetime = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
            if start_datetime >= end_datetime:
                raise ValueError("De startdatum moet vóór de einddatum liggen.")
            return start_datetime, end_datetime
    except FileNotFoundError:
        logging.error(f"Fout: Het Vector tijdsbestand {txt_file_path} is niet gevonden.")
        raise
    except ValueError as ve:
        logging.error(f"Fout bij het lezen van Vector tijdsbestand {txt_file_path}: {ve}")
        raise

def create_vector_datetime_column(data_sen):
    """Maakt een datetime kolom uit de Vector .sen data."""
    try:
        data_sen['Datetime'] = pd.to_datetime(data_sen[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
        return data_sen
    except Exception as e:
        logging.error(f"Vector: Fout bij het maken van de Datetime kolom: {e}")
        raise

def calculate_resultant_speed_direction_vector(data_filtered):
    """Bereken de resulterende snelheid en richting voor Vector data."""
    # Bereken de resulterende snelheid als de norm van de beam-snelheden
    data_filtered['Resultant_Speed'] = np.sqrt(
        data_filtered['Velocity_Beam1']**2 +
        data_filtered['Velocity_Beam2']**2 +
        data_filtered['Velocity_Beam3']**2
    )

    # Bereken de richting op basis van Velocity_Beam1 en Velocity_Beam2 (aangenomen als Oost en Noord)
    data_filtered['Direction'] = (np.degrees(np.arctan2(data_filtered['Velocity_Beam2'], data_filtered['Velocity_Beam1'])) + 360) % 360

    # Bereken U en V componenten
    data_filtered['U'] = data_filtered['Velocity_Beam1']
    data_filtered['V'] = data_filtered['Velocity_Beam2']

    return data_filtered

def load_vector_data(dat_file_path, sen_file_path, dat_column_names, sen_column_names, start_datetime, end_datetime):
    """Laadt en filtert de Vector data."""
    try:
        data_dat = pd.read_csv(dat_file_path, sep=r'\s+', header=None, names=dat_column_names, comment='#', engine='python')
        logging.info("Vector: Het .dat-bestand is succesvol ingelezen.")
    except Exception as e:
        logging.error(f"Vector: Fout bij het lezen van het .dat bestand: {e}")
        raise

    try:
        data_sen = pd.read_csv(sen_file_path, sep=r'\s+', header=None, names=sen_column_names, comment='#', engine='python')
        logging.info("Vector: Het .sen-bestand is succesvol ingelezen.")
    except Exception as e:
        logging.error(f"Vector: Fout bij het lezen van het .sen bestand: {e}")
        raise

    if len(data_dat) != len(data_sen):
        logging.warning("Vector: Het aantal rijen in .dat en .sen bestanden komt niet overeen.")
        logging.warning(f"Vector: .dat rijen: {len(data_dat)}, .sen rijen: {len(data_sen)}")
        min_length = min(len(data_dat), len(data_sen))
        data_dat = data_dat.iloc[:min_length].reset_index(drop=True)
        data_sen = data_sen.iloc[:min_length].reset_index(drop=True)
        logging.info(f"Vector: Data is getrimd tot {min_length} rijen.")
    else:
        logging.info("Vector: Aantal rijen in .dat en .sen bestanden komt overeen.")

    data_sen = create_vector_datetime_column(data_sen)
    data_dat['Datetime'] = data_sen['Datetime']

    # Filter op geldige metingen (Checksum == 0) en tijdsperiode
    data_valid = data_dat[data_dat['Checksum'] == 0].reset_index(drop=True)
    logging.info(f"Vector: Aantal geldige metingen: {len(data_valid)}")

    data_filtered = data_valid[(data_valid['Datetime'] >= start_datetime) & (data_valid['Datetime'] <= end_datetime)].reset_index(drop=True)
    logging.info(f"Vector: Aantal metingen in de geselecteerde periode: {len(data_filtered)}")

    if data_filtered.empty:
        logging.warning("Vector: Geen data beschikbaar na filtering.")
        return None, None, None

    # Bereken resulterende snelheid en richting zonder transformatie
    data_filtered = calculate_resultant_speed_direction_vector(data_filtered)

    # Bereken de diepte uit de druksensor (Pressure)
    # Aannemend dat Pressure in decibar (dbar) is en 1 dbar ≈ 1 meter diepte
    # Pas deze conversie aan indien de Pressure in een andere eenheid is
    try:
        if 'Pressure' in data_filtered.columns:
            # Gebruik gemiddelde druk om vaste diepte te bepalen
            average_pressure = data_filtered['Pressure'].mean()
            fixed_vector_depth = average_pressure  # 1 dbar ≈ 1m
            logging.info(f"Vector: Gemiddelde druk = {average_pressure:.2f} dbar, aangenomen diepte = {fixed_vector_depth:.2f} m")
        else:
            raise ValueError("De kolom 'Pressure' is niet gevonden in de Vector DataFrame.")
    except Exception as e:
        logging.error(f"Vector: Fout bij het berekenen van de diepte uit Pressure: {e}")
        raise

    return data_filtered, None, fixed_vector_depth

# --------------------- Data Smoothing Function ---------------------

def apply_running_average(series, window_size):
    """Past een lopend gemiddelde toe op een pandas Series."""
    return series.rolling(window=window_size, min_periods=1, center=True).mean()

# --------------------- Vector Plot Function ---------------------

def plot_vector_plot(U, V, instrument_name, depth, directory):
    """
    Visualiseert de snelheidsvectoren in een aparte 2D-plot.

    Parameters:
    - U (array-like): Velocity East (m/s)
    - V (array-like): Velocity North (m/s)
    - instrument_name (str): Naam van het instrument (AWAC of Vector).
    - depth (float): Diepte van het instrument in meters.
    - directory (str): Directory om de plot op te slaan.
    """
    plt.figure(figsize=(10, 8))

    # Alle vectoren laten beginnen bij de oorsprong
    X = np.zeros_like(U)
    Y = np.zeros_like(V)

    # Bereken de resulterende snelheid voor schaling
    resultant_speed = np.sqrt(U**2 + V**2)
    max_speed = np.max(resultant_speed)
    desired_max_arrow_length = 1  # Pas dit aan naar behoefte

    if max_speed > 0:
        scale = max_speed / desired_max_arrow_length
    else:
        scale = 1

    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=scale, color='r', width=0.003, alpha=0.6)
    plt.title(f'{instrument_name} Snelheidsvectoren op Diepte {depth:.2f}m')
    plt.xlabel('Velocity East (m/s)')
    plt.ylabel('Velocity North (m/s)')
    plt.grid(True)
    plt.axis('equal')

    # Stel limieten in op basis van maximale snelheid
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)

    # Sla de vector plot op
    vector_plot_path = os.path.join(directory, f'{instrument_name}_velocity_vectors_depth_{depth:.2f}m.png')
    plt.savefig(vector_plot_path, dpi=300)
    logging.info(f"{instrument_name} vector plot opgeslagen in: {vector_plot_path}")

    # Toon de plot
    plt.show()

# --------------------- Combined Visualization Function ---------------------

def plot_combined_data(time_awac, speed_awac, direction_awac, speed_awac_smoothed, direction_awac_smoothed,
                       time_vector, speed_vector, direction_vector, speed_vector_smoothed, direction_vector_smoothed,
                       depth_awac, fixed_vector_depth, directory, awac_window_size, vector_window_size):
    """Genereert en slaat gecombineerde plots op voor zowel AWAC als Vector data."""
    fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True, constrained_layout=True)

    # Plot Snelheid
    axs[0].plot(time_awac, speed_awac, label='AWAC Snelheid Origineel', color='tab:blue', alpha=0.5)
    axs[0].plot(time_awac, speed_awac_smoothed, label=f'AWAC Snelheid (lopende gemiddelde over {awac_window_size})', color='tab:blue')
    axs[0].plot(time_vector, speed_vector, label='Vector Snelheid Origineel', color='tab:orange', alpha=0.5)
    axs[0].plot(time_vector, speed_vector_smoothed, label=f'Vector Snelheid (lopende gemiddelde over {vector_window_size})', color='tab:orange')
    axs[0].set_ylabel('Snelheid (m/s)')
    axs[0].set_title(f'Snelheid op Dieptes: AWAC {depth_awac:.2f}m vs Vector {fixed_vector_depth:.2f}m')
    axs[0].legend(loc='upper right')
    axs[0].grid(True)

    # Plot Richting
    axs[1].plot(time_awac, direction_awac, label='AWAC Richting Origineel', color='tab:green', alpha=0.5)
    axs[1].plot(time_awac, direction_awac_smoothed, label=f'AWAC Richting (lopende gemiddelde over {awac_window_size})', color='tab:green')
    axs[1].plot(time_vector, direction_vector, label='Vector Richting Origineel', color='tab:red', alpha=0.5)
    axs[1].plot(time_vector, direction_vector_smoothed, label=f'Vector Richting (lopende gemiddelde over {vector_window_size})', color='tab:red')
    axs[1].set_ylabel('Richting (°)')
    axs[1].set_xlabel('Tijd')
    axs[1].legend(loc='upper right')
    axs[1].grid(True)

    # Formatteren van de x-as voor datums
    date_format = mdates.DateFormatter('%d/%m/%Y\n%H:%M:%S')
    axs[1].xaxis.set_major_formatter(date_format)

    # Opslaan van de gecombineerde grafiek
    save_path = os.path.join(directory, f'combined_velocity_direction_depth_{depth_awac:.2f}m_{fixed_vector_depth:.2f}m.png')
    fig.savefig(save_path, dpi=300)
    logging.info(f"Gecombineerde grafiek opgeslagen als {save_path}")

    # Sla de geselecteerde data op als CSV-bestanden
    # Originele en gegladdde AWAC data
    awac_csv_path = os.path.join(directory, f'AWAC_selected_depth_{depth_awac:.2f}m.csv')
    awac_selected_df = pd.DataFrame({
        'Datetime': time_awac,
        'Speed_AWAC_Origineel': speed_awac,
        'Direction_AWAC_Origineel': direction_awac,
        'Speed_AWAC_Smoothed': speed_awac_smoothed,
        'Direction_AWAC_Smoothed': direction_awac_smoothed
    })
    awac_selected_df.to_csv(awac_csv_path, index=False)
    logging.info(f"AWAC geselecteerde data opgeslagen als {awac_csv_path}")

    # Originele en gegladdde Vector data
    vector_csv_path = os.path.join(directory, f'Vector_selected_depth_{fixed_vector_depth:.2f}m.csv')
    vector_selected_df = pd.DataFrame({
        'Datetime': time_vector,
        'Resultant_Speed_Vector_Origineel': speed_vector,
        'Direction_Vector_Origineel': direction_vector,
        'Resultant_Speed_Vector_Smoothed': speed_vector_smoothed,
        'Direction_Vector_Smoothed': direction_vector_smoothed,
        'Depth_Vector_m': fixed_vector_depth  # Voeg de diepte toe als een constante kolom
    })
    vector_selected_df.to_csv(vector_csv_path, index=False)
    logging.info(f"Vector geselecteerde data opgeslagen als {vector_csv_path}")

    # Toon de plot
    plt.show()

# --------------------- GUI and Integration ---------------------

def main():
    # --------------------- AWAC Data Paths ---------------------
    awac_csv_file_path = r'C:\Users\claeysst\Desktop\werkfiles\verwerken AWAC\kanne04.csv'
    awac_time_bounds_file = os.path.join(os.path.dirname(awac_csv_file_path), 'begin-eind.txt')

    # --------------------- Vector Data Paths ---------------------
    vector_dat_file_path = r'C:\Users\claeysst\Desktop\werkfiles\Verwerken Vector\KANNE05.dat'
    vector_sen_file_path = r'C:\Users\claeysst\Desktop\werkfiles\Verwerken Vector\KANNE05.sen'
    vector_time_bounds_file = os.path.join(os.path.dirname(vector_dat_file_path), 'begin-eind.txt')

    # --------------------- Vector Column Names ---------------------
    dat_column_names = [
        'Burst_counter',
        'Ensemble_counter',
        'Velocity_Beam1',  # V1 (m/s)
        'Velocity_Beam2',  # V2 (m/s)
        'Velocity_Beam3',  # V3 (m/s)
        'Amplitude_Beam1',
        'Amplitude_Beam2',
        'Amplitude_Beam3',
        'SNR_Beam1',
        'SNR_Beam2',
        'SNR_Beam3',
        'Correlation_Beam1',
        'Correlation_Beam2',
        'Correlation_Beam3',
        'Pressure',         # Druksensor
        'Analog_input1',
        'Analog_input2',
        'Checksum'
    ]

    sen_column_names = [
        'Month',            # 1
        'Day',              # 2
        'Year',             # 3
        'Hour',             # 4
        'Minute',           # 5
        'Second',           # 6
        'Error_code',       # 7
        'Status_code',      # 8
        'Battery_voltage',  # 9 (V)
        'Soundspeed',       # 10 (m/s)
        'Heading',          # 11 (degrees)
        'Pitch',            # 12 (degrees)
        'Roll',             # 13 (degrees)
        'Temperature',      # 14 (degrees C)
        'Analog_input',     # 15
        'Checksum'          # 16 (1=failed)
    ]

    # --------------------- Load and Process AWAC Data ---------------------
    try:
        awac_start_time, awac_end_time = read_awac_time_period(awac_time_bounds_file)
        logging.info(f"AWAC: Starttijd: {awac_start_time}")
        logging.info(f"AWAC: Eindtijd: {awac_end_time}")
    except Exception:
        messagebox.showerror("Fout", "Fout bij het lezen van AWAC tijdsbestanden.")
        return

    try:
        awac_df_filtered, awac_speed_cols, awac_dir_cols, awac_cell_positions = load_awac_data(
            awac_csv_file_path, awac_start_time, awac_end_time)
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het laden van AWAC data: {e}")
        return

    # --------------------- Load and Process Vector Data ---------------------
    try:
        vector_start_time, vector_end_time = read_vector_time_bounds(vector_time_bounds_file)
        logging.info(f"Vector: Startdatum en -tijd: {vector_start_time}")
        logging.info(f"Vector: Einddatum en -tijd: {vector_end_time}")
    except Exception:
        messagebox.showerror("Fout", "Fout bij het lezen van Vector tijdsbestanden.")
        return

    try:
        vector_data_filtered, _, fixed_vector_depth = load_vector_data(
            vector_dat_file_path,
            vector_sen_file_path,
            dat_column_names,
            sen_column_names,
            vector_start_time,
            vector_end_time
        )
        if vector_data_filtered is None:
            logging.warning("Vector: Geen data beschikbaar na filtering.")
            messagebox.showwarning("Waarschuwing", "Geen Vector data beschikbaar na filtering.")
            return
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het laden van Vector data: {e}")
        return

    # --------------------- GUI Setup ---------------------
    root = tk.Tk()
    root.title("Diepte Selectie en Vergelijking")
    root.geometry("900x700")  # Vergroot de grootte indien nodig

    # Configureer grid
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=2)
    root.columnconfigure(2, weight=0)  # Voor de slider label

    # Functie om de selectie te verwerken
    def on_select():
        selected_depth_awac = depth_var_awac.get()
        awac_window_size = awac_window_entry.get()
        vector_window_size = vector_window_entry.get()
        time_shift_minutes = time_shift_var.get()  # Haal de tijdverschuiving op

        if not selected_depth_awac:
            messagebox.showerror("Fout", "Selecteer een AWAC diepte uit de lijst.")
            return

        try:
            depth_awac = float(selected_depth_awac)
        except ValueError:
            messagebox.showerror("Fout", "Ongeldige diepte geselecteerd.")
            return

        try:
            awac_window_size = int(awac_window_size)
            if awac_window_size < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Fout", "Voer een geldig integer waarde in voor het lopend gemiddelde van AWAC.")
            return

        try:
            vector_window_size = int(vector_window_size)
            if vector_window_size < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Fout", "Voer een geldig integer waarde in voor het lopend gemiddelde van Vector.")
            return

        # Zoek de dichtstbijzijnde diepte voor AWAC
        closest_idx_awac = (np.abs(awac_cell_positions - depth_awac)).argmin()
        closest_depth_awac = awac_cell_positions[closest_idx_awac]
        selected_speed_col_awac = awac_speed_cols[closest_idx_awac]
        selected_dir_col_awac = awac_dir_cols[closest_idx_awac]
        speed_awac = awac_df_filtered[selected_speed_col_awac].values
        direction_awac = awac_df_filtered[selected_dir_col_awac].values
        time_awac = awac_df_filtered.index

        # Toepassen van de tijdverschuiving op Vector data
        shifted_vector_data = vector_data_filtered.copy()
        shifted_vector_data['Datetime'] = shifted_vector_data['Datetime'] + pd.to_timedelta(time_shift_minutes, unit='m')

        speed_vector = shifted_vector_data['Resultant_Speed'].values
        direction_vector = shifted_vector_data['Direction'].values
        U_vector = shifted_vector_data['U'].values
        V_vector = shifted_vector_data['V'].values
        time_vector = shifted_vector_data['Datetime']

        # Toepassen van het lopend gemiddelde
        speed_awac_smoothed = apply_running_average(pd.Series(speed_awac), awac_window_size).values
        direction_awac_smoothed = apply_running_average(pd.Series(direction_awac), awac_window_size).values
        speed_vector_smoothed = apply_running_average(pd.Series(speed_vector), vector_window_size).values
        direction_vector_smoothed = apply_running_average(pd.Series(direction_vector), vector_window_size).values

        # Plot en sla de gegevens op
        plot_combined_data(
            time_awac, speed_awac, direction_awac, speed_awac_smoothed, direction_awac_smoothed,
            time_vector, speed_vector, direction_vector, speed_vector_smoothed, direction_vector_smoothed,
            closest_depth_awac, fixed_vector_depth,
            os.path.dirname(awac_csv_file_path),
            awac_window_size, vector_window_size  # Voeg beide window_sizes toe als parameters
        )

        # Bereken U en V componenten voor AWAC
        U_awac = speed_awac * np.cos(np.radians(direction_awac))
        V_awac = speed_awac * np.sin(np.radians(direction_awac))

        # Maak en plot de vectorplots voor beide instrumenten
        try:
            # AWAC Vector Plot
            plot_vector_plot(
                U_awac, V_awac, "AWAC",
                closest_depth_awac, os.path.dirname(awac_csv_file_path)
            )

            # Vector Instrument Vector Plot
            plot_vector_plot(
                U_vector, V_vector, "Vector",
                fixed_vector_depth, os.path.dirname(awac_csv_file_path)
            )
        except Exception as e:
            messagebox.showerror("Fout", f"Fout bij het genereren van vector plots: {e}")
            return

    # Label voor AWAC diepte
    label_awac = ttk.Label(root, text="Selecteer de gewenste AWAC diepte boven de bodem (m):")
    label_awac.grid(row=0, column=0, padx=10, pady=10, sticky='w')

    # Dropdown-menu voor AWAC dieptes
    available_depths_awac = sorted(awac_cell_positions)
    depth_var_awac = tk.StringVar()
    depth_dropdown_awac = ttk.Combobox(root, textvariable=depth_var_awac, values=available_depths_awac, state="readonly")
    depth_dropdown_awac.grid(row=0, column=1, padx=10, pady=10, sticky='ew')
    depth_dropdown_awac.set(available_depths_awac[0])  # Stel standaard waarde in

    # Label voor Vector diepte (automatisch berekend)
    label_vector = ttk.Label(root, text=f"Vector data is op een automatisch berekende diepte van {fixed_vector_depth:.2f}m.")
    label_vector.grid(row=1, column=0, padx=10, pady=10, sticky='w')

    # Label voor Lopend Gemiddelde AWAC
    label_awac_window = ttk.Label(root, text="Voer de grootte van het lopend gemiddelde voor AWAC in (aantal datapunten):")
    label_awac_window.grid(row=2, column=0, padx=10, pady=10, sticky='w')

    # Invoerveld voor Lopend Gemiddelde AWAC
    awac_window_var = tk.StringVar()
    awac_window_entry = ttk.Entry(root, textvariable=awac_window_var)
    awac_window_entry.grid(row=2, column=1, padx=10, pady=10, sticky='ew')
    awac_window_entry.insert(0, "1")  # Stel standaard waarde in

    # Label voor Lopend Gemiddelde Vector
    label_vector_window = ttk.Label(root, text="Voer de grootte van het lopend gemiddelde voor Vector in (aantal datapunten):")
    label_vector_window.grid(row=3, column=0, padx=10, pady=10, sticky='w')

    # Invoerveld voor Lopend Gemiddelde Vector
    vector_window_var = tk.StringVar()
    vector_window_entry = ttk.Entry(root, textvariable=vector_window_var)
    vector_window_entry.grid(row=3, column=1, padx=10, pady=10, sticky='ew')
    vector_window_entry.insert(0, "1")  # Stel standaard waarde in

    # Label voor Tijdverschuiving
    label_time_shift = ttk.Label(root, text="Tijdverschuiving tussen AWAC en Vector (minuten):")
    label_time_shift.grid(row=4, column=0, padx=10, pady=10, sticky='w')

    # Slider voor Tijdverschuiving
    time_shift_var = tk.IntVar()
    time_shift_slider = ttk.Scale(
        root,
        from_=-180,
        to=180,
        orient='horizontal',
        variable=time_shift_var,
        command=lambda val: time_shift_var.set(int(float(val)))  # Zorgt voor integer waarden
    )
    time_shift_slider.grid(row=4, column=1, padx=10, pady=10, sticky='ew')
    time_shift_slider.set(0)  # Stel standaard waarde in op 0

    # Label om de huidige waarde van de slider weer te geven
    time_shift_label = ttk.Label(root, textvariable=time_shift_var)
    time_shift_label.grid(row=4, column=2, padx=10, pady=10, sticky='w')

    # Knop om te bevestigen
    confirm_button = ttk.Button(root, text="Plot en Sla Op", command=on_select)
    confirm_button.grid(row=5, column=0, columnspan=2, padx=10, pady=20)

    # Knop om af te sluiten
    exit_button = ttk.Button(root, text="Afsluiten", command=root.destroy)
    exit_button.grid(row=6, column=0, columnspan=3, padx=10, pady=5)

    # Zorg ervoor dat de kolommen zich aanpassen aan de beschikbare ruimte
    root.grid_columnconfigure(1, weight=1)

    root.mainloop()

if __name__ == "__main__":
    main()
