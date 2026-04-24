import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000  # Voorkom OverflowError bij grote datasets

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import os
import re
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Bibliotheek voor kalender
from tkcalendar import DateEntry

# -------------------------------------------------------
# Functie om een (eventueel negatieve) hoek te normaliseren naar [0, 360)
# -------------------------------------------------------
def parse_angle(angle_str):
    """
    Converteert een string naar float en normaliseert deze in de range [0, 360).
    Bijvoorbeeld: -113 -> 247, 400 -> 40, etc.
    """
    val = float(angle_str)
    return val % 360

# -------------------------------------------------------
# Functie voor optionele transformatie (volgens Transform.m)
# -------------------------------------------------------
def transform_data(speed, direction, heading, pitch, roll, status_bit):
    """
    Past een transformatie toe op speed en direction op basis van heading, pitch, roll en status_bit.
    Gebaseerd op een Transform.m-achtig script.
    """
    # Transformatiematrix zoals in Transform.m
    T = np.array([[2896, 2896, 0],
                  [-2896, 2896, 0],
                  [-2896, -2896, 5792]], dtype=float) / 4096.0

    # Als status_bit == 1, pas wat extra flips toe
    if status_bit == 1:
        T[1, :] = -T[1, :]
        T[2, :] = -T[2, :]

    # Bereken rotatiematrices op basis van heading/pitch/roll
    hh = np.pi * (heading - 90) / 180.0
    pp = np.pi * pitch / 180.0
    rr = np.pi * roll / 180.0

    H = np.array([[np.cos(hh), np.sin(hh), 0],
                  [-np.sin(hh), np.cos(hh), 0],
                  [0, 0, 1]])

    P = np.array([
        [np.cos(pp), -np.sin(pp)*np.sin(rr), -np.cos(rr)*np.sin(pp)],
        [0,          np.cos(rr),             -np.sin(rr)],
        [np.sin(pp), np.sin(rr)*np.cos(pp),   np.cos(pp)*np.cos(rr)]
    ])

    R = H @ P @ T

    # Zet speed en direction om in (u,v)
    rad = np.radians(direction)
    u = speed * np.cos(rad)
    v = speed * np.sin(rad)

    # Pas de transformatie toe (vooral op de horizontale componenten)
    enu_x = R[0,0]*u + R[0,1]*v  # R[0,2]*0 wordt genegeerd
    enu_y = R[1,0]*u + R[1,1]*v

    # Herbereken speed en direction
    new_speed = np.sqrt(enu_x**2 + enu_y**2)
    new_direction = (np.degrees(np.arctan2(enu_y, enu_x)) + 360) % 360

    return new_speed, new_direction

# -------------------------------------------------------
# Functie om 'getekende' snelheid te bepalen
# -------------------------------------------------------
def signed_speed(speed, direction, pos_min, pos_max):
    """
    Berekent een 'getekende' snelheid op basis van een range [pos_min, pos_max].
    - Als de richting in [pos_min, pos_max] ligt (rekening houdend met wrap-around), is de snelheid positief.
    - Anders is de snelheid negatief.

    direction, speed: numpy arrays van gelijke lengte.
    pos_min, pos_max: floats in [0, 360).
    """
    # Normaliseer richting
    direction = direction % 360

    # Als pos_min <= pos_max, dan is de range direct [pos_min, pos_max].
    # Anders (wrap-around), is de positieve range [pos_min, 360) en [0, pos_max].
    if pos_min <= pos_max:
        mask_positive = (direction >= pos_min) & (direction <= pos_max)
    else:
        mask_positive = (direction >= pos_min) | (direction <= pos_max)

    # Positieve snelheid waar mask_positive True is, anders negatief
    signed = np.where(mask_positive, speed, -speed)
    return signed

# -------------------------------------------------------
# Data inlezen en filteren, met optionele transformatie
# -------------------------------------------------------
def process_data(csv_file, date_file,
                 start_datetime, end_datetime,
                 perform_transformation, heading, pitch, roll, status_bit,
                 pos_min, pos_max):
    """
    Leest CSV in, filtert op de tijdsperiode en berekent (optioneel) getransformeerde en/of getekende snelheid.

    Parameters:
    -----------
    csv_file : str
        Pad naar het CSV-bestand
    date_file : str of None
        Pad naar een tekstbestand met begin- en eindtijd (eerste twee regels),
        of leeg als dit niet wordt gebruikt
    start_datetime, end_datetime : datetime.datetime
        Start- en eindtijd uit de GUI
    perform_transformation : bool
        True als transformatie moet worden toegepast
    heading, pitch, roll : float
        Hoeken in graden
    status_bit : int
        0 of 1
    pos_min, pos_max : float
        Hoekgrenzen in graden voor de positieve richting

    Returns:
    --------
    (processed_data, directory)
        processed_data: dict met data per diepte
        directory: map waarin de CSV staat (handig om plots op te slaan)
    """
    # Probeer CSV in te lezen
    try:
        df = pd.read_csv(csv_file, sep=';', header=0, encoding='utf-8')
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het lezen van het CSV bestand: {e}")
        return None

    # Kolomnamen opschonen
    df.columns = df.columns.str.strip()
    
    if 'DateTime' not in df.columns:
        messagebox.showerror("Fout", "De kolom 'DateTime' is niet gevonden in het DataFrame.")
        return None

    # DateTime kolom converteren
    try:
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M:%S')
    except Exception as e:
        messagebox.showerror("Fout", f"Fout bij het converteren van 'DateTime': {e}")
        return None

    # Als er een date_file is opgegeven, lees de eerste twee regels als start- en eindtijd
    if date_file:
        try:
            with open(date_file, 'r') as f:
                start_time_file = f.readline().strip()
                end_time_file = f.readline().strip()
            start_time = pd.to_datetime(start_time_file, format='%d/%m/%Y %H:%M:%S')
            end_time = pd.to_datetime(end_time_file, format='%d/%m/%Y %H:%M:%S')
        except Exception as e:
            messagebox.showerror("Fout", f"Fout bij het lezen van het datum bestand: {e}")
            return None
    else:
        # Gebruik de datetime-objecten uit de GUI
        start_time = start_datetime
        end_time = end_datetime

    # Filteren op de gekozen tijdsrange
    df_filtered = df[(df['DateTime'] >= start_time) & (df['DateTime'] <= end_time)]
    if df_filtered.empty:
        messagebox.showerror("Fout", "Geen data beschikbaar binnen de opgegeven tijdsperiode.")
        return None

    df_filtered.set_index('DateTime', inplace=True)

    # Zoeken naar kolommen met snelheid (Speed#) en richting (Dir#)
    speed_cols = [col for col in df_filtered.columns if col.startswith('Speed#')]
    dir_cols = [col for col in df_filtered.columns if col.startswith('Dir#')]
    speed_cols.sort()
    dir_cols.sort()

    if len(speed_cols) != len(dir_cols):
        messagebox.showerror("Fout", "Het aantal snelheids- en richtingskolommen komt niet overeen.")
        return None

    # Bepaal dieptes uit de kolomnamen
    depth_pattern = re.compile(r'\((\d+(?:\.\d+)?)m\)')
    cell_positions = []
    for col in speed_cols:
        match = depth_pattern.search(col)
        if match:
            cell_positions.append(float(match.group(1)))
        else:
            # Als geen diepte is gevonden, zet er een placeholder in
            cell_positions.append(np.nan)

    # Tijd als numpy array
    times = df_filtered.index.values.astype('datetime64[ns]')
    processed_data = {}

    # Doorloop elke cel (elk Speed#/Dir#-paar)
    for i, speed_col in enumerate(speed_cols):
        dir_col = dir_cols[i]
        speed_data = df_filtered[speed_col].values
        direction_data = df_filtered[dir_col].values

        # Eventueel transformeren
        if perform_transformation:
            speed_data, direction_data = transform_data(speed_data, direction_data, heading, pitch, roll, status_bit)

        # Bepaal de getekende snelheid
        speed_signed = signed_speed(speed_data, direction_data, pos_min, pos_max)

        # Stel de diepte in; als die niet bekend was, gebruik i om iets te verzinnen
        depth = cell_positions[i] if not np.isnan(cell_positions[i]) else (0.9 + 0.5 * i)

        processed_data[depth] = {
            'time': times,
            'speed': speed_data,
            'direction': direction_data,
            'speed_signed': speed_signed
        }
    
    return processed_data, os.path.dirname(csv_file)

# -------------------------------------------------------
# Plot-functie: tijdreeks van snelheid & richting
# -------------------------------------------------------
def plot_and_save(time, speed, direction, depth, directory):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    ax1.set_ylabel('Snelheid (m/s)', color='tab:blue')
    ax1.plot(time, speed, color='tab:blue', label='Snelheid')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title(f'Snelheid en Richting op Diepte {depth} m')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2.set_ylabel('Richting (graden)', color='tab:red')
    ax2.plot(time, direction, color='tab:red', label='Richting')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_xlabel('Tijd')
    ax2.grid(True)
    ax2.legend(loc='upper left')

    date_format = mdates.DateFormatter('%d/%m/%Y\n%H:%M:%S')
    ax2.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.tight_layout()
    save_path = os.path.join(directory, f'snelheid_richting_{depth}m.png')
    fig.savefig(save_path, dpi=300)
    plt.show()
    messagebox.showinfo("Info", f"Grafiek opgeslagen als {save_path}")

# -------------------------------------------------------
# Plot-functie: tijdreeks van getekende snelheid
# -------------------------------------------------------
def plot_signed_speed(time, speed_signed, depth, directory):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_ylabel('Getekende snelheid (m/s)')
    ax.plot(time, speed_signed, color='tab:purple', label='Getekende Snelheid')
    ax.set_xlabel('Tijd')
    ax.grid(True)
    ax.legend(loc='upper left')
    ax.set_title(f'Getekende Snelheid op Diepte {depth} m')

    date_format = mdates.DateFormatter('%d/%m/%Y\n%H:%M:%S')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.tight_layout()
    save_path = os.path.join(directory, f'getekende_snelheid_{depth}m.png')
    fig.savefig(save_path, dpi=300)
    plt.show()
    messagebox.showinfo("Info", f"Grafiek (getekende snelheid) opgeslagen als {save_path}")

# -------------------------------------------------------
# Plot-functie: vector plot
# -------------------------------------------------------
def plot_vector_graphs(time, speed, direction, depth, directory):
    """
    Maakt een quiver-plot van u en v, ingekleurd door de resulterende snelheidsgrootte.
    """
    # Bereken east- en north-componenten
    u = speed * np.cos(np.radians(direction))
    v = speed * np.sin(np.radians(direction))
    resultant = np.sqrt(u**2 + v**2)

    plt.figure(figsize=(10, 8))
    plt.quiver(u, v, u, v, resultant, angles='xy', scale_units='xy', scale=1, cmap='viridis', width=0.005)
    plt.title(f'Snelheidsvectoren op diepte {depth} m')
    plt.xlabel('Velocity East (m/s)')
    plt.ylabel('Velocity North (m/s)')
    plt.grid(True)
    plt.colorbar(label='Resultante Snelheid (m/s)')
    vector_save_path = os.path.join(directory, f'vector_plot_{depth}m.png')
    plt.savefig(vector_save_path, dpi=300)
    plt.show()
    messagebox.showinfo("Info", f"Vector plot opgeslagen als {vector_save_path}")

# -------------------------------------------------------
# Plot-functie: windrose/polar plot
# -------------------------------------------------------
def plot_wind_rose(speed, direction, depth, directory):
    """
    Maakt een polar-scatter-plot van speed en direction, met 0°=noord (op top).
    """
    # Bereken de kompasrichting: 0° = noorden
    compass_bearing = (90 - direction) % 360
    angles = np.radians(compass_bearing)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    sc = ax.scatter(angles, speed, c=speed, cmap='viridis', alpha=0.75)
    ax.set_title(f"Snelheid-Richting-Roos op diepte {depth} m")
    plt.colorbar(sc, label='Snelheid (m/s)')

    rose_save_path = os.path.join(directory, f'windrose_{depth}m.png')
    plt.savefig(rose_save_path, dpi=300)
    plt.show()
    messagebox.showinfo("Info", f"Windrose plot opgeslagen als {rose_save_path}")

# -------------------------------------------------------
# Hoofd-GUI
# -------------------------------------------------------
def main():
    root = tk.Tk()
    root.title("Data Verwerking")

    # ---------------------
    # GUI-variabelen
    # ---------------------
    csv_file_path = tk.StringVar()
    date_file_path = tk.StringVar()

    # Standaard waarden voor de tijd
    start_hour_var = tk.StringVar(value="00")
    start_min_var = tk.StringVar(value="00")
    start_sec_var = tk.StringVar(value="00")

    end_hour_var = tk.StringVar(value="00")
    end_min_var = tk.StringVar(value="00")
    end_sec_var = tk.StringVar(value="00")

    perform_transformation = tk.BooleanVar()
    heading_var = tk.StringVar()
    pitch_var = tk.StringVar()
    roll_var = tk.StringVar()
    status_bit_var = tk.StringVar()

    # Default-waarden voor de richtingsrange (positief)
    # Bijvoorbeeld 68 en -113 (equivalent aan 247) -> effectively [68, 247].
    pos_min_var = tk.StringVar(value="68")
    pos_max_var = tk.StringVar(value="-113")

    processed_data_global = {}
    data_directory = ""

    # ---------------------
    # Functies voor bestandselectie
    # ---------------------
    def select_csv_file():
        path = filedialog.askopenfilename(title="Selecteer CSV bestand", 
                                          filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if path:
            csv_file_path.set(path)

    def select_date_file():
        path = filedialog.askopenfilename(title="Selecteer datum bestand (bijv. begin-eind.txt)", 
                                          filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if path:
            date_file_path.set(path)

    # ---------------------
    # Functie om de data te verwerken
    # ---------------------
    def process_button():
        csv_path = csv_file_path.get()
        if not csv_path:
            messagebox.showerror("Fout", "Selecteer alstublieft een CSV bestand.")
            return

        date_path = date_file_path.get()  # Optioneel

        # Lees de datums uit de DateEntry-widgets
        start_date = start_cal.get_date()  # datetime.date
        end_date = end_cal.get_date()      # datetime.date

        # Lees de tijden uit de spinboxen
        try:
            sh = int(start_hour_var.get())
            sm = int(start_min_var.get())
            ss = int(start_sec_var.get())
            eh = int(end_hour_var.get())
            em = int(end_min_var.get())
            es = int(end_sec_var.get())
        except ValueError:
            messagebox.showerror("Fout", "Uur, minuut en seconde moeten gehele getallen zijn.")
            return

        # Bouw volledige datetime-objects
        import datetime
        start_datetime = datetime.datetime(start_date.year, start_date.month, start_date.day, sh, sm, ss)
        end_datetime = datetime.datetime(end_date.year, end_date.month, end_date.day, eh, em, es)

        if start_datetime >= end_datetime:
            messagebox.showerror("Fout", "Startdatum/tijd moet vóór einddatum/tijd liggen.")
            return

        # Haal heading, pitch, roll en status op
        try:
            head = float(heading_var.get()) if heading_var.get() else 0.0
            pit = float(pitch_var.get()) if pitch_var.get() else 0.0
            rol = float(roll_var.get()) if roll_var.get() else 0.0
            stat = int(status_bit_var.get()) if status_bit_var.get() else 0
        except ValueError:
            messagebox.showerror("Fout", "Heading, pitch, roll en status moeten numeriek zijn (status int).")
            return

        # Parse de min/max richtingshoek
        try:
            pos_min = parse_angle(pos_min_var.get())
            pos_max = parse_angle(pos_max_var.get())
        except ValueError:
            messagebox.showerror("Fout", "Positieve richtingsrange moet numeriek zijn.")
            return

        # Verwerk de data
        result = process_data(
            csv_file=csv_path,
            date_file=date_path,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            perform_transformation=perform_transformation.get(),
            heading=head,
            pitch=pit,
            roll=rol,
            status_bit=stat,
            pos_min=pos_min,
            pos_max=pos_max
        )
        if result is None:
            return

        nonlocal processed_data_global, data_directory
        processed_data_global, data_directory = result

        # Open venster om diepte te selecteren
        open_depth_selection(processed_data_global, data_directory)

    # ---------------------
    # Venster om diepte te selecteren en grafieken te plotten
    # ---------------------
    def open_depth_selection(processed_data, directory):
        depth_window = tk.Toplevel(root)
        depth_window.title("Diepte Selectie")

        label = ttk.Label(depth_window, text="Selecteer de gewenste diepte boven de bodem (m):")
        label.pack(pady=10)

        depths = sorted(list(processed_data.keys()))
        depth_var = tk.StringVar()
        depth_dropdown = ttk.Combobox(depth_window, textvariable=depth_var, values=depths, state="readonly")
        depth_dropdown.pack(pady=5)
        depth_dropdown.set(depths[0])  # Standaard eerste waarde

        def plot_time_series():
            selected_depth = float(depth_var.get())
            cell_data = processed_data[selected_depth]
            plot_and_save(cell_data['time'], cell_data['speed'], cell_data['direction'], selected_depth, directory)

        def plot_vector():
            selected_depth = float(depth_var.get())
            cell_data = processed_data[selected_depth]
            plot_vector_graphs(cell_data['time'], cell_data['speed'], cell_data['direction'], selected_depth, directory)

        def plot_rose():
            selected_depth = float(depth_var.get())
            cell_data = processed_data[selected_depth]
            plot_wind_rose(cell_data['speed'], cell_data['direction'], selected_depth, directory)

        def plot_signed():
            selected_depth = float(depth_var.get())
            cell_data = processed_data[selected_depth]
            plot_signed_speed(cell_data['time'], cell_data['speed_signed'], selected_depth, directory)

        # Knoppen om verschillende plots te maken
        ts_button = ttk.Button(depth_window, text="Plot Tijdreeks (Snelheid & Richting)", command=plot_time_series)
        ts_button.pack(pady=5)

        vector_button = ttk.Button(depth_window, text="Plot Vector", command=plot_vector)
        vector_button.pack(pady=5)

        rose_button = ttk.Button(depth_window, text="Plot Windrose", command=plot_rose)
        rose_button.pack(pady=5)

        signed_button = ttk.Button(depth_window, text="Plot Getekende Snelheid", command=plot_signed)
        signed_button.pack(pady=5)

        close_button = ttk.Button(depth_window, text="Afsluiten", command=depth_window.destroy)
        close_button.pack(pady=5)

    # ---------------------
    # Layout van het hoofdscherm
    # ---------------------
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Rij 0
    ttk.Label(frame, text="CSV Bestand:").grid(row=0, column=0, sticky=tk.W)
    csv_entry = ttk.Entry(frame, textvariable=csv_file_path, width=50)
    csv_entry.grid(row=0, column=1, sticky=tk.W)
    ttk.Button(frame, text="Selecteer CSV", command=select_csv_file).grid(row=0, column=2, padx=5)

    # Rij 1
    ttk.Label(frame, text="Datum Bestand (optioneel):").grid(row=1, column=0, sticky=tk.W)
    date_entry = ttk.Entry(frame, textvariable=date_file_path, width=50)
    date_entry.grid(row=1, column=1, sticky=tk.W)
    ttk.Button(frame, text="Selecteer Datum Bestand", command=select_date_file).grid(row=1, column=2, padx=5)

    # Rij 2: startdatum
    ttk.Label(frame, text="Startdatum:").grid(row=2, column=0, sticky=tk.W)
    start_cal = DateEntry(frame, date_pattern='dd/mm/yyyy')
    start_cal.grid(row=2, column=1, sticky=tk.W)

    # Spinboxen voor starttijd
    ttk.Label(frame, text="Start (hh:mm:ss):").grid(row=2, column=2, sticky=tk.E)
    spin_sh = ttk.Spinbox(frame, from_=0, to=23, textvariable=start_hour_var, width=3)
    spin_sh.grid(row=2, column=3, sticky=tk.W)
    spin_sm = ttk.Spinbox(frame, from_=0, to=59, textvariable=start_min_var, width=3)
    spin_sm.grid(row=2, column=4, sticky=tk.W)
    spin_ss = ttk.Spinbox(frame, from_=0, to=59, textvariable=start_sec_var, width=3)
    spin_ss.grid(row=2, column=5, sticky=tk.W)

    # Rij 3: einddatum
    ttk.Label(frame, text="Einddatum:").grid(row=3, column=0, sticky=tk.W)
    end_cal = DateEntry(frame, date_pattern='dd/mm/yyyy')
    end_cal.grid(row=3, column=1, sticky=tk.W)

    # Spinboxen voor eindtijd
    ttk.Label(frame, text="Eind (hh:mm:ss):").grid(row=3, column=2, sticky=tk.E)
    spin_eh = ttk.Spinbox(frame, from_=0, to=23, textvariable=end_hour_var, width=3)
    spin_eh.grid(row=3, column=3, sticky=tk.W)
    spin_em = ttk.Spinbox(frame, from_=0, to=59, textvariable=end_min_var, width=3)
    spin_em.grid(row=3, column=4, sticky=tk.W)
    spin_es = ttk.Spinbox(frame, from_=0, to=59, textvariable=end_sec_var, width=3)
    spin_es.grid(row=3, column=5, sticky=tk.W)

    # Rij 4: transformatie-checkbox
    trans_check = ttk.Checkbutton(frame, text="Voer transformatie uit", variable=perform_transformation)
    trans_check.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)

    # Rij 5-8: heading, pitch, roll, status
    ttk.Label(frame, text="Heading (graden):").grid(row=5, column=0, sticky=tk.W)
    heading_entry = ttk.Entry(frame, textvariable=heading_var, width=25)
    heading_entry.grid(row=5, column=1, sticky=tk.W)

    ttk.Label(frame, text="Pitch (graden):").grid(row=6, column=0, sticky=tk.W)
    pitch_entry = ttk.Entry(frame, textvariable=pitch_var, width=25)
    pitch_entry.grid(row=6, column=1, sticky=tk.W)

    ttk.Label(frame, text="Roll (graden):").grid(row=7, column=0, sticky=tk.W)
    roll_entry = ttk.Entry(frame, textvariable=roll_var, width=25)
    roll_entry.grid(row=7, column=1, sticky=tk.W)

    ttk.Label(frame, text="Status Bit (0 of 1):").grid(row=8, column=0, sticky=tk.W)
    status_entry = ttk.Entry(frame, textvariable=status_bit_var, width=25)
    status_entry.grid(row=8, column=1, sticky=tk.W)

    # Rij 9: positieve richtingsrange
    ttk.Label(frame, text="Positieve Range (min/max graden):").grid(row=9, column=0, sticky=tk.W)
    pos_min_entry = ttk.Entry(frame, textvariable=pos_min_var, width=5)
    pos_min_entry.grid(row=9, column=1, sticky=tk.W)
    pos_max_entry = ttk.Entry(frame, textvariable=pos_max_var, width=5)
    pos_max_entry.grid(row=9, column=2, sticky=tk.W, padx=5)

    # Rij 10: Verwerkknop
    process_btn = ttk.Button(frame, text="Verwerk Data", command=process_button)
    process_btn.grid(row=10, column=0, columnspan=6, pady=10)

    root.mainloop()

# Start de GUI
if __name__ == "__main__":
    main()
