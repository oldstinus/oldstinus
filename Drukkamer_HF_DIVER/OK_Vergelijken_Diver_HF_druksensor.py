import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from datetime import datetime, timedelta
import chardet
from scipy.stats import linregress
import numpy as np

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

# Functie om de aanmaakdatum van een bestand te verkrijgen
def get_file_creation_time(file_path):
    creation_time = os.path.getctime(file_path)
    return datetime.fromtimestamp(creation_time)

# Functie om een wave CSV-bestand te verwerken
def process_wave_file(file_path, pressure_offset=0):
    print(f"Bestand inlezen: {file_path}")
    try:
        data_numeric = pd.read_csv(file_path, encoding='latin1', skiprows=10, header=None)
        filtered_data = data_numeric[data_numeric[1] == 'C1']
        measurements = (filtered_data[2].astype(float) * 1000) + pressure_offset
        print(f"Meetwaarden gevonden: {len(measurements)}")
        return measurements
    except Exception as e:
        print(f"Fout bij het verwerken van {file_path}: {e}")
        return None

# Functie om de bestandsencoding te detecteren
def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
        result = chardet.detect(rawdata)
        return result['encoding']

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
        if len(parts) >= 4 and "/" in parts[0] and ":" in parts[1]:
            try:
                date_str = parts[0] + " " + parts[1]
                datetime_obj = datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S.%f") + timedelta(seconds=time_offset)
                pressure = float(parts[2].replace(',', '.')) + pressure_offset
                data.append({'Datetime': datetime_obj, 'Pressure': pressure})
            except ValueError as ve:
                print(f"Fout bij het parsen van regel: {line}. Error: {ve}")
    return pd.DataFrame(data)

# Functie om druk om te rekenen naar verschillende eenheden
def convert_pressure_units(pressure_pa):
    pressure_mmH2O = pressure_pa / 9.80665  # Omrekening van Pa naar mmH2O
    pressure_mmHg = pressure_pa / 133.322   # Omrekening van Pa naar mmHg
    pressure_bar = pressure_pa / 100000     # Omrekening van Pa naar bar
    return pressure_pa, pressure_mmH2O, pressure_mmHg, pressure_bar

# Functie om de maximum-, begin- en einddrukken weer te geven in verschillende eenheden
def display_pressure_summary(df_wave, df_mon):
    try:
        # Verkrijg de begin-, eind- en maximumdruk in Pa
        wave_start_pressure = df_wave['Pressure'].iloc[0]
        wave_end_pressure = df_wave['Pressure'].iloc[-1]
        wave_max_pressure = df_wave['Pressure'].max()

        mon_start_pressure = df_mon['Pressure'].iloc[0]
        mon_end_pressure = df_mon['Pressure'].iloc[-1]
        mon_max_pressure = df_mon['Pressure'].max()

        # Zet drukken om naar verschillende eenheden
        summary_data = {
            'Druksoort': ['HF Druksonde Start', 'HF Druksonde Eind', 'HF Druksonde Max',
                          'Diver Druk Start', 'Diver Druk Eind', 'Diver Druk Max'],
            'Druk (Pa)': [
                wave_start_pressure, wave_end_pressure, wave_max_pressure,
                mon_start_pressure, mon_end_pressure, mon_max_pressure
            ],
            'Druk (mmH₂O)': [
                convert_pressure_units(wave_start_pressure)[1], convert_pressure_units(wave_end_pressure)[1],
                convert_pressure_units(wave_max_pressure)[1], convert_pressure_units(mon_start_pressure)[1],
                convert_pressure_units(mon_end_pressure)[1], convert_pressure_units(mon_max_pressure)[1]
            ],
            'Druk (mmHg)': [
                convert_pressure_units(wave_start_pressure)[2], convert_pressure_units(wave_end_pressure)[2],
                convert_pressure_units(wave_max_pressure)[2], convert_pressure_units(mon_start_pressure)[2],
                convert_pressure_units(mon_end_pressure)[2], convert_pressure_units(mon_max_pressure)[2]
            ],
            'Druk (bar)': [
                convert_pressure_units(wave_start_pressure)[3], convert_pressure_units(wave_end_pressure)[3],
                convert_pressure_units(wave_max_pressure)[3], convert_pressure_units(mon_start_pressure)[3],
                convert_pressure_units(mon_end_pressure)[3], convert_pressure_units(mon_max_pressure)[3]
            ]
        }

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
            tree.insert("", "end", values=(row['Druksoort'],
                                          f"{row['Druk (Pa)']:.2f}",
                                          f"{row['Druk (mmH₂O)']:.2f}",
                                          f"{row['Druk (mmHg)']:.2f}",
                                          f"{row['Druk (bar)']:.5f}"))

        tree.pack(expand=True, fill='both')

        # Voeg een sluitknop toe
        close_button = ttk.Button(summary_window, text="Sluiten", command=summary_window.destroy)
        close_button.pack(pady=10)

    except Exception as e:
        print(f"Fout bij het genereren van de druk samenvatting: {e}")

# Functie om de tijdselectie GUI te creëren
def create_time_selection_gui(df_wave, df_mon, selected_directory, update_plot_callback, export_and_close_callback):
    root = tk.Tk()
    root.title("Tijdselectie voor Grafiek")

    min_time = min(df_wave['Datetime'].min(), df_mon['Datetime'].min())
    max_time = max(df_wave['Datetime'].max(), df_mon['Datetime'].max())

    # Labels en invoervelden voor begin- en eindtijd
    ttk.Label(root, text="Begin tijd (YYYY-MM-DD HH:MM:SS):").grid(row=0, column=0, padx=5, pady=5, sticky='e')
    start_time_entry = ttk.Entry(root, width=25)
    start_time_entry.insert(0, min_time.strftime("%Y-%m-%d %H:%M:%S"))
    start_time_entry.grid(row=0, column=1, padx=5, pady=5)

    ttk.Label(root, text="Eind tijd (YYYY-MM-DD HH:MM:SS):").grid(row=1, column=0, padx=5, pady=5, sticky='e')
    end_time_entry = ttk.Entry(root, width=25)
    end_time_entry.insert(0, max_time.strftime("%Y-%m-%d %H:%M:%S"))
    end_time_entry.grid(row=1, column=1, padx=5, pady=5)

    # Functie om de plot bij te werken bij het klikken op de knop
    def update_plot():
        try:
            start_time = datetime.strptime(start_time_entry.get(), "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time_entry.get(), "%Y-%m-%d %H:%M:%S")
            if start_time >= end_time:
                messagebox.showerror("Ongeldige Tijd", "Begin tijd moet voor eind tijd zijn.")
                return
            update_plot_callback(start_time, end_time)
        except ValueError:
            messagebox.showerror("Ongeldige Invoer", "Gebruik het juiste formaat: YYYY-MM-DD HH:MM:SS")

    # Functie om te exporteren en sluiten
    def export_and_close():
        try:
            start_time = datetime.strptime(start_time_entry.get(), "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time_entry.get(), "%Y-%m-%d %H:%M:%S")
            if start_time >= end_time:
                messagebox.showerror("Ongeldige Tijd", "Begin tijd moet voor eind tijd zijn.")
                return
            export_and_close_callback(start_time, end_time)
            root.destroy()  # Sluit de GUI na export
        except ValueError:
            messagebox.showerror("Ongeldige Invoer", "Gebruik het juiste formaat: YYYY-MM-DD HH:MM:SS")

    # Knop om de grafiek te updaten
    update_button = ttk.Button(root, text="Update Grafiek", command=update_plot)
    update_button.grid(row=2, column=0, padx=5, pady=10, sticky='e')

    # Knop om de data te exporteren en de GUI te sluiten
    export_button = ttk.Button(root, text="Export Data en Sluiten", command=export_and_close)
    export_button.grid(row=2, column=1, padx=5, pady=10, sticky='w')

    root.mainloop()

# Functie om de gecombineerde grafiek te plotten
def plot_combined_graph(df_wave, df_mon, start_time=None, end_time=None):
    try:
        print("Start plot_combined_graph")
        fig, ax = plt.subplots(figsize=(12, 6))

        if start_time and end_time:
            mask_wave = (df_wave['Datetime'] >= start_time) & (df_wave['Datetime'] <= end_time)
            mask_mon = (df_mon['Datetime'] >= start_time) & (df_mon['Datetime'] <= end_time)
            df_wave_plot = df_wave[mask_wave]
            df_mon_plot = df_mon[mask_mon]
        else:
            df_wave_plot = df_wave
            df_mon_plot = df_mon

        # Filteren van drukwaarden <=2100
        df_wave_plot = df_wave_plot[df_wave_plot['Pressure'] <= 2100]
        df_mon_plot = df_mon_plot[df_mon_plot['Pressure'] <= 2100]

        ax.plot(df_wave_plot['Datetime'], df_wave_plot['Pressure'], label='HF druksonde druk', color='red', marker='o', markersize=2, linewidth=0.5)
        ax.plot(df_mon_plot['Datetime'], df_mon_plot['Pressure'], label='Diver druk', color='blue', marker='x', markersize=2, linewidth=0.5)

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
def plot_xy_regression_with_slider(df_wave, df_mon, start_time=None, end_time=None):
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

# Functie om data te exporteren naar aparte CSV-bestanden
def export_data(df_wave, df_mon, selected_directory, start_time, end_time):
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

# Hoofdprogramma
def main():
    # Verwerking van de wave bestanden
    selected_directory = select_directory()
    if selected_directory:
        # Selecteer alleen .csv bestanden die niet eindigen op '_filtered.csv' en niet 'combined_data_filtered.csv'
        files = [f for f in os.listdir(selected_directory) if f.lower().endswith('.csv') and not f.endswith('_filtered.csv') and f != 'combined_data_filtered.csv']
        if files:
            all_measurements = []
            all_timestamps = []
            
            # Eenmalige prompt voor HF druksonde druk offset
            wave_pressure_offset = get_offset_input('HF druksonde druk')
            if wave_pressure_offset is None:
                wave_pressure_offset = 0.0

            for file in files:
                file_path = os.path.join(selected_directory, file)
                print(f"Verwerken van HF druksonde bestand: {file_path}")

                measurements = process_wave_file(file_path, wave_pressure_offset)

                if measurements is not None:
                    creation_time = get_file_creation_time(file_path)
                    timestamps = [creation_time + timedelta(seconds=i/8) for i in range(len(measurements))]
                    all_measurements.extend(measurements)
                    all_timestamps.extend(timestamps)

            df_wave = pd.DataFrame({'Datetime': all_timestamps, 'Pressure': all_measurements})
        else:
            messagebox.showwarning("Geen CSV-bestanden", "Geen CSV-bestanden gevonden in de geselecteerde directory (geen ongefilterde .csv bestanden).")
            return
    else:
        messagebox.showwarning("Geen Directory Geselecteerd", "Er is geen directory geselecteerd.")
        return

    # Verwerking van .mon bestanden
    mon_file = filedialog.askopenfilename(title="Selecteer een .mon bestand", filetypes=[("MON bestanden", "*.mon")])
    if mon_file:
        print(f"Verwerken van .mon bestand: {mon_file}")

        # Eenmalige prompts voor Diver druk en tijd offset
        mon_pressure_offset = get_offset_input('Diver druk')
        if mon_pressure_offset is None:
            mon_pressure_offset = 0.0

        mon_time_offset = get_offset_input('Diver tijd (in seconden)')
        if mon_time_offset is None:
            mon_time_offset = 0.0

        data_lines = read_mon_file(mon_file)
        if data_lines:
            df_mon = parse_mon_data(data_lines, mon_pressure_offset, mon_time_offset)
            if not df_mon.empty:
                # Toon druk samenvatting
                display_pressure_summary(df_wave, df_mon)
                
                # Tijdselectie GUI
                def update_plot_with_time_range(start_time, end_time):
                    plot_combined_graph(df_wave, df_mon, start_time, end_time)
                    plot_xy_regression_with_slider(df_wave, df_mon, start_time, end_time)

                def export_and_close(start_time, end_time):
                    plot_combined_graph(df_wave, df_mon, start_time, end_time)
                    plot_xy_regression_with_slider(df_wave, df_mon, start_time, end_time)
                    export_data(df_wave, df_mon, selected_directory, start_time, end_time)

                create_time_selection_gui(df_wave, df_mon, selected_directory, update_plot_with_time_range, export_and_close)
            else:
                messagebox.showerror("Geen Data", "Geen geldige data geparsed uit het .mon bestand.")
        else:
            messagebox.showerror("Geen Data", "Geen data gevonden in het .mon bestand.")
    else:
        messagebox.showwarning("Geen Bestand Geselecteerd", "Er is geen .mon bestand geselecteerd.")

if __name__ == "__main__":
    main()
