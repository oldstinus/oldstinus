import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import chardet

# Functie om de directory te selecteren
def select_directory():
    root = tk.Tk()
    root.withdraw()  # Verberg het hoofdvenster
    folder_selected = filedialog.askdirectory()
    return folder_selected

# Functie om offset in te voeren
def get_offset_input(label):
    root = tk.Tk()
    root.withdraw()  # Verberg het hoofdvenster
    offset_value = simpledialog.askfloat(f"Voer {label} offset in", f"Voer de {label} offset in:")
    return offset_value if offset_value is not None else 0.0

# Functie om de aanmaakdatum van een bestand te verkrijgen
def get_file_creation_time(file_path):
    creation_time = os.path.getctime(file_path)
    return datetime.fromtimestamp(creation_time)

# Functie om een bestand te verwerken en meetwaarden te extraheren (wave CSV)
# Functie om een bestand te verwerken en meetwaarden te extraheren (wave CSV)
def process_wave_file(file_path, pressure_offset=0):
    print(f"Bestand inlezen: {file_path}")
    
    try:
        # Lees het bestand in, filter de juiste kolom en vermenigvuldig de druk met 1000 voordat de offset wordt toegepast
        data_numeric = pd.read_csv(file_path, encoding='latin1', skiprows=10, header=None)
        filtered_data = data_numeric[data_numeric[1] == 'C1']
        
        # Vermenigvuldig de meetwaarden met 1000 en voeg daarna de offset toe
        measurements = (filtered_data[2].astype(float) * 1000) + pressure_offset
        print(f"Meetwaarden gevonden: {len(measurements)}")
        return measurements
    except Exception as e:
        print(f"Fout bij het verwerken van {file_path}: {e}")
        return None


# Functie om de grafiek te plotten voor wave-data
def plot_wave_graph(timestamps, measurements, label='Wave Meetwaarden'):
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, measurements, label=label, color='red', marker='o', markersize=2, linewidth=0.5)
    plt.title(f"{label} over tijd")
    plt.xlabel('Tijd')
    plt.ylabel('Meetwaarde')
    plt.gcf().autofmt_xdate()  # Automatisch roteren van de datums op de x-as
    plt.legend()
    plt.grid(True)
    plt.show()

# Functie om een .mon-bestand te verwerken
def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
        result = chardet.detect(rawdata)
        return result['encoding']

def read_mon_file(file_path):
    encoding = detect_file_encoding(file_path)
    header_info = {}
    data_lines = []
    
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if 'END OF DATA' in line:
                    print("Einde van gegevensbestand bereikt.")
                    break  # Stop de verwerking als we het einde van het bestand bereiken
                if line_num < 54:
                    if '=' in line:
                        key_value = line.split('=')
                        header_info[key_value[0].strip()] = key_value[1].strip()
                else:
                    data_lines.append(line)
                    
    except Exception as e:
        print(f"Fout bij het lezen van het .mon bestand: {e}")
    
    return header_info, data_lines

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
        else:
            print(f"Onverwachte regelindeling: {line}")
    if not data:
        print("Geen geldige data gevonden in het .mon-bestand.")
    return pd.DataFrame(data)

# Functie om de grafiek te plotten voor MON data
def plot_mon_graph(df, label='Mon Druk'):
    plt.figure(figsize=(10, 6))
    plt.plot(df['Datetime'], df['Pressure'], label=label, color='blue', marker='x', markersize=2, linewidth=0.5)
    plt.xlabel('Datum/Tijd')
    plt.ylabel('Druk')
    plt.gcf().autofmt_xdate()  # Formatteer de datum
    plt.title(f'{label} over tijd')
    plt.legend()
    plt.grid(True)
    plt.show()

# Functie om de gecombineerde grafiek te plotten (wave en mon samen) met tijdselectie
def plot_combined_graph(df_wave, df_mon, start_time=None, end_time=None, wave_label='Wave Druk', mon_label='Mon Druk'):
    if start_time is not None and end_time is not None:
        df_wave = df_wave[(df_wave['Datetime'] >= start_time) & (df_wave['Datetime'] <= end_time)]
        df_mon = df_mon[(df_mon['Datetime'] >= start_time) & (df_mon['Datetime'] <= end_time)]
    
    plt.figure(figsize=(12, 6))

    # Plot voor Wave data
    plt.plot(df_wave['Datetime'], df_wave['Pressure'], label=wave_label, color='red', marker='o', markersize=2, linewidth=0.5)

    # Plot voor Mon data
    plt.plot(df_mon['Datetime'], df_mon['Pressure'], label=mon_label, color='blue', marker='x', markersize=2, linewidth=0.5)

    plt.title('Wave vs Mon Druk over Tijd')
    plt.xlabel('Tijd')
    plt.ylabel('Druk')
    plt.gcf().autofmt_xdate()
    plt.legend()
    plt.grid(True)
    plt.show()

# Functie voor GUI tijdselectie
def create_time_selection_gui(df_wave, df_mon, update_plot_callback):
    root = tk.Tk()
    root.title("Tijdselectie voor grafiek")

    min_time = min(df_wave['Datetime'].min(), df_mon['Datetime'].min())
    max_time = max(df_wave['Datetime'].max(), df_mon['Datetime'].max())

    # Labels en invoervelden voor begin- en eindtijd
    tk.Label(root, text="Begin tijd (YYYY-MM-DD HH:MM:SS):").grid(row=0, column=0)
    start_time_entry = tk.Entry(root, width=25)
    start_time_entry.insert(0, min_time.strftime("%Y-%m-%d %H:%M:%S"))
    start_time_entry.grid(row=0, column=1)

    tk.Label(root, text="Eind tijd (YYYY-MM-DD HH:MM:SS):").grid(row=1, column=0)
    end_time_entry = tk.Entry(root, width=25)
    end_time_entry.insert(0, max_time.strftime("%Y-%m-%d %H:%M:%S"))
    end_time_entry.grid(row=1, column=1)

    # Functie om de plot bij te werken bij het klikken op de knop
    def update_plot():
        try:
            start_time = datetime.strptime(start_time_entry.get(), "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(end_time_entry.get(), "%Y-%m-%d %H:%M:%S")
            update_plot_callback(start_time, end_time)
        except ValueError:
            tk.messagebox.showerror("Fout", "Ongeldige tijdinvoer. Gebruik het juiste formaat: YYYY-MM-DD HH:MM:SS")

    # Knop om de grafiek te updaten
    update_button = ttk.Button(root, text="Update Grafiek", command=update_plot)
    update_button.grid(row=2, column=1)

    root.mainloop()

# Functie om CSV-bestand te exporteren
def export_to_csv(df_wave, df_mon, filename='output.csv'):
    merged_df = pd.DataFrame({'WaveTime': df_wave['Datetime'], 'WavePressure': df_wave['Pressure'], 
                              'MonTime': df_mon['Datetime'], 'MonPressure': df_mon['Pressure']})
    merged_df.to_csv(filename, index=False)
    print(f"Data geëxporteerd naar {filename}")

# Hoofdprogramma
def main():
    # Verwerking van de wave bestanden
    selected_directory = select_directory()
    if selected_directory:
        files = [f for f in os.listdir(selected_directory) if f.endswith('.CSV')]
        if files:
            first_file = os.path.join(selected_directory, files[0])
            print(f"Verwerken van wave bestand: {first_file}")
            
            wave_pressure_offset = get_offset_input('Wave druk')
            measurements = process_wave_file(first_file, wave_pressure_offset)
            
            if measurements is not None:
                creation_time = get_file_creation_time(first_file)
                timestamps = [creation_time + timedelta(seconds=i/8) for i in range(len(measurements))]
                df_wave = pd.DataFrame({'Datetime': timestamps, 'Pressure': measurements})
                plot_wave_graph(timestamps, measurements, label='Wave Druk')

    # Verwerking van .mon bestanden
    mon_file = filedialog.askopenfilename(title="Selecteer een .mon bestand", filetypes=[("MON bestanden", "*.mon")])
    if mon_file:
        print(f"Verwerken van .mon bestand: {mon_file}")
        
        mon_pressure_offset = get_offset_input('Mon druk')
        mon_time_offset = get_offset_input('Mon tijd (in seconden)')
        
        header_info, data_lines = read_mon_file(mon_file)
        df_mon = parse_mon_data(data_lines, mon_pressure_offset, mon_time_offset)
        plot_mon_graph(df_mon, label='Mon Druk')

        if df_wave is not None:
            # Functie om de plot te updaten vanuit de GUI
            def update_plot_with_time_range(start_time, end_time):
                plot_combined_graph(df_wave, df_mon, start_time, end_time)

            # Toon GUI voor tijdselectie
            create_time_selection_gui(df_wave, df_mon, update_plot_with_time_range)
            export_to_csv(df_wave, df_mon)

if __name__ == "__main__":
    main()
