import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, simpledialog
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import chardet

# Functie om de directory te selecteren
def select_directory():
    root = tk.Tk()
    root.withdraw()  # Verberg het hoofdvenster
    folder_selected = filedialog.askdirectory(title="Selecteer de directory met CSV-bestanden")
    return folder_selected

# Functie om offset in te voeren
def get_offset_input(label):
    root = tk.Tk()
    root.withdraw()  # Verberg het hoofdvenster
    while True:
        try:
            offset_value = simpledialog.askfloat(f"Voer {label} offset in", f"Voer de {label} offset in:")
            return offset_value if offset_value is not None else 0.0
        except ValueError:
            print("Ongeldige invoer. Probeer het opnieuw.")

# Functie om encoding te detecteren
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))  # Lees de eerste 10KB voor detectie
    return result['encoding'] if result['encoding'] else 'utf-8'

# Functie om sampling frequentie en startdatum/tijd uit de header te lezen
def extract_file_info(file_path):
    encoding = detect_encoding(file_path)
    with open(file_path, 'r', encoding=encoding) as f:
        lines = f.readlines()

    # Debug: toon de relevante headerregels zodat je kunt checken of de juiste data aanwezig is.
    print(f"\nDEBUG - In bestand: {file_path}")
    if len(lines) > 4:
        print(f"DEBUG - line[4] (verwacht sampling freq): {lines[4].rstrip()}")
    if len(lines) > 9:
        print(f"DEBUG - line[9] (verwacht datum):         {lines[9].rstrip()}")
    if len(lines) > 10:
        print(f"DEBUG - line[10] (verwacht tijd):         {lines[10].rstrip()}")

    # 1) Sampling frequentie staat in rij 5 (index 4), gescheiden door komma's
    sampling_frequency = 8.0
    try:
        sampling_line = lines[4].strip()
        parts_sampling = sampling_line.split(",")
        sampling_frequency = float(parts_sampling[1].strip())
    except Exception as e:
        print(f"Fout bij uitlezen van de sampling frequentie: {e} - fallback = 8 Hz")

    # 2) Startdatum: rij 10 (index 9) na de 2e komma
    # 3) Starttijd: rij 11 (index 10) na de 2e komma
    start_datetime = datetime.now()
    try:
        date_line = lines[9].strip()
        parts_date = date_line.split(",")
        start_date_str = parts_date[2].strip()
        
        time_line = lines[10].strip()
        parts_time = time_line.split(",")
        start_time_str = parts_time[2].strip()
        
        # Combineer datum en tijd tot één datetime-object
        start_datetime = datetime.strptime(start_date_str + " " + start_time_str, "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Fout bij uitlezen datum/tijd: {e} - fallback = {start_datetime}")

    return sampling_frequency, start_datetime

# Functie om een bestand te verwerken en meetwaarden te extraheren
def process_wave_file(file_path, pressure_offset=0):
    print(f"\nBestand inlezen: {file_path}")
    try:
        encoding = detect_encoding(file_path)
        # Extraheer sampling frequentie en startdatum/tijd uit de header
        sampling_frequency, start_datetime = extract_file_info(file_path)
        
        # Lees de data, sla de eerste 11 rijen (header) over
        data_numeric = pd.read_csv(file_path, encoding=encoding, skiprows=11, header=None)
        
        # Filter op 'C1' in kolom 1
        filtered_data = data_numeric[data_numeric[1] == 'C1']
        
        # Indien offset 0 is, gebruik 'coerce' zodat mogelijke conversiefouten niet leiden tot een Exception
        if pressure_offset == 0:
            measurements = pd.to_numeric(filtered_data[2], errors='coerce')
            # Verwijder NaN-waarden indien er conversiefouten waren
            measurements = measurements.dropna().reset_index(drop=True)
        else:
            measurements = pd.to_numeric(filtered_data[2], errors='raise') + pressure_offset
        
        print(f"Sampling frequentie = {sampling_frequency} Hz, start_datetime = {start_datetime}")
        print(f"Meetwaarden gevonden: {len(measurements)}")
        
        return measurements, sampling_frequency, start_datetime
    except Exception as e:
        print(f"Fout bij het verwerken van {file_path}: {e}")
        return None, None, None

# Functie om de grafiek te plotten
def plot_combined_graph(df_combined, label='Wave Druk'):
    plt.figure(figsize=(12, 7))
    plt.plot(df_combined['Datetime'], df_combined['Pressure'],
             label=label, color='blue', marker='o', markersize=2, linewidth=0.5)
    plt.title(f"{label} over Tijd")
    plt.xlabel('Tijd')
    plt.ylabel('Meetwaarde')
    plt.gcf().autofmt_xdate()  # Automatisch roteren van de datums op de x-as
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'lijngrafiek.png')
    plt.savefig(output_path)
    print(f"Lijngrafiek opgeslagen als '{output_path}'")
    plt.show()

# Functie om gecombineerde gegevens te exporteren
def export_combined_data(df_combined, output_path):
    try:
        df_combined.to_csv(output_path, index=False)
        print(f"Gecombineerde gegevens succesvol geëxporteerd naar {output_path}")
    except Exception as e:
        print(f"Fout bij het exporteren van gegevens: {e}")

# Hoofdprogramma
def main():
    # Selecteer directory
    selected_directory = select_directory()
    if not selected_directory:
        print("Geen directory geselecteerd. Programma wordt afgesloten.")
        return

    # Vraag om offset
    wave_pressure_offset = get_offset_input('Wave druk')

    # Verzamelen van alle CSV-bestanden, maar sluit 'gecombineerde_data.csv' uit
    csv_files = [f for f in os.listdir(selected_directory) 
                 if f.lower().endswith('.csv') and f.lower() != 'gecombineerde_data.csv']
    if not csv_files:
        print("Geen CSV-bestanden gevonden in de geselecteerde directory.")
        return

    combined_data = []

    # Verwerken van elk CSV-bestand
    for file_name in csv_files:
        file_path = os.path.join(selected_directory, file_name)
        measurements, sampling_frequency, start_datetime = process_wave_file(file_path, wave_pressure_offset)
        if measurements is not None and sampling_frequency and start_datetime:
            # Maak tijdstempels op basis van de gevonden sampling frequentie
            dt_step = 1.0 / sampling_frequency
            timestamps = [start_datetime + timedelta(seconds=i * dt_step) for i in range(len(measurements))]
            df_temp = pd.DataFrame({'Datetime': timestamps, 'Pressure': measurements})
            combined_data.append(df_temp)

    if not combined_data:
        print("Geen meetgegevens gevonden om te combineren.")
        return

    # Combineer alle dataframes en sorteer op tijd
    df_combined = pd.concat(combined_data).sort_values('Datetime').reset_index(drop=True)
    print(f"\nTotaal aantal meetwaarden: {len(df_combined)}")

    # Plot de gecombineerde gegevens
    plot_combined_graph(df_combined, label='Wave Druk')

    # Exporteren van de gecombineerde gegevens
    output_file = os.path.join(selected_directory, 'gecombineerde_data.csv')
    export_combined_data(df_combined, output_file)

if __name__ == "__main__":
    main()
