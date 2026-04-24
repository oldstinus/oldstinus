import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Functie om de directory te selecteren
def select_directory():
    root = tk.Tk()
    root.withdraw()  # Verberg het hoofdvenster
    folder_selected = filedialog.askdirectory()
    return folder_selected

# Functie om een bestand te verwerken en meetwaarden te extraheren
def process_wave_file(file_path):
    print(f"Bestand inlezen: {file_path}")
    
    try:
        # Lees het bestand in, sla de eerste 10 rijen met metagegevens over
        data_numeric = pd.read_csv(file_path, encoding='latin1', skiprows=10, header=None)

        # Filter de meetwaarden (alleen rijen waar kolom 1 gelijk is aan 'C1')
        filtered_data = data_numeric[data_numeric[1] == 'C1']
        
        # Meetwaarden in de derde kolom
        measurements = filtered_data[2].astype(float)

        print(f"Meetwaarden gevonden: {len(measurements)}")
        return measurements

    except Exception as e:
        print(f"Fout bij het verwerken van {file_path}: {e}")
        return None

# Functie om de aanmaakdatum van een bestand te verkrijgen
def get_file_creation_time(file_path):
    # Haal de aanmaakdatum van het bestand op (in seconden sinds epoch)
    creation_time = os.path.getctime(file_path)
    # Converteer naar een datetime-object
    return datetime.fromtimestamp(creation_time)

# Functie om alle bestanden in een directory te verwerken en grafieken te maken
def process_files_in_directory(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.CSV')]  # Selecteer alleen .CSV-bestanden
    if not files:
        print("Geen .CSV-bestanden gevonden in de directory.")
        return

    for file in files:
        file_path = os.path.join(directory, file)
        print(f"Verwerken van bestand: {file_path}")

        # Haal de aanmaakdatum van het bestand op
        creation_time = get_file_creation_time(file_path)
        print(f"Aanmaakdatum van het bestand: {creation_time}")

        # Verwerk het bestand om de meetwaarden te extraheren
        measurements = process_wave_file(file_path)
        if measurements is not None:
            # Genereer tijdstempels op basis van de aanmaakdatum en samplefrequentie (8 Hz)
            timestamps = [creation_time + timedelta(seconds=i/8) for i in range(len(measurements))]

            # Plot de meetwaarden met de gegenereerde tijdstempels
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, measurements, label='Meetwaarden')
            plt.title(f"Meetwaarden voor bestand: {file}")
            plt.xlabel('Tijd')
            plt.ylabel('Meetwaarde')
            plt.gcf().autofmt_xdate()  # Automatisch roteren van de datums op de x-as
            plt.legend()
            plt.show()

# Hoofdprogramma
if __name__ == "__main__":
    selected_directory = select_directory()
    if selected_directory:
        print(f"Geselecteerde directory: {selected_directory}")
        process_files_in_directory(selected_directory)
    else:
        print("Geen directory geselecteerd.")
