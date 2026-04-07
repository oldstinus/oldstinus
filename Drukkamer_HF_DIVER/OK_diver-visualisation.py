import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import chardet
import sys
import os

def select_file():
    try:
        print("Initialiseren van de bestandskiezer...")
        root = tk.Tk()
        root.withdraw()  # Verberg het hoofdvenster

        # Zorg ervoor dat de bestandskiezer boven alle andere vensters verschijnt
        root.attributes('-topmost', True)
        root.update()

        # Open de bestandskiezer
        file_path = filedialog.askopenfilename(
            title="Selecteer een .mon bestand",
            filetypes=[("MON bestanden", "*.mon"), ("Alle bestanden", "*.*")]
        )

        root.destroy()  # Sluit het venster na selectie

        if file_path:
            print(f"Geselecteerd bestand: {file_path}")
        else:
            print("Geen bestand geselecteerd.")
        return file_path
    except Exception as e:
        print(f"Fout bij het selecteren van een bestand: {e}")
        return None

def detect_file_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            rawdata = f.read()
            result = chardet.detect(rawdata)
            encoding = result['encoding']
            confidence = result['confidence']
            print(f"Detecteerde encoding: {encoding} met vertrouwen {confidence}")
            return encoding
    except Exception as e:
        print(f"Fout bij het detecteren van encoding: {e}")
        return None

def read_mon_file(file_path):
    header_info = {}
    data_lines = []
    encoding = detect_file_encoding(file_path)

    if not encoding:
        print("Kon de encoding niet detecteren. Gebruik standaard encoding 'utf-8'.")
        encoding = 'utf-8'

    try:
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if line_num < 54:
                    # Extract header info
                    if '=' in line:
                        key_value = line.split('=')
                        if len(key_value) >= 2:
                            key = key_value[0].strip()
                            value = '='.join(key_value[1:]).strip()
                            header_info[key] = value
                else:
                    # Data starts from line 54
                    if line:  # Negeer lege regels
                        data_lines.append(line)
        return header_info, data_lines
    except Exception as e:
        print(f"Fout bij het lezen van het bestand: {e}")
        return {}, []

def parse_data(data_lines):
    data = []
    for idx, line in enumerate(data_lines):
        if line:
            parts = line.split()
            if len(parts) >= 4:
                date_str = parts[0]
                time_str = parts[1]
                pressure_str = parts[2]
                temperature_str = parts[3]
                try:
                    # Combine date and time
                    datetime_str = f"{date_str} {time_str}"
                    # Convert to datetime object
                    datetime_obj = datetime.datetime.strptime(datetime_str, "%Y/%m/%d %H:%M:%S.%f")
                    # Convert strings to float, replace comma with dot if necessary
                    pressure = float(pressure_str.replace(',', '.'))
                    temperature = float(temperature_str.replace(',', '.'))
                    data.append({
                        'DatumTijd': datetime_obj,
                        'Druk': pressure,
                        'Temperatuur': temperature
                    })
                except ValueError as ve:
                    print(f"Fout bij het parsen van regel {idx + 54}: {ve}")
                    print(f"Regelinhoud: {line}")
            else:
                print(f"Onverwacht aantal velden in regel {idx + 54}: {line}")
    df = pd.DataFrame(data)
    return df

def plot_data(df, header_info):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Druk op de eerste as
    lijn1, = ax1.plot(df['DatumTijd'], df['Druk'], label='Druk', color='blue', marker='o', markersize=2,linewidth=0.1)
    ax1.set_xlabel('Datum/Tijd')
    ax1.set_ylabel('Druk')

    # Maak een tweede as voor Temperatuur
    ax2 = ax1.twinx()
    lijn2, = ax2.plot(df['DatumTijd'], df['Temperatuur'], label='Temperatuur', color='red', marker='x', markersize=2,linewidth=0.1)
    ax2.set_ylabel('Temperatuur (°C)')

    # Voeg legenda's toe
    # Haal de lijnen en labels van beide assen
    lijnen = [lijn1, lijn2]
    labels = [lijn1.get_label(), lijn2.get_label()]

    # Voeg de legenda toe aan figuur
    ax1.legend(lijnen, labels, loc='upper right')

    # Formatteer de datum op de x-as
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    fig.autofmt_xdate()

    # Voeg informatie uit de header toe aan de titel
    location = header_info.get('Location', 'Onbekende Locatie')
    serial_number = header_info.get('Serial number', 'Onbekend Serienummer')
    plt.title(f"Druk en Temperatuur vs Datum/Tijd\nLocatie: {location}, Serienummer: {serial_number}")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("Het script wordt geladen.")

def main():
    print("Script gestart.")
    file_path = select_file()
    print(f"Bestandspad na select_file(): {file_path}")
    if not file_path:
        print("Geen bestand geselecteerd. Script wordt beëindigd.")
        input("Druk op Enter om af te sluiten...")
        return

    print("Lees het .mon bestand...")
    header_info, data_lines = read_mon_file(file_path)
    print(f"Header informatie: {header_info}")
    print(f"Aantal dataregels: {len(data_lines)}")
    if not data_lines:
        print("Geen data gevonden in het bestand of er is een probleem met de encoding.")
        input("Druk op Enter om af te sluiten...")
        return

    print("Parsen van data...")
    df = parse_data(data_lines)
    print(f"Aantal rijen in DataFrame: {len(df)}")

    if df.empty:
        print("Geen geldige data gevonden in het bestand.")
        input("Druk op Enter om af te sluiten...")
        return

    print("Data succesvol geparsed. Genereren van de grafiek...")
    plot_data(df, header_info)
    print("Script voltooid.")
    input("Druk op Enter om af te sluiten...")

if __name__ == "__main__":
    print("Het script wordt direct uitgevoerd.")
    main()
else:
    print("Het script is geïmporteerd als module.")
