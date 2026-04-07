import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Functie om de start- en einddatum/tijd uit het txt bestand te lezen
def read_time_bounds(txt_file_path):
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
        print(f"Fout: Het bestand {txt_file_path} is niet gevonden.")
        exit(1)
    except ValueError as ve:
        print(f"Fout bij het lezen van {txt_file_path}: {ve}")
        exit(1)

# Definieer het pad naar je .dat en .sen bestanden
dat_file_path = r'C:\Users\claeysst\Desktop\werkfiles\Verwerken Vector/KANNE05.dat'
sen_file_path = r'C:\Users\claeysst\Desktop\werkfiles\Verwerken Vector/KANNE05.sen'

# Definieer het pad naar het begin-eind.txt bestand (zelfde directory als .dat bestand)
data_dir = os.path.dirname(dat_file_path)
time_bounds_file = os.path.join(data_dir, 'begin-eind.txt')

# Lees de tijdsgrenzen uit het txt bestand
start_datetime, end_datetime = read_time_bounds(time_bounds_file)
print(f"Startdatum en -tijd: {start_datetime}")
print(f"Einddatum en -tijd: {end_datetime}")

# Definieer de kolomnamen voor het .dat bestand op basis van de gegeven structuur
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
    'Pressure',
    'Analog_input1',
    'Analog_input2',
    'Checksum'
]

# Definieer de kolomnamen voor het .sen bestand op basis van de gegeven structuur
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

# Lees het .dat bestand in een pandas DataFrame
try:
    data_dat = pd.read_csv(dat_file_path, sep='\s+', header=None, names=dat_column_names, comment='#')
    print("Het .dat-bestand is succesvol ingelezen.")
except Exception as e:
    print(f"Fout bij het lezen van het .dat bestand: {e}")
    exit(1)

# Lees het .sen bestand in een pandas DataFrame
try:
    data_sen = pd.read_csv(sen_file_path, sep='\s+', header=None, names=sen_column_names, comment='#')
    print("Het .sen-bestand is succesvol ingelezen.")
except Exception as e:
    print(f"Fout bij het lezen van het .sen bestand: {e}")
    exit(1)

# Controleer of het aantal rijen in .dat en .sen hetzelfde is
if len(data_dat) != len(data_sen):
    print("Waarschuwing: Het aantal rijen in .dat en .sen bestanden komt niet overeen.")
    print(f".dat rijen: {len(data_dat)}, .sen rijen: {len(data_sen)}")
    # Afhankelijk van de situatie kun je besluiten om verder te gaan of het script te stoppen
    # Hier gaan we ervan uit dat ze overeenkomen
else:
    print("Aantal rijen in .dat en .sen bestanden komt overeen.")

# Maak een datetime kolom uit de .sen data
try:
    data_sen['Datetime'] = pd.to_datetime(
        data_sen[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']],
        format='%Y %m %d %H %M %S'
    )
except Exception as e:
    print(f"Fout bij het maken van de Datetime kolom: {e}")
    exit(1)

# Voeg de datetime kolom toe aan de .dat DataFrame
data_dat['Datetime'] = data_sen['Datetime']

# Filter op geldige metingen (Checksum == 0)
data_valid = data_dat[data_dat['Checksum'] == 0].reset_index(drop=True)
print(f"Aantal geldige metingen: {len(data_valid)}")

# Filter de data op basis van de ingelezen tijdsperiode
mask = (data_valid['Datetime'] >= start_datetime) & (data_valid['Datetime'] <= end_datetime)
data_filtered = data_valid.loc[mask].reset_index(drop=True)
print(f"Aantal metingen in de geselecteerde periode: {len(data_filtered)}")

# Extraheer de beam-snelheden
V_beam = data_filtered[['Velocity_Beam1', 'Velocity_Beam2', 'Velocity_Beam3']].values

# Definieer de transformatie matrix
transformation_matrix = np.array([
    [2.7305, -1.3511, -1.3767],
    [0.0154, 2.3816, -2.3945],
    [0.3401, 0.3420, 0.3484]
])

# Bereken de ENU-snelheden
V_ENU = V_beam.dot(transformation_matrix.T)  # Transpose voor juiste matrixvermenigvuldiging

# Voeg de ENU-snelheden toe aan de DataFrame
data_filtered['Velocity_East'] = V_ENU[:, 0]
data_filtered['Velocity_North'] = V_ENU[:, 1]
data_filtered['Velocity_Up'] = V_ENU[:, 2]

# Bereken de resulterende snelheid (horizontaal) en richting
data_filtered['Resultant_Speed'] = np.sqrt(data_filtered['Velocity_East']**2 + data_filtered['Velocity_North']**2)
data_filtered['Direction'] = np.degrees(np.arctan2(data_filtered['Velocity_North'], data_filtered['Velocity_East']))
# Zorg ervoor dat de richting positief is (0-360 graden)
data_filtered['Direction'] = (data_filtered['Direction'] + 360) % 360

# Reorder de kolommen zodat Datetime de eerste kolom is
columns_order = ['Datetime'] + [
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
    'Checksum',
    'Velocity_East',
    'Velocity_North',
    'Velocity_Up',
    'Resultant_Speed',
    'Direction'
]

data_filtered = data_filtered[columns_order]

# Bekijk de eerste paar rijen met de nieuwe ENU-snelheden, resultaat, en richting
print("\nEerste 5 rijen met Datetime, ENU-snelheden, Resultante Snelheid en Richting:")
print(data_filtered[['Datetime', 'Velocity_East', 'Velocity_North', 'Velocity_Up', 'Resultant_Speed', 'Direction']].head())

# Definieer de output CSV-paden
output_csv_all_valid = os.path.join(data_dir, 'KANNE05_ENU_with_Time.csv')
output_csv_filtered = os.path.join(data_dir, 'KANNE05_ENU_with_Time_Filtered.csv')

# Opslaan van alle geldige metingen naar een CSV-bestand
data_valid.to_csv(output_csv_all_valid, index=False)
print(f"\nENU-snelheden met tijd, resultaat en richting opgeslagen in: {output_csv_all_valid}")

# Opslaan van de gefilterde resultaten naar een apart CSV-bestand
data_filtered.to_csv(output_csv_filtered, index=False)
print(f"Gefilterde ENU-snelheden met tijd, resultaat en richting opgeslagen in: {output_csv_filtered}")

# Optioneel: Visualisatie van de ENU-snelheden, Resultante Snelheid en Richting
plt.figure(figsize=(15, 20))

# Plot East Snelheid
plt.subplot(5, 1, 1)
plt.plot(data_filtered['Datetime'], data_filtered['Velocity_East'], label='East (E)', color='r')
plt.xlabel('Tijd')
plt.ylabel('Snelheid (m/s)')
plt.title('East (E) Snelheid')
plt.legend()

# Plot North Snelheid
plt.subplot(5, 1, 2)
plt.plot(data_filtered['Datetime'], data_filtered['Velocity_North'], label='North (N)', color='g')
plt.xlabel('Tijd')
plt.ylabel('Snelheid (m/s)')
plt.title('North (N) Snelheid')
plt.legend()

# Plot Up Snelheid
plt.subplot(5, 1, 3)
plt.plot(data_filtered['Datetime'], data_filtered['Velocity_Up'], label='Up (U)', color='b')
plt.xlabel('Tijd')
plt.ylabel('Snelheid (m/s)')
plt.title('Up (U) Snelheid')
plt.legend()

# Plot Resultante Snelheid
plt.subplot(5, 1, 4)
plt.plot(data_filtered['Datetime'], data_filtered['Resultant_Speed'], label='Resultant Speed', color='m')
plt.xlabel('Tijd')
plt.ylabel('Snelheid (m/s)')
plt.title('Resultante Snelheid (√E² + N²)')
plt.legend()

# Plot Richting
plt.subplot(5, 1, 5)
plt.plot(data_filtered['Datetime'], data_filtered['Direction'], label='Direction', color='c')
plt.xlabel('Tijd')
plt.ylabel('Richting (°)')
plt.title('Richting (0-360°)')
plt.legend()

plt.tight_layout()
plt.show()
