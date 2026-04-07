import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import StringIO
from datetime import datetime, timedelta

# Definieer het pad naar je .dat en .vhd bestanden
dat_file_path = r'P:\PA029-BMeetinstr-Cmp\3_Uitvoering\0_03_MEETCAMPAGNES_CALIBRATION_MEETNET\2022_06_13-2022_07_10_vergelijkende_meting_Albertkanaal_Kanne\data\Vector_AWAC_2022_05_31_stationair_ metingen\VECTOR\data\processing\export\KANNE05.dat'
vhd_file_path = r'P:\PA029-BMeetinstr-Cmp\3_Uitvoering\0_03_MEETCAMPAGNES_CALIBRATION_MEETNET\2022_06_13-2022_07_10_vergelijkende_meting_Albertkanaal_Kanne\data\Vector_AWAC_2022_05_31_stationair_ metingen\VECTOR\data\processing\export\KANNE05.vhd'

output_csv_path = r'C:\Users\claeysst\AppData\Roaming\Python\Python312\Scripts\oldstinus\verwerking_vector_ENU.csv'

# Controleer of de bestanden bestaan
if not os.path.exists(dat_file_path):
    print(f"Het .dat bestand {dat_file_path} bestaat niet. Controleer het pad.")
    exit(1)

if not os.path.exists(vhd_file_path):
    print(f"Het .vhd bestand {vhd_file_path} bestaat niet. Controleer het pad.")
    exit(1)

# Definieer de kolomnamen voor .vhd op basis van het gegeven voorbeeld
vhd_column_names = [
    'Month',
    'Day',
    'Year',
    'Hour',
    'Minute',
    'Second',
    'Burst_counter',
    'Unknown1',
    'Unknown2',
    'Unknown3',
    'Unknown4',
    'Unknown5',
    'Unknown6',
    'Unknown7',
    'Unknown8',
    'Unknown9',
    'Unknown10',
    'Unknown11',
    'Unknown12',
    'Unknown13',
    'Unknown14'
]

# Lees het .vhd bestand
try:
    with open(vhd_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        cleaned_content_vhd = f.read().replace('\x00', ' ')
    
    vhd_data = pd.read_csv(StringIO(cleaned_content_vhd), sep='\s+', header=None, names=vhd_column_names)
except Exception as e:
    print(f"Fout bij het lezen van het .vhd bestand: {e}")
    exit(1)

# Controleer de eerste paar rijen van .vhd data
print("Eerste 5 rijen van het .vhd bestand:")
print(vhd_data.head())

# Zorg ervoor dat de tijdkolommen numeriek zijn
time_columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']
for col in time_columns:
    vhd_data[col] = pd.to_numeric(vhd_data[col], errors='coerce')

# Controleer op NaN waarden in tijdkolommen
if vhd_data[time_columns].isnull().any().any():
    print("Waarschuwing: Er zijn NaN waarden in de tijdkolommen. Deze rijen zullen als NaT worden gemarkeerd.")
    
# Maak een datetime kolom aan in .vhd data
vhd_data['Datetime'] = pd.to_datetime(vhd_data[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']], errors='coerce')

# Selecteer relevante kolommen (Burst_counter en Datetime)
vhd_time = vhd_data[['Burst_counter', 'Datetime']].copy()

# Controleer of er duplicaten zijn in Burst_counter
if vhd_time['Burst_counter'].duplicated().any():
    print("Waarschuwing: Er zijn duplicaten in Burst_counter in het .vhd bestand. Dit kan problemen veroorzaken bij het mergen.")
    # Optioneel: beheers de duplicaten, bijvoorbeeld door de eerste te behouden
    vhd_time = vhd_time.drop_duplicates(subset='Burst_counter', keep='first')

# Lees het .dat bestand
# Definieer de kolomnamen op basis van de gegeven structuur
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

try:
    data = pd.read_csv(dat_file_path, sep='\s+', header=None, names=dat_column_names, encoding='utf-8', errors='ignore')
except Exception as e:
    print(f"Fout bij het lezen van het .dat bestand: {e}")
    exit(1)

# Controleer de eerste paar rijen van .dat data
print("\nEerste 5 rijen van het .dat bestand:")
print(data.head())

# Filter op geldige metingen (Checksum == 0)
data_valid = data[data['Checksum'] == 0].copy()
print(f"\nAantal geldige metingen: {len(data_valid)}")

# Merge de .dat data met de .vhd tijd data op Burst_counter
data_merged = pd.merge(data_valid, vhd_time, on='Burst_counter', how='left')

# Controleer of er Burst_counter zijn zonder tijdinformatie
missing_time = data_merged['Datetime'].isnull().sum()
if missing_time > 0:
    print(f"Waarschuwing: {missing_time} metingen hebben geen bijbehorende tijdinformatie in het .vhd bestand.")

# Bereken de absolute tijd voor elke meting
# Aangenomen dat de Ensemble_counter het aantal seconden na de Burst_start aangeeft
# (Aangezien de sampling rate 1 Hz)

# Zorg ervoor dat 'Ensemble_counter' numeriek is
data_merged['Ensemble_counter'] = pd.to_numeric(data_merged['Ensemble_counter'], errors='coerce')

# Voeg een kolom toe die het aantal seconden representeert
data_merged['Ensemble_seconds'] = data_merged['Ensemble_counter'].fillna(0).astype(int)

# Bereken de Absolute_Time
data_merged['Absolute_Time'] = data_merged.apply(
    lambda row: row['Datetime'] + timedelta(seconds=row['Ensemble_seconds']) if pd.notnull(row['Datetime']) else pd.NaT,
    axis=1
)

# Extraheer de beam-snelheden
V_beam = data_merged[['Velocity_Beam1', 'Velocity_Beam2', 'Velocity_Beam3']].values

# Definieer de transformatie matrix
transformation_matrix = np.array([
    [2.7305, -1.3511, -1.3767],
    [0.0154, 2.3816, -2.3945],
    [0.3401, 0.3420, 0.3484]
])

# Bereken de ENU-snelheden
V_ENU = V_beam.dot(transformation_matrix.T)

# Voeg de ENU-snelheden toe aan de DataFrame
data_merged['Velocity_East'] = V_ENU[:, 0]
data_merged['Velocity_North'] = V_ENU[:, 1]
data_merged['Velocity_Up'] = V_ENU[:, 2]

# Bekijk de eerste paar rijen met de nieuwe ENU-snelheden en Absolute_Time
print("\nEerste 5 rijen met ENU-snelheden en tijdstempels:")
print(data_merged[['Absolute_Time', 'Velocity_Beam1', 'Velocity_Beam2', 'Velocity_Beam3', 'Velocity_East', 'Velocity_North', 'Velocity_Up']].head())

# Opslaan van de resultaten naar een nieuw CSV-bestand
data_merged.to_csv(output_csv_path, index=False)
print(f"\nENU-snelheden en tijdstempels opgeslagen in: {output_csv_path}")

# Optioneel: Visualisatie van de ENU-snelheden met tijdstempels
plt.figure(figsize=(15, 10))

# Plot East Snelheid
plt.subplot(3, 1, 1)
plt.plot(data_merged['Absolute_Time'], data_merged['Velocity_East'], label='East (E)', color='r')
plt.xlabel('Tijd')
plt.ylabel('Snelheid (m/s)')
plt.title('East (E) Snelheid over Tijd')
plt.legend()

# Plot North Snelheid
plt.subplot(3, 1, 2)
plt.plot(data_merged['Absolute_Time'], data_merged['Velocity_North'], label='North (N)', color='g')
plt.xlabel('Tijd')
plt.ylabel('Snelheid (m/s)')
plt.title('North (N) Snelheid over Tijd')
plt.legend()

# Plot Up Snelheid
plt.subplot(3, 1, 3)
plt.plot(data_merged['Absolute_Time'], data_merged['Velocity_Up'], label='Up (U)', color='b')
plt.xlabel('Tijd')
plt.ylabel('Snelheid (m/s)')
plt.title('Up (U) Snelheid over Tijd')
plt.legend()

plt.tight_layout()
plt.show()
