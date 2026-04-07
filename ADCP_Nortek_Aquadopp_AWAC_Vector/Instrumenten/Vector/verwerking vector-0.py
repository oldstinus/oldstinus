import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definieer het pad naar je .dat bestand
dat_file_path = r'P:\PA029-BMeetinstr-Cmp\3_Uitvoering\0_03_MEETCAMPAGNES_CALIBRATION_MEETNET\2022_06_13-2022_07_10_vergelijkende_meting_Albertkanaal_Kanne\data\Vector_AWAC_2022_05_31_stationair_ metingen\VECTOR\data\processing\export/KANNE05.dat'

# Definieer de kolomnamen op basis van de gegeven structuur
column_names = [
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

# Lees het .dat bestand in een pandas DataFrame
try:
    data = pd.read_csv(dat_file_path, sep='\s+', header=None, names=column_names)
except Exception as e:
    print(f"Fout bij het lezen van het .dat bestand: {e}")
    exit(1)

# Controleer de eerste paar rijen
print("Eerste 5 rijen van de gegevens:")
print(data.head())

# Filter op geldige metingen (Checksum == 0)
data_valid = data[data['Checksum'] == 0].copy()
print(f"\nAantal geldige metingen: {len(data_valid)}")

# Extraheer de beam-snelheden
V_beam = data_valid[['Velocity_Beam1', 'Velocity_Beam2', 'Velocity_Beam3']].values

# Definieer de transformatie matrix
transformation_matrix = np.array([
    [2.7305, -1.3511, -1.3767],
    [0.0154, 2.3816, -2.3945],
    [0.3401, 0.3420, 0.3484]
])

# Bereken de ENU-snelheden
V_ENU = V_beam.dot(transformation_matrix.T)  # Transponeer voor juiste matrixvermenigvuldiging

# Voeg de ENU-snelheden toe aan de DataFrame
data_valid['Velocity_East'] = V_ENU[:, 0]
data_valid['Velocity_North'] = V_ENU[:, 1]
data_valid['Velocity_Up'] = V_ENU[:, 2]

# Bekijk de eerste paar rijen met de nieuwe ENU-snelheden
print("\nEerste 5 rijen met ENU-snelheden:")
print(data_valid[['Velocity_Beam1', 'Velocity_Beam2', 'Velocity_Beam3', 'Velocity_East', 'Velocity_North', 'Velocity_Up']].head())

# Opslaan van de resultaten naar een nieuw CSV-bestand
output_csv_path = 'C:/Nortek/Vector/KANNE05_ENU.csv'
data_valid.to_csv(output_csv_path, index=False)
print(f"\nENU-snelheden opgeslagen in: {output_csv_path}")

# Optioneel: Visualisatie van de ENU-snelheden
plt.figure(figsize=(15, 10))

# Plot East Snelheid
plt.subplot(3, 1, 1)
plt.plot(data_valid['Velocity_East'], label='East (E)', color='r')
plt.xlabel('Metingen')
plt.ylabel('Snelheid (m/s)')
plt.title('East (E) Snelheid')
plt.legend()

# Plot North Snelheid
plt.subplot(3, 1, 2)
plt.plot(data_valid['Velocity_North'], label='North (N)', color='g')
plt.xlabel('Metingen')
plt.ylabel('Snelheid (m/s)')
plt.title('North (N) Snelheid')
plt.legend()

# Plot Up Snelheid
plt.subplot(3, 1, 3)
plt.plot(data_valid['Velocity_Up'], label='Up (U)', color='b')
plt.xlabel('Metingen')
plt.ylabel('Snelheid (m/s)')
plt.title('Up (U) Snelheid')
plt.legend()

plt.tight_layout()
plt.show()
