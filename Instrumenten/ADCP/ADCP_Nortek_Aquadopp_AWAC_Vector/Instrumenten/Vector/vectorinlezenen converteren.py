import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from io import StringIO
from datetime import timedelta
import logging
import matplotlib.dates as mdates

# Configureer logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Definieer het pad naar je .dat en .sen bestanden
dat_file_path = r'C:\Users\claeysst\Desktop\werkfiles\Verwerken Vector\KANNE05.dat'
sen_file_path = r'C:\Users\claeysst\Desktop\werkfiles\Verwerken Vector\KANNE05.sen'

# Definieer het pad waar het output CSV-bestand opgeslagen moet worden
output_csv_path = r'C:\Users\claeysst\Desktop\werkfiles\Verwerken Vector\processed_data.csv'  # Pas dit pad aan naar jouw gewenste locatie

# Zorg ervoor dat de output directory bestaat, maak deze indien nodig aan
output_dir = os.path.dirname(output_csv_path)
os.makedirs(output_dir, exist_ok=True)

# Controleer of de .dat en .sen bestanden bestaan
if not os.path.exists(dat_file_path):
    logging.error(f"Het .dat bestand {dat_file_path} bestaat niet. Controleer het pad.")
    exit(1)

if not os.path.exists(sen_file_path):
    logging.error(f"Het .sen bestand {sen_file_path} bestaat niet. Controleer het pad.")
    exit(1)

# Definieer de kolomnamen voor het .sen bestand op basis van het gegeven voorbeeld
sen_column_names = [
    'Month',            # Maand (MM)
    'Day',              # Dag (DD)
    'Year',             # Jaar (JJJJ)
    'Hour',             # Uur (UU)
    'Minute',           # Minuut (MM)
    'Second',           # Seconde (SS)
    'Burst_counter',    # Burst_counter
    'Unknown1',
    'Unknown2',
    'Unknown3',
    'Unknown4',
    'Unknown5',
    'Unknown6',
    'Unknown7',
    'Unknown8'
]

# Lees het .sen bestand
try:
    with open(sen_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        cleaned_content_sen = f.read().replace('\x00', ' ')
    
    sen_data = pd.read_csv(StringIO(cleaned_content_sen), sep='\s+', header=None, names=sen_column_names)
    logging.info("Succesvol het .sen bestand gelezen.")
except Exception as e:
    logging.error(f"Fout bij het lezen van het .sen bestand: {e}")
    exit(1)

# Toon de eerste paar rijen van het .sen bestand
logging.info("Eerste 5 rijen van het .sen bestand:")
logging.info(f"\n{sen_data.head()}")

# Zorg ervoor dat de tijdkolommen numeriek zijn
time_columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']
for col in time_columns:
    sen_data[col] = pd.to_numeric(sen_data[col], errors='coerce')

# Controleer op NaN waarden in tijdkolommen
if sen_data[time_columns].isnull().any().any():
    logging.warning("Waarschuwing: Er zijn NaN waarden in de tijdkolommen. Deze rijen zullen als NaT worden gemarkeerd.")

# Maak een datetime kolom aan in .sen data
sen_data['Datetime'] = pd.to_datetime(sen_data[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']], errors='coerce')

# Sorteer de sen_data op Burst_counter om de volgorde te garanderen
sen_data = sen_data.sort_values('Burst_counter').reset_index(drop=True)

# Selecteer alleen de relevante kolommen
sen_time = sen_data[['Burst_counter', 'Datetime']].copy()

# Controleer of er duplicaten zijn in Burst_counter
if sen_time['Burst_counter'].duplicated().any():
    logging.warning("Waarschuwing: Er zijn duplicaten in Burst_counter in het .sen bestand. Deze zullen worden genegeerd bij toewijzing.")
    # Verwijder duplicaten, behoud de eerste
    sen_time = sen_time.drop_duplicates(subset='Burst_counter', keep='first').reset_index(drop=True)

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

# Lees het .dat bestand
try:
    data = pd.read_csv(dat_file_path, sep='\s+', header=None, names=dat_column_names, encoding='utf-8', engine='python')
    logging.info("Succesvol het .dat bestand gelezen.")
except Exception as e:
    logging.error(f"Fout bij het lezen van het .dat bestand: {e}")
    exit(1)

# Toon de eerste paar rijen van het .dat bestand
logging.info("\nEerste 5 rijen van het .dat bestand:")
logging.info(f"\n{data.head()}")

# Filter op geldige metingen (Checksum == 0)
data_valid = data[data['Checksum'] == 0].copy()
logging.info(f"\nAantal geldige metingen: {len(data_valid)}")

# Sorteer de .dat data op Burst_counter en Ensemble_counter
data_valid = data_valid.sort_values(['Burst_counter', 'Ensemble_counter']).reset_index(drop=True)

# Controleer of Burst_counter in .dat uniek is
if data_valid['Burst_counter'].duplicated().any():
    logging.warning("Burst_counter in .dat is niet uniek. We zullen een sequentiële Burst_ID toewijzen.")
    # Creëer een unieke Burst_ID door te cumuleren wanneer Burst_counter verandert
    data_valid['Burst_ID'] = (data_valid['Burst_counter'] != data_valid['Burst_counter'].shift()).cumsum()
else:
    # Gebruik Burst_counter direct als Burst_ID
    data_valid['Burst_ID'] = data_valid['Burst_counter']

# Creëer een lijst van unieke Burst_ID's in .dat
unique_bursts_dat = data_valid['Burst_ID'].unique()

# Bereken het aantal bursts in .sen en .dat
num_bursts_sen = len(sen_time)
num_bursts_dat = len(unique_bursts_dat)

logging.info(f"Aantal bursts in .sen: {num_bursts_sen}")
logging.info(f"Aantal unieke bursts in .dat: {num_bursts_dat}")

# Controleer of er meer bursts in .dat zijn dan in .sen
if num_bursts_dat > num_bursts_sen:
    logging.warning("Waarschuwing: Meer unieke bursts in .dat dan in .sen. De extra bursts zullen geen tijdstempel krijgen.")

# Bepaal het minimum aantal bursts om te voorkomen dat we buiten de grenzen van de arrays gaan
min_bursts = min(num_bursts_dat, num_bursts_sen)

# Creëer een DataFrame met Burst_ID en Datetime uit .sen
burst_time_mapping = pd.DataFrame({
    'Burst_ID': unique_bursts_dat[:min_bursts],
    'Datetime': sen_time['Datetime'].iloc[:min_bursts].values
})

# Creëer een mapping dictionary
burst_id_to_datetime = pd.Series(burst_time_mapping.Datetime.values, index=burst_time_mapping.Burst_ID).to_dict()

# Wijs 'Datetime' toe aan data_valid op basis van 'Burst_ID'
data_valid['Datetime'] = data_valid['Burst_ID'].map(burst_id_to_datetime)

# Controleer of er Burst_ID's zijn zonder tijdinformatie
missing_time = data_valid['Datetime'].isnull().sum()
if missing_time > 0:
    logging.warning(f"Waarschuwing: {missing_time} bursts hebben geen bijbehorende tijdinformatie in het .sen bestand.")

# **Vervangen van de relatieve tijd met de tijd uit het .sen bestand**
# In deze versie wordt de 'Absolute_Time' direct gebaseerd op de 'Datetime' uit het .sen bestand,
# zonder toevoeging van 'Ensemble_seconds'. Dit betekent dat alle metingen binnen een burst dezelfde tijd krijgen.

data_valid['Absolute_Time'] = data_valid['Datetime']

# Extraheer de beam-snelheden
V_beam = data_valid[['Velocity_Beam1', 'Velocity_Beam2', 'Velocity_Beam3']].values

# Definieer de transformatie matrix
transformation_matrix = np.array([
    [2.7305, -1.3511, -1.3767],
    [0.0154, 2.3816, -2.3945],
    [0.3401, 0.3420, 0.3484]
])

# Bereken de ENU-snelheden
V_ENU = V_beam.dot(transformation_matrix.T)

# Voeg de ENU-snelheden toe aan de DataFrame
data_valid['Velocity_East'] = V_ENU[:, 0]
data_valid['Velocity_North'] = V_ENU[:, 1]
data_valid['Velocity_Up'] = V_ENU[:, 2]

# Bekijk de eerste paar rijen met de nieuwe ENU-snelheden en Absolute_Time
logging.info("\nEerste 5 rijen met ENU-snelheden en tijdstempels:")
logging.info(f"\n{data_valid[['Absolute_Time', 'Velocity_Beam1', 'Velocity_Beam2', 'Velocity_Beam3', 'Velocity_East', 'Velocity_North', 'Velocity_Up']].head()}")

# Opslaan van de resultaten naar een nieuw CSV-bestand
try:
    data_valid.to_csv(output_csv_path, index=False)
    logging.info(f"\nENU-snelheden en tijdstempels opgeslagen in: {output_csv_path}")
except Exception as e:
    logging.error(f"Fout bij het opslaan van het CSV-bestand: {e}")

# Optioneel: Visualisatie van de ENU-snelheden met tijdstempels
# Filter alleen de metingen met geldige Absolute_Time
data_plot = data_valid.dropna(subset=['Absolute_Time']).copy()

if data_plot.empty:
    logging.warning("Geen gegevens om te plotten na filtering op geldige Absolute_Time.")
else:
    # Sorteer data_plot op Absolute_Time
    data_plot = data_plot.sort_values('Absolute_Time')

    plt.figure(figsize=(15, 10))

    # Plot East Snelheid
    plt.subplot(3, 1, 1)
    plt.plot(data_plot['Absolute_Time'], data_plot['Velocity_East'], label='East (E)', color='r')
    plt.xlabel('Tijd')
    plt.ylabel('Snelheid (m/s)')
    plt.title('East (E) Snelheid over Tijd')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    # Plot North Snelheid
    plt.subplot(3, 1, 2)
    plt.plot(data_plot['Absolute_Time'], data_plot['Velocity_North'], label='North (N)', color='g')
    plt.xlabel('Tijd')
    plt.ylabel('Snelheid (m/s)')
    plt.title('North (N) Snelheid over Tijd')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    # Plot Up Snelheid
    plt.subplot(3, 1, 3)
    plt.plot(data_plot['Absolute_Time'], data_plot['Velocity_Up'], label='Up (U)', color='b')
    plt.xlabel('Tijd')
    plt.ylabel('Snelheid (m/s)')
    plt.title('Up (U) Snelheid over Tijd')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
