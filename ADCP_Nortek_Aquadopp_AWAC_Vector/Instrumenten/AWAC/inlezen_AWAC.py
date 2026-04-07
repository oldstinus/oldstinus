import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import os

# Pad naar de directory waar de bestanden zich bevinden
directory = os.path.dirname(r'C:\Users\claeysst\Desktop\werkfiles\verwerken AWAC\kanne04.csv')  # Pas aan indien nodig

# Lees de start- en eindtijd uit het bestand 'begin-eind.txt'
with open(os.path.join(directory, 'begin-eind.txt'), 'r') as f:
    start_time_str = f.readline().strip()
    end_time_str = f.readline().strip()

# Converteer de start- en eindtijd naar datetime-objecten
start_time = pd.to_datetime(start_time_str, format='%d/%m/%Y %H:%M:%S')
end_time = pd.to_datetime(end_time_str, format='%d/%m/%Y %H:%M:%S')

print(f"Starttijd: {start_time}")
print(f"Eindtijd: {end_time}")

# Lees het CSV-bestand met het juiste scheidingsteken (';') en encoding
df = pd.read_csv(os.path.join(directory, 'kanne04.csv'), sep=';', header=0, encoding='utf-8')

# Verwijder eventuele voor- en achterliggende spaties uit kolomnamen
df.columns = df.columns.str.strip()

# Controleer of 'DateTime' in de kolommen zit
if 'DateTime' in df.columns:
    datetime_col = 'DateTime'
else:
    raise ValueError("De kolom 'DateTime' is niet gevonden in het DataFrame.")

# Converteer de 'DateTime' kolom naar datetime-objecten
df[datetime_col] = pd.to_datetime(df[datetime_col], format='%d/%m/%Y %H:%M:%S')

# Filter de DataFrame op basis van start- en eindtijd
df_filtered = df[(df[datetime_col] >= start_time) & (df[datetime_col] <= end_time)]

# Controleer of er data is na het filteren
if df_filtered.empty:
    raise ValueError("Geen data beschikbaar binnen de opgegeven tijdsperiode.")

# Zet 'DateTime' als index
df_filtered.set_index(datetime_col, inplace=True)

# Zoek alle kolommen die beginnen met 'Speed#' en 'Dir#'
speed_cols = [col for col in df_filtered.columns if col.startswith('Speed#')]
dir_cols = [col for col in df_filtered.columns if col.startswith('Dir#')]

# Controleer of we evenveel snelheids- en richtingskolommen hebben
if len(speed_cols) != len(dir_cols):
    raise ValueError("Het aantal snelheids- en richtingskolommen komt niet overeen.")

# Sorteer de kolommen op celnummer
speed_cols.sort()
dir_cols.sort()

# Bepaal het aantal cellen
num_cells = len(speed_cols)

print(f"Aantal cellen gedetecteerd: {num_cells}")

# Haal de diepte-informatie uit de kolomnamen
cell_positions = []
for col in speed_cols:
    # Voorbeeld kolomnaam: 'Speed#1(0.9m)'
    start_idx = col.find('(')
    end_idx = col.find('m)')
    if start_idx != -1 and end_idx != -1:
        depth = float(col[start_idx+1:end_idx])
        cell_positions.append(depth)
    else:
        # Als diepte niet gevonden wordt, gebruik dan een standaardwaarde
        cell_positions.append(np.nan)

# Controleer of alle dieptes zijn gevonden
if np.any(np.isnan(cell_positions)):
    # Als diepte-informatie ontbreekt, gebruik standaard waarden
    cell_positions = [0.90 + 0.50 * i for i in range(num_cells)]

cell_positions = np.array(cell_positions)

# Bereid de tijd en diepte voor
times = df_filtered.index.values.astype('datetime64[ns]')
depths = cell_positions

# Maak meshgrids voor contour plot
T, Z = np.meshgrid(times, depths)

# Maak een 2D-array voor snelheid en richting
speed_data = df_filtered[speed_cols].T.values  # Vorm: (num_cells, num_times)
direction_data = df_filtered[dir_cols].T.values  # Vorm: (num_cells, num_times)

# Maak de eerste figuur voor stroomsnelheid
fig1, ax1 = plt.subplots(figsize=(15, 8))

# Plot snelheid als functie van tijd en diepte
im1 = ax1.pcolormesh(T, Z, speed_data, shading='auto', cmap='jet')

# Kleurbalk toevoegen
cbar1 = fig1.colorbar(im1, ax=ax1)
cbar1.set_label('Snelheid (m/s)')

# Formatteren van de x-as voor datums
ax1.xaxis_date()
date_format = mdates.DateFormatter('%d/%m/%Y\n%H:%M:%S')
ax1.xaxis.set_major_formatter(date_format)
fig1.autofmt_xdate()

# Labels en titel voor de eerste figuur
ax1.set_xlabel('Tijd')
ax1.set_ylabel('Diepte (m)')
ax1.set_title('Stroomsnelheid als functie van tijd en diepte')

# Keer de y-as om zodat grotere dieptes naar beneden gaan
ax1.invert_yaxis()

# Maak de tweede figuur voor stroomrichting
fig2, ax2 = plt.subplots(figsize=(15, 8))

# Plot richting als functie van tijd en diepte
im2 = ax2.pcolormesh(T, Z, direction_data, shading='auto', cmap='hsv')

# Kleurbalk toevoegen
cbar2 = fig2.colorbar(im2, ax=ax2)
cbar2.set_label('Richting (graden)')

# Formatteren van de x-as voor datums
ax2.xaxis_date()
ax2.xaxis.set_major_formatter(date_format)
fig2.autofmt_xdate()

# Labels en titel voor de tweede figuur
ax2.set_xlabel('Tijd')
ax2.set_ylabel('Diepte (m)')
ax2.set_title('Stroomrichting als functie van tijd en diepte')

# Keer de y-as om zodat grotere dieptes naar beneden gaan
ax2.invert_yaxis()

plt.show()
