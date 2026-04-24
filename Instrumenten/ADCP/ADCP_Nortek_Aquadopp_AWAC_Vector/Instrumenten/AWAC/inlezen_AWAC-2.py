import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import os

# Voor het berekenen van de cirkelgemiddelde richting
from scipy.stats import circmean

# Voor nauwkeurige controle over de subplots
import matplotlib.gridspec as gridspec

# Pad naar de directory waar de bestanden zich bevinden
directory = os.path.dirname(r'C:\Users\claeysst\Desktop\werkfiles\verwerken AWAC\'kanne04.csv')  # Pas aan indien nodig

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
        # Als diepte-informatie ontbreekt, gebruik dan een standaardwaarde
        cell_positions.append(np.nan)

# Controleer of alle dieptes zijn gevonden
if np.any(np.isnan(cell_positions)):
    # Als diepte-informatie ontbreekt, gebruik standaard waarden
    cell_positions = [0.90 + 0.50 * i for i in range(num_cells)]

cell_positions = np.array(cell_positions)

# Sorteer dieptes en data op toenemende diepte
sorted_indices = np.argsort(cell_positions)
cell_positions = cell_positions[sorted_indices]
speed_data = df_filtered[speed_cols].T.values[sorted_indices]
direction_data = df_filtered[dir_cols].T.values[sorted_indices]

# Bereid de tijd en diepte voor
times = df_filtered.index.values.astype('datetime64[ns]')
depths = cell_positions

# Converteer tijden naar numerieke waarden voor plotting
time_nums = mdates.date2num(times)

# Bereken gemiddelde snelheid en richting over de waterkolom
avg_speed = np.nanmean(speed_data, axis=0)
avg_direction = circmean(direction_data, high=360, low=0, axis=0, nan_policy='omit')

# Bereken minimale en maximale diepte voor consistente y-assen
min_depth = np.min(depths)
max_depth = np.max(depths)

# Maak één figuur met vier subplots met behulp van GridSpec
fig = plt.figure(figsize=(15, 16))
gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 3, 1])

# Subplot 1: Contourplot van stroomsnelheid met imshow
ax1 = fig.add_subplot(gs[0])
extent = [time_nums[0], time_nums[-1], min_depth, max_depth]
im1 = ax1.imshow(speed_data, aspect='auto', extent=extent, cmap='jet', origin='lower')
ax1.set_ylabel('Diepte (m)')
ax1.set_title('Stroomsnelheid als functie van tijd en diepte')
ax1.set_ylim([max_depth, min_depth])  # Omkeren van y-as
cbar1 = fig.colorbar(im1, ax=ax1)
cbar1.set_label('Snelheid (m/s)')

# Subplot 2: Gemiddelde stroomsnelheid over de waterkolom
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(time_nums, avg_speed, color='black', label='Gemiddelde snelheid over diepte')
ax2.set_ylabel('Gemiddelde snelheid (m/s)')
ax2.set_title('Gemiddelde stroomsnelheid over de waterkolom')
ax2.legend()
ax2.grid(True)

# Subplot 3: Contourplot van stroomrichting met imshow
ax3 = fig.add_subplot(gs[2], sharex=ax1)
im2 = ax3.imshow(direction_data, aspect='auto', extent=extent, cmap='hsv', origin='lower')
ax3.set_ylabel('Diepte (m)')
ax3.set_title('Stroomrichting als functie van tijd en diepte')
ax3.set_ylim([max_depth, min_depth])  # Omkeren van y-as
cbar2 = fig.colorbar(im2, ax=ax3)
cbar2.set_label('Richting (graden)')

# Subplot 4: Gemiddelde stroomrichting over de waterkolom
ax4 = fig.add_subplot(gs[3], sharex=ax1)
ax4.plot(time_nums, avg_direction, color='black', label='Gemiddelde richting over diepte')
ax4.set_ylabel('Gemiddelde richting (graden)')
ax4.set_xlabel('Tijd')
ax4.set_title('Gemiddelde stroomrichting over de waterkolom')
ax4.legend()
ax4.grid(True)

# Formatteren van de x-as voor datums
for ax in [ax1, ax2, ax3, ax4]:
    ax.xaxis_date()
    date_format = mdates.DateFormatter('%d/%m/%Y\n%H:%M:%S')
    ax.xaxis.set_major_formatter(date_format)

# Zorg ervoor dat alle x-assen dezelfde limieten hebben
x_limits = [time_nums[0], time_nums[-1]]
for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim(x_limits)

# Minimaliseer ruimte tussen subplots en zorg voor correcte uitlijning
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)

# Pas de lay-out aan
plt.tight_layout()
plt.show()
