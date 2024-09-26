import conda
import pandas as pd
import plotly.graph_objects as go

# Bestandspad naar je CSV-bestand
file_path = "C:\Users\claeysst\Desktop\werkfiles\Scripts & 3D & config\Python\test-data.csv'

# Inlezen van het CSV-bestand met een aangepaste delimiter en correctie van de structuur
data_fixed = pd.read_csv(file_path, delimiter='.', skiprows=2)

# Alle numerieke waarden met een decimale komma vervangen door een decimale punt
data_fixed.replace({',': '.'}, regex=True, inplace=True)

# Nieuwe kolomnamen toepassen zoals eerder gegeven
new_column_names = [
    "Date Time D/M/Y HH:MM:SS", "Temp C", "SpCond uS", "Sal ppt", 
    "Depth meters", "pH", "Turbid+ NTU", "Battery volts"
]
data_fixed.columns = new_column_names

# Tijdkolom omzetten naar datetime-formaat
data_fixed[new_column_names[0]] = pd.to_datetime(data_fixed[new_column_names[0]], format='%d/%m/%y %H:%M:%S')

# Plotly interactieve grafiek maken
fig = go.Figure()

# Elke parameter toevoegen aan de grafiek
for param in new_column_names[1:]:
    fig.add_trace(go.Scatter(x=data_fixed[new_column_names[0]], y=data_fixed[param].astype(float), mode='lines', name=param))

# Layout instellen voor interactieve weergave en lineaire schaal
fig.update_layout(
    title="Parameterplot op de tijdlijn",
    xaxis_title="Tijd",
    yaxis_title="Waardes",
    legend_title="Parameters",
    hovermode="x unified",
    yaxis_type="linear"
)

# Interactieve periode-selector toevoegen
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )
)

# De grafiek opslaan als HTML-bestand
output_file_path = 'interactieve_grafiek_decimaal.html'
fig.write_html(output_file_path)

print(f"De interactieve grafiek is opgeslagen als {output_file_path}")
