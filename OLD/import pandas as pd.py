import pandas as pd

# Data representing the columns 'Lijnnr begrotingsexcel', 'Omschrijving', and 'Budget 2024'
data = {
    'Lijnnr begrotingsexcel': [72, 73, 74, 76, 77, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
    'Omschrijving': [
        'onderhoud MIKE11', 'Onderhoud GIStools', 'Raamovereenkomst waterbeheer', 'Ondersteuning Databeheer en meetnet',
        'Onderhoud Wiski / Kiwis / 24/7', 'HIC-loggingtool - implementatie', 'Ondersteuning validatie en databeheer',
        'Upgrade performantie R shiny apps - dienst R studio', 'Samenwerkingsovereenkomst KMI', 'Ontwikkelingen Waterinfo',
        'Verlenging Onderhoud portaal CMS', 'Jaarlijkse hosting en onderhoud YAMI', 'Onderhoud en verbetering FEWS Cloud -  2D',
        'Onderhoud en verbetering FEWS Cloud -  2D', 'Bijstand -   (technisch tekenen, onderhoud terrein)',
        'Opmeting, ontwerp en plaatsing van meetinfrastructuur', 'Onderhoudscontract Fabricom-Aanderaa', 
        'Vervanging ADM loopttijdmeters', 'Onderhoudscontract Elscolab', 'Onderhoudscontract peil- en pluviomeetnet HIC',
        'assistentie metingen op terrein (1 VTE)', 'aankoop peilmeettoestellen GOG gebieden', 
        'werkingskosten meetnet (algemene kleine bestellingen)', 'Ondersteuning metingen op terrein'
    ],
    'Budget 2024': [
        70000, 15000, 150000, 223000, 270000, 50000, 150000, 1250, 45000, 50000, 50000, 9500, 125000, 
        75000, 289000, 55000, 30000, 100000, 50000, 150000, 90000, 10000, 75000, 100000
    ]
}

# Create a dataframe
df = pd.DataFrame(data)

# Plotting the data using matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.barh(df['Omschrijving'], df['Budget 2024'], color='skyblue')
plt.xlabel('Budget 2024 (EUR)')
plt.title('Budget Allocations for 2024 by Project')
plt.tight_layout()

# Show the plot
plt.show()
