# Lippenbroek sedimentmodel

Bronbestand: `omzetten oude files/lippenbroek/LPB_sediment_150506-250907_sedimentballans_in_uit_spring_doodtij.xls`

Afgeleide modelrelaties uit de Excel:

- `SSC_in (mg/L) = 15.688660 + 20.618557 * Q_in (m3/s)`
- `SSC_out (mg/L) = 28.0`
- `Sed_in (kg/15 min) = 0.9 * Q_in * SSC_in`
- `Sed_out (kg/15 min) = 0.9 * Q_out * SSC_out`
- `Sedimentbalans = Sed_in - Sed_out`

Resultaat voor de historische meetreeks:

- Totale instroom sediment: `1,515,829.745 kg`
- Totale uitstroom sediment: `510,842.135 kg`
- Netto sedimentretentie: `1,004,987.609 kg` (`1004.988 ton`)

Belangrijkste outputbestanden:

- `sediment_tijdreeks.csv`
- `sediment_maandbalans.csv`
- `sediment_getijtype.csv`
- `sediment_tijdreeks.png`
- `sediment_maandbalans.png`
- `ssc_regressie.png`
- `sediment_getijtype.png`
- `samenvatting.json`

Opnieuw draaien:

```powershell
& .\.venv_word_excel_ppt\Scripts\python.exe simulate_lpb_sediment_balance.py
```

Opslag/erosiemodel draaien:

```powershell
& .\.venv_word_excel_ppt\Scripts\python.exe simulate_lpb_sediment_balance.py --mode storage --output-dir output/spreadsheet/lpb_sediment_storage_model
```

Voorbeeldscenario met 10% meer instroomdebiet en 5% hogere uitstroomconcentratie:

```powershell
& .\.venv_word_excel_ppt\Scripts\python.exe simulate_lpb_sediment_balance.py --q-in-scale 1.10 --ssc-out-scale 1.05
```

Interactief dashboard starten:

```powershell
& .\.venv_word_excel_ppt\Scripts\python.exe lpb_sediment_dashboard.py
```

Astronomisch getij als input gebruiken:

```powershell
& .\.venv_word_excel_ppt\Scripts\python.exe simulate_lpb_sediment_balance.py --mode astronomical-storage --tide-amplitude 1.4 --spring-neap-strength 0.45 --inflow-gain 2.0 --outflow-gain 1.6 --output-dir output/spreadsheet/lpb_sediment_astronomical
```

In het dashboard kun je ook `Debietbron = Astronomisch getij` kiezen en daar amplitude, periode, fase en spring-neap modulatie via sliders aanpassen.

Waterpeil als input gebruiken:

- Gemeten waterpeil: `Lippenbroek GOG_Zeeschelde_Waterpeil.csv`
- Voorspeld getij: `Driegoten tij_Zeeschelde_Voorspeld waterpeil getij.csv`

Waterpeil-naar-debietformule:

- `Q_in = max(k_in * dH/dt, 0)`
- `Q_out = max(k_out * (-dH/dt), 0)`
- `SSC_in = 15.688660 + 20.618557 * Q_in`
- `Sed_in = dt * Q_in * SSC_in * 0.001`
- `Sed_out = dt * Q_out * SSC_out * 0.001` of in opslagmodus via opslag/erosie

Voorbeeld met voorspeld waterpeil en animatie:

```powershell
& .\.venv_word_excel_ppt\Scripts\python.exe simulate_lpb_sediment_balance.py --mode storage --waterlevel-source forecast --make-animation --output-dir output/spreadsheet/lpb_sediment_forecast_level
```

Voorbeeld met gemeten waterpeil:

```powershell
& .\.venv_word_excel_ppt\Scripts\python.exe simulate_lpb_sediment_balance.py --mode storage --waterlevel-source measured --output-dir output/spreadsheet/lpb_sediment_measured_level
```
