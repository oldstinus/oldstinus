#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LOG-A-LEVEL -> CSV (hoogte)

- Input : *.log
- Output: CSV met tijd + hoogte
- Druk wordt genegeerd
- Hoogte = L_raw / 1000  (mm -> m, pas factor aan indien nodig)

Geschikt voor 8 Hz LOG-A-LEVEL data
"""

import re
import pandas as pd
from pathlib import Path

# ============================================================
# INSTELLINGEN
# ============================================================

INPUT_LOG = r"STA220260203.log"     # pad naar logbestand
OUTPUT_CSV = r"STA220260203_hoogte.csv"

HOOGTE_FACTOR = 1 / 1000.0          # mm → m (pas aan indien nodig)
FALLBACK_FS = 5.0                   # Hz indien niet afleidbaar uit log

# ============================================================
# REGEX PATRONEN
# ============================================================

PAT_T = re.compile(r"^T=(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}\.\d{3})")
PAT_L = re.compile(r"^L=([-+]?\d+)")

# ============================================================
# INLEZEN
# ============================================================

lines = []
with open(INPUT_LOG, "r", errors="ignore") as f:
    for ln in f:
        ln = ln.strip()
        if ln:
            lines.append(ln)

# index van alle T= lijnen
t_idx = [i for i, ln in enumerate(lines) if ln.startswith("T=")]

if len(t_idx) < 1:
    raise RuntimeError("Geen T= tijdmarkeringen gevonden")

# ============================================================
# Timestamps van T= lijnen
# ============================================================

T_times = []
for i in t_idx:
    m = PAT_T.match(lines[i])
    if m:
        T_times.append(
            pd.to_datetime(m.group(1), format="%m/%d/%Y %H:%M:%S.%f")
        )
    else:
        T_times.append(pd.NaT)

# ============================================================
# Bepaal sample-interval
# ============================================================

L_counts = []
for a, b in zip(t_idx, t_idx[1:]):
    seg = lines[a+1:b]
    L_counts.append(sum(1 for ln in seg if ln.startswith("L=")))

if len(L_counts) > 0 and len(T_times) > 1:
    nL = max(set(L_counts), key=L_counts.count)
    dt = (T_times[1] - T_times[0]).total_seconds() / nL
else:
    dt = 1 / FALLBACK_FS

# ============================================================
# DATA OPBOUW
# ============================================================

records = []

for k, start in enumerate(t_idx):
    t0 = T_times[k]
    end = t_idx[k+1] if k+1 < len(t_idx) else len(lines)

    lvals = []
    for ln in lines[start+1:end]:
        m = PAT_L.match(ln)
        if m:
            lvals.append(int(m.group(1)))

    for i, lv in enumerate(lvals):
        records.append({
            "tijd": t0 + pd.to_timedelta(i * dt, unit="s"),
            "L_raw": lv,
            "hoogte_m": lv * HOOGTE_FACTOR
        })

df = pd.DataFrame(records)

# ============================================================
# EXPORT CSV (Excel-vriendelijk)
# ============================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
    sep=";",          # geschikt voor NL Excel
    decimal=","       # NL decimaal
)

# ============================================================
# SAMENVATTING
# ============================================================

print("Conversie klaar")
print(f"Samples        : {len(df)}")
print(f"Eerste tijd    : {df['tijd'].iloc[0]}")
print(f"Laatste tijd   : {df['tijd'].iloc[-1]}")
print(f"dt (s)         : {dt:.6f}")
print(f"Output bestand : {Path(OUTPUT_CSV).resolve()}")
