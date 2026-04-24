#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog, simpledialog
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import zipfile, glob, os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTDIR = SCRIPT_DIR / "output_report"
os.makedirs(OUTDIR, exist_ok=True)

# =========================
# LOADERS
# =========================
def load_waterinfo(zip_path):

    tmp = OUTDIR / ("tmp_" + os.path.basename(zip_path).replace(" ", "_"))
    os.makedirs(tmp, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(tmp)

    csvs = [c for c in glob.glob(str(tmp / "*.csv")) if "disclaimer" not in c.lower()]
    df = pd.read_csv(csvs[0], sep=';', comment='#', encoding='latin1')

    df.columns = ["time","value","q","x1","x2"]
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df["value"] = df["value"].astype(str).str.replace(",",".").astype(float)

    return df[["time","value"]].dropna()

def load_sasdruk_csv(path):
    """
    Specifieke loader voor CSV met kolommen:
    tijd;dokpeil;saspeil;zeepeil (sep=';', decimal=',', dayfirst).
    Retourneert time + h_sas kolom; andere kolommen worden genegeerd.
    """
    if not path:
        return pd.DataFrame(columns=["time", "h_sas"])
    try:
        df = pd.read_csv(path, sep=";", decimal=",", encoding="latin1")
    except Exception as exc:
        print(f"Waarschuwing: sasdruk-CSV kon niet worden ingelezen: {exc}")
        return pd.DataFrame(columns=["time", "h_sas"])

    # Harmoniseer kolomnamen
    df.columns = [str(c).strip().lower() for c in df.columns]

    # tijdkolom = eerste of kolom met tijd/tijdstip
    time_col = df.columns[0]
    for c in df.columns:
        if "tijd" in c or "time" in c:
            time_col = c
            break

    df["time"] = pd.to_datetime(df[time_col], format="%d-%m-%Y %H:%M:%S", errors="coerce", dayfirst=True)
    df["time"] = df["time"].dt.tz_localize(None)

    sas_col = None
    for c in df.columns:
        if "sas" in c and ("peil" in c or "druk" in c or "niveau" in c):
            sas_col = c
            break
    if sas_col is None and len(df.columns) >= 3:
        sas_col = df.columns[2]  # derde kolom volgens opgave

    df["h_sas"] = pd.to_numeric(df[sas_col], errors="coerce")

    return df[["time", "h_sas"]].dropna()

def load_adcp(path):
    # Specifiek formaat (tab-delimited) zoals vandamme2E21_export_window-4dagen_hans.tsv
    try:
        df_tab = pd.read_csv(path, sep="\t", encoding="latin1")
        if {"DateTime", "x_avg"}.issubset(df_tab.columns):
            df = df_tab.copy()

            # helpers
            def to_float(series):
                return pd.to_numeric(series.astype(str).str.replace(",", ".").str.strip(), errors="coerce")

            df["time"] = pd.to_datetime(df["DateTime"], errors="coerce").dt.tz_localize(None)
            df["v"] = to_float(df["x_avg"])  # kolom 5 (1-based)
            if "Pressure" in df.columns:
                df["p"] = to_float(df["Pressure"])  # kolom 12 (1-based)

            keep_cols = ["time", "v"]
            if "p" in df.columns:
                keep_cols.append("p")

            df = df[keep_cols].dropna(subset=["time", "v"])
            if not df.empty:
                return df
    except Exception:
        pass

    # Probeer verschillende delimiters zonder header; kies de variant met de meeste kolommen
    seps = [None, ",", ";", "\t", r"\s+"]
    df = None
    for sep in seps:
        try:
            df_try = pd.read_csv(path, sep=sep, engine="python", header=None)
            if df is None or df_try.shape[1] > df.shape[1]:
                df = df_try
            # kies meteen zodra we >=5 kolommen hebben
            if df.shape[1] >= 5:
                break
        except Exception:
            continue

    if df is None or df.shape[1] < 2:
        print("Waarschuwing: ADCP-bestand kon niet worden ingelezen; onbekende delimiter.")
        return pd.DataFrame(columns=["time", "v"])

    df.columns = [f"c{i}" for i in range(df.shape[1])]

    # tijdkolom (eerste)
    time_col = "c0"
    df["time"] = pd.to_datetime(df[time_col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if df["time"].isna().all():
        df["time"] = pd.to_datetime(df[time_col], dayfirst=True, errors="coerce")
    df["time"] = df["time"].dt.tz_localize(None)

    # kies snelheidskolom: vraag aan gebruiker, val terug op heuristiek (meeste valide waarden)
    v_col_idx = None
    try:
        preview = df.head(3).iloc[:, : min(8, df.shape[1])].to_string(index=False)
        v_col_idx = simpledialog.askinteger(
            "ADCP snelheidskolom",
            f"Kolomindex voor snelheid (0-based, 0 = tijd)\nTotaal kolommen: {df.shape[1]}\nVoorbeeld (eerste 3 rijen):\n{preview}\nStandaard = 4",
            initialvalue=min(4, df.shape[1]-1),
            minvalue=1,
            maxvalue=df.shape[1]-1,
        )
    except Exception:
        v_col_idx = None

    if v_col_idx is not None:
        v_series = pd.to_numeric(df.iloc[:, v_col_idx], errors="coerce")
        chosen_idx = v_col_idx
    else:
        numeric_cols = []
        for c in df.columns[1:]:
            numeric_cols.append(pd.to_numeric(df[c], errors="coerce"))
        if not numeric_cols:
            print("Waarschuwing: geen numerieke kolommen gevonden na de tijdkolom.")
            return pd.DataFrame(columns=["time", "v"])
        valid_counts = [s.count() for s in numeric_cols]
        best_idx = int(np.argmax(valid_counts))
        v_series = numeric_cols[best_idx]
        chosen_idx = best_idx + 1  # +1 omdat we c1.. zijn

    df["v"] = v_series
    df = df[["time", "v"]].dropna()

    if not df.empty:
        v_max = df["v"].abs().max()
        print(f"ADCP: kolom {chosen_idx} gekozen als snelheid. v-mean={df['v'].mean():.3f}, v-max={df['v'].max():.3f}, v-min={df['v'].min():.3f}")
        if v_max > 10:  # vermoedelijk mm/s of cm/s
            try:
                scale = simpledialog.askfloat(
                    "Snelheid schalen",
                    f"Snelheden lijken groot (|v|max = {v_max:.1f}). Vermenigvuldig met factor (bv 0.001 voor mm/s -> m/s).",
                    initialvalue=0.001,
                )
                if scale:
                    df["v"] = df["v"] * scale
                    print(f"Snelheden geschaald met factor {scale}. Nieuwe |v|max = {df['v'].abs().max():.3f}")
            except Exception:
                pass
    if df.empty:
        print("Waarschuwing: ADCP-bestand gaf geen valide tijd/velocity rijen. Controleer delimiter en kolomvolgorde (tijd in kolom 0, snelheid in numerieke kolom).")
    return df

def _parse_datetime_column(df, candidates):
    """Return first datetime-like Series found in candidate columns or any column."""
    for col in candidates:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True, format="%d-%m-%Y %H:%M:%S")
            dt = dt.dt.tz_localize(None)
            if dt.notna().any():
                return dt
    for col in df.columns:
        dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True, format="%d-%m-%Y %H:%M:%S")
        dt = dt.dt.tz_localize(None)
        if dt.notna().any():
            return dt
    return None

def load_versassingsdata(path):
    """Load versassingsdata (lockage events) from CSV/TSV/Excel and return start/end/label."""
    if not path:
        return pd.DataFrame(columns=["start", "end", "label"])
    try:
        ext = Path(path).suffix.lower()
        if ext in [".xlsx", ".xls"]:
            raw = pd.read_excel(path)
        else:
            raw = None
            # force common Vlaanderen export: ; delimited, decimal comma
            try:
                raw = pd.read_csv(path, sep=";", decimal=",")
            except Exception:
                raw = None
            if raw is None or raw.shape[1] == 1:
                for sep in [";", ",", "\t", r"\s+"]:
                    try:
                        raw = pd.read_csv(path, sep=sep)
                        break
                    except Exception:
                        continue
            if raw is None:
                raw = pd.read_csv(path)
    except Exception as exc:
        print(f"Waarschuwing: versassingsdata kon niet worden ingelezen: {exc}")
        return pd.DataFrame(columns=["start", "end", "label"])

    raw.columns = [str(c).strip() for c in raw.columns]

    start_candidates = [c for c in raw.columns if any(k in c.lower() for k in ["start", "begin", "tijd", "time"])]
    end_candidates = [c for c in raw.columns if any(k in c.lower() for k in ["end", "stop", "einde"])]

    start = _parse_datetime_column(raw, start_candidates)
    if start is None:
        print("Waarschuwing: geen geldige tijdkolom gevonden in versassingsdata.")
        return pd.DataFrame(columns=["start", "end", "label"])

    end = _parse_datetime_column(raw, end_candidates)

    label_col = None
    for c in raw.columns:
        if any(k in c.lower() for k in ["schip", "ship", "naam", "name", "actie", "action", "opmerking", "remark", "id", "nr"]):
            label_col = c
            break

    events = pd.DataFrame({"start": start})
    events["end"] = end if end is not None else events["start"]
    events["label"] = raw[label_col] if label_col else None
    events = events.dropna(subset=["start"]).sort_values("start")
    return events

def _merge_events(df, min_gap=pd.Timedelta(minutes=5)):
    """Merge events closer than min_gap into spans to avoid overplotting."""
    if df.empty:
        return df
    merged = []
    current_start = df.iloc[0]["start"]
    current_end = df.iloc[0]["end"] if not pd.isna(df.iloc[0]["end"]) else df.iloc[0]["start"]
    for _, row in df.iloc[1:].iterrows():
        s = row["start"]
        e = row["end"] if not pd.isna(row["end"]) else s
        if s - current_end <= min_gap:
            current_end = max(current_end, e)
        else:
            merged.append({"start": current_start, "end": current_end})
            current_start, current_end = s, e
    merged.append({"start": current_start, "end": current_end})
    return pd.DataFrame(merged)

# =========================
# MAIN
# =========================
def run(adcp_file, sea_zip, dock_zip, versassing_file=None, sas_csv=None):

    # vraag doorsnede
    root = tk.Tk()
    root.withdraw()
    A = simpledialog.askfloat("Doorsnede", "Geef doorsnede (m²):", initialvalue=1.0)

    adcp = load_adcp(adcp_file)
    sea = load_waterinfo(sea_zip)
    dock = load_waterinfo(dock_zip)
    sas = load_sasdruk_csv(sas_csv)
    versass = load_versassingsdata(versassing_file)

    if adcp.empty:
        print("Waarschuwing: geen ADCP-data ingelezen. Controleer het bestand en delimiter.")
        return
    if sea.empty or dock.empty:
        print("Waarschuwing: waterinfo-bestanden zijn leeg of ongeldig.")
        return

    # resample
    adcp = adcp.set_index("time").resample("5min").mean()
    sea = sea.set_index("time").resample("5min").mean()
    dock = dock.set_index("time").resample("5min").mean()
    if not sas.empty:
        sas = sas.set_index("time").resample("5min").mean()

    # overlap
    start_candidates = [adcp.index.min(), sea.index.min(), dock.index.min()]
    end_candidates = [adcp.index.max(), sea.index.max(), dock.index.max()]
    if not sas.empty:
        start_candidates.append(sas.index.min())
        end_candidates.append(sas.index.max())
    start = max(start_candidates)
    end = min(end_candidates)

    adcp = adcp.loc[start:end]
    sea = sea.loc[start:end]
    dock = dock.loc[start:end]
    if not sas.empty:
        sas = sas.loc[start:end]
    if not versass.empty:
        versass = versass[(versass["start"] >= start - pd.Timedelta(hours=1)) & (versass["start"] <= end + pd.Timedelta(hours=1))]
        versass = _merge_events(versass)
        n_raw = len(versass)
        VERSASS_MAX = 200
        if n_raw > VERSASS_MAX:
            step = max(1, n_raw // VERSASS_MAX)
            versass = versass.iloc[::step]
            print(f"Versassingsdata: {n_raw} gebeurtenissen, gedecimeerd tot {len(versass)} voor leesbare plot.")
        else:
            print(f"Versassingsdata: {n_raw} gebeurtenissen ingelezen (geplot op tijdlijn).")

    df = adcp.join(sea.rename(columns={"value":"h_sea"}))
    df = df.join(dock.rename(columns={"value":"h_dock"}))
    if not sas.empty:
        df = df.join(sas.rename(columns={"h_sas":"h_sas"}))
    # optionele druk uit ADCP
    if "p" in adcp.columns:
        df["p_adcp"] = adcp["p"]
    df = df.interpolate().dropna()

    if df.empty:
        print("Waarschuwing: geen overlappende meetpunten na resampling/interpolatie. Controleer of de tijdreeksen elkaar overlappen.")
        return

    # hydraulica
    df["dh"] = df["h_sea"] - df["h_dock"]
    df["dh_abs"] = np.sqrt(np.abs(df["dh"]))

    # debiet
    df["Q"] = A * df["v"]

    # regimes
    flood = df[df["dh"] > 0]
    ebb = df[df["dh"] < 0]

    warnings = []

    Cf = Ce = None
    if len(flood) >= 2:
        Cf = np.polyfit(flood["dh_abs"], flood["Q"], 1)
    else:
        warnings.append("Onvoldoende punten voor flood-regressie (dh > 0).")

    if len(ebb) >= 2:
        Ce = np.polyfit(ebb["dh_abs"], ebb["Q"], 1)
    else:
        warnings.append("Onvoldoende punten voor ebb-regressie (dh < 0).")

    if Cf is not None and Ce is not None and Cf[0] < 0 and Ce[0] < 0:
        df["Q"] = -df["Q"]
        flood = df[df["dh"] > 0]
        ebb = df[df["dh"] < 0]
        if len(flood) >= 2:
            Cf = np.polyfit(flood["dh_abs"], flood["Q"], 1)
        if len(ebb) >= 2:
            Ce = np.polyfit(ebb["dh_abs"], ebb["Q"], 1)
        warnings.append("Snelheidsteken automatisch omgedraaid (Q*(-1)) omdat beide hellingen negatief waren.")

    print("\n=== DEBIETMODEL ===")
    if Cf is not None:
        print(f"Flood: Q = {Cf[0]:.3f} * sqrt(dh)")
    else:
        print("Flood: onvoldoende data voor model.")
    if Ce is not None:
        print(f"Ebb  : Q = {Ce[0]:.3f} * sqrt(|dh|)")
    else:
        print("Ebb  : onvoldoende data.")
    if warnings:
        for w in warnings:
            print("Waarschuwing:", w)

    # =========================
    # MODELLEN
    # =========================
    C = None
    rmse = r2 = None
    res = None

    # 1) Globaal model zonder intercept: v = C sign(Δh) √|Δh|
    if len(df) >= 2:
        C = np.polyfit(df["dh_abs"], df["v"], 1)[0]
    else:
        warnings.append("Onvoldoende punten voor globaal C-model.")

    # 2) Piecewise model met intercepts (flood/ebb), levert betere fit bij grotere debieten
    df["Q_model"] = np.nan
    if Cf is not None:
        mask_f = df["dh"] > 0
        df.loc[mask_f, "Q_model"] = Cf[0] * df.loc[mask_f, "dh_abs"] + (Cf[1] if len(Cf) > 1 else 0)
    if Ce is not None:
        mask_e = df["dh"] < 0
        df.loc[mask_e, "Q_model"] = Ce[0] * df.loc[mask_e, "dh_abs"] + (Ce[1] if len(Ce) > 1 else 0)

    # fallback: globaal model naar Q
    if df["Q_model"].isna().any() and C is not None:
        df.loc[df["Q_model"].isna(), "Q_model"] = A * (C * np.sign(df["dh"]) * df["dh_abs"])

    # v_model afgeleid uit Q_model
    df["v_model"] = df["Q_model"] / A

    # metrics op basis van beschikbare modelpunten
    valid = df["v_model"].notna()
    if valid.any():
        res = (df.loc[valid, "v"] - df.loc[valid, "v_model"])
        rmse = np.sqrt(np.mean(res**2))
        r2 = 1 - np.sum(res**2) / np.sum((df.loc[valid, "v"] - df.loc[valid, "v"].mean())**2)

    # =========================
    # ARX-achtig realtime model
    # =========================
    coef_arx = None
    if len(df) >= 10:
        df["dh_dt"] = np.gradient(df["dh"], 300)
        df["v_lag"] = df["v"].shift(1)
        df_arx = df.dropna(subset=["dh_abs", "dh", "dh_dt", "v_lag", "v"])
        if len(df_arx) >= 5:
            X = np.column_stack([df_arx["dh_abs"], df_arx["dh"], df_arx["dh_dt"], df_arx["v_lag"]])
            y = df_arx["v"]
            coef_arx = np.linalg.lstsq(X, y, rcond=None)[0]
            df.loc[df_arx.index, "v_arx"] = X @ coef_arx
            df["Q_arx"] = A * df["v_arx"]
        else:
            warnings.append("Onvoldoende punten voor ARX-model (na lag/gradient).")
    else:
        warnings.append("Onvoldoende punten voor ARX-model (min. 10).")

    # =========================
    # DETECTIE
    # =========================
    # gradient kan falen bij identieke dh_abs; gebruik veilige deling
    dv = np.gradient(df["v"])
    ddh = np.gradient(df["dh_abs"])
    denom = np.where(np.abs(ddh) < 1e-12, np.nan, ddh)
    df["slope"] = dv / denom
    slope_thresh = 0.1
    df["limit_flag"] = (df["slope"] < slope_thresh) & df["slope"].notna()

    if res is not None and rmse is not None:
        df["pump_flag"] = np.abs(res) > 2 * rmse
    else:
        df["pump_flag"] = False

    median = df["v"].median()
    mad = np.median(np.abs(df["v"] - median))
    df["spike_flag"] = np.abs(df["v"] - median) > 5 * mad

    n_limit = int(df["limit_flag"].sum())
    n_pump = int(df["pump_flag"].sum())
    n_spike = int(df["spike_flag"].sum())
    n_versass = len(versass)

    # Detect sprongen in saspeil (versassingen) op basis van eerste verschil
    jump_events = pd.DataFrame(columns=["time", "delta"])
    thresh = 0.0  # default zodat downstream code altijd een waarde heeft
    if "h_sas" in df.columns:
        diff_sas = df["h_sas"].diff()
        thresh = max(0.05, 3 * diff_sas.std(skipna=True))  # min 5 cm of 3σ
        jump_mask = diff_sas.abs() >= thresh
        jump_events = pd.DataFrame({"time": df.index[jump_mask], "delta": diff_sas[jump_mask]})
        jump_events = jump_events.dropna()

    # =========================
    # FIGUREN
    # =========================

    # debietcurve
    plt.figure()
    plt.scatter(df["dh"], df["Q"], s=5, label="Q gemeten [m^3/s]")

    x = np.linspace(df["dh"].min(), df["dh"].max(), 200)
    if Cf is not None:
        plt.plot(x[x>0], Cf[0]*np.sqrt(x[x>0]) + (Cf[1] if len(Cf)>1 else 0), label="Flood model [m^3/s]")
    if Ce is not None:
        plt.plot(x[x<0], Ce[0]*np.sqrt(-x[x<0]) + (Ce[1] if len(Ce)>1 else 0), label="Ebb model [m^3/s]")

    plt.legend()
    plt.title("Debietcurve")
    plt.xlabel("Delta h [m]")
    plt.ylabel("Q [m^3/s]")
    f1 = OUTDIR / "debietcurve.png"
    plt.savefig(f1)
    plt.close()

    # tijdreeks
    if "h_sas" in df.columns:
        fig, (ax, ax_sas) = plt.subplots(
            2,
            1,
            figsize=(10, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax_sas = None

    ax.plot(df.index, df["Q"], label="Q gemeten [m^3/s]")
    if "Q_model" in df:
        ax.plot(df.index, df["Q_model"], label="Q model (piecewise) [m^3/s]")
    if "Q_arx" in df:
        ax.plot(df.index, df["Q_arx"], label="Q model (ARX) [m^3/s]", alpha=0.7)

    if ax_sas is not None:
        ax_sas.plot(df.index, df["h_sas"], color="tab:purple", label="Saspeil [m]")
        ax_sas.set_ylabel("Saspeil [m]")
        ax_sas.set_xlabel("tijd")
        ax_sas.grid(True, alpha=0.3)
        ax_sas.legend(loc="upper left")
    else:
        ax.set_xlabel("tijd")

    ax.set_title("Debiet tijdreeks")
    ax.set_ylabel("Q [m^3/s]")
    ax.legend(loc="upper left")
    f2 = OUTDIR / "tijdreeks_Q.png"
    fig.savefig(f2)
    plt.close(fig)

    # residuen (optioneel)
    res_fig = None
    if res is not None:
        plt.figure(figsize=(10,4))
        plt.plot(df.index, res, label="v - v_model [m/s]")
        plt.axhline(0, color="k", linewidth=0.8)
        plt.title("Residuen snelheid")
        plt.xlabel("tijd")
        plt.ylabel("v - v_model [m/s]")
        plt.legend()
        res_fig = OUTDIR / "residuen.png"
        plt.savefig(res_fig)
        plt.close()

    # drukfiguur (indien aanwezig)
    p_fig = None
    if {"h_sea","h_dock"}.issubset(df.columns) or "p_adcp" in df.columns or "h_sas" in df.columns:
        plt.figure(figsize=(10,4))
        if "h_sea" in df:
            plt.plot(df.index, df["h_sea"], label="Zeepeil [m]")
        if "h_dock" in df:
            plt.plot(df.index, df["h_dock"], label="Dokpeil [m]")
        if "h_sas" in df:
            plt.plot(df.index, df["h_sas"], label="Saspeil [m]", linestyle="--")
        if "p_adcp" in df:
            plt.plot(df.index, df["p_adcp"], label="ADCP druk [bron-eenheid]", alpha=0.7)
        plt.title("Druk / Waterpeilen")
        plt.xlabel("tijd")
        plt.ylabel("peil [m] / druk [bron-eenheid]")
        plt.legend()
        p_fig = OUTDIR / "druk.png"
        plt.savefig(p_fig)
        plt.close()

    # =========================
    # RAPPORT
    # =========================
    # helper voor nette NA-format
    def fmt3(x):
        return f"{x:.3f}" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "n.v.t."

    arx_text = "ARX: onvoldoende data"
    if coef_arx is not None:
        arx_text = (
            f"ARX: v_t = {coef_arx[0]:.3f}|Δh| + {coef_arx[1]:.3f}Δh + "
            f"{coef_arx[2]:.3f}·dΔh/dt + {coef_arx[3]:.3f}·v_(t-1)"
        )

    rmse_text = f"RMSE = sqrt(mean((v - v_model)²)) = {fmt3(rmse)} (N={len(df)})" if rmse is not None else "RMSE: n.v.t."

    html = f"""
    <html><body>

    <h1>WL Debietrapport</h1>

    <h2>Doorsnede</h2>
    A = {A} m²

    <h2>Model</h2>
    {"Flood: Q = {:.3f} √Δh {:+.3f}".format(Cf[0], Cf[1]) if Cf is not None else "Flood: onvoldoende data"}<br>
    {"Ebb: Q = {:.3f} √|Δh| {:+.3f}".format(Ce[0], Ce[1]) if Ce is not None else "Ebb: onvoldoende data"}

    <h2>Kwaliteit</h2>
    Globale C: {fmt3(C)}<br>
    RMSE (v-model): {fmt3(rmse)}<br>
    R² (v-model): {fmt3(r2)}<br>
    {arx_text}

    <h2>Formules ingevuld</h2>
    <ul>
      <li>{rmse_text}</li>
      <li>{arx_text}</li>
    </ul>

    <h2>ARX model uitleg</h2>
    <ul>
      <li>Doel: kleine oscillaties en vertragingen modelleren die het statische √Δh-model mist.</li>
      <li>Inputvariabelen: |Δh| (amplitude), Δh (teken/asymmetrie), dΔh/dt (snelheid van peilverandering), v_(t-1) (traagheid/viscous memory).</li>
      <li>Vorm: v_t = a1|Δh| + a2Δh + a3·dΔh/dt + a4·v_(t-1).</li>
      <li>Interpretatie coëfficiënten: a1/a2 sturen hoofdrespons, a3 vangt pomp- of golfcomponenten, a4 dempt/ versterkt vorige snelheid (0&lt;a4&lt;1 = stabiliserend, a4&gt;1 = mogelijk overshoot).</li>
      <li>Gebruik: levert Q_arx = A·v_t en wordt naast het piecewise √Δh-model in de tijdreeks getoond.</li>
    </ul>

    <h2>Interpretatie</h2>
    <ul>
    <li>Niet-lineair gedrag bevestigd</li>
    <li>Verschil tussen ebb en flood = hysterese</li>
    <li>Debiet gelimiteerd door hydraulische capaciteit</li>
    {"".join(f"<li>{w}</li>" for w in warnings)}
    </ul>

    <h2>Detectie</h2>
    <ul>
    <li>Verzadiging (slope &lt; {slope_thresh}): {n_limit} punten</li>
    <li>Pompen domineren (|res| &gt; 2·RMSE): {n_pump} punten</li>
    <li>ADCP spikes (|v-median| &gt; 5·MAD): {n_spike} punten</li>
    <li>Versassingen uitgezet op tijdlijn: {n_versass}</li>
    <li>Saspeil-sprongen (>|{thresh:.2f}| m): {len(jump_events)}</li>
    </ul>

    <h2>Debietcurve</h2>
    <img src="debietcurve.png" width="700">

    <h2>Tijdreeks</h2>
    <img src="tijdreeks_Q.png" width="900">

    {'<h2>Residuen</h2><img src="residuen.png" width="900">' if res_fig else ""}
    {'<h2>Druk</h2><img src="druk.png" width="900">' if p_fig else ""}

    </body></html>
    """

    (OUTDIR / "debietrapport.html").write_text(html, encoding="utf-8")

    print("\n✅ Debietrapport klaar in:", OUTDIR)

# =========================
# GUI
# =========================
def main():

    root = tk.Tk()
    root.withdraw()

    adcp = filedialog.askopenfilename(title="ADCP TSV")
    sea = filedialog.askopenfilename(title="Zee ZIP")
    dock = filedialog.askopenfilename(title="Dok ZIP")
    sas_csv = filedialog.askopenfilename(
        title="Saspeil/druk CSV (optioneel, tijd;dok;sas;zee)",
        filetypes=[("CSV", "*.csv"), ("Alle bestanden", "*.*")],
    )
    versassing = filedialog.askopenfilename(
        title="Versassingsdata (optioneel)",
        filetypes=[("CSV/TSV/Excel", "*.csv *.tsv *.txt *.xlsx *.xls"), ("Alle bestanden", "*.*")],
    )
    if not versassing:
        versassing = None
    if not sas_csv:
        sas_csv = None

    run(adcp, sea, dock, versassing, sas_csv)

if __name__ == "__main__":
    main()
