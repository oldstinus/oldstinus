#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.io import loadmat
import tkinter as tk
from tkinter import filedialog

# ---------------- MATLAB helpers ----------------
def _is_mat_struct(x):
    return str(type(x)).endswith("mat_struct'>") or isinstance(x, np.void)

def _from_mat(x):
    if _is_mat_struct(x):
        return _todict(x)
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            if x.shape == ():
                return _from_mat(x.item())
            out = []
            for idx in np.ndindex(x.shape):
                out.append(_from_mat(x[idx]))
            return out
        return x
    return x

def _todict(matobj):
    out = {}
    fns = getattr(matobj, '_fieldnames', None)
    if fns is not None:
        for f in fns: out[f] = _from_mat(getattr(matobj, f))
        return out
    if hasattr(matobj, 'dtype') and matobj.dtype.names:
        for f in matobj.dtype.names: out[f] = _from_mat(matobj[f])
        return out
    for k in dir(matobj):
        if not k.startswith('_'):
            out[k] = _from_mat(getattr(matobj, k))
    return out

def deep_find_first(obj, key_substr):
    key_substr = key_substr.lower()
    def _walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                p = path + [str(k)]
                if key_substr in "/".join(p).lower():
                    return v, p
                r = _walk(v, p)
                if r is not None: return r
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                r = _walk(v, path + [f"[{i}]"])
                if r is not None: return r
        elif isinstance(o, np.ndarray) and o.dtype == object:
            for idx in np.ndindex(o.shape):
                r = _walk(o[idx], path + [f"[{','.join(map(str, idx))}]"])
                if r is not None: return r
        return None
    got = _walk(obj, [])
    return got if got is not None else (None, None)

# ---------------- tijd & vorm ----------------
MATLAB_EPOCH_DAYS = 719529.0
def datenum_to_datetime(dn, tz="Europe/Brussels", serial_is="local"):
    secs = (np.asarray(dn, dtype=float) - MATLAB_EPOCH_DAYS) * 86400.0
    if serial_is.lower() == "utc":
        return pd.to_datetime(secs, unit="s", utc=True).tz_convert(tz)
    else:
        return pd.to_datetime(secs, unit="s").tz_localize(tz)

def orient_depth_ens(M):
    M = np.asarray(M, dtype=float)
    if M.ndim != 2: raise ValueError("matrix niet 2D")
    r, c = M.shape
    return M if r <= c else M.T

def col_weighted_mean(V, W, mask=None):
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        V = np.where(mask, V, np.nan)
    num = np.nansum(V * W, axis=0)
    den = np.nansum(W * np.isfinite(V), axis=0)
    den[den == 0] = np.nan
    out = num / den
    out[~np.isfinite(out)] = np.nan
    return out

def project_uv(U, V, deg):
    th = np.deg2rad(deg)
    return U*np.cos(th) + V*np.sin(th)

# ---------------- één raai verwerken ----------------
def process_transect(tr, ti, tz, serial_is, mode, theta_deg=0.0):
    if not isinstance(tr, dict): raise RuntimeError("ongeldige transect-struct")

    wVel   = tr.get("wVel", {})
    depths = tr.get("depths", {})
    dt     = tr.get("dateTime", {})

    start_dn = np.asarray(dt.get("startSerialTime", np.nan), dtype=float).ravel()
    ens_dur  = np.asarray(dt.get("ensDuration_sec", []), dtype=float).ravel()
    if start_dn.size == 0 or ens_dur.size == 0:
        raise RuntimeError("tijdvelden ontbreken")
    start_dn = start_dn[0]

    U = wVel.get("uProcessed_mps", None)
    V = wVel.get("vProcessed_mps", None)
    D = wVel.get("d_mps", None)
    valid = wVel.get("validData", None)

    U_mat = V_mat = D_mat = None
    if D is not None: D_mat = orient_depth_ens(np.asarray(D, dtype=float))
    if U is not None and V is not None:
        U_mat = orient_depth_ens(np.asarray(U, dtype=float))
        V_mat = orient_depth_ens(np.asarray(V, dtype=float))
        if U_mat.shape != V_mat.shape:
            if U_mat.T.shape == V_mat.shape: U_mat = U_mat.T
            elif V_mat.T.shape == U_mat.shape: V_mat = V_mat.T
            else: raise RuntimeError("u/v dims mismatch")

    if mode == "d_mps" and D_mat is None: raise RuntimeError("d_mps niet beschikbaar")
    if mode == "proj"  and (U_mat is None or V_mat is None): raise RuntimeError("u/v ontbreken")

    mask = None
    if valid is not None:
        vd = np.asarray(valid)
        if vd.ndim == 3: mask = vd[:,:,0].astype(bool)
        elif vd.ndim == 2: mask = vd.astype(bool)

    weights = None
    w = None
    for k in ("btDepths", "vbDepths"):
        blk = depths.get(k, {})
        if isinstance(blk, dict) and "depthCellSize_m" in blk:
            w = np.asarray(blk["depthCellSize_m"], dtype=float); break
    if w is not None:
        tgt = D_mat.shape if mode == "d_mps" else U_mat.shape
        if w.shape == tgt: weights = w
        elif w.T.shape == tgt: weights = w.T

    nEns = (D_mat.shape[1] if mode == "d_mps" else U_mat.shape[1])
    if ens_dur.size >= nEns: ens_dur = ens_dur[:nEns]
    offs = (np.cumsum(ens_dur) - ens_dur) if ens_dur.size == nEns else \
           (np.concatenate(([0.0], np.cumsum(ens_dur))) if ens_dur.size == nEns-1 else np.zeros(nEns))
    t0 = datenum_to_datetime(start_dn, tz=tz, serial_is=serial_is)
    time_vec = (t0 + pd.to_timedelta(offs, unit="s")).round("S")

    keep = None
    itx = tr.get("inTransectIdx", None)
    if itx is not None:
        try:
            idx = np.asarray(itx, dtype=int).ravel()
            idx = idx[(idx >= 1) & (idx <= nEns)] - 1
            if idx.size: keep = idx
        except Exception: pass
    if keep is not None:
        if mode == "d_mps": D_mat = D_mat[:, keep]
        else: U_mat = U_mat[:, keep]; V_mat = V_mat[:, keep]
        if mask is not None: mask = mask[:, keep]
        if weights is not None: weights = weights[:, keep]
        time_vec = time_vec[keep]; nEns = keep.size

    if mode == "d_mps":
        comp2d = D_mat
        comp_label = "d_mps (QRev streamwise)"
    else:
        comp2d = project_uv(U_mat, V_mat, theta_deg)
        if comp2d.ndim == 1: comp2d = comp2d[np.newaxis, :]
        comp_label = f"u/v geprojecteerd op θ={theta_deg:.1f}°"

    if weights is not None:
        mean_speed = col_weighted_mean(comp2d, weights, mask=mask)
        col_label = "gewogen (depthCellSize_m)"
    else:
        X = comp2d.copy()
        if mask is not None: X = np.where(mask, X, np.nan)
        mean_speed = np.nanmean(X, axis=0)
        col_label = "ongewogen (mask toegepast)" if mask is not None else "ongewogen"

    tid = tr.get("transectID", None) or f"T{ti:03d}"
    df = pd.DataFrame({
        "time": time_vec,
        "mean_speed_mps": mean_speed.astype(float),
        "transect_id": str(tid),
        "transect_index": ti,
        "component": comp_label,
        "column_mean": col_label
    })
    return df

# ---------------- scoringsfunctie voor hoek-keuze ----------------
def score_series(y):
    y = np.asarray(y, dtype=float)
    m = np.nanmedian(np.abs(y))
    # gladheid: std van eerste verschil (na median filter 9 p)
    if np.sum(np.isfinite(y)) > 10:
        yy = pd.Series(y).rolling(9, center=True, min_periods=3).median().to_numpy()
    else:
        yy = y.copy()
    dy = np.diff(yy)
    s = np.nanstd(dy) if np.isfinite(dy).any() else np.inf
    # tekenwissels (na smoothing)
    sign = np.sign(yy)
    flips = np.nansum(np.abs(np.diff(sign)) > 0)
    # score: groter is beter
    return m / (s + 1e-6) - 0.005*flips

def make_concat(mat_path, tz, serial_is, try_d_mps=True, th_min=0.0, th_max=179.0, th_step=1.0):
    raw = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    dat = {k: _from_mat(v) for k, v in raw.items() if not k.startswith("__")}
    transects, _ = deep_find_first(dat, "transects")
    if transects is None: raise RuntimeError("Kon 'transects' niet vinden.")
    if isinstance(transects, list) and len(transects) == 1 and isinstance(transects[0], list):
        transects = transects[0]
    elif isinstance(transects, dict):
        transects = list(transects.values())
    if not isinstance(transects, list) or len(transects) == 0:
        raise RuntimeError("Geen transecten.")

    # 1) kandidaat d_mps
    best_d = None
    if try_d_mps:
        dfs = []
        ok = 0
        for i, tr in enumerate(transects, 1):
            try:
                dfs.append(process_transect(tr, i, tz, serial_is, mode="d_mps"))
                ok += 1
            except Exception as e:
                print(f"[INFO] d_mps transect {i} overgeslagen: {e}", file=sys.stderr)
        if ok:
            d_all = pd.concat(dfs, ignore_index=True).sort_values("time")
            best_d = d_all

    # 2) hoek sweep met u/v projectie
    thetas = np.arange(th_min, th_max + 1e-9, th_step, dtype=float)
    best_theta = None; best_score = -np.inf; best_df = None
    for th in thetas:
        dfs = []
        ok = 0
        for i, tr in enumerate(transects, 1):
            try:
                dfs.append(process_transect(tr, i, tz, serial_is, mode="proj", theta_deg=float(th)))
                ok += 1
            except Exception:
                pass
        if not ok: continue
        df_all = pd.concat(dfs, ignore_index=True).sort_values("time")
        sc = score_series(df_all["mean_speed_mps"].to_numpy())
        if sc > best_score:
            best_score = sc; best_theta = float(th); best_df = df_all

    # kies beste
    if best_d is None and best_df is None:
        raise RuntimeError("Geen bruikbare snelheidsreeks gevonden.")
    if best_d is None:
        chosen = ("proj", best_theta, best_df)
    elif best_df is None:
        chosen = ("d_mps", np.nan, best_d)
    else:
        # vergelijk scores nog eens eerlijk
        sc_d  = score_series(best_d["mean_speed_mps"].to_numpy())
        sc_uv = score_series(best_df["mean_speed_mps"].to_numpy())
        chosen = ("d_mps", np.nan, best_d) if sc_d >= sc_uv else ("proj", best_theta, best_df)
    return chosen

# ---------------- CLI ----------------
def pick_file():
    root = tk.Tk(); root.withdraw()
    return filedialog.askopenfilename(title="Selecteer QRev .mat", filetypes=(("MAT","*.mat"),("Alle","*.*")))

def main():
    ap = argparse.ArgumentParser(description="Bepaal automatisch de juiste streamwise snelheden uit QRev MAT.")
    ap.add_argument("--file", "-f", help="QRev .mat (klassiek v5/v7). Laat leeg voor dialoog.")
    ap.add_argument("--tz", default="Europe/Brussels", help="Plot/uitvoer tijdzone.")
    ap.add_argument("--serial", choices=["local","utc"], default="local", help="Interpretatie van startSerialTime.")
    ap.add_argument("--th-min", type=float, default=0.0)
    ap.add_argument("--th-max", type=float, default=179.0)
    ap.add_argument("--th-step", type=float, default=1.0)
    ap.add_argument("--no-dmps", action="store_true", help="Sla d_mps-kandidaat over.")
    ap.add_argument("--out-csv", default="QRev_streamwise_auto.csv")
    ap.add_argument("--out-png", default="QRev_streamwise_auto.png")
    ap.add_argument("--out-report", default="QRev_streamwise_auto_REPORT.txt")
    args = ap.parse_args()

    mat_path = args.file or pick_file()
    if not mat_path:
        print("Geen bestand gekozen.", file=sys.stderr); sys.exit(1)

    mode, theta, df = make_concat(
        mat_path, tz=args.tz, serial_is=args.serial,
        try_d_mps=(not args.no_dmps),
        th_min=args.th_min, th_max=args.th_max, th_step=args.th_step
    )

    # schrijfsels
    time_col = f"time_{args.tz.replace('/','_')}"
    out = df.rename(columns={"time": time_col})
    out.to_csv(args.out_csv, index=False)
    print(f"CSV geschreven: {args.out_csv}")

    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write("QRev automatische componentselectie\n")
        f.write(f"Bestand: {mat_path}\n")
        f.write(f"Tijdzone: {args.tz}   startSerialTime: {args.serial}\n")
        f.write(f"Gekozen: {mode}\n")
        if mode == "proj": f.write(f"Optimale hoek θ (° vanaf oost, ccw): {theta:.2f}\n")
        f.write(f"Records: {len(out)}  raaien: {len(np.unique(out['transect_index']))}\n")

    # plot
    plt.figure(figsize=(14,4))
    tt = pd.to_datetime(out[time_col]).dt.tz_localize(None)
    yy = out["mean_speed_mps"].astype(float)
    plt.plot(tt, yy, linewidth=1.1, color="#0b7285", alpha=0.9, label=("d_mps" if mode=="d_mps" else f"proj θ={theta:.1f}°"))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, alpha=0.25)
    plt.xlabel(f"Tijd ({args.tz})")
    plt.ylabel("Gemiddelde stroomsnelheid (m/s)")
    plt.title("Kolomgemiddelde aangepaste snelheid (alle raaien samengevoegd)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"Plot geschreven: {args.out_png}")
def gui_main():
    def run_script():
        mat_path = filedialog.askopenfilename(title="Selecteer QRev .mat", filetypes=(("MAT","*.mat"),("Alle","*.*")))
        if not mat_path:
            tk.messagebox.showerror("Fout", "Geen bestand gekozen.")
            return

        tz = tz_var.get()
        serial = serial_var.get()
        th_min = float(th_min_var.get())
        th_max = float(th_max_var.get())
        th_step = float(th_step_var.get())
        no_dmps = no_dmps_var.get()
        out_csv = out_csv_var.get()
        out_png = out_png_var.get()
        out_report = out_report_var.get()

        try:
            mode, theta, df = make_concat(
                mat_path, tz=tz, serial_is=serial,
                try_d_mps=(not no_dmps),
                th_min=th_min, th_max=th_max, th_step=th_step
            )
            time_col = f"time_{tz.replace('/','_')}"
            out = df.rename(columns={"time": time_col})
            out.to_csv(out_csv, index=False)
            with open(out_report, "w", encoding="utf-8") as f:
                f.write("QRev automatische componentselectie\n")
                f.write(f"Bestand: {mat_path}\n")
                f.write(f"Tijdzone: {tz}   startSerialTime: {serial}\n")
                f.write(f"Gekozen: {mode}\n")
                if mode == "proj": f.write(f"Optimale hoek θ (° vanaf oost, ccw): {theta:.2f}\n")
                f.write(f"Records: {len(out)}  raaien: {len(np.unique(out['transect_index']))}\n")
            plt.figure(figsize=(14,4))
            tt = pd.to_datetime(out[time_col]).dt.tz_localize(None)
            yy = out["mean_speed_mps"].astype(float)
            plt.plot(tt, yy, linewidth=1.1, color="#0b7285", alpha=0.9, label=("d_mps" if mode=="d_mps" else f"proj θ={theta:.1f}°"))
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=30, ha="right")
            plt.grid(True, alpha=0.25)
            plt.xlabel(f"Tijd ({tz})")
            plt.ylabel("Gemiddelde stroomsnelheid (m/s)")
            plt.title("Kolomgemiddelde aangepaste snelheid (alle raaien samengevoegd)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            tk.messagebox.showinfo("Klaar", f"CSV, rapport en plot zijn opgeslagen.")
        except Exception as e:
            tk.messagebox.showerror("Fout", str(e))

    root = tk.Tk()
    root.title("QRev streamwise - parameters kiezen")

    # Parameters
    tk.Label(root, text="Tijdzone:").grid(row=0, column=0, sticky="e")
    tz_var = tk.StringVar(value="Europe/Brussels")
    tk.Entry(root, textvariable=tz_var).grid(row=0, column=1)

    tk.Label(root, text="Serial time interpretatie:").grid(row=1, column=0, sticky="e")
    serial_var = tk.StringVar(value="local")
    tk.OptionMenu(root, serial_var, "local", "utc").grid(row=1, column=1)

    tk.Label(root, text="Hoek minimum (th_min):").grid(row=2, column=0, sticky="e")
    th_min_var = tk.StringVar(value="0.0")
    tk.Entry(root, textvariable=th_min_var).grid(row=2, column=1)

    tk.Label(root, text="Hoek maximum (th_max):").grid(row=3, column=0, sticky="e")
    th_max_var = tk.StringVar(value="179.0")
    tk.Entry(root, textvariable=th_max_var).grid(row=3, column=1)

    tk.Label(root, text="Hoek stapgrootte (th_step):").grid(row=4, column=0, sticky="e")
    th_step_var = tk.StringVar(value="1.0")
    tk.Entry(root, textvariable=th_step_var).grid(row=4, column=1)

    no_dmps_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="Sla d_mps-kandidaat over", variable=no_dmps_var).grid(row=5, columnspan=2)

    tk.Label(root, text="CSV-bestand:").grid(row=6, column=0, sticky="e")
    out_csv_var = tk.StringVar(value="QRev_streamwise_auto.csv")
    tk.Entry(root, textvariable=out_csv_var).grid(row=6, column=1)

    tk.Label(root, text="PNG-bestand:").grid(row=7, column=0, sticky="e")
    out_png_var = tk.StringVar(value="QRev_streamwise_auto.png")
    tk.Entry(root, textvariable=out_png_var).grid(row=7, column=1)

    tk.Label(root, text="Rapport-bestand:").grid(row=8, column=0, sticky="e")
    out_report_var = tk.StringVar(value="QRev_streamwise_auto_REPORT.txt")
    tk.Entry(root, textvariable=out_report_var).grid(row=8, column=1)

    tk.Button(root, text="Start", command=run_script).grid(row=9, columnspan=2, pady=10)

    root.mainloop()
    
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        # main()  # commentaar uitzetten als je GUI wilt gebruiken
        gui_main()