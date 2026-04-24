import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import importlib.util


_HTML_STATE_HELPER_ROOT = Path(__file__).resolve().parents[1]
if not (_HTML_STATE_HELPER_ROOT / "html_state_saver.py").exists():
    _HTML_STATE_HELPER_ROOT = Path(__file__).resolve().parents[2]
_HTML_STATE_HELPER_SPEC = importlib.util.spec_from_file_location(
    "html_state_saver", _HTML_STATE_HELPER_ROOT / "html_state_saver.py"
)
if _HTML_STATE_HELPER_SPEC is None or _HTML_STATE_HELPER_SPEC.loader is None:
    raise ImportError("html_state_saver.py kon niet worden geladen.")
_html_state_saver = importlib.util.module_from_spec(_HTML_STATE_HELPER_SPEC)
_HTML_STATE_HELPER_SPEC.loader.exec_module(_html_state_saver)
add_interactive_html_saver = _html_state_saver.add_interactive_html_saver


# ----------------------------
# IO helpers
# ----------------------------
def read_rtk_offsets_csv(path: str) -> pd.DataFrame:
    """
    Leest *_filtered_offsets.csv met ';' separator.
    Vereist: rtk_anchor_time, rtk_E_m, rtk_N_m, rtk_H_m
    """
    p = Path(path)
    df = pd.read_csv(p, sep=";")

    req = ["rtk_anchor_time", "rtk_E_m", "rtk_N_m", "rtk_H_m"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"RTK file mist kolom '{c}'")

    df["rtk_anchor_time"] = pd.to_datetime(df["rtk_anchor_time"], utc=True, errors="coerce")
    df = df.sort_values("rtk_anchor_time").reset_index(drop=True)
    return df


def read_bathy_csv(path: str) -> pd.DataFrame:
    """
    Leest Bathy_*_bottom_track.csv met kolommen: x,y,z (comma-separated).
    Verwacht: x = E, y = N, z = diepte (positief omlaag).
    """
    p = Path(path)
    df = pd.read_csv(p)

    df.columns = [c.strip().lower() for c in df.columns]
    if not set(["x", "y", "z"]).issubset(df.columns):
        raise ValueError("Bathy file verwacht kolommen: x,y,z")

    df["x"] = df["x"].astype(float)
    df["y"] = df["y"].astype(float)
    df["z"] = df["z"].astype(float)
    return df


# ----------------------------
# Figure builders
# ----------------------------
def _xy_mode(E, N, mode):
    E = np.asarray(E, dtype=float)
    N = np.asarray(N, dtype=float)
    E0, N0 = E[0], N[0]

    if mode == "relative":
        return (E - E0), (N - N0), "ΔE (m)", "ΔN (m)"
    return E, N, "E (m)", "N (m)"


def build_single_figure(E, N, depth, title, mode="absolute", as_line=True):
    """
    Eén dataset in 3D:
      - depth positief omlaag
      - markers gekleurd op depth
      - z-axis reversed
      - z-exaggeration slider
    as_line=True => lines+markers, anders markers.
    """
    E = np.asarray(E, dtype=float)
    N = np.asarray(N, dtype=float)
    depth = np.asarray(depth, dtype=float)

    x, y, xlab, ylab = _xy_mode(E, N, mode)
    zlab = "Diepte (m) (positief omlaag)"

    factors = [0.5, 1, 2, 5, 10]
    z0 = depth
    z_disp = z0 * 1.0

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z_disp,
            mode="lines+markers" if as_line else "markers",
            name=title,
            marker=dict(
                size=3,
                color=depth,
                colorbar=dict(title="Diepte (m)"),
            ),
            line=dict(width=3) if as_line else None,
            hovertemplate=(
                f"{xlab}=%{{x:.3f}} m<br>"
                f"{ylab}=%{{y:.3f}} m<br>"
                "Diepte=%{marker.color:.3f} m<br>"
                f"{zlab}=%{{z:.3f}} m<extra></extra>"
            ),
        )
    )

    steps = []
    for f in factors:
        steps.append(dict(method="restyle", args=[{"z": [z0 * f]}], label=f"{f}×"))

    fig.update_layout(
        title=f"{title} — {'relatief' if mode=='relative' else 'absolute'}",
        scene=dict(
            xaxis_title=xlab,
            yaxis_title=ylab,
            zaxis=dict(title=zlab, autorange="reversed"),
            aspectmode="data",
        ),
        sliders=[dict(
            active=1,
            currentvalue={"prefix": "Z-overdrijving: "},
            pad={"t": 35},
            steps=steps
        )],
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def build_combined_figure(rtk_E, rtk_N, rtk_depth, bathy_E, bathy_N, bathy_depth,
                          title="Combined (RTK + Bathy)", mode="absolute"):
    """
    Gecombineerde 3D figuur:
      - RTK: lijn+markers
      - Bathy: puntenwolk
      - Beide gekleurd op diepte
      - Eén z-overdrijving slider die beide traces tegelijk schaalt
    """
    rtk_E = np.asarray(rtk_E, dtype=float)
    rtk_N = np.asarray(rtk_N, dtype=float)
    rtk_depth = np.asarray(rtk_depth, dtype=float)

    bathy_E = np.asarray(bathy_E, dtype=float)
    bathy_N = np.asarray(bathy_N, dtype=float)
    bathy_depth = np.asarray(bathy_depth, dtype=float)

    # Gebruik dezelfde XY-transformatie per dataset (relatief per eigen eerste punt)
    # (Als je liever relatieve bathy t.o.v. RTK startpunt wil, zeg het: dan zet ik beide op RTK E0/N0.)
    rtk_x, rtk_y, xlab, ylab = _xy_mode(rtk_E, rtk_N, mode)
    bathy_x, bathy_y, _, _ = _xy_mode(bathy_E, bathy_N, mode)

    zlab = "Diepte (m) (positief omlaag)"
    factors = [0.5, 1, 2, 5, 10]

    rtk_z0 = rtk_depth
    bathy_z0 = bathy_depth

    fig = go.Figure()

    # RTK trace
    fig.add_trace(
        go.Scatter3d(
            x=rtk_x, y=rtk_y, z=rtk_z0 * 1.0,
            mode="lines+markers",
            name="RTK (lijn+punten)",
            marker=dict(size=3, color=rtk_depth),
            line=dict(width=4),
            hovertemplate=(
                f"RTK<br>{xlab}=%{{x:.3f}} m<br>{ylab}=%{{y:.3f}} m<br>"
                "Diepte=%{marker.color:.3f} m<br>"
                f"{zlab}=%{{z:.3f}} m<extra></extra>"
            ),
        )
    )

    # Bathy trace
    fig.add_trace(
        go.Scatter3d(
            x=bathy_x, y=bathy_y, z=bathy_z0 * 1.0,
            mode="markers",
            name="Bathy bottom-track (punten)",
            marker=dict(
                size=2,
                color=bathy_depth,
                colorbar=dict(title="Diepte (m)")  # één colorbar (hier aan bathy)
            ),
            hovertemplate=(
                f"Bathy<br>{xlab}=%{{x:.3f}} m<br>{ylab}=%{{y:.3f}} m<br>"
                "Diepte=%{marker.color:.3f} m<br>"
                f"{zlab}=%{{z:.3f}} m<extra></extra>"
            ),
        )
    )

    # Slider update beide traces
    steps = []
    for f in factors:
        steps.append(dict(
            method="restyle",
            args=[{"z": [rtk_z0 * f, bathy_z0 * f]}],  # trace 0 en 1
            label=f"{f}×"
        ))

    fig.update_layout(
        title=f"{title} — {'relatief' if mode=='relative' else 'absolute'}",
        scene=dict(
            xaxis_title=xlab,
            yaxis_title=ylab,
            zaxis=dict(title=zlab, autorange="reversed"),
            aspectmode="data",
        ),
        sliders=[dict(
            active=1,
            currentvalue={"prefix": "Z-overdrijving: "},
            pad={"t": 35},
            steps=steps
        )],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=70, b=0),
    )

    return fig


# ----------------------------
# GUI
# ----------------------------
def pick_files():
    root = tk.Tk()
    root.withdraw()
    root.update()

    rtk_path = filedialog.askopenfilename(
        title="Selecteer *_filtered_offsets.csv (RTK)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    bathy_path = filedialog.askopenfilename(
        title="Selecteer Bathy_*_bottom_track.csv (x,y,z) (optioneel - annuleer indien niet)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    root.destroy()
    return rtk_path, bathy_path


def main():
    rtk_path, bathy_path = pick_files()
    if not rtk_path and not bathy_path:
        return

    out_files = []

    try:
        rtk_ok = bool(rtk_path)
        bathy_ok = bool(bathy_path)

        # --- RTK ---
        if rtk_ok:
            df_rtk = read_rtk_offsets_csv(rtk_path)
            rtk_E = df_rtk["rtk_E_m"].to_numpy()
            rtk_N = df_rtk["rtk_N_m"].to_numpy()
            H = df_rtk["rtk_H_m"].to_numpy()
            H0 = H[0]
            rtk_depth = (H0 - H)  # positief omlaag

            base = Path(rtk_path)
            fig_rtk_abs = build_single_figure(rtk_E, rtk_N, rtk_depth, "RTK (H→diepte)", mode="absolute", as_line=True)
            fig_rtk_rel = build_single_figure(rtk_E, rtk_N, rtk_depth, "RTK (H→diepte)", mode="relative", as_line=True)

            out_rtk_abs = base.with_name(base.stem + "_3D_RTK_depth_absolute.html")
            out_rtk_rel = base.with_name(base.stem + "_3D_RTK_depth_relative.html")

            fig_rtk_abs.write_html(out_rtk_abs, include_plotlyjs="cdn")
            fig_rtk_rel.write_html(out_rtk_rel, include_plotlyjs="cdn")
            add_interactive_html_saver(out_rtk_abs)
            add_interactive_html_saver(out_rtk_rel)

            out_files += [str(out_rtk_abs), str(out_rtk_rel)]

        # --- Bathy ---
        if bathy_ok:
            df_b = read_bathy_csv(bathy_path)
            bathy_E = df_b["x"].to_numpy()
            bathy_N = df_b["y"].to_numpy()
            bathy_depth = df_b["z"].to_numpy()  # z = diepte positief omlaag

            base = Path(bathy_path)
            fig_b_abs = build_single_figure(bathy_E, bathy_N, bathy_depth, "Bathy bottom-track (z=diepte)", mode="absolute", as_line=True)
            fig_b_rel = build_single_figure(bathy_E, bathy_N, bathy_depth, "Bathy bottom-track (z=diepte)", mode="relative", as_line=True)

            out_b_abs = base.with_name(base.stem + "_3D_BATHY_depth_absolute.html")
            out_b_rel = base.with_name(base.stem + "_3D_BATHY_depth_relative.html")

            fig_b_abs.write_html(out_b_abs, include_plotlyjs="cdn")
            fig_b_rel.write_html(out_b_rel, include_plotlyjs="cdn")
            add_interactive_html_saver(out_b_abs)
            add_interactive_html_saver(out_b_rel)

            out_files += [str(out_b_abs), str(out_b_rel)]

        # --- Combined ---
        if rtk_ok and bathy_ok:
            base = Path(rtk_path)  # schrijf combined naast RTK file

            fig_c_abs = build_combined_figure(
                rtk_E, rtk_N, rtk_depth,
                bathy_E, bathy_N, bathy_depth,
                title="Combined (RTK + Bathy)", mode="absolute"
            )
            fig_c_rel = build_combined_figure(
                rtk_E, rtk_N, rtk_depth,
                bathy_E, bathy_N, bathy_depth,
                title="Combined (RTK + Bathy)", mode="relative"
            )

            out_c_abs = base.with_name(base.stem + "_3D_COMBINED_depth_absolute.html")
            out_c_rel = base.with_name(base.stem + "_3D_COMBINED_depth_relative.html")

            fig_c_abs.write_html(out_c_abs, include_plotlyjs="cdn")
            fig_c_rel.write_html(out_c_rel, include_plotlyjs="cdn")
            add_interactive_html_saver(out_c_abs)
            add_interactive_html_saver(out_c_rel)

            out_files += [str(out_c_abs), str(out_c_rel)]

        messagebox.showinfo("Klaar", "HTML's opgeslagen:\n\n" + "\n".join(out_files))

    except Exception as e:
        messagebox.showerror("Fout", str(e))


if __name__ == "__main__":
    main()
