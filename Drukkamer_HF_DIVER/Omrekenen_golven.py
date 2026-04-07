import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, simpledialog
from datetime import timedelta

# ==========================
#  HULPFUNCTIES
# ==========================

def select_file():
    """Laat gebruiker een CSV-file kiezen (bv. gecombineerde_data.csv)."""
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecteer gecombineerde_data.csv",
        filetypes=[("CSV bestanden", "*.csv"), ("Alle bestanden", "*.*")]
    )
    return file_path

def ask_float(title, prompt, default):
    """Vraag een float via dialoogvenster, met default."""
    root = Tk()
    root.withdraw()
    while True:
        value = simpledialog.askstring(title, f"{prompt}\n(default = {default})")
        if value is None or value.strip() == "":
            return float(default)
        try:
            return float(value.replace(",", "."))
        except ValueError:
            print("Ongeldige invoer, probeer opnieuw.")

def compute_dispersion_relation(omega, h, g=9.81, max_iter=50, tol=1e-6):
    """
    Los de dispersierelatie ω^2 = g k tanh(k h) op voor k (vectorized).

    omega : array [rad/s]
    h     : waterdiepte [m]
    """
    k = (omega**2) / g  # diepe-water startwaarde
    k[omega == 0] = 0.0

    for _ in range(max_iter):
        kh = k * h
        tanh_kh = np.tanh(kh)
        # vermijd deling door nul
        tanh_kh[tanh_kh == 0] = 1e-12

        f = g * k * tanh_kh - omega**2
        df = g * tanh_kh + g * k * h * (1 - tanh_kh**2)

        # idem
        df[df == 0] = 1e-12

        k_new = k - f / df
        if np.max(np.abs(k_new - k)) < tol:
            k = k_new
            break
        k = k_new

    k[k < 0] = 0.0
    return k

def pressure_to_surface_elevation_fft(time, p_bar, h, z, rho=1025.0, g=9.81, min_T_ratio=0.02):
    """
    Zet drukserie (in bar) om naar oppervlakteschommeling η(t) via FFT + dieptecorrectie.

    time : array van datetime64
    p_bar: array druk [bar]
    h    : waterdiepte [m]
    z    : sensor-diepte onder MSL [m] (positief neerwaarts)
    rho  : dichtheid [kg/m³]
    g    : zwaartekracht [m/s²]
    min_T_ratio : minimum verhouding |p(z)| / |p(0)|; frequenties met té sterke attenuatie worden onderdrukt
    """

    # 1) Zorg dat tijdstap constant is
    t_seconds = (time - time[0]).astype("timedelta64[ns]").astype(float) * 1e-9
    dt_arr = np.diff(t_seconds)
    dt = np.median(dt_arr)   # typische tijdstap [s]
    fs = 1.0 / dt            # sampling frequentie [Hz]

    # 2) Druk → Pa en detrenden
    p_pa = p_bar * 1e5
    p_prime = p_pa - np.mean(p_pa)

    N = len(p_prime)
    # 3) FFT van druk
    p_hat = np.fft.rfft(p_prime)
    freqs = np.fft.rfftfreq(N, d=dt)  # [Hz]
    omega = 2 * np.pi * freqs         # [rad/s]

    # 4) Dispersierelatie → golfgetal k
    k = compute_dispersion_relation(omega, h, g=g)

    # 5) Transferfunctie H_p_eta(f):
    # p(z, f) = ρ g η(f) * cosh(k(h - z)) / cosh(k h)
    # → η(f) = p(z, f) * [cosh(kh) / cosh(k(h - z))] / (ρ g)

    kh = k * h
    kz = k * (h - z)

    # vermijd overflows
    cosh_kh = np.cosh(np.clip(kh, -50, 50))
    cosh_kz = np.cosh(np.clip(kz, -50, 50))

    # attenuatie op sensor-diepte
    with np.errstate(divide="ignore", invalid="ignore"):
        T_p = cosh_kz / cosh_kh  # |p(z)| / |p(0)|
    T_p[0] = 1.0  # DC-term

    # sterkte van druksignaal op diepte: frequenties met te kleine T_p niet corrigeren (ruisonderdrukking)
    mask_valid = T_p >= min_T_ratio

    # H_p_eta: van druk naar η
    with np.errstate(divide="ignore", invalid="ignore"):
        H = (cosh_kh / cosh_kz) / (rho * g)
    H[~mask_valid] = 0.0
    H[0] = 0.0  # DC-offset weg

    # 6) η(f) = p_hat * H en inverse FFT
    eta_hat = p_hat * H
    eta = np.fft.irfft(eta_hat, n=N)

    return eta, freqs, fs

def compute_wave_parameters(eta, fs):
    """
    Berekent basis-golfparameters uit η(t):

    - Hm0 (significante golfhoogte, m)
    - Hz (zero-crossing hoogte, m, hier afgeleid uit variantie)
    - Tz (gemiddelde zero-crossing periode, s)
    - Tp (piekperiode, s) op basis van spectrum
    """
    dt = 1.0 / fs
    N = len(eta)

    # Variantie
    eta_prime = eta - np.mean(eta)
    m0 = np.var(eta_prime)
    Hm0 = 4.0 * np.sqrt(m0)  # Hm0

    # Zero-crossing periode Tz
    sgn = np.sign(eta_prime)
    sgn[sgn == 0] = 1
    zero_crossings = np.where(np.diff(sgn) != 0)[0]
    if len(zero_crossings) > 1:
        periods = np.diff(zero_crossings) * dt
        Tz = np.mean(periods)
    else:
        Tz = np.nan

    # Spectrum (eenvoudige FFT-methode)
    eta_hat = np.fft.rfft(eta_prime)
    freqs = np.fft.rfftfreq(N, d=dt)
    T = N * dt
    # eenzijdig spectrum: S(f) ≈ 2/T * |η_hat|^2 * dt^2
    S = (2.0 / T) * (np.abs(eta_hat) ** 2) * (dt**2)
    S[0] = 0.0

    # piekfrequentie en -periode
    if len(S) > 1:
        idx_max = np.argmax(S[1:]) + 1
        fp = freqs[idx_max]
        Tp = 1.0 / fp if fp > 0 else np.nan
    else:
        Tp = np.nan

    return {"Hm0": Hm0, "Tz": Tz, "Tp": Tp, "freqs": freqs, "S": S}

def plot_results(time, eta, wave_params, fs):
    """Maak figuren: η(t) volledig, ingezoomd en spectrum."""
    Hm0 = wave_params["Hm0"]
    Tz = wave_params["Tz"]
    Tp = wave_params["Tp"]
    freqs = wave_params["freqs"]
    S = wave_params["S"]

    # 1) volledige tijdreeks
    plt.figure(figsize=(12, 6))
    plt.plot(time, eta)
    plt.title(f"Oppervlakteschommeling η(t) – volledige tijdreeks\nHm0 ≈ {Hm0:.2f} m, Tz ≈ {Tz:.1f} s, Tp ≈ {Tp:.1f} s")
    plt.xlabel("Tijd")
    plt.ylabel("η [m]")
    plt.grid(True)
    plt.tight_layout()

    # 2) ingezoomd segment (bv. eerste 2 minuten of alles als korter)
    dt = 1.0 / fs
    N = len(eta)
    max_samples_zoom = int(2 * 60 * fs)  # 2 minuten
    n_zoom = min(N, max_samples_zoom)

    plt.figure(figsize=(12, 6))
    plt.plot(time[:n_zoom], eta[:n_zoom])
    plt.title("Oppervlakteschommeling η(t) – ingezoomd (eerste 2 min)")
    plt.xlabel("Tijd")
    plt.ylabel("η [m]")
    plt.grid(True)
    plt.tight_layout()

    # 3) spectrum
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs[1:], S[1:])  # sla f=0 over
    plt.title("Golfspectrum Sη(f)")
    plt.xlabel("Frequentie [Hz]")
    plt.ylabel("Sη(f) [m²/Hz]")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()

    plt.show()

# ==========================
#  HOOFDPROGRAMMA
# ==========================

def main():
    # 1) Kies gecombineerde_data.csv
    file_path = select_file()
    if not file_path:
        print("Geen bestand geselecteerd. Programma stopt.")
        return

    print(f"Bestand geselecteerd: {file_path}")

    # 2) Data inlezen
    try:
        df = pd.read_csv(file_path, parse_dates=["Datetime"])
    except Exception as e:
        print(f"Fout bij inlezen CSV: {e}")
        return

    if "Datetime" not in df.columns or "Pressure" not in df.columns:
        print("CSV moet minimaal kolommen 'Datetime' en 'Pressure' bevatten.")
        return

    df = df.sort_values("Datetime").reset_index(drop=True)
    time = df["Datetime"].to_numpy()
    pressure_bar = df["Pressure"].to_numpy(dtype=float)

    # 3) Vraag fysische parameters op
    h = ask_float("Waterdiepte", "Geef de waterdiepte h [m]", default=10.0)
    z = ask_float("Sensor-diepte", "Geef de sensor-diepte onder MSL z [m]", default=h - 0.5)
    rho = ask_float("Dichtheid", "Geef de waterdichtheid ρ [kg/m³]", default=1025.0)
    g = ask_float("Zwaartekracht", "Geef zwaartekrachtsversnelling g [m/s²]", default=9.81)
    min_T_ratio = ask_float(
        "Diepte-attenuatie-filter",
        "Minimale verhouding |p(z)|/|p(0)| waarvoor frequenties nog worden gecorrigeerd\n"
        "(typisch 0.01–0.05)",
        default=0.02
    )

    print("\n--- Stap 1: druk → oppervlakteschommeling via FFT + dieptecorrectie ---")
    eta, freqs, fs = pressure_to_surface_elevation_fft(
        time, pressure_bar, h=h, z=z, rho=rho, g=g, min_T_ratio=min_T_ratio
    )

    print(f"Geschatte samplingfrequentie fs ≈ {fs:.3f} Hz")
    print(f"Aantal punten: {len(eta)}")

    print("\n--- Stap 2: golfparameters uit η(t) ---")
    wave_params = compute_wave_parameters(eta, fs)

    print(f"Hm0 (significante golfhoogte)\t≈ {wave_params['Hm0']:.3f} m")
    print(f"Tz  (zero-crossing periode)\t≈ {wave_params['Tz']:.3f} s")
    print(f"Tp  (piekperiode)\t\t≈ {wave_params['Tp']:.3f} s")

    print("\n--- Stap 3: figuren tonen ---")
    plot_results(time, eta, wave_params, fs)

if __name__ == "__main__":
    main()
