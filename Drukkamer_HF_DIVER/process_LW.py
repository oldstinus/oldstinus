import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy import signal
import math
from numpy.fft import fft
import matplotlib.pyplot as plt

# Functie om spectrale parameters te berekenen (overgenomen uit je eerdere scripts)
def calculate_Sf_params(S, f, fmin, fmax, delta_f):
    if np.isnan(S).all():
        return np.nan, np.nan, np.nan, np.nan
    else:
        Sf_pd = pd.DataFrame({'f': f, 'Sf': S})
        fp = f[Sf_pd['Sf'][(Sf_pd['f'] >= fmin) & (Sf_pd['f'] <= fmax)].idxmax()]

        m0, m1, m2, m_1 = 0, 0, 0, 0
        index_f = [i[0] for i in np.argwhere((f >= min(fp/3, fmin)) & (f <= min(3*fp, fmax)))]
        for j in range(index_f[0], index_f[-1] + 1):
            m0 += S[j] * delta_f
            m1 += f[j] * S[j] * delta_f
            m2 += (f[j]**2) * S[j] * delta_f
            m_1 += (f[j]**-1) * S[j] * delta_f
        Hm0 = 4 * np.sqrt(m0)
        Tm01 = m0 / m1
        Tm02 = m0 / m2
        Tm_10 = m_1 / m0

        return Hm0, Tm01, Tm02, Tm_10

# Functie om de directory te selecteren
def select_directory():
    root = tk.Tk()
    root.withdraw()  # Verberg het hoofdvenster
    folder_selected = filedialog.askdirectory()
    return folder_selected

# Functie om bestanden in een directory te verwerken
def process_files_in_directory(directory):
    print(f"Verwerken van bestanden in directory: {directory}")
    files = [f for f in os.listdir(directory) if f.endswith('.dat')]  # Selecteer alleen .dat-bestanden
    if not files:
        print("Geen bestanden gevonden in de directory.")
        return
    for file in files:
        file_path = os.path.join(directory, file)
        print(f"Verwerken van bestand: {file_path}")
        process_wave_file(file_path)

# Functie om een bestand te verwerken (hier pas je de berekeningen toe)
def process_wave_file(file_path):
    print(f"Bestand inlezen: {file_path}")
    # Probeer het bestand in te lezen (voeg een try-except toe voor fouten)
    try:
        eta = pd.read_csv(file_path, header=None).values.flatten()
    except Exception as e:
        print(f"Fout bij het inlezen van {file_path}: {e}")
        return

    # Parameters voor spectrale analyse
    fs = 8  # Samplefrequentie
    m = 15 * 60 * fs
    M = 128 * fs
    p = 13
    delta_f = fs / M
    f = np.array([i * delta_f for i in range(int(M / 2))])

    Neta = len(eta)
    Nwindows = Neta // m
    print(f"Aantal windows: {Nwindows}")

    # Voor elk tijdsvenster de spectrale parameters berekenen
    for i in range(Nwindows):
        eta_np_window = eta[i * m:(i + 1) * m]
        notnan = np.argwhere(~np.isnan(eta_np_window))
        eta_np_window_notnan = eta_np_window[notnan].flatten()

        if len(eta_np_window_notnan) == 0:
            continue  # Sla lege vensters over

        # Spectrale analyse
        S = np.zeros(int(M / 2))
        for q in range(0, len(eta_np_window_notnan) - M, M):
            etaseg = eta_np_window_notnan[q:q + M]
            etaseg_detrended = signal.detrend(etaseg)
            S_eta = fft(etaseg_detrended * np.hanning(M))
            S_k = (np.real(S_eta) ** 2 + np.imag(S_eta) ** 2) / delta_f
            S += S_k[:int(M / 2)]
        S /= p

        # Bereken golfparameters
        Hm0, Tm01, Tm02, Tm_10 = calculate_Sf_params(S, f, 0.05, 0.43, delta_f)
        print(f"Window {i+1}/{Nwindows}: Hm0={Hm0}, Tm01={Tm01}, Tm02={Tm02}, Tm_10={Tm_10}")

# Hoofdprogramma
if __name__ == "__main__":
    selected_directory = select_directory()
    if selected_directory:
        process_files_in_directory(selected_directory)
    else:
        print("Geen directory geselecteerd.")
