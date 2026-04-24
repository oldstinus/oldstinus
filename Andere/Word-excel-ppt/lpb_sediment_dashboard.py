from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from simulate_lpb_sediment_balance import (
    AstronomicalTideParameters,
    SedimentBalanceModel,
    StorageErosionParameters,
    WaterLevelForcingParameters,
    apply_astronomical_tide,
    apply_waterlevel_forcing,
    load_workbook,
    load_waterlevel_csv,
    make_animation,
    monthly_summary,
    piecewise_formula_lines,
    simulate_storage_erosion,
)


DEFAULT_INPUT = Path("omzetten oude files/lippenbroek/LPB_sediment_150506-250907_sedimentballans_in_uit_spring_doodtij.xls")


class SedimentDashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Lippenbroek sedimentbalans")
        self.geometry("1420x920")

        self.input_path = DEFAULT_INPUT
        self.data = load_workbook(self.input_path)
        self.model = SedimentBalanceModel.fit(self.data)
        self.default_storage = StorageErosionParameters.from_empirical_model(self.model)
        self.measured_level_path = Path("Lippenbroek GOG_Zeeschelde_Waterpeil.csv")
        self.forecast_level_path = Path("Driegoten tij_Zeeschelde_Voorspeld waterpeil getij.csv")

        self.mode_var = tk.StringVar(value="storage")
        self.forcing_var = tk.StringVar(value="measured_discharge")
        self.calibration_var = tk.StringVar(value="all_excels")
        self.q_in_scale = tk.DoubleVar(value=1.0)
        self.q_out_scale = tk.DoubleVar(value=1.0)
        self.ssc_in_scale = tk.DoubleVar(value=1.0)
        self.ssc_out_scale = tk.DoubleVar(value=1.0)
        self.capture_fraction = tk.DoubleVar(value=self.default_storage.capture_fraction)
        self.background_out_ssc = tk.DoubleVar(value=self.default_storage.background_out_ssc_mgL)
        self.erosion_coeff = tk.DoubleVar(value=self.default_storage.erosion_coeff_kg_per_step_per_m3s)
        self.spring_erosion = tk.DoubleVar(value=self.default_storage.spring_erosion_factor)
        self.initial_storage = tk.DoubleVar(value=0.0)
        self.tide_mean = tk.DoubleVar(value=0.0)
        self.tide_amplitude = tk.DoubleVar(value=1.2)
        self.tide_period = tk.DoubleVar(value=12.42)
        self.spring_neap_strength = tk.DoubleVar(value=0.35)
        self.spring_neap_period = tk.DoubleVar(value=14.77)
        self.tide_phase = tk.DoubleVar(value=0.0)
        self.spring_neap_phase = tk.DoubleVar(value=0.0)
        self.inflow_gain = tk.DoubleVar(value=1.8)
        self.outflow_gain = tk.DoubleVar(value=1.4)
        self.wl_inflow_gain = tk.DoubleVar(value=10.0)
        self.wl_outflow_gain = tk.DoubleVar(value=10.0)
        self.wl_resample = tk.DoubleVar(value=15.0)
        self.wl_offset = tk.DoubleVar(value=0.0)

        self.summary_var = tk.StringVar()
        self.formula_var = tk.StringVar()
        self.figure = plt.Figure(figsize=(12, 7), dpi=100)
        self.axes = self.figure.subplots(3, 1, sharex=False)

        self.measured_level_df, self.measured_level_meta = load_waterlevel_csv(self.measured_level_path)
        self.forecast_level_df, self.forecast_level_meta = load_waterlevel_csv(self.forecast_level_path)

        self._build_ui()
        self.run_simulation()

    def reload_model(self) -> None:
        try:
            include_extra = self.calibration_var.get() == "all_excels"
            self.model = SedimentBalanceModel.fit(self.data, include_extra_calibration=include_extra)
            self.default_storage = StorageErosionParameters.from_empirical_model(self.model)
            self.run_simulation()
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Kalibratie mislukt", str(exc))

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls_host = ttk.Frame(self, padding=8)
        controls_host.grid(row=0, column=0, sticky="nsw")
        controls_host.rowconfigure(0, weight=1)
        controls_host.columnconfigure(0, weight=1)

        controls_canvas = tk.Canvas(controls_host, width=340, highlightthickness=0)
        controls_scrollbar = ttk.Scrollbar(controls_host, orient="vertical", command=controls_canvas.yview)
        controls_hscrollbar = ttk.Scrollbar(controls_host, orient="horizontal", command=controls_canvas.xview)
        controls_canvas.configure(yscrollcommand=controls_scrollbar.set, xscrollcommand=controls_hscrollbar.set)
        controls_canvas.grid(row=0, column=0, sticky="nsw")
        controls_scrollbar.grid(row=0, column=1, sticky="ns")
        controls_hscrollbar.grid(row=1, column=0, sticky="ew")

        controls = ttk.Frame(controls_canvas, padding=12)
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor="nw")

        def _sync_scroll_region(_event=None) -> None:
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

        controls.bind("<Configure>", _sync_scroll_region)

        def _on_mousewheel(event) -> None:
            controls_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        controls_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        charts = ttk.Frame(self, padding=8)
        charts.grid(row=0, column=1, sticky="nsew")
        charts.columnconfigure(0, weight=1)
        charts.rowconfigure(0, weight=1)

        ttk.Label(controls, text="Bronbestand sediment").grid(row=0, column=0, sticky="w")
        ttk.Button(controls, text="Kies .xls", command=self.choose_file).grid(row=1, column=0, sticky="ew", pady=(2, 6))
        self.source_label = ttk.Label(controls, text=self.input_path.name, wraplength=300)
        self.source_label.grid(row=2, column=0, sticky="w", pady=(0, 10))

        ttk.Label(controls, text="Gemeten waterpeilbestand").grid(row=3, column=0, sticky="w")
        ttk.Button(controls, text="Kies gemeten waterpeil CSV", command=self.choose_measured_level_file).grid(row=4, column=0, sticky="ew", pady=(2, 6))
        self.measured_label = ttk.Label(controls, text=self.measured_level_path.name, wraplength=300)
        self.measured_label.grid(row=5, column=0, sticky="w", pady=(0, 10))

        ttk.Label(controls, text="Voorspeld waterpeilbestand").grid(row=6, column=0, sticky="w")
        ttk.Button(controls, text="Kies voorspeld waterpeil CSV", command=self.choose_forecast_level_file).grid(row=7, column=0, sticky="ew", pady=(2, 10))
        self.forecast_label = ttk.Label(controls, text=self.forecast_level_path.name, wraplength=300)
        self.forecast_label.grid(row=8, column=0, sticky="w", pady=(0, 10))

        ttk.Label(controls, text="Modeltype").grid(row=9, column=0, sticky="w")
        ttk.Radiobutton(controls, text="Baseline", variable=self.mode_var, value="baseline", command=self.run_simulation).grid(row=10, column=0, sticky="w")
        ttk.Radiobutton(controls, text="Opslag/Erosie", variable=self.mode_var, value="storage", command=self.run_simulation).grid(row=11, column=0, sticky="w")

        ttk.Label(controls, text="Kalibratie").grid(row=12, column=0, sticky="w", pady=(8, 0))
        ttk.Radiobutton(controls, text="Alle Excel-kalibraties", variable=self.calibration_var, value="all_excels", command=self.reload_model).grid(row=13, column=0, sticky="w")
        ttk.Radiobutton(controls, text="Alleen hoofdwerkboek", variable=self.calibration_var, value="main_only", command=self.reload_model).grid(row=14, column=0, sticky="w")

        ttk.Label(controls, text="Debietbron").grid(row=15, column=0, sticky="w", pady=(8, 0))
        ttk.Radiobutton(controls, text="Gemeten debiet", variable=self.forcing_var, value="measured_discharge", command=self.run_simulation).grid(row=16, column=0, sticky="w")
        ttk.Radiobutton(controls, text="Astronomisch getij", variable=self.forcing_var, value="astronomical", command=self.run_simulation).grid(row=17, column=0, sticky="w")
        ttk.Radiobutton(controls, text="Gemeten waterpeil", variable=self.forcing_var, value="measured_level", command=self.run_simulation).grid(row=18, column=0, sticky="w")
        ttk.Radiobutton(controls, text="Voorspeld waterpeil", variable=self.forcing_var, value="forecast_level", command=self.run_simulation).grid(row=19, column=0, sticky="w")

        slider_specs = [
            ("Qin schaal", self.q_in_scale, 0.5, 1.5),
            ("Qout schaal", self.q_out_scale, 0.5, 1.5),
            ("SSCin schaal", self.ssc_in_scale, 0.5, 1.5),
            ("SSCout schaal", self.ssc_out_scale, 0.5, 1.5),
            ("Capture", self.capture_fraction, 0.0, 1.0),
            ("Bg SSC uit", self.background_out_ssc, 0.0, 30.0),
            ("Erosie coeff", self.erosion_coeff, 0.0, 40.0),
            ("Spring erosie", self.spring_erosion, 0.0, 1.0),
            ("Beginopslag t", self.initial_storage, 0.0, 500.0),
            ("Gem. getij m", self.tide_mean, -2.0, 2.0),
            ("Amp. getij m", self.tide_amplitude, 0.1, 3.0),
            ("Periode u", self.tide_period, 10.0, 14.0),
            ("Spring-neap", self.spring_neap_strength, 0.0, 1.0),
            ("SN periode d", self.spring_neap_period, 10.0, 20.0),
            ("Getij fase u", self.tide_phase, 0.0, 24.0),
            ("SN fase d", self.spring_neap_phase, 0.0, 20.0),
            ("Inflow gain", self.inflow_gain, 0.1, 5.0),
            ("Outflow gain", self.outflow_gain, 0.1, 5.0),
            ("WL Qin gain", self.wl_inflow_gain, 0.1, 50.0),
            ("WL Qout gain", self.wl_outflow_gain, 0.1, 50.0),
            ("WL stap min", self.wl_resample, 5.0, 60.0),
            ("WL offset m", self.wl_offset, -2.0, 2.0),
        ]

        row = 20
        for label, variable, min_value, max_value in slider_specs:
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", pady=(8 if row == 5 else 6, 0))
            scale = tk.Scale(
                controls,
                from_=min_value,
                to=max_value,
                resolution=0.01,
                orient="horizontal",
                variable=variable,
                command=lambda _value: self.run_simulation(),
                length=210,
            )
            scale.grid(row=row + 1, column=0, sticky="ew")
            row += 2

        ttk.Button(controls, text="Exporteer scenario CSV", command=self.export_csv).grid(row=row, column=0, sticky="ew", pady=(12, 4))
        ttk.Button(controls, text="Exporteer animatie GIF", command=self.export_animation).grid(row=row + 1, column=0, sticky="ew", pady=(4, 4))
        ttk.Button(controls, text="Reset defaults", command=self.reset_defaults).grid(row=row + 2, column=0, sticky="ew")

        ttk.Label(controls, textvariable=self.formula_var, justify="left", anchor="w", foreground="#0b3c6f").grid(row=row + 3, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(controls, textvariable=self.summary_var, justify="left", anchor="w").grid(row=row + 4, column=0, sticky="ew", pady=(12, 0))

        canvas = FigureCanvasTkAgg(self.figure, master=charts)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas = canvas

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Kies sediment .xls-bestand",
            filetypes=[("Excel 97-2003", "*.xls"), ("Alle bestanden", "*.*")],
            initialdir=str(self.input_path.parent),
        )
        if not path:
            return
        try:
            self.input_path = Path(path)
            self.data = load_workbook(self.input_path)
            self.source_label.configure(text=self.input_path.name)
            self.reload_model()
            self.reset_defaults()
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Bestand kon niet geladen worden", str(exc))

    def choose_measured_level_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Kies gemeten waterpeil CSV",
            filetypes=[("CSV", "*.csv"), ("Alle bestanden", "*.*")],
            initialdir=str(self.measured_level_path.parent),
        )
        if not path:
            return
        try:
            self.measured_level_path = Path(path)
            self.measured_level_df, self.measured_level_meta = load_waterlevel_csv(self.measured_level_path)
            self.measured_label.configure(text=self.measured_level_path.name)
            self.run_simulation()
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("CSV kon niet geladen worden", str(exc))

    def choose_forecast_level_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Kies voorspeld waterpeil CSV",
            filetypes=[("CSV", "*.csv"), ("Alle bestanden", "*.*")],
            initialdir=str(self.forecast_level_path.parent),
        )
        if not path:
            return
        try:
            self.forecast_level_path = Path(path)
            self.forecast_level_df, self.forecast_level_meta = load_waterlevel_csv(self.forecast_level_path)
            self.forecast_label.configure(text=self.forecast_level_path.name)
            self.run_simulation()
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("CSV kon niet geladen worden", str(exc))

    def reset_defaults(self) -> None:
        self.mode_var.set("storage")
        self.forcing_var.set("measured_discharge")
        self.calibration_var.set("all_excels")
        self.q_in_scale.set(1.0)
        self.q_out_scale.set(1.0)
        self.ssc_in_scale.set(1.0)
        self.ssc_out_scale.set(1.0)
        self.capture_fraction.set(self.default_storage.capture_fraction)
        self.background_out_ssc.set(self.default_storage.background_out_ssc_mgL)
        self.erosion_coeff.set(self.default_storage.erosion_coeff_kg_per_step_per_m3s)
        self.spring_erosion.set(self.default_storage.spring_erosion_factor)
        self.initial_storage.set(0.0)
        self.tide_mean.set(0.0)
        self.tide_amplitude.set(1.2)
        self.tide_period.set(12.42)
        self.spring_neap_strength.set(0.35)
        self.spring_neap_period.set(14.77)
        self.tide_phase.set(0.0)
        self.spring_neap_phase.set(0.0)
        self.inflow_gain.set(1.8)
        self.outflow_gain.set(1.4)
        self.wl_inflow_gain.set(10.0)
        self.wl_outflow_gain.set(10.0)
        self.wl_resample.set(15.0)
        self.wl_offset.set(0.0)
        self.reload_model()

    def current_simulation(self):
        base_data = self.data
        forcing = self.forcing_var.get()
        if forcing == "astronomical":
            tide_params = AstronomicalTideParameters(
                mean_level_m=self.tide_mean.get(),
                semidiurnal_amplitude_m=self.tide_amplitude.get(),
                semidiurnal_period_hours=self.tide_period.get(),
                spring_neap_strength=self.spring_neap_strength.get(),
                spring_neap_period_days=self.spring_neap_period.get(),
                phase_hours=self.tide_phase.get(),
                spring_neap_phase_days=self.spring_neap_phase.get(),
                inflow_gain_m3s_per_m=self.inflow_gain.get(),
                outflow_gain_m3s_per_m=self.outflow_gain.get(),
            )
            base_data = apply_astronomical_tide(self.data, tide_params)
        elif forcing in {"measured_level", "forecast_level"}:
            level_df = self.measured_level_df if forcing == "measured_level" else self.forecast_level_df
            base_data = apply_waterlevel_forcing(
                level_df,
                self.data,
                WaterLevelForcingParameters(
                    inflow_gain_m3s_per_m_per_h=self.wl_inflow_gain.get(),
                    outflow_gain_m3s_per_m_per_h=self.wl_outflow_gain.get(),
                    resample_minutes=int(self.wl_resample.get()),
                    level_offset_m=self.wl_offset.get(),
                ),
            )

        if self.mode_var.get() == "baseline":
            sim = self.model.simulate(
                base_data,
                q_in_scale=self.q_in_scale.get(),
                q_out_scale=self.q_out_scale.get(),
                ssc_in_scale=self.ssc_in_scale.get(),
                ssc_out_scale=self.ssc_out_scale.get(),
            )
        else:
            params = StorageErosionParameters(
                capture_fraction=self.capture_fraction.get(),
                background_out_ssc_mgL=self.background_out_ssc.get(),
                erosion_coeff_kg_per_step_per_m3s=self.erosion_coeff.get(),
                spring_erosion_factor=self.spring_erosion.get(),
                initial_storage_kg=self.initial_storage.get() * 1000,
            )
            sim = simulate_storage_erosion(
                base_data,
                self.model,
                params,
                q_in_scale=self.q_in_scale.get(),
                q_out_scale=self.q_out_scale.get(),
                ssc_in_scale=self.ssc_in_scale.get(),
            )
        return sim

    def run_simulation(self) -> None:
        try:
            sim = self.current_simulation()
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Simulatie mislukt", str(exc))
            return

        monthly = monthly_summary(sim)
        for ax in self.axes:
            ax.clear()

        if "water_level_m" in sim.columns:
            self.axes[0].plot(sim["time"], sim["water_level_m"], color="#1f77b4", label="Waterpeil")
            self.axes[0].set_ylabel("m TAW")
            self.axes[0].set_title("Waterpeil forcing")
            ax0b = self.axes[0].twinx()
            ax0b.plot(sim["time"], sim["q_in_m3s_sim"], color="#2ca02c", alpha=0.7, label="Q_in")
            ax0b.plot(sim["time"], sim["q_out_m3s_sim"], color="#d62728", alpha=0.7, label="Q_out")
            ax0b.set_ylabel("m3/s")
        else:
            self.axes[0].plot(sim["time"], sim["q_in_m3s_sim"], color="#2ca02c", label="Q_in")
            self.axes[0].plot(sim["time"], sim["q_out_m3s_sim"], color="#d62728", label="Q_out")
            self.axes[0].set_ylabel("m3/s")
            self.axes[0].set_title("Debiet forcing")
        self.axes[0].legend(loc="upper left")

        self.axes[1].plot(sim["time"], sim["sed_in_kg_15m_sim"] / 1000, color="#1f77b4", label="In")
        self.axes[1].plot(sim["time"], sim["sed_out_kg_15m_sim"] / 1000, color="#ff7f0e", label="Uit")
        self.axes[1].set_ylabel("ton / stap")
        self.axes[1].set_title("Gesimuleerde in- en uitstroom")
        self.axes[1].legend(loc="upper right")

        self.axes[2].plot(sim["time"], sim["cum_balance_kg_sim"] / 1000, color="#2ca02c", label="Netto cumulatief")
        if "storage_kg_sim" in sim.columns:
            self.axes[2].plot(sim["time"], sim["storage_kg_sim"] / 1000, color="#6a3d9a", label="Opslag")
        self.axes[2].set_ylabel("ton")
        self.axes[2].set_title("Cumulatieve balans en opslag")
        self.axes[2].legend(loc="upper left")

        self.figure.tight_layout()
        self.canvas.draw_idle()

        forcing = self.forcing_var.get()
        if forcing in {"measured_level", "forecast_level"}:
            source_name = "Lippenbroek gemeten waterpeil" if forcing == "measured_level" else "Driegoten voorspeld astronomisch getij"
            formula = (
                f"Bron: {source_name}\n"
                f"Q_in = max({self.wl_inflow_gain.get():.2f} * dH/dt, 0)\n"
                f"Q_out = max({self.wl_outflow_gain.get():.2f} * (-dH/dt), 0)\n"
                f"SSC_in = {self.model.ssc_in_intercept:.3f} + {self.model.ssc_in_slope:.3f} * Q_in\n"
            )
        elif forcing == "astronomical":
            formula = (
                "H(t) = H0 + A * (1 + S * sin(2pi t/Tsn)) * sin(2pi t/T)\n"
                f"Q_in = max(v(t) * {self.inflow_gain.get():.2f}, 0)\n"
                f"Q_out = max(-v(t) * {self.outflow_gain.get():.2f}, 0)\n"
                f"SSC_in = {self.model.ssc_in_intercept:.3f} + {self.model.ssc_in_slope:.3f} * Q_in\n"
            )
        else:
            formula = (
                "Gebaseerd op Excel-debieten\n"
                f"SSC_in = {self.model.ssc_in_intercept:.3f} + {self.model.ssc_in_slope:.3f} * Q_in\n"
            )
        piecewise = "\n".join(piecewise_formula_lines(self.model))
        if piecewise:
            formula += "SSC_out per ebfase:\n" + piecewise
        else:
            formula += f"SSC_out = {self.model.ssc_out_constant:.3f} mg/L\n"
        if self.model.calibration_sources:
            formula += "\nExtra kalibratie:\n" + "\n".join(self.model.calibration_sources)
        else:
            formula += "\nExtra kalibratie: geen"
        self.formula_var.set(formula)

        total_in_t = sim["sed_in_kg_15m_sim"].sum() / 1000
        total_out_t = sim["sed_out_kg_15m_sim"].sum() / 1000
        net_t = sim["sed_balance_kg_15m_sim"].sum() / 1000
        summary_lines = [
            f"Bestand: {self.input_path.name}",
            f"Periode: {sim['time'].min().date()} t/m {sim['time'].max().date()}",
            f"Model: {self.mode_var.get()}",
            f"Forcing: {forcing}",
            f"Kalibratie: {'alle excelfiles' if self.calibration_var.get() == 'all_excels' else 'alleen hoofdwerkboek'}",
            f"In: {total_in_t:,.1f} ton".replace(",", " "),
            f"Uit: {total_out_t:,.1f} ton".replace(",", " "),
            f"Netto: {net_t:,.1f} ton".replace(",", " "),
        ]
        if self.model.calibration_sources:
            summary_lines.append(f"Extra IN punten: {self.model.extra_in_count}")
            summary_lines.append(f"Extra UIT punten: {self.model.extra_out_count}")
        if "storage_kg_sim" in sim.columns:
            summary_lines.append(f"Eindopslag: {sim['storage_kg_sim'].iloc[-1] / 1000:,.1f} ton".replace(",", " "))
            summary_lines.append(f"Gem. SSC uit: {sim['ssc_out_mgL_sim'].mean():.1f} mg/L")
        else:
            summary_lines.append(f"SSC uit: {self.model.ssc_out_constant * self.ssc_out_scale.get():.1f} mg/L")
        if forcing == "astronomical":
            summary_lines.append(f"Getij amp.: {self.tide_amplitude.get():.2f} m")
            summary_lines.append(f"Spring-neap: {self.spring_neap_strength.get():.2f}")
        elif forcing in {"measured_level", "forecast_level"}:
            summary_lines.append(f"dH/dt inflow gain: {self.wl_inflow_gain.get():.2f}")
            summary_lines.append(f"dH/dt outflow gain: {self.wl_outflow_gain.get():.2f}")
        self.summary_var.set("\n".join(summary_lines))
        self.latest_sim = sim

    def export_csv(self) -> None:
        if not hasattr(self, "latest_sim"):
            return
        path = filedialog.asksaveasfilename(
            title="Sla scenario op",
            defaultextension=".csv",
            initialfile="lpb_sediment_scenario.csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        self.latest_sim.to_csv(path, index=False)
        messagebox.showinfo("Export voltooid", f"Scenario opgeslagen als:\n{path}")

    def export_animation(self) -> None:
        if not hasattr(self, "latest_sim"):
            return
        path = filedialog.asksaveasfilename(
            title="Sla animatie op",
            defaultextension=".gif",
            initialfile="lpb_sediment_animatie.gif",
            filetypes=[("GIF", "*.gif")],
        )
        if not path:
            return
        make_animation(self.latest_sim, Path(path), title="Lippenbroek sedimentbalans")
        messagebox.showinfo("Export voltooid", f"Animatie opgeslagen als:\n{path}")


def main() -> None:
    app = SedimentDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
