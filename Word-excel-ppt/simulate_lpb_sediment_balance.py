from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SECONDS_PER_STEP = 15 * 60
KG_PER_MG_PER_L_PER_M3 = 0.001
STEP_FACTOR = SECONDS_PER_STEP * KG_PER_MG_PER_L_PER_M3
PHASE_BINS = [-0.001, 0.25, 0.5, 0.75, 1.0]
PHASE_LABELS = ["early", "mid1", "mid2", "late"]


@dataclass
class SedimentBalanceModel:
    ssc_in_intercept: float
    ssc_in_slope: float
    ssc_out_constant: float
    ssc_out_piecewise: dict[str, tuple[float, float]]
    calibration_sources: list[str]
    extra_in_count: int = 0
    extra_out_count: int = 0

    @classmethod
    def fit(cls, data: pd.DataFrame, include_extra_calibration: bool = True) -> "SedimentBalanceModel":
        base_dir = Path("omzetten oude files/lippenbroek")
        extras = (
            load_additional_calibration_data(base_dir)
            if include_extra_calibration and base_dir.exists()
            else {"in_q_ssc": pd.DataFrame(), "out_q_ssc": pd.DataFrame(), "sources": []}
        )

        inflow = data.loc[(data["q_in_m3s"] > 0) & data["ssc_in_mgL"].notna(), ["time", "q_in_m3s", "ssc_in_mgL"]]
        if not extras["in_q_ssc"].empty:
            inflow = pd.concat([inflow, extras["in_q_ssc"]], ignore_index=True)
        inflow = inflow.dropna(subset=["q_in_m3s", "ssc_in_mgL"])
        slope, intercept = np.polyfit(inflow["q_in_m3s"], inflow["ssc_in_mgL"], 1)

        outflow = data.loc[(data["q_out_m3s"] > 0) & (data["sed_out_kg_15m_obs"] > 0), ["q_out_m3s", "sed_out_kg_15m_obs"]]
        ssc_out = (outflow["sed_out_kg_15m_obs"] / (STEP_FACTOR * outflow["q_out_m3s"])).mean()
        piecewise = fit_piecewise_outflow_power_law(data, extras["out_q_ssc"])
        return cls(
            ssc_in_intercept=float(intercept),
            ssc_in_slope=float(slope),
            ssc_out_constant=float(ssc_out),
            ssc_out_piecewise=piecewise,
            calibration_sources=extras.get("sources", []),
            extra_in_count=int(len(extras["in_q_ssc"])),
            extra_out_count=int(len(extras["out_q_ssc"])),
        )

    def predict_out_ssc(self, q_out_m3s: pd.Series, ebb_phase: pd.Series | None = None) -> pd.Series:
        if not self.ssc_out_piecewise:
            return pd.Series(np.where(q_out_m3s > 0, self.ssc_out_constant, 0.0), index=q_out_m3s.index)

        phase = ebb_phase.fillna(0.5) if ebb_phase is not None else pd.Series(0.5, index=q_out_m3s.index)
        phase_bin = pd.cut(phase, bins=PHASE_BINS, labels=PHASE_LABELS)
        out = pd.Series(0.0, index=q_out_m3s.index, dtype=float)
        for label in PHASE_LABELS:
            mask = (phase_bin == label) & (q_out_m3s > 0)
            if not mask.any():
                continue
            a, b = self.ssc_out_piecewise[label]
            out.loc[mask] = a * np.power(np.maximum(q_out_m3s.loc[mask], 1e-6), b)
        return out.clip(lower=0.0)

    def simulate(
        self,
        data: pd.DataFrame,
        q_in_scale: float = 1.0,
        q_out_scale: float = 1.0,
        ssc_in_scale: float = 1.0,
        ssc_out_scale: float = 1.0,
    ) -> pd.DataFrame:
        sim = data.copy()
        sim["q_in_m3s_sim"] = sim["q_in_m3s"] * q_in_scale
        sim["q_out_m3s_sim"] = sim["q_out_m3s"] * q_out_scale
        sim["ssc_in_mgL_sim"] = np.where(
            sim["q_in_m3s_sim"] > 0,
            (self.ssc_in_intercept + self.ssc_in_slope * sim["q_in_m3s_sim"]) * ssc_in_scale,
            0.0,
        )
        sim["ssc_out_mgL_sim"] = self.predict_out_ssc(sim["q_out_m3s_sim"], sim.get("ebb_phase")).to_numpy() * ssc_out_scale
        sim["sed_in_kg_15m_sim"] = STEP_FACTOR * sim["q_in_m3s_sim"] * sim["ssc_in_mgL_sim"]
        sim["sed_out_kg_15m_sim"] = STEP_FACTOR * sim["q_out_m3s_sim"] * sim["ssc_out_mgL_sim"]
        sim["sed_balance_kg_15m_sim"] = sim["sed_in_kg_15m_sim"] - sim["sed_out_kg_15m_sim"]
        sim["cum_balance_kg_sim"] = sim["sed_balance_kg_15m_sim"].cumsum()
        return sim


@dataclass
class StorageErosionParameters:
    capture_fraction: float
    background_out_ssc_mgL: float
    erosion_coeff_kg_per_step_per_m3s: float
    spring_erosion_factor: float
    initial_storage_kg: float = 0.0

    @classmethod
    def from_empirical_model(cls, model: SedimentBalanceModel) -> "StorageErosionParameters":
        total_out_capacity = STEP_FACTOR * model.ssc_out_constant
        return cls(
            capture_fraction=0.88,
            background_out_ssc_mgL=7.0,
            erosion_coeff_kg_per_step_per_m3s=max(total_out_capacity - STEP_FACTOR * 7.0, 0.0),
            spring_erosion_factor=0.15,
            initial_storage_kg=0.0,
        )


@dataclass
class AstronomicalTideParameters:
    mean_level_m: float = 0.0
    semidiurnal_amplitude_m: float = 1.2
    semidiurnal_period_hours: float = 12.42
    spring_neap_strength: float = 0.35
    spring_neap_period_days: float = 14.77
    phase_hours: float = 0.0
    spring_neap_phase_days: float = 0.0
    inflow_gain_m3s_per_m: float = 1.8
    outflow_gain_m3s_per_m: float = 1.4


@dataclass
class WaterLevelForcingParameters:
    inflow_gain_m3s_per_m_per_h: float = 10.0
    outflow_gain_m3s_per_m_per_h: float = 10.0
    resample_minutes: int = 15
    level_offset_m: float = 0.0


def annotate_tide_phase(data: pd.DataFrame) -> pd.DataFrame:
    phased = data.copy()
    phased["ebb_active"] = (phased["q_out_m3s"] > 0).astype(int)
    phased["ebb_event_id"] = ((phased["ebb_active"].diff().fillna(0) == 1)).cumsum()
    phased.loc[phased["ebb_active"] == 0, "ebb_event_id"] = np.nan
    phased["ebb_event_len"] = phased.groupby("ebb_event_id")["q_out_m3s"].transform("size")
    phased["ebb_event_idx"] = phased.groupby("ebb_event_id").cumcount()
    phased["ebb_phase"] = phased["ebb_event_idx"] / phased["ebb_event_len"].replace(0, np.nan)

    phased["flood_active"] = (phased["q_in_m3s"] > 0).astype(int)
    phased["flood_event_id"] = ((phased["flood_active"].diff().fillna(0) == 1)).cumsum()
    phased.loc[phased["flood_active"] == 0, "flood_event_id"] = np.nan
    phased["flood_event_len"] = phased.groupby("flood_event_id")["q_in_m3s"].transform("size")
    phased["flood_event_idx"] = phased.groupby("flood_event_id").cumcount()
    phased["flood_phase"] = phased["flood_event_idx"] / phased["flood_event_len"].replace(0, np.nan)
    return phased


def fit_piecewise_outflow_power_law(data: pd.DataFrame, extra_out: pd.DataFrame | None = None) -> dict[str, tuple[float, float]]:
    observed = data.loc[(data["q_out_m3s"] > 0) & data["ssc_out_obs_mgL"].notna(), ["time", "q_out_m3s", "ssc_out_obs_mgL", "ebb_phase"]].copy()
    if extra_out is not None and not extra_out.empty:
        extra = extra_out.copy().drop(columns=["ebb_phase"], errors="ignore")
        extra["time"] = pd.to_datetime(extra["time"], errors="coerce")
        phased = pd.merge_asof(
            extra.sort_values("time"),
            data[["time", "ebb_phase"]].sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=pd.Timedelta("30min"),
        )
        phased = phased.rename(columns={"ssc_out_mgL": "ssc_out_obs_mgL"})
        observed = pd.concat(
            [observed, phased[["time", "q_out_m3s", "ssc_out_obs_mgL", "ebb_phase"]]],
            ignore_index=True,
        )
    observed = observed.dropna(subset=["q_out_m3s", "ssc_out_obs_mgL"])
    if observed.empty:
        return {}

    observed["phase_bin"] = pd.cut(observed["ebb_phase"].fillna(0.5), bins=PHASE_BINS, labels=PHASE_LABELS)
    coeffs: dict[str, tuple[float, float]] = {}
    global_x = observed["q_out_m3s"].to_numpy()
    global_y = observed["ssc_out_obs_mgL"].to_numpy()
    global_b, global_a = np.polyfit(np.log(np.maximum(global_x, 1e-6)), np.log(np.maximum(global_y, 1e-6)), 1)
    global_coeff = (float(np.exp(global_a)), float(global_b))

    for label in PHASE_LABELS:
        grp = observed.loc[observed["phase_bin"] == label]
        if len(grp) < 20:
            coeffs[label] = global_coeff
            continue
        x = grp["q_out_m3s"].to_numpy()
        y = grp["ssc_out_obs_mgL"].to_numpy()
        b, a = np.polyfit(np.log(np.maximum(x, 1e-6)), np.log(np.maximum(y, 1e-6)), 1)
        coeffs[label] = (float(np.exp(a)), float(b))
    return coeffs


def load_additional_calibration_data(base_dir: Path) -> dict[str, pd.DataFrame]:
    in_frames: list[pd.DataFrame] = []
    out_frames: list[pd.DataFrame] = []
    sources: list[str] = []

    in_path = base_dir / "LPB_sediment_13uur_IN_kalibratie_OBS.xls"
    if in_path.exists():
        try:
            df = pd.read_excel(in_path, sheet_name="sscgegevens")
            cols = [str(c) for c in df.columns]
            time_col = next((c for c in cols if "Tijd" in c), cols[0])
            ssc_col = next((c for c in cols if "concentratie" in c.lower()), cols[1])
            q_col = next((c for c in cols if "debiet" in c.lower()), cols[2] if len(cols) > 2 else cols[1])
            extra_in = pd.DataFrame(
                {
                    "time": pd.to_datetime(df[time_col], errors="coerce"),
                    "q_in_m3s": pd.to_numeric(df[q_col], errors="coerce"),
                    "ssc_in_mgL": pd.to_numeric(df[ssc_col], errors="coerce"),
                }
            ).dropna()
            in_frames.append(extra_in)
            sources.append(in_path.name)
        except Exception:
            pass

    long_path = base_dir / "LPB_sediment_langdurige metingen analyse IN&UIT 2007-turbiditeit.xls"
    if long_path.exists():
        try:
            q_df = pd.read_excel(long_path, sheet_name="QINQUIT", header=2)
            q_df = q_df.rename(columns={q_df.columns[0]: "time", q_df.columns[1]: "q_in_m3s", q_df.columns[4]: "q_out_m3s"})
            q_df["time"] = pd.to_datetime(q_df["time"], errors="coerce")
            q_df["q_in_m3s"] = pd.to_numeric(q_df["q_in_m3s"], errors="coerce")
            q_df["q_out_m3s"] = pd.to_numeric(q_df["q_out_m3s"], errors="coerce")
            q_df = q_df.dropna(subset=["time"]).sort_values("time")

            in_samples = pd.read_excel(long_path, sheet_name="INLAATTOTAAL", header=2)
            in_samples["sample_time"] = combine_excel_date_time(
                in_samples["Datum"],
                in_samples["Startuur staalname (om de 30min)"],
            )
            in_samples["ssc_in_mgL"] = pd.to_numeric(in_samples["concentratie (mg/l)"], errors="coerce")
            in_samples = in_samples.dropna(subset=["sample_time", "ssc_in_mgL"]).sort_values("sample_time")
            in_join = pd.merge_asof(
                in_samples[["sample_time", "ssc_in_mgL"]].rename(columns={"sample_time": "time"}),
                q_df[["time", "q_in_m3s"]],
                on="time",
                direction="nearest",
                tolerance=pd.Timedelta("30min"),
            ).dropna()
            in_frames.append(in_join)
            sources.append(long_path.name + " [INLAATTOTAAL/QINQUIT]")

            out_samples = pd.read_excel(long_path, sheet_name="uitlaatverwerking 1", header=None)
            out_samples = out_samples.iloc[1:].copy()
            out_extra = pd.DataFrame(
                {
                    "time": pd.to_datetime(out_samples.iloc[:, 0], errors="coerce"),
                    "ssc_out_mgL": pd.to_numeric(out_samples.iloc[:, 11], errors="coerce"),
                    "q_out_m3s": pd.to_numeric(out_samples.iloc[:, 18], errors="coerce"),
                }
            ).dropna()
            out_frames.append(out_extra)
            sources.append(long_path.name + " [uitlaatverwerking 1]")
        except Exception:
            pass

    result = {
        "in_q_ssc": pd.concat(in_frames, ignore_index=True) if in_frames else pd.DataFrame(columns=["time", "q_in_m3s", "ssc_in_mgL"]),
        "out_q_ssc": pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame(columns=["time", "q_out_m3s", "ssc_out_mgL"]),
        "sources": sorted(set(sources)),
    }
    for key in ["in_q_ssc", "out_q_ssc"]:
        result[key] = result[key].drop_duplicates().reset_index(drop=True)
    return result


def combine_excel_date_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    date_part = pd.to_datetime(date_series, errors="coerce").dt.strftime("%Y-%m-%d")
    time_part = pd.to_datetime(time_series.astype(str), format="%H:%M:%S", errors="coerce").dt.strftime("%H:%M:%S")
    combined = pd.to_datetime(date_part + " " + time_part, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return combined


def simulate_storage_erosion(
    data: pd.DataFrame,
    empirical_model: SedimentBalanceModel,
    params: StorageErosionParameters,
    q_in_scale: float = 1.0,
    q_out_scale: float = 1.0,
    ssc_in_scale: float = 1.0,
) -> pd.DataFrame:
    sim = data.copy()
    sim["q_in_m3s_sim"] = sim["q_in_m3s"] * q_in_scale
    sim["q_out_m3s_sim"] = sim["q_out_m3s"] * q_out_scale
    phase_driver = pd.DataFrame({"q_in_m3s": sim["q_in_m3s_sim"], "q_out_m3s": sim["q_out_m3s_sim"]})
    phase_driver = annotate_tide_phase(phase_driver)
    sim["ebb_phase"] = phase_driver["ebb_phase"]
    sim["flood_phase"] = phase_driver["flood_phase"]
    sim["ssc_in_mgL_sim"] = np.where(
        sim["q_in_m3s_sim"] > 0,
        (empirical_model.ssc_in_intercept + empirical_model.ssc_in_slope * sim["q_in_m3s_sim"]) * ssc_in_scale,
        0.0,
    )
    sim["sed_in_kg_15m_sim"] = STEP_FACTOR * sim["q_in_m3s_sim"] * sim["ssc_in_mgL_sim"]

    storage = params.initial_storage_kg
    sed_out = []
    direct_out = []
    erosion_out = []
    storage_series = []
    ssc_out_series = []

    for row in sim.itertuples(index=False):
        incoming = float(row.sed_in_kg_15m_sim)
        q_out = float(row.q_out_m3s_sim)
        spring_flag = 1.0 if getattr(row, "tide_type", "unknown") == "spring" else 0.0
        phase = getattr(row, "ebb_phase", np.nan)

        obs_like_ssc = empirical_model.predict_out_ssc(pd.Series([q_out]), pd.Series([phase])).iloc[0] if q_out > 0 else 0.0
        direct_flux = incoming * (1.0 - params.capture_fraction) + (STEP_FACTOR * q_out * params.background_out_ssc_mgL)
        erosion_capacity = params.erosion_coeff_kg_per_step_per_m3s * q_out * (1.0 + params.spring_erosion_factor * spring_flag)
        available = storage + incoming
        target_flux = STEP_FACTOR * q_out * obs_like_ssc
        outgoing = min(max(direct_flux + erosion_capacity, target_flux, 0.0), available)

        erosion_component = max(outgoing - direct_flux, 0.0)
        storage = max(available - outgoing, 0.0)
        out_ssc = (outgoing / (STEP_FACTOR * q_out)) if q_out > 0 else 0.0

        sed_out.append(outgoing)
        direct_out.append(min(direct_flux, outgoing))
        erosion_out.append(erosion_component)
        storage_series.append(storage)
        ssc_out_series.append(out_ssc)

    sim["sed_out_kg_15m_sim"] = sed_out
    sim["direct_out_kg_15m_sim"] = direct_out
    sim["erosion_out_kg_15m_sim"] = erosion_out
    sim["storage_kg_sim"] = storage_series
    sim["ssc_out_mgL_sim"] = ssc_out_series
    sim["sed_balance_kg_15m_sim"] = sim["sed_in_kg_15m_sim"] - sim["sed_out_kg_15m_sim"]
    sim["cum_balance_kg_sim"] = sim["sed_balance_kg_15m_sim"].cumsum()
    return sim


def apply_astronomical_tide(data: pd.DataFrame, tide_params: AstronomicalTideParameters) -> pd.DataFrame:
    forced = data.copy()
    elapsed_hours = (forced["time"] - forced["time"].iloc[0]).dt.total_seconds() / 3600.0
    phase = 2.0 * np.pi * (elapsed_hours - tide_params.phase_hours) / tide_params.semidiurnal_period_hours
    spring_phase = (
        2.0 * np.pi * ((elapsed_hours / 24.0) - tide_params.spring_neap_phase_days) / tide_params.spring_neap_period_days
    )

    envelope = 1.0 + tide_params.spring_neap_strength * np.sin(spring_phase)
    tide_level = tide_params.mean_level_m + tide_params.semidiurnal_amplitude_m * envelope * np.sin(phase)
    tide_velocity = (
        tide_params.semidiurnal_amplitude_m
        * envelope
        * (2.0 * np.pi / tide_params.semidiurnal_period_hours)
        * np.cos(phase)
    )

    forced["astronomical_tide_m"] = tide_level
    forced["astronomical_velocity_mph"] = tide_velocity
    forced["q_in_m3s"] = np.clip(tide_velocity, 0.0, None) * tide_params.inflow_gain_m3s_per_m
    forced["q_out_m3s"] = np.clip(-tide_velocity, 0.0, None) * tide_params.outflow_gain_m3s_per_m
    forced["tide_type"] = np.where(np.sin(spring_phase) >= 0.0, "spring", "neap")
    forced["vol_in_m3_15m_obs"] = forced["q_in_m3s"] * SECONDS_PER_STEP
    forced["ssc_in_mgL"] = np.nan
    forced["sed_in_kg_15m_obs"] = 0.0
    forced["sed_out_kg_15m_obs"] = 0.0
    forced["sed_balance_kg_15m_obs"] = 0.0
    forced["cum_balance_kg_obs"] = 0.0
    forced["ssc_out_obs_mgL"] = np.nan
    return annotate_tide_phase(forced)


def load_waterlevel_csv(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("#"):
                continue
            parts = line.split(";", 1)
            if len(parts) == 2 and parts[0] != "#Timestamp":
                metadata[parts[0].lstrip("#")] = parts[1]
            if line.startswith("#Timestamp;"):
                break

    df = pd.read_csv(path, sep=";", skiprows=8)
    df["time"] = pd.to_datetime(df["#Timestamp"], errors="coerce", utc=True).dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
    df["water_level_m"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.loc[df["time"].notna() & df["water_level_m"].notna(), ["time", "water_level_m"]].copy()
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return df, metadata


def apply_waterlevel_forcing(
    level_df: pd.DataFrame,
    reference_data: pd.DataFrame,
    params: WaterLevelForcingParameters,
) -> pd.DataFrame:
    series = level_df.copy()
    series["water_level_m"] = series["water_level_m"] + params.level_offset_m
    regular = (
        series.set_index("time")
        .resample(f"{params.resample_minutes}min")
        .interpolate(method="time")
        .reset_index()
    )
    dt_hours = regular["time"].diff().dt.total_seconds().div(3600).fillna(params.resample_minutes / 60.0)
    dh = regular["water_level_m"].diff().fillna(0.0)
    dh_dt = dh.div(dt_hours.replace(0.0, np.nan)).fillna(0.0)

    forced = pd.DataFrame({"time": regular["time"]})
    forced["water_level_m"] = regular["water_level_m"]
    forced["dwater_level_dt_m_per_h"] = dh_dt
    forced["q_in_m3s"] = np.clip(dh_dt, 0.0, None) * params.inflow_gain_m3s_per_m_per_h
    forced["q_out_m3s"] = np.clip(-dh_dt, 0.0, None) * params.outflow_gain_m3s_per_m_per_h
    forced["vol_in_m3_15m_obs"] = forced["q_in_m3s"] * params.resample_minutes * 60
    rolling = forced["water_level_m"].rolling(4 * 24 * 2, min_periods=1).mean()
    forced["tide_type"] = np.where(forced["water_level_m"] >= rolling, "spring", "neap")
    forced["ssc_in_mgL"] = np.nan
    forced["sed_in_kg_15m_obs"] = 0.0
    forced["sed_out_kg_15m_obs"] = 0.0
    forced["sed_balance_kg_15m_obs"] = 0.0
    forced["cum_balance_kg_obs"] = 0.0
    forced["ssc_out_obs_mgL"] = np.nan
    return annotate_tide_phase(forced)


def load_workbook(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="debiet-ssc", header=None)
    tide = pd.read_excel(path, sheet_name="spring doodtij", header=None, names=["timestamp", "tide_type", "marker"])

    data = pd.DataFrame(
        {
            "time": pd.to_datetime(raw.iloc[2:, 0], errors="coerce"),
            "q_in_m3s": pd.to_numeric(raw.iloc[2:, 1], errors="coerce").fillna(0.0),
            "vol_in_m3_15m_obs": pd.to_numeric(raw.iloc[2:, 2], errors="coerce").fillna(0.0),
            "ssc_in_mgL": pd.to_numeric(raw.iloc[2:, 3], errors="coerce"),
            "sed_in_kg_15m_obs": pd.to_numeric(raw.iloc[2:, 5], errors="coerce").fillna(0.0),
            "q_out_m3s": pd.to_numeric(raw.iloc[2:, 13], errors="coerce").fillna(0.0),
            "ssc_out_obs_mgL": pd.to_numeric(raw.iloc[2:, 11], errors="coerce"),
            "sed_out_kg_15m_obs": pd.to_numeric(raw.iloc[2:, 15], errors="coerce").fillna(0.0),
        }
    )

    data = data.loc[data["time"].notna()].copy()
    data = data.sort_values("time").reset_index(drop=True)
    data["sed_balance_kg_15m_obs"] = data["sed_in_kg_15m_obs"] - data["sed_out_kg_15m_obs"]
    data["cum_balance_kg_obs"] = data["sed_balance_kg_15m_obs"].cumsum()

    tide = tide.dropna(subset=["timestamp"]).copy()
    tide["timestamp"] = pd.to_datetime(tide["timestamp"], errors="coerce")
    tide["tide_type"] = (
        tide["tide_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"springtij": "spring", "doodtij": "neap"})
    )
    tide = tide.dropna(subset=["timestamp"]).sort_values("timestamp")

    data = pd.merge_asof(
        data,
        tide[["timestamp", "tide_type"]],
        left_on="time",
        right_on="timestamp",
        direction="backward",
    )
    data["tide_type"] = data["tide_type"].fillna("unknown")
    return annotate_tide_phase(data)


def monthly_summary(sim: pd.DataFrame) -> pd.DataFrame:
    monthly = sim.copy()
    monthly["month"] = monthly["time"].dt.to_period("M").dt.to_timestamp()
    return (
        monthly.groupby("month", as_index=False)[
            [
                "sed_in_kg_15m_obs",
                "sed_out_kg_15m_obs",
                "sed_balance_kg_15m_obs",
                "sed_in_kg_15m_sim",
                "sed_out_kg_15m_sim",
                "sed_balance_kg_15m_sim",
            ]
        ]
        .sum()
    )


def piecewise_formula_lines(model: SedimentBalanceModel) -> list[str]:
    lines = []
    for label in PHASE_LABELS:
        if label not in model.ssc_out_piecewise:
            continue
        a, b = model.ssc_out_piecewise[label]
        lines.append(f"{label}: SSC_out = {a:.3f} * Q_out^{b:.3f}")
    return lines


def tide_summary(sim: pd.DataFrame) -> pd.DataFrame:
    return (
        sim.groupby("tide_type", as_index=False)[
            ["sed_in_kg_15m_sim", "sed_out_kg_15m_sim", "sed_balance_kg_15m_sim"]
        ]
        .sum()
        .sort_values("tide_type")
    )


def make_plots(sim: pd.DataFrame, monthly: pd.DataFrame, tide: pd.DataFrame, model: SedimentBalanceModel, out_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(sim["time"], sim["sed_in_kg_15m_obs"], label="In gemeten", color="#1f77b4", alpha=0.65)
    axes[0].plot(sim["time"], sim["sed_in_kg_15m_sim"], label="In gesimuleerd", color="#0b3c6f", linewidth=1.0)
    axes[0].set_ylabel("kg / 15 min")
    axes[0].set_title("Instromend sediment")
    axes[0].legend()

    axes[1].plot(sim["time"], sim["sed_out_kg_15m_obs"], label="Uit gemeten", color="#ff7f0e", alpha=0.65)
    axes[1].plot(sim["time"], sim["sed_out_kg_15m_sim"], label="Uit gesimuleerd", color="#8c3b00", linewidth=1.0)
    axes[1].set_ylabel("kg / 15 min")
    axes[1].set_title("Uitstromend sediment")
    axes[1].legend()

    axes[2].plot(sim["time"], sim["cum_balance_kg_obs"] / 1000, label="Cumulatief gemeten", color="#2ca02c", alpha=0.65)
    axes[2].plot(sim["time"], sim["cum_balance_kg_sim"] / 1000, label="Cumulatief gesimuleerd", color="#145a32", linewidth=1.1)
    axes[2].set_ylabel("ton")
    axes[2].set_title("Cumulatieve sedimentbalans")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "sediment_tijdreeks.png", dpi=180)
    plt.close(fig)

    if "storage_kg_sim" in sim.columns:
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(sim["time"], sim["storage_kg_sim"] / 1000, color="#6a3d9a", linewidth=1.2)
        ax.set_title("Gesimuleerde opslag in het systeem")
        ax.set_ylabel("ton")
        fig.tight_layout()
        fig.savefig(out_dir / "sediment_opslag.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 5))
    width = 10
    ax.bar(monthly["month"] - pd.Timedelta(days=width), monthly["sed_in_kg_15m_sim"] / 1000, width=20, label="In", color="#1f77b4")
    ax.bar(monthly["month"] + pd.Timedelta(days=width), monthly["sed_out_kg_15m_sim"] / 1000, width=20, label="Uit", color="#ff7f0e")
    ax.plot(monthly["month"], monthly["sed_balance_kg_15m_sim"] / 1000, color="#2ca02c", label="Netto", linewidth=2)
    ax.set_title("Maandelijkse gesimuleerde sedimentbalans")
    ax.set_ylabel("ton / maand")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "sediment_maandbalans.png", dpi=180)
    plt.close(fig)

    inflow = sim.loc[(sim["q_in_m3s"] > 0) & sim["ssc_in_mgL"].notna(), ["q_in_m3s", "ssc_in_mgL"]].copy()
    if len(inflow) > 4000:
        inflow = inflow.sample(4000, random_state=42)
    q_line = np.linspace(0, sim["q_in_m3s"].max(), 200)
    ssc_line = model.ssc_in_intercept + model.ssc_in_slope * q_line

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(inflow["q_in_m3s"], inflow["ssc_in_mgL"], s=8, alpha=0.25, color="#1f77b4", label="Metingen")
    ax.plot(q_line, ssc_line, color="#d62728", linewidth=2, label="Lineaire fit")
    ax.set_xlabel("Q_in (m3/s)")
    ax.set_ylabel("SSC_in (mg/L)")
    ax.set_title("Afgeleide relatie voor instromende sedimentconcentratie")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ssc_regressie.png", dpi=180)
    plt.close(fig)

    known_tide = tide.loc[tide["tide_type"] != "unknown"].copy()
    if not known_tide.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(known_tide))
        ax.bar(x - 0.2, known_tide["sed_in_kg_15m_sim"] / 1000, width=0.2, label="In", color="#1f77b4")
        ax.bar(x, known_tide["sed_out_kg_15m_sim"] / 1000, width=0.2, label="Uit", color="#ff7f0e")
        ax.bar(x + 0.2, known_tide["sed_balance_kg_15m_sim"] / 1000, width=0.2, label="Netto", color="#2ca02c")
        ax.set_xticks(x)
        ax.set_xticklabels(known_tide["tide_type"])
        ax.set_ylabel("ton")
        ax.set_title("Gesimuleerde balans per getijtype")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "sediment_getijtype.png", dpi=180)
        plt.close(fig)


def make_animation(sim: pd.DataFrame, out_path: Path, title: str = "Sedimentbalans") -> None:
    if sim.empty:
        return
    step = max(len(sim) // 120, 1)
    indices = list(range(0, len(sim), step))
    if indices[-1] != len(sim) - 1:
        indices.append(len(sim) - 1)

    temp_dir = out_path.parent / "_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []

    for frame_no, idx in enumerate(indices):
        subset = sim.iloc[: idx + 1]
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        if "water_level_m" in sim.columns:
            axes[0].plot(sim["time"], sim["water_level_m"], color="#1f77b4", alpha=0.2)
            axes[0].plot(subset["time"], subset["water_level_m"], color="#1f77b4", linewidth=2)
            axes[0].scatter(subset["time"].iloc[-1], subset["water_level_m"].iloc[-1], color="#d62728", s=40)
            axes[0].set_ylabel("m TAW")
            axes[0].set_title(f"{title}: waterpeil")
        else:
            axes[0].plot(sim["time"], sim["q_in_m3s_sim"], color="#1f77b4", alpha=0.2)
            axes[0].plot(subset["time"], subset["q_in_m3s_sim"], color="#1f77b4", linewidth=2)
            axes[0].set_ylabel("m3/s")
            axes[0].set_title(f"{title}: instroomdebiet")

        axes[1].plot(sim["time"], sim["sed_in_kg_15m_sim"] / 1000, color="#1f77b4", alpha=0.2, label="In")
        axes[1].plot(subset["time"], subset["sed_in_kg_15m_sim"] / 1000, color="#1f77b4", linewidth=2)
        axes[1].plot(sim["time"], sim["sed_out_kg_15m_sim"] / 1000, color="#ff7f0e", alpha=0.2, label="Uit")
        axes[1].plot(subset["time"], subset["sed_out_kg_15m_sim"] / 1000, color="#ff7f0e", linewidth=2)
        axes[1].set_ylabel("ton/step")
        axes[1].set_title("Sediment in en uit")
        axes[1].legend(loc="upper right")

        axes[2].plot(sim["time"], sim["cum_balance_kg_sim"] / 1000, color="#2ca02c", alpha=0.2, label="Netto")
        axes[2].plot(subset["time"], subset["cum_balance_kg_sim"] / 1000, color="#2ca02c", linewidth=2)
        if "storage_kg_sim" in sim.columns:
            axes[2].plot(sim["time"], sim["storage_kg_sim"] / 1000, color="#6a3d9a", alpha=0.2, label="Opslag")
            axes[2].plot(subset["time"], subset["storage_kg_sim"] / 1000, color="#6a3d9a", linewidth=2)
        axes[2].set_ylabel("ton")
        axes[2].set_title("Cumulatieve balans")
        axes[2].legend(loc="upper left")

        for ax in axes:
            ax.set_xlim(sim["time"].iloc[0], sim["time"].iloc[-1])
        fig.tight_layout()
        frame_path = temp_dir / f"frame_{frame_no:04d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frames.append(imageio.imread(frame_path))

    imageio.mimsave(out_path, frames, duration=0.08)

    for frame_path in temp_dir.glob("frame_*.png"):
        frame_path.unlink()
    temp_dir.rmdir()


def summary_dict(sim: pd.DataFrame, model: SedimentBalanceModel) -> dict[str, float | str]:
    summary = {
        "tijdstappen": int(len(sim)),
        "start": sim["time"].min().isoformat(),
        "einde": sim["time"].max().isoformat(),
        "ssc_in_formule_mgL": f"{model.ssc_in_intercept:.6f} + {model.ssc_in_slope:.6f} * Q_in",
        "totale_in_kg": round(float(sim["sed_in_kg_15m_sim"].sum()), 3),
        "totale_uit_kg": round(float(sim["sed_out_kg_15m_sim"].sum()), 3),
        "netto_balans_kg": round(float(sim["sed_balance_kg_15m_sim"].sum()), 3),
        "netto_balans_ton": round(float(sim["sed_balance_kg_15m_sim"].sum() / 1000), 3),
    }
    if "storage_kg_sim" in sim.columns:
        summary["eindopslag_kg"] = round(float(sim["storage_kg_sim"].iloc[-1]), 3)
        summary["gemiddelde_ssc_out_mgL"] = round(float(sim["ssc_out_mgL_sim"].replace([np.inf, -np.inf], np.nan).fillna(0).mean()), 3)
    else:
        summary["ssc_out_constant_mgL"] = round(model.ssc_out_constant, 6)
    summary["ssc_out_piecewise_power_law"] = piecewise_formula_lines(model)
    summary["extra_calibration_sources"] = model.calibration_sources
    summary["extra_in_calibration_points"] = model.extra_in_count
    summary["extra_out_calibration_points"] = model.extra_out_count
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Simuleer sedimentbalans uit de Lippenbroek .xls-meetfile.")
    parser.add_argument(
        "--input",
        default="omzetten oude files/lippenbroek/LPB_sediment_150506-250907_sedimentballans_in_uit_spring_doodtij.xls",
        help="Pad naar de bron .xls-file.",
    )
    parser.add_argument("--output-dir", default="output/spreadsheet/lpb_sediment_model", help="Map voor CSV/JSON/PNG-output.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "storage", "astronomical-baseline", "astronomical-storage"],
        default="baseline",
        help="Kies tussen gemeten debieten en astronomisch gegenereerde debieten.",
    )
    parser.add_argument("--q-in-scale", type=float, default=1.0, help="Schaalfactor op instromend debiet.")
    parser.add_argument("--q-out-scale", type=float, default=1.0, help="Schaalfactor op uitstromend debiet.")
    parser.add_argument("--ssc-in-scale", type=float, default=1.0, help="Schaalfactor op afgeleide instroom-SSC.")
    parser.add_argument("--ssc-out-scale", type=float, default=1.0, help="Schaalfactor op vaste uitstroom-SSC.")
    parser.add_argument("--capture-fraction", type=float, default=0.88, help="Fractie van inkomend sediment die eerst wordt opgeslagen.")
    parser.add_argument("--background-out-ssc", type=float, default=7.0, help="Achtergrondconcentratie in de uitstroom (mg/L).")
    parser.add_argument("--erosion-coeff", type=float, default=None, help="Erosiecoefficient in kg/15 min per (m3/s).")
    parser.add_argument("--spring-erosion-factor", type=float, default=0.15, help="Extra erosie tijdens springtij.")
    parser.add_argument("--initial-storage-kg", type=float, default=0.0, help="Beginschatting van opgeslagen sediment.")
    parser.add_argument("--waterlevel-source", choices=["none", "measured", "forecast"], default="none", help="Gebruik een waterpeilreeks als forcing.")
    parser.add_argument("--waterlevel-file", default=None, help="Optioneel pad naar waterpeil-CSV.")
    parser.add_argument("--waterlevel-inflow-gain", type=float, default=10.0, help="Gain van stijgend waterpeil naar instroomdebiet.")
    parser.add_argument("--waterlevel-outflow-gain", type=float, default=10.0, help="Gain van dalend waterpeil naar uitstroomdebiet.")
    parser.add_argument("--waterlevel-resample-minutes", type=int, default=15, help="Resample-stap voor waterpeil forcing.")
    parser.add_argument("--waterlevel-offset", type=float, default=0.0, help="Offset op waterpeil in meter.")
    parser.add_argument("--tide-mean-level", type=float, default=0.0, help="Gemiddeld astronomisch getijniveau (m).")
    parser.add_argument("--tide-amplitude", type=float, default=1.2, help="Semidiurnale getij-amplitude (m).")
    parser.add_argument("--tide-period-hours", type=float, default=12.42, help="Semidiurnale periode (uur).")
    parser.add_argument("--spring-neap-strength", type=float, default=0.35, help="Sterkte van spring-neap modulatie.")
    parser.add_argument("--spring-neap-period-days", type=float, default=14.77, help="Periode van spring-neap cyclus (dagen).")
    parser.add_argument("--tide-phase-hours", type=float, default=0.0, help="Faseverschuiving van de semidiurnale golf (uur).")
    parser.add_argument("--spring-neap-phase-days", type=float, default=0.0, help="Faseverschuiving van spring-neap modulatie (dagen).")
    parser.add_argument("--inflow-gain", type=float, default=1.8, help="Omzetting van getij-velocity naar instroomdebiet.")
    parser.add_argument("--outflow-gain", type=float, default=1.4, help="Omzetting van getij-velocity naar uitstroomdebiet.")
    parser.add_argument("--make-animation", action="store_true", help="Maak een gekoppelde GIF-animatie.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_workbook(input_path)
    model = SedimentBalanceModel.fit(data)

    if args.waterlevel_source != "none":
        default_files = {
            "measured": Path("Lippenbroek GOG_Zeeschelde_Waterpeil.csv"),
            "forecast": Path("Driegoten tij_Zeeschelde_Voorspeld waterpeil getij.csv"),
        }
        wl_path = Path(args.waterlevel_file) if args.waterlevel_file else default_files[args.waterlevel_source]
        level_df, metadata = load_waterlevel_csv(wl_path)
        simulation_data = apply_waterlevel_forcing(
            level_df,
            data,
            WaterLevelForcingParameters(
                inflow_gain_m3s_per_m_per_h=args.waterlevel_inflow_gain,
                outflow_gain_m3s_per_m_per_h=args.waterlevel_outflow_gain,
                resample_minutes=args.waterlevel_resample_minutes,
                level_offset_m=args.waterlevel_offset,
            ),
        )
    elif args.mode.startswith("astronomical"):
        tide_params = AstronomicalTideParameters(
            mean_level_m=args.tide_mean_level,
            semidiurnal_amplitude_m=args.tide_amplitude,
            semidiurnal_period_hours=args.tide_period_hours,
            spring_neap_strength=args.spring_neap_strength,
            spring_neap_period_days=args.spring_neap_period_days,
            phase_hours=args.tide_phase_hours,
            spring_neap_phase_days=args.spring_neap_phase_days,
            inflow_gain_m3s_per_m=args.inflow_gain,
            outflow_gain_m3s_per_m=args.outflow_gain,
        )
        simulation_data = apply_astronomical_tide(data, tide_params)
    else:
        simulation_data = data

    if args.mode in {"baseline", "astronomical-baseline"}:
        sim = model.simulate(
            simulation_data,
            q_in_scale=args.q_in_scale,
            q_out_scale=args.q_out_scale,
            ssc_in_scale=args.ssc_in_scale,
            ssc_out_scale=args.ssc_out_scale,
        )
    else:
        default_storage = StorageErosionParameters.from_empirical_model(model)
        storage_params = StorageErosionParameters(
            capture_fraction=args.capture_fraction,
            background_out_ssc_mgL=args.background_out_ssc,
            erosion_coeff_kg_per_step_per_m3s=(
                args.erosion_coeff
                if args.erosion_coeff is not None
                else default_storage.erosion_coeff_kg_per_step_per_m3s
            ),
            spring_erosion_factor=args.spring_erosion_factor,
            initial_storage_kg=args.initial_storage_kg,
        )
        sim = simulate_storage_erosion(
            simulation_data,
            model,
            storage_params,
            q_in_scale=args.q_in_scale,
            q_out_scale=args.q_out_scale,
            ssc_in_scale=args.ssc_in_scale,
        )
    monthly = monthly_summary(sim)
    tide = tide_summary(sim)

    sim.to_csv(out_dir / "sediment_tijdreeks.csv", index=False)
    monthly.to_csv(out_dir / "sediment_maandbalans.csv", index=False)
    tide.to_csv(out_dir / "sediment_getijtype.csv", index=False)

    summary = summary_dict(sim, model)
    summary["mode"] = args.mode
    summary["waterlevel_source"] = args.waterlevel_source
    if args.waterlevel_source != "none":
        summary["waterlevel_file"] = str(wl_path)
        summary["waterlevel_formula"] = (
            f"Q_in=max({args.waterlevel_inflow_gain:.3f}*dH/dt,0), "
            f"Q_out=max({args.waterlevel_outflow_gain:.3f}*(-dH/dt),0)"
        )
    with (out_dir / "samenvatting.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    make_plots(sim, monthly, tide, model, out_dir)
    if args.make_animation:
        animation_title = "Sedimentbalans"
        if args.waterlevel_source != "none":
            animation_title = f"Sedimentbalans op waterpeil ({args.waterlevel_source})"
        make_animation(sim, out_dir / "sediment_animatie.gif", title=animation_title)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
