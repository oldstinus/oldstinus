from __future__ import annotations

import argparse
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import scipy.io as sio


def load_mat(path: Path):
    return sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)


def _bearing_from_en(east: np.ndarray, north: np.ndarray) -> np.ndarray:
    ang = np.degrees(np.arctan2(east, north))
    return (ang + 360.0) % 360.0


def _circular_mean_deg(values_deg: np.ndarray, weights: np.ndarray | None = None) -> float:
    vals = np.asarray(values_deg, dtype=float).reshape(-1)
    valid = np.isfinite(vals)
    if weights is None:
        w = np.ones_like(vals, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        valid &= np.isfinite(w)
    if not np.any(valid):
        return float("nan")
    vals = vals[valid]
    w = w[valid]
    s = np.sum(np.sin(np.radians(vals)) * w)
    c = np.sum(np.cos(np.radians(vals)) * w)
    return float((np.degrees(np.arctan2(s, c)) + 360.0) % 360.0)


def _angle_diff_deg(a: float, b: float) -> float:
    return float(((a - b + 180.0) % 360.0) - 180.0)


def _rotate_en(east: np.ndarray, north: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    ang = np.radians(angle_deg)
    ca = np.cos(ang)
    sa = np.sin(ang)
    e2 = east * ca - north * sa
    n2 = east * sa + north * ca
    return e2, n2


def parse_kmz_line(path: Path) -> tuple[tuple[float, float], tuple[float, float]]:
    with zipfile.ZipFile(path) as zf:
        with zf.open("doc.kml") as fh:
            root = ET.parse(fh).getroot()
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    coord_text = root.findtext(".//kml:LineString/kml:coordinates", namespaces=ns)
    if not coord_text:
        raise ValueError("Geen LineString coordinates gevonden in KMZ.")
    parts = [p for p in coord_text.strip().split() if p]
    if len(parts) < 2:
        raise ValueError("Te weinig coördinaten in KMZ.")
    lon1, lat1, *_ = map(float, parts[0].split(","))
    lon2, lat2, *_ = map(float, parts[1].split(","))
    return (lon1, lat1), (lon2, lat2)


def geodesic_bearing_deg(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    lon1, lat1 = p1
    lon2, lat2 = p2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    lam1, lam2 = math.radians(lon1), math.radians(lon2)
    dlam = lam2 - lam1
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return float((math.degrees(math.atan2(x, y)) + 360.0) % 360.0)


def track_axis_bearing(track_en: np.ndarray) -> float:
    xy = np.asarray(track_en, dtype=float)
    valid = np.isfinite(xy).all(axis=1)
    xy = xy[valid]
    if xy.shape[0] < 2:
        return float("nan")
    xy = xy - np.mean(xy, axis=0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    vec = vt[0]
    return float(_bearing_from_en(np.array([vec[0]]), np.array([vec[1]]))[0])


def choose_best_reference(measured_bearing: float, candidates: list[float]) -> tuple[float, float]:
    best = min(candidates, key=lambda cand: abs(_angle_diff_deg(cand, measured_bearing)))
    delta = _angle_diff_deg(best, measured_bearing)
    return best, delta


def project_on_bearing(east: np.ndarray, north: np.ndarray, bearing_deg: float) -> np.ndarray:
    ang = np.radians(bearing_deg)
    unit_e = np.sin(ang)
    unit_n = np.cos(ang)
    return np.asarray(east, dtype=float) * unit_e + np.asarray(north, dtype=float) * unit_n


def signed_runs(values: np.ndarray, zero_threshold: float = 0.01) -> list[dict[str, int | float | str]]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    state = np.zeros(arr.shape[0], dtype=int)
    state[arr > zero_threshold] = 1
    state[arr < -zero_threshold] = -1

    runs: list[dict[str, int | float | str]] = []
    if state.size == 0:
        return runs

    start = 0
    for i in range(1, state.size):
        if state[i] != state[start]:
            segment = arr[start:i]
            runs.append(
                {
                    "start": start,
                    "end": i - 1,
                    "sign": int(state[start]),
                    "label": "positive" if state[start] > 0 else ("negative" if state[start] < 0 else "near_zero"),
                    "length": i - start,
                    "mean": float(np.nanmean(segment)),
                    "min": float(np.nanmin(segment)),
                    "max": float(np.nanmax(segment)),
                }
            )
            start = i

    segment = arr[start:]
    runs.append(
        {
            "start": start,
            "end": state.size - 1,
            "sign": int(state[start]),
            "label": "positive" if state[start] > 0 else ("negative" if state[start] < 0 else "near_zero"),
            "length": state.size - start,
            "mean": float(np.nanmean(segment)),
            "min": float(np.nanmin(segment)),
            "max": float(np.nanmax(segment)),
        }
    )
    return runs


def summarize_runs(runs: list[dict[str, int | float | str]], min_length: int = 5) -> list[dict[str, int | float | str]]:
    return [r for r in runs if int(r["length"]) >= min_length and int(r["sign"]) != 0]


def cluster_indices(indices: np.ndarray, max_gap: int = 2) -> list[tuple[int, int]]:
    idx = np.asarray(indices, dtype=int).reshape(-1)
    if idx.size == 0:
        return []
    clusters: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for val in idx[1:]:
        val = int(val)
        if val - prev <= max_gap:
            prev = val
            continue
        clusters.append((start, prev))
        start = prev = val
    clusters.append((start, prev))
    return clusters


def segments_between_clusters(length: int, clusters: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    if not clusters:
        return [(0, length - 1)]
    segs: list[tuple[int, int]] = []
    start = 0
    for a, b in clusters:
        if start <= a - 1:
            segs.append((start, a - 1))
        start = b + 1
    if start <= length - 1:
        segs.append((start, length - 1))
    return segs


def analyse_file(mat_path: Path, kmz_path: Path) -> None:
    p1, p2 = parse_kmz_line(kmz_path)
    raai_bearing = geodesic_bearing_deg(p1, p2)
    raai_bearings = [raai_bearing, (raai_bearing + 180.0) % 360.0]
    stroom_bearings = [(raai_bearing + 90.0) % 360.0, (raai_bearing + 270.0) % 360.0]

    mat = load_mat(mat_path)
    setup = mat["Setup"]
    system = mat["System"]
    summary = mat["Summary"]
    compass = mat["Compass"]

    step = np.asarray(system.Step).astype(int).reshape(-1)
    mask = step == 3 if np.any(step == 3) else np.ones_like(step, dtype=bool)

    track = np.asarray(summary.Track, dtype=float)[mask, :2]
    boat = np.asarray(summary.Boat_Vel, dtype=float)[mask, :2]
    water = np.asarray(summary.Mean_Vel, dtype=float)[mask, :2]
    heading = np.asarray(system.Heading, dtype=float)[mask]
    true_heading = np.asarray(system.True_North_ADP_Heading, dtype=float)[mask]
    mag_error = np.asarray(compass.Magnetic_error, dtype=float)[mask]

    boat_speed = np.linalg.norm(boat, axis=1)
    water_speed = np.linalg.norm(water, axis=1)
    boat_bearing = _bearing_from_en(boat[:, 0], boat[:, 1])
    water_bearing = _bearing_from_en(water[:, 0], water[:, 1])

    measured_track_axis = track_axis_bearing(track)
    measured_boat_course = _circular_mean_deg(boat_bearing, weights=np.where(boat_speed > 0, boat_speed, 0.0))

    ref_raai, delta_raai = choose_best_reference(measured_boat_course, raai_bearings)
    ref_stroom, delta_stroom = choose_best_reference(measured_boat_course, stroom_bearings)

    boat_e_raai, boat_n_raai = _rotate_en(boat[:, 0], boat[:, 1], delta_raai)
    water_e_raai, water_n_raai = _rotate_en(water[:, 0], water[:, 1], delta_raai)

    mean_water_bearing_rot = _circular_mean_deg(
        _bearing_from_en(water_e_raai, water_n_raai),
        weights=np.where(water_speed > 0, water_speed, 0.0),
    )
    water_normal_to_raai = project_on_bearing(water_e_raai, water_n_raai, (ref_raai + 90.0) % 360.0)
    water_along_raai = project_on_bearing(water_e_raai, water_n_raai, ref_raai)
    boat_along_raai = project_on_bearing(boat_e_raai, boat_n_raai, ref_raai)
    water_normal_true = project_on_bearing(water[:, 0], water[:, 1], (raai_bearing + 90.0) % 360.0)
    track_rel = track - track[0]
    track_along_true = project_on_bearing(track_rel[:, 0], track_rel[:, 1], raai_bearing)
    track_normal_true = project_on_bearing(track_rel[:, 0], track_rel[:, 1], (raai_bearing + 90.0) % 360.0)
    if track.shape[0] >= 2:
        inc = np.diff(track, axis=0)
        inc_along_true = project_on_bearing(inc[:, 0], inc[:, 1], raai_bearing)
        inc_normal_true = project_on_bearing(inc[:, 0], inc[:, 1], (raai_bearing + 90.0) % 360.0)
        inc_abs = np.linalg.norm(inc, axis=1)
    else:
        inc_along_true = np.array([], dtype=float)
        inc_normal_true = np.array([], dtype=float)
        inc_abs = np.array([], dtype=float)

    # Rotatie-/flip-detectie: alleen indicatief. Een pure yaw-rotatie is niet zichtbaar in BT alleen.
    low_bt = boat_speed < 0.03
    sign = np.sign(water_normal_to_raai)
    sign[np.abs(water_normal_to_raai) < 0.01] = 0.0
    heading_jump = np.abs(np.array([_angle_diff_deg(b, a) for a, b in zip(true_heading[:-1], true_heading[1:])]))
    sign_change = sign[1:] * sign[:-1] < 0
    flip_idx = np.where(sign_change & low_bt[1:] & low_bt[:-1] & (heading_jump > 120.0))[0] + 1

    print(f"Bestand: {mat_path}")
    print(f"Raai uit KMZ: {kmz_path}")
    print(f"Raai-azimut true north: {raai_bearing:.3f}° / {(raai_bearing + 180.0) % 360.0:.3f}°")
    print(f"Verwachte stroomrichtingen loodrecht op raai: {stroom_bearings[0]:.3f}° / {stroom_bearings[1]:.3f}°")
    print()
    print("Instellingen uit MAT:")
    print(f"  coordinateSystem={getattr(setup, 'coordinateSystem', None)}  trackReference={getattr(setup, 'trackReference', None)}")
    print(f"  magneticDeclination={getattr(setup, 'magneticDeclination', None)}  headingCorrection={getattr(setup, 'headingCorrection', None)}")
    print()
    print("Gemeten oriëntatie:")
    print(f"  gemiddelde boat-course uit ENU={measured_boat_course:.3f}°")
    print(f"  hoofd-as van track={measured_track_axis:.3f}° / {(measured_track_axis + 180.0) % 360.0:.3f}°")
    print(f"  gemiddelde waterrichting uit ENU={_circular_mean_deg(water_bearing, weights=np.where(water_speed > 0, water_speed, 0.0)):.3f}°")
    print()
    print("Optie A: rotatie zodat bootbeweging op de raai valt")
    print(f"  gekozen raai-richting={ref_raai:.3f}°")
    print(f"  toe te passen rotatie={delta_raai:.3f}°")
    print(f"  gemiddelde waterrichting na rotatie={mean_water_bearing_rot:.3f}°")
    print(f"  gemiddelde watercomponent langs raai={np.nanmean(water_along_raai):.4f} m/s")
    print(f"  gemiddelde watercomponent loodrecht op raai={np.nanmean(water_normal_to_raai):.4f} m/s")
    print(f"  gemiddelde boat-component langs raai={np.nanmean(boat_along_raai):.4f} m/s")
    print()
    print("Alternatief ter controle: als boat-course eigenlijk stroomas volgt")
    print(f"  dichtste stroomrichting={ref_stroom:.3f}°")
    print(f"  vereiste rotatie naar stroomas={delta_stroom:.3f}°")
    print()
    print("Kompas / flip-indicatie:")
    print(f"  magnetic_error mean={np.nanmean(mag_error):.3f}%  max={np.nanmax(mag_error):.3f}%")
    print(f"  True_North_ADP_Heading == Heading: {bool(np.allclose(true_heading, heading, equal_nan=True))}")
    if flip_idx.size:
        print(f"  mogelijke rotatie/flip-samples: {flip_idx.tolist()}")
    else:
        print("  geen harde flip gedetecteerd met de huidige criteriumcombinatie")
    print("  opmerking: een pure draaiing ter plaatse is niet betrouwbaar uit bottom-track alleen af te leiden.")
    print()

    if inc_abs.size:
        event_idx = np.where((np.abs(inc_normal_true) > 0.20) | (inc_abs > 0.25))[0] + 1
        events = cluster_indices(event_idx, max_gap=2)
        print("Bottom-track verplaatsing t.o.v. raai:")
        print(f"  cumulatieve verplaatsing langs raai: {track_along_true[0]:+.3f} -> {track_along_true[-1]:+.3f} m")
        print(f"  cumulatieve verplaatsing loodrecht op raai: {track_normal_true[0]:+.3f} -> {track_normal_true[-1]:+.3f} m")
        print(f"  extremen loodrecht op raai: min {np.nanmin(track_normal_true):+.3f} m, max {np.nanmax(track_normal_true):+.3f} m")
        if events:
            print("  grote verplaatsingsclusters:")
            for a, b in events:
                s = slice(max(0, a - 1), min(track_normal_true.size, b + 1))
                dn = float(track_normal_true[min(track_normal_true.size - 1, b)] - track_normal_true[max(0, a - 1)])
                da = float(track_along_true[min(track_along_true.size - 1, b)] - track_along_true[max(0, a - 1)])
                print(
                    f"    {a:03d}-{b:03d}  "
                    f"d_normal={dn:+.3f} m  d_along={da:+.3f} m  "
                    f"max_step={float(np.nanmax(inc_abs[a-1:b])):.3f} m"
                )
        else:
            print("  geen grote verplaatsingsclusters boven de ingestelde drempel")
        print()
    else:
        events = []

    runs = signed_runs(water_normal_true, zero_threshold=0.01)
    pos_runs = [r for r in runs if r["sign"] > 0]
    neg_runs = [r for r in runs if r["sign"] < 0]
    zero_runs = [r for r in runs if r["sign"] == 0]

    main_runs = summarize_runs(runs, min_length=5)
    pos_main = [r for r in main_runs if r["sign"] > 0]
    neg_main = [r for r in main_runs if r["sign"] < 0]

    print("Projectie op raai-normaal:")
    print(f"  normaalrichting op basis van KMZ={(raai_bearing + 90.0) % 360.0:.3f}°")
    print(f"  gemiddelde geprojecteerde snelheid={np.nanmean(water_normal_true):.4f} m/s")
    print(f"  min/max geprojecteerde snelheid={np.nanmin(water_normal_true):.4f} / {np.nanmax(water_normal_true):.4f} m/s")
    print(f"  blokken: positief={len(pos_runs)}  negatief={len(neg_runs)}  near_zero={len(zero_runs)}")
    print(f"  betekenisvolle blokken (n >= 5): positief={len(pos_main)}  negatief={len(neg_main)}")
    print()

    print("Positieve blokken op raai-normaal (n >= 5):")
    if pos_main:
        for r in pos_main:
            print(
                f"  {r['start']:03d}-{r['end']:03d}  n={r['length']:3d}  "
                f"mean={r['mean']:+.4f}  range=[{r['min']:+.4f}, {r['max']:+.4f}] m/s"
            )
    else:
        print("  geen")
    print()

    print("Negatieve blokken op raai-normaal (n >= 5):")
    if neg_main:
        for r in neg_main:
            print(
                f"  {r['start']:03d}-{r['end']:03d}  n={r['length']:3d}  "
                f"mean={r['mean']:+.4f}  range=[{r['min']:+.4f}, {r['max']:+.4f}] m/s"
            )
    else:
        print("  geen")

    print()
    segs = segments_between_clusters(len(water_normal_true), events)
    print("Segmenten tussen verplaatsingsclusters:")
    candidate_segments: list[tuple[int, int, float, float, float, float]] = []
    for i, (a, b) in enumerate(segs, start=1):
        seg_vn = water_normal_true[a:b + 1]
        seg_va = project_on_bearing(water[a:b + 1, 0], water[a:b + 1, 1], raai_bearing)
        seg_bt = boat_speed[a:b + 1]
        seg_norm_disp = track_normal_true[b] - track_normal_true[a]
        seg_along_disp = track_along_true[b] - track_along_true[a]
        seg_mean = float(np.nanmean(seg_vn))
        seg_std = float(np.nanstd(seg_vn))
        seg_bt_mean = float(np.nanmean(seg_bt))
        print(
            f"  S{i}: {a:03d}-{b:03d}  n={b-a+1:3d}  "
            f"v_norm_mean={seg_mean:+.4f}  v_norm_std={seg_std:.4f}  "
            f"v_along_mean={float(np.nanmean(seg_va)):+.4f}  "
            f"bt_mean={seg_bt_mean:.4f}  "
            f"d_norm={seg_norm_disp:+.3f} m  d_along={seg_along_disp:+.3f} m"
        )
        if (b - a + 1) >= 20 and (abs(seg_mean) >= 0.05 or seg_std <= 0.08):
            candidate_segments.append((i, a, b, seg_mean, seg_std, seg_bt_mean))

    print()
    print("Voorstel apart behandelen:")
    if candidate_segments:
        for i, a, b, seg_mean, seg_std, seg_bt_mean in candidate_segments:
            reasons = []
            if abs(seg_mean) >= 0.10:
                reasons.append("duidelijke gemiddelde dwarscomponent")
            elif abs(seg_mean) >= 0.05:
                reasons.append("matige maar consistente dwarscomponent")
            if seg_std <= 0.08:
                reasons.append("relatief stabiel intern signaal")
            if seg_bt_mean <= 0.08:
                reasons.append("beperkte gemiddelde bodemverplaatsing")
            reason_txt = ", ".join(reasons) if reasons else "segment valt op in vergelijking met de rest"
            print(f"  S{i} ({a:03d}-{b:03d}): {reason_txt}")
    else:
        print("  geen duidelijke kandidaatsegmenten volgens de huidige drempels")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyseer M9 MAT t.o.v. een raai in KMZ.")
    parser.add_argument("mat_path", type=Path)
    parser.add_argument("kmz_path", type=Path)
    args = parser.parse_args()
    analyse_file(args.mat_path, args.kmz_path)


if __name__ == "__main__":
    main()
