#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
KMZ -> Shapefile (pure Python)
- Geen GDAL nodig
- Geen admin nodig
- Leest KMZ (zip) -> KML
- Parseert Placemark geometrie: Point, LineString, Polygon
- Schrijft aparte shapefiles per geometrie-type:
    points.shp, lines.shp, polygons.shp
- CRS: WGS84 (EPSG:4326) (lon,lat)
"""

import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime

# PyShp import: pip install --user pyshp  (module heet "shapefile")
try:
    import shapefile  # pyshp
except ImportError:
    print("FOUT: pyshp ontbreekt. Installeer met: py -m pip install --user pyshp")
    print("Of zet shapefile.py (PyShp) naast dit script.")
    sys.exit(1)

WGS84_PRJ = """GEOGCS["WGS 84",
DATUM["WGS_1984",
SPHEROID["WGS 84",6378137,298.257223563]],
PRIMEM["Greenwich",0],
UNIT["degree",0.0174532925199433],
AXIS["Longitude",EAST],
AXIS["Latitude",NORTH]]"""


def find_kml_in_kmz(kmz_path: str) -> str:
    """Extract KML content from KMZ and return KML text (string)."""
    with zipfile.ZipFile(kmz_path, "r") as z:
        kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
        if not kml_names:
            raise FileNotFoundError("Geen .kml gevonden in de KMZ.")
        # meestal is dit doc.kml
        kml_name = sorted(kml_names, key=lambda s: ("/" in s, len(s)))[0]
        return z.read(kml_name).decode("utf-8", errors="replace")


def ns_tag(tag: str, ns: str) -> str:
    return f"{{{ns}}}{tag}"


def get_text(elem):
    if elem is None or elem.text is None:
        return ""
    return elem.text.strip()


def parse_coords(coord_text: str):
    """
    KML coords: "lon,lat,alt lon,lat,alt ..."
    Return list of (lon, lat) floats.
    """
    pts = []
    for token in coord_text.replace("\n", " ").replace("\t", " ").split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                pts.append((lon, lat))
            except ValueError:
                continue
    return pts


def extract_extended_data(placemark, ns):
    """
    Pakt eenvoudige ExtendedData (Data name/value) mee.
    Beperkt tot shapefile-veldnaam limiet (10 chars) -> we nemen enkel enkele velden.
    """
    data = {}
    ext = placemark.find(".//" + ns_tag("ExtendedData", ns))
    if ext is None:
        return data

    # <Data name="..."><value>...</value></Data>
    for d in ext.findall(".//" + ns_tag("Data", ns)):
        key = d.attrib.get("name", "").strip()
        val = get_text(d.find(ns_tag("value", ns)))
        if key:
            data[key] = val

    # <SchemaData><SimpleData name="...">...</SimpleData>
    for sd in ext.findall(".//" + ns_tag("SimpleData", ns)):
        key = sd.attrib.get("name", "").strip()
        val = get_text(sd)
        if key:
            data[key] = val

    return data


def sanitize_fieldname(name: str):
    """
    Shapefile veldnamen: max 10 tekens, geen rare chars.
    """
    keep = []
    for c in name:
        if c.isalnum() or c == "_":
            keep.append(c)
        else:
            keep.append("_")
    s = "".join(keep).strip("_")
    if not s:
        s = "FIELD"
    return s[:10].upper()


def write_prj(base_path_no_ext: str):
    with open(base_path_no_ext + ".prj", "w", encoding="utf-8") as f:
        f.write(WGS84_PRJ)


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def main(kmz_path: str, out_dir: str):
    ensure_outdir(out_dir)

    kml_text = find_kml_in_kmz(kmz_path)

    # Parse KML
    root = ET.fromstring(kml_text)

    # Namespace detect (meestal "http://www.opengis.net/kml/2.2")
    if root.tag.startswith("{") and "}" in root.tag:
        ns = root.tag.split("}")[0].strip("{")
    else:
        ns = "http://www.opengis.net/kml/2.2"

    placemarks = root.findall(".//" + ns_tag("Placemark", ns))
    if not placemarks:
        print("Geen Placemark gevonden in de KML.")
        return

    points = []
    lines = []
    polys = []

    # Basisvelden (shp limiet veldnaam 10)
    # We houden het bewust simpel en stabiel.
    for pm in placemarks:
        name = get_text(pm.find(ns_tag("name", ns)))
        desc = get_text(pm.find(ns_tag("description", ns)))
        ext = extract_extended_data(pm, ns)

        # Point
        p = pm.find(".//" + ns_tag("Point", ns))
        if p is not None:
            c = get_text(p.find(ns_tag("coordinates", ns)))
            pts = parse_coords(c)
            if pts:
                # Point verwacht 1 punt
                lon, lat = pts[0]
                points.append({
                    "geom": (lon, lat),
                    "NAME": name,
                    "DESC": desc,
                    "EXTRA": "; ".join([f"{k}={v}" for k, v in list(ext.items())[:5]])
                })

        # LineString
        ls = pm.find(".//" + ns_tag("LineString", ns))
        if ls is not None:
            c = get_text(ls.find(ns_tag("coordinates", ns)))
            pts = parse_coords(c)
            if len(pts) >= 2:
                lines.append({
                    "geom": pts,
                    "NAME": name,
                    "DESC": desc,
                    "EXTRA": "; ".join([f"{k}={v}" for k, v in list(ext.items())[:5]])
                })

        # Polygon (outer ring; inner holes worden genegeerd of je kan uitbreiden)
        pol = pm.find(".//" + ns_tag("Polygon", ns))
        if pol is not None:
            ring = pol.find(".//" + ns_tag("outerBoundaryIs", ns) + "/" + ns_tag("LinearRing", ns))
            if ring is None:
                ring = pol.find(".//" + ns_tag("LinearRing", ns))
            if ring is not None:
                c = get_text(ring.find(ns_tag("coordinates", ns)))
                pts = parse_coords(c)
                if len(pts) >= 3:
                    # Zorg dat ring gesloten is
                    if pts[0] != pts[-1]:
                        pts.append(pts[0])
                    polys.append({
                        "geom": pts,  # enkel outer ring
                        "NAME": name,
                        "DESC": desc,
                        "EXTRA": "; ".join([f"{k}={v}" for k, v in list(ext.items())[:5]])
                    })

    base = os.path.splitext(os.path.basename(kmz_path))[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    written = []

    # Write points
    if points:
        shp_base = os.path.join(out_dir, f"{base}_points_{stamp}")
        w = shapefile.Writer(shp_base, shapeType=shapefile.POINT)
        w.field("NAME", "C", size=254)
        w.field("DESC", "C", size=254)
        w.field("EXTRA", "C", size=254)
        for r in points:
            lon, lat = r["geom"]
            w.point(lon, lat)
            w.record(r["NAME"], r["DESC"], r["EXTRA"])
        w.close()
        write_prj(shp_base)
        written.append(shp_base + ".shp")

    # Write lines
    if lines:
        shp_base = os.path.join(out_dir, f"{base}_lines_{stamp}")
        w = shapefile.Writer(shp_base, shapeType=shapefile.POLYLINE)
        w.field("NAME", "C", size=254)
        w.field("DESC", "C", size=254)
        w.field("EXTRA", "C", size=254)
        for r in lines:
            w.line([r["geom"]])  # 1 part
            w.record(r["NAME"], r["DESC"], r["EXTRA"])
        w.close()
        write_prj(shp_base)
        written.append(shp_base + ".shp")

    # Write polygons
    if polys:
        shp_base = os.path.join(out_dir, f"{base}_polygons_{stamp}")
        w = shapefile.Writer(shp_base, shapeType=shapefile.POLYGON)
        w.field("NAME", "C", size=254)
        w.field("DESC", "C", size=254)
        w.field("EXTRA", "C", size=254)
        for r in polys:
            w.poly([r["geom"]])  # 1 ring = 1 part
            w.record(r["NAME"], r["DESC"], r["EXTRA"])
        w.close()
        write_prj(shp_base)
        written.append(shp_base + ".shp")

    if not written:
        print("Geen ondersteunde geometrieën gevonden (Point/LineString/Polygon).")
    else:
        print("KLAAR. Shapefiles geschreven:")
        for p in written:
            print(" -", p)
        print("\nCRS: WGS84 (EPSG:4326). (lon/lat)")

    print("\nTip: als je Lambert72 nodig hebt, herprojecteer daarna in QGIS of met GDAL (als beschikbaar).")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Gebruik:")
        print('  py kmz_to_shapefile_purepython.py "C:\\pad\\file.kmz" "C:\\pad\\output"')
        sys.exit(1)

    kmz = sys.argv[1]
    out = sys.argv[2]
    main(kmz, out)
