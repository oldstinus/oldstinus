# 3D-kist voor 1 L flessen (licht + stevig)

Dit pakket bevat een eenvoudig parametrisch 3D-ontwerp van een flessenkist voor standaard 1 L flessen, plus een virtuele productafbeelding.

## Gekozen route

Lokaal zijn `python`, `numpy`, `matplotlib` en `trimesh` beschikbaar.
Er is geen lokale bevestiging gevonden voor `Blender`, `OpenSCAD` of `FreeCAD`.
Daarom is gekozen voor een pragmatische Python-workflow:

1. Parametrische opbouw van de kist als verzameling balken/frames (trimesh).
2. Export van het model naar `OBJ` en `STL`.
3. Generatie van een pseudo-render (axonometisch) naar `PNG` met matplotlib.

## Aannames

- Standaard 1 L flesdiameter: ongeveer `85 mm`.
- Binnenruimte per fles (met speling): `92 mm` celmaat.
- Indeling: `3 x 2` flessen (6 stuks totaal).
- Buitenmaat van het concept: ongeveer `300 x 208 x 330 mm`.
- Doel: zo licht mogelijk via open frame + bodemrunners, maar met stijfheid via hoekstijlen, middenrails, bovenrand en interne verdelers.

## Bestanden

- `generate_crate_model.py`: script dat model + render maakt.
- `crate_1l_lightweight.obj`: 3D-model (mesh).
- `crate_1l_lightweight.stl`: 3D-model (mesh).
- `crate_1l_render.png`: virtuele productafbeelding.
- `crate_dimensions.txt`: samenvatting van afmetingen.

## Uitvoeren

```powershell
python agent_outputs/render/generate_crate_model.py
```

Na uitvoeren worden `OBJ`, `STL`, `PNG` en dimensiebestand automatisch (opnieuw) gegenereerd in deze map.
