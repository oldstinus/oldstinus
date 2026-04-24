# Repository guide

## Purpose

This workspace contains small operational scripts for different measuring instruments and reporting workflows.
The codebase is organized by instrument family rather than by a single deployable application.

## Top-level projects

- `ADCP_Nortek_Aquadopp_AWAC_Vector`
- `ADCP_RDI_schepen_stationair`
- `ADCP_Sontek_Riversurveyor_M9_IQ`
- `Drukkamer_HF_DIVER`
- `Druksensor_HR_OSSI`
- `EXO-verwerking`
- `Log-a-level`
- `Multiparameter_Aanderaa`
- `Multiparameter_Aquatroll`
- `Multiparameter_YSI`
- `Radar_Vega_via_CR850`
- `TOPO-RTK_GNSS-GPS`
- `WATERINFO`
- `Word-excel-ppt`
- `OLD` for archived work

## Conventions

- Put source code near the project it belongs to.
- Treat `.venv`, `venv`, logs, installers, and exported charts/maps as local artifacts.
- Add a `README.md` to folders that are placeholders or work-in-progress.
- Keep long-lived documentation in `docs/`.
- Add repository-level checks in `tests/` using the standard library when possible.

## Known placeholders

- `Druksensor_HR_OSSI` is intentionally present but currently has no implementation files.

## Suggested next cleanup steps

- Split generated output into dedicated `output/` folders inside each project.
- Replace broad top-level `requirements.txt` with smaller per-project environment files.
- Add project-specific smoke tests for the scripts that are actively maintained.
