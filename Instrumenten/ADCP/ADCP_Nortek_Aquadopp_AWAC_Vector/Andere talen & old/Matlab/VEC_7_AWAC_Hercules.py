from __future__ import annotations

from pathlib import Path

from awac_core import load_wave_file


def main(path: str | Path = "Herculesa660602_23092015_14112015.wap") -> dict:
    return load_wave_file(path)


if __name__ == "__main__":
    data = main()
    print(f"Sea level range: {data['sea_level'].min():.2f} to {data['sea_level'].max():.2f} m")
