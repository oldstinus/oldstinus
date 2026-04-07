from __future__ import annotations

from pathlib import Path

from awac_core import load_wave_file


def main(path: str | Path = "Herculesa660602_23092015_14112015.wap") -> dict:
    return load_wave_file(path)


if __name__ == "__main__":
    data = main()
    print(f"Loaded {len(data['wave'])} AWAC wave records")
