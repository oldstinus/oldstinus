from __future__ import annotations

from pathlib import Path
from pprint import pprint

from awac_core import load_awac_currents


def main(path: str | Path = "Herculesa660602_23092015_14112015_p.dat") -> dict:
    data = load_awac_currents(path)
    pprint({key: value.shape for key, value in data.items() if hasattr(value, "shape")})
    return data


if __name__ == "__main__":
    main()
