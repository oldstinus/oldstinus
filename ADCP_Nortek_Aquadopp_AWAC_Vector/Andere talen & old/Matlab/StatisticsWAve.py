from __future__ import annotations

from pathlib import Path
from pprint import pprint

from awac_core import wave_statistics


def main(path: str | Path = "Herculesa660602_23092015_14112015.wap") -> dict:
    stats = wave_statistics(path)
    pprint(stats)
    return stats


if __name__ == "__main__":
    main()
