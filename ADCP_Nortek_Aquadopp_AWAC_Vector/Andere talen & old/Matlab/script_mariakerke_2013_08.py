from __future__ import annotations

import numpy as np

from campaign_runner import run_campaign_entries, show


def main() -> list[dict]:
    entries = [
        {
            "dpl_param": {
                "folder_name": "../data/2013_08_mariakerke/A847102",
                "dpl_name": "A847102",
                "reference_height": "aquadopp",
                "vertical_dir": "downward",
                "time_gap": 0,
            },
            "out_param": {"save_format": "png"},
            "days_per_subplot": 7,
            "ebb_flood": False,
            "ellipse_bounding_box_display": [np.nan, np.nan, np.nan, np.nan],
        }
    ]
    return run_campaign_entries(entries)


if __name__ == "__main__":
    main()
    show()
