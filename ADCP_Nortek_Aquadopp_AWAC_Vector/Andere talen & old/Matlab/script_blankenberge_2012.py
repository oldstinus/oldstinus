from __future__ import annotations

import numpy as np

from campaign_runner import run_campaign_entries, show


def main() -> list[dict]:
    entries = [
        {
            "dpl_param": {
                "folder_name": "../data/2012_06_blankenberge/aquadopp_1",
                "dpl_name": "Averti01",
                "z0": -5.3,
                "vertical_dir": "downward",
            },
            "out_param": {
                "save_format": "png",
                "time_start": "5-Jun-2012 11:10:00",
                "cell_stop": 14,
            },
            "days_per_subplot": 5,
            "ebb_flood": False,
            "ellipse_bounding_box_display": [np.nan, np.nan, np.nan, np.nan],
        },
        {
            "dpl_param": {
                "folder_name": "../data/2012_06_blankenberge/aquadopp_2",
                "dpl_name": "Ahoriz01",
                "z0": -4.6,
                "vertical_dir": "downward",
            },
            "out_param": {
                "save_format": "png",
                "time_start": "5-Jun-2012 11:30:00",
                "time_stop": "26-Jun-2012 08:25:00",
                "cell_stop": 17,
            },
            "days_per_subplot": 5,
            "ebb_flood": False,
            "ellipse_bounding_box_display": [np.nan, np.nan, np.nan, np.nan],
        },
    ]
    return run_campaign_entries(entries)


if __name__ == "__main__":
    main()
    show()
