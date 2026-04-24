from __future__ import annotations

import numpy as np

from campaign_runner import run_campaign_entries, show


def main() -> list[dict]:
    entries = [
        {
            "dpl_param": {
                "folder_name": "../data/2013_07_scheldt/aquadopp",
                "dpl_name": "ALNOT02",
                "reference_height": "aquadopp",
                "z0": 0,
                "vertical_dir": "upward",
                "data_treatment": "storm",
            },
            "out_param": {
                "time_start": "13-Jul-2013 00:00:00",
                "time_stop": "13-Aug-2013 15:30:00",
                "save_format": "png",
            },
            "days_per_subplot": 7,
            "ebb_flood": False,
            "ellipse_bounding_box_display": [np.nan, np.nan, np.nan, np.nan],
        }
    ]
    return run_campaign_entries(entries)


if __name__ == "__main__":
    main()
    show()
