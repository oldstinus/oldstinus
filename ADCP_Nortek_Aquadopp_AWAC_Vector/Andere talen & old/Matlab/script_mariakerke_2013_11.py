from __future__ import annotations

import numpy as np

from campaign_runner import run_campaign_entries, show


def main() -> list[dict]:
    entries = [
        {
            "dpl_param": {
                "folder_name": "F:/MyMatlab/CampaignAnalysis/nortek_matlab/data/2013_11_mariakerke/HerculesI/AQD8481",
                "dpl_name": "A848103",
                "z0": -6.5,
                "reference_height": "aquadopp",
                "vertical_dir": "downward",
                "time_gap": 3.969,
                "data_treatment": "storm",
            },
            "out_param": {
                "save_format": "png",
                "time_start": "13-Nov-2013 18:20:00",
                "time_stop": "11-Dec-2013 00:00:00",
            },
            "days_per_subplot": 7,
            "ebb_flood": False,
            "ellipse_bounding_box_display": [np.nan, np.nan, np.nan, np.nan],
        },
        {
            "dpl_param": {
                "folder_name": "F:/MyMatlab/CampaignAnalysis/nortek_matlab/data/2013_11_mariakerke/HerculesI/AWAC",
                "dpl_name": "W202202",
                "z0": -6.5,
                "reference_height": "awac",
                "vertical_dir": "upward",
                "time_gap": -1,
                "data_treatment": "storm",
            },
            "out_param": {
                "save_format": "png",
                "time_start": "13-Nov-2013 00:00:00",
                "time_stop": "11-Dec-2013 00:00:00",
            },
            "days_per_subplot": 7,
            "ebb_flood": False,
            "ellipse_bounding_box_display": [np.nan, np.nan, np.nan, np.nan],
        },
    ]
    return run_campaign_entries(entries)


if __name__ == "__main__":
    main()
    show()
