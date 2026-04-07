from __future__ import annotations

import numpy as np

from campaign_runner import run_campaign_entries, show


def main() -> list[dict]:
    entries = [
        {
            "dpl_param": {
                "folder_name": "F:/MyMatlab/CampaignAnalysis/nortek_matlab/data/2015_09_mariakerke/M1_Hercules/AWAC/Processed23092015_14112015",
                "dpl_name": "Herculesa660602_23092015_14112015",
                "z0": -6.5,
                "reference_height": "awac",
                "vertical_dir": "upward",
                "time_gap": -14,
                "data_treatment": "storm",
            },
            "out_param": {
                "save_format": "png",
                "time_start": "23-Sep-2015 00:01:01",
                "time_stop": "14-Nov-2015 00:01:01",
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
