from __future__ import annotations

from campaign_runner import run_campaign_entries, show


def main() -> list[dict]:
    entries = [
        {
            "dpl_param": {
                "folder_name": "../data/2013_06_blankenberge/aquadopp",
                "dpl_name": "Q847902_5July_2013",
                "z0": -4.5,
                "reference_height": "TAW",
                "vertical_dir": "upward",
                "data_treatment": "storm",
            },
            "out_param": {
                "save_format": "png",
                "time_start": "19-Jun-2013 10:00:00",
                "time_stop": "4-Jul-2013 11:30:00",
            },
            "days_per_subplot": 3,
            "min_depth_display": -4.2,
            "max_depth_display": 3.5,
            "max_velocity_display": 1.6654,
            "ebb_flood": True,
            "ellipse_bounding_box_display": [-0.9, 1.4, -0.7, 1.1],
        },
        {
            "dpl_param": {
                "folder_name": "../data/2013_06_blankenberge/awac_sea",
                "dpl_name": "W660601_AWAC_sea_5_july_2013",
                "z0": -5.1,
                "reference_height": "TAW",
                "vertical_dir": "upward",
                "data_treatment": "storm",
            },
            "out_param": {
                "save_format": "png",
                "time_start": "19-Jun-2013 10:05:00",
                "time_stop": "4-Jul-2013 10:40:00",
            },
            "days_per_subplot": 3,
            "min_depth_display": -4.2,
            "max_depth_display": 3.5,
            "max_velocity_display": 1.6654,
            "ebb_flood": True,
            "ellipse_bounding_box_display": [-0.9, 1.4, -0.7, 1.1],
        },
        {
            "dpl_param": {
                "folder_name": "../data/2013_06_blankenberge/awac_harbor",
                "dpl_name": "W659404",
                "z0": -1.2,
                "reference_height": "TAW",
                "vertical_dir": "upward",
                "data_treatment": "storm",
            },
            "out_param": {
                "save_format": "png",
                "time_start": "20-Jun-2013 13:00:00",
                "time_stop": "5-Jul-2013 11:20:00",
            },
            "days_per_subplot": 3,
            "min_depth_display": -4.2,
            "max_depth_display": 3.5,
            "max_velocity_display": 1.6654,
            "ebb_flood": False,
            "ellipse_bounding_box_display": [-0.9, 1.4, -0.7, 1.1],
        },
    ]
    return run_campaign_entries(entries)


if __name__ == "__main__":
    main()
    show()
