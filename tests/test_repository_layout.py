from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_core_docs_exist(self) -> None:
        self.assertTrue((ROOT / "README.md").is_file())
        self.assertTrue((ROOT / "docs" / "REPOSITORY_GUIDE.md").is_file())
        self.assertTrue((ROOT / ".gitignore").is_file())

    def test_placeholder_project_is_documented(self) -> None:
        self.assertTrue((ROOT / "Druksensor_HR_OSSI" / "README.md").is_file())

    def test_expected_project_directories_exist(self) -> None:
        expected = [
            "ADCP_Nortek_Aquadopp_AWAC_Vector",
            "ADCP_Sontek_Riversurveyor_M9_IQ",
            "Drukkamer_HF_DIVER",
            "Druksensor_HR_OSSI",
            "Multiparameter_YSI",
            "Radar_Vega_via_CR850",
            "TOPO-RTK_GNSS-GPS",
            "WATERINFO",
            "Word-excel-ppt",
        ]
        for name in expected:
            with self.subTest(name=name):
                self.assertTrue((ROOT / name).is_dir())


if __name__ == "__main__":
    unittest.main()
