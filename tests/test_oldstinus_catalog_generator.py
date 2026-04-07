from pathlib import Path
import unittest
from unittest import mock

from docs import generate_oldstinus_catalog as catalog


ROOT = Path(__file__).resolve().parents[1]


class OldstinusCatalogGeneratorTests(unittest.TestCase):
    def test_docs_directory_is_excluded(self) -> None:
        self.assertTrue(catalog.is_excluded(Path("docs/launchers/example.bat")))

    def test_transient_temp_directories_are_excluded(self) -> None:
        self.assertTrue(catalog.is_excluded(Path("oldstinus/.tmp/cache.py")))
        self.assertTrue(catalog.is_excluded(Path("tests/tmpait4jnna/example.py")))

    def test_extract_doc_hint_skips_encoding_line(self) -> None:
        lines = [
            "#!/usr/bin/env python",
            "# -*- coding: utf-8 -*-",
            "",
            "# Generieke GUI om CSV-data snel te visualiseren.",
            "import os",
        ]
        self.assertEqual(
            catalog.extract_doc_hint(lines),
            "Generieke GUI om CSV-data snel te visualiseren.",
        )

    def test_root_level_script_uses_root_directory_as_project_dir(self) -> None:
        item = catalog.classify_file(ROOT / "Super-generiek_csv_tijdgrafiek_gui.py")
        self.assertEqual(item.top_level, ROOT.name)
        self.assertEqual(item.project, "Losse hulpscripts")
        self.assertEqual(item.project_dir, ".")

    def test_root_level_launcher_uses_directory_not_script_path(self) -> None:
        item = catalog.classify_file(ROOT / "Super-generiek_csv_tijdgrafiek_gui.py")
        original_launchers_dir = catalog.LAUNCHERS_DIR
        try:
            catalog.LAUNCHERS_DIR = ROOT / "tests"
            with mock.patch.object(Path, "write_text", autospec=True) as write_text:
                catalog.generate_launcher(item)
        finally:
            catalog.LAUNCHERS_DIR = original_launchers_dir

        launcher_path = write_text.call_args.args[0]
        launcher_text = write_text.call_args.args[1]

        self.assertEqual(launcher_path, ROOT / "tests" / Path(item.launcher_rel_path).name)
        self.assertIn(f'set "PROJECT_DIR={ROOT}"', launcher_text)
        self.assertIn(f'set "SCRIPT_PATH={ROOT / "Super-generiek_csv_tijdgrafiek_gui.py"}"', launcher_text)
        self.assertIn(f'set "VENV_ACTIVATE={ROOT / ".venv" / "Scripts" / "activate.bat"}"', launcher_text)
        self.assertIn('start "" "%CODE_EXE%" --new-window "%PROJECT_DIR%" "%SCRIPT_PATH%"', launcher_text)
        self.assertNotIn(f'set "PROJECT_DIR={ROOT / "Super-generiek_csv_tijdgrafiek_gui.py"}"', launcher_text)

    def test_root_relative_path_rejects_paths_outside_root(self) -> None:
        self.assertIsNone(catalog.root_relative_path(ROOT.parent / "outside.py"))


if __name__ == "__main__":
    unittest.main()
