"""Simple helper to browse for a MAT file and print its structure."""

import h5py
import tkinter as tk
from tkinter import filedialog
from scipy import io as scipy_io


def choose_mat_file() -> str | None:
    """Show a file-selection dialog for HDF5 (.mat) files."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select MAT file to inspect",
        filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
    )
    root.destroy()
    return path


def dump_mat_structure(path: str) -> None:
    """Open an HDF5-based MAT file and print every group/dataset name."""
    try:
        with h5py.File(path, "r") as mat_file:
            def print_name(name: str) -> None:
                print(name)

            mat_file.visit(print_name)
    except OSError as exc:
        print("Failed to read file as HDF5:", exc)
        dump_classic_mat_structure(path)


def dump_classic_mat_structure(path: str) -> None:
    """Try to load a legacy (pre-7.3) MAT file and print its variable names."""
    try:
        data = scipy_io.loadmat(path, struct_as_record=False, squeeze_me=True)
    except Exception as exc:
        print("Also failed to read file as classic MAT:", exc)
        return

    variables = [key for key in data if not key.startswith("__")]
    if not variables:
        print("Loaded legacy MAT but no user variables found.")
        return

    print("Legacy MAT variables:")
    for name in variables:
        print(f"  {name}")


def main() -> None:
    mat_path = choose_mat_file()
    if not mat_path:
        print("No file selected; exiting.")
        return

    print(f"Inspecting '{mat_path}' …")
    dump_mat_structure(mat_path)


if __name__ == "__main__":
    main()
