"""Produce fixed epoch comparisons only after eight-epoch fitting is frozen."""
from pathlib import Path
import runpy

if __name__=="__main__":
    runpy.run_path(str(Path(__file__).with_name("evaluate_gears.py")),run_name="__main__")
