"""Write outputs/environment.txt (Python, OS, CPU, RAM, library versions)."""
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import psutil

OUT = Path(__file__).resolve().parents[1] / "outputs" / "environment.txt"

lines = [
    f"python: {sys.version}",
    f"os: {platform.platform()}",
    f"cpu: {platform.processor()}",
    f"cpu_logical_cores: {os.cpu_count()}",
    f"cpu_physical_cores: {psutil.cpu_count(logical=False)}",
    f"ram_total_gb: {psutil.virtual_memory().total / 1024**3:.2f}",
]
for lib in ["pandas", "numpy", "pyarrow", "matplotlib", "psutil"]:
    lines.append(f"{lib}: {version(lib)}")
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
