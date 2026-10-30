"""Build the parquet cache for all files, reporting runtime and memory."""
import time

import psutil

from src import io_utils as io

proc = psutil.Process()
rows = []
for name, fn in [("views_11-30", io.load_views)] + [
    (f"schedule_{p}", lambda p=p: io.load_schedule(p)) for p in io.SCHEDULE_FILES
]:
    t0 = time.perf_counter()
    df = fn()
    dt = time.perf_counter() - t0
    mem = df.memory_usage(deep=True).sum() / 1024**2
    rss = proc.memory_info().rss / 1024**2
    rows.append((name, len(df), dt, mem, rss))
    print(f"{name}: rows={len(df):,} load_s={dt:.1f} df_mem_MB={mem:.0f} process_rss_MB={rss:.0f}")
    print(df.dtypes.to_string(), "\n")
    del df

import pandas as pd
pd.DataFrame(rows, columns=["table", "rows", "load_seconds", "df_memory_mb", "process_rss_mb"]).to_csv(
    io.ROOT / "outputs" / "load_runtime_memory.csv", index=False)
