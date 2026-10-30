"""Efficient loaders for the sweet.tv data with a parquet cache in ./cache/.

Raw CSVs are parsed once with pyarrow (explicit column types) and cached as
parquet; later loads read the parquet file only.

Column mapping (verified in Step 1 audit, see outputs/LOG.md):
  view log raw `stop_time` -> `end_time`, raw `duraton` -> `duration`.
Timestamps are naive local-time strings in the raw files (no timezone):
  view log  : 'YYYY-MM-DD HH:MM:SS'
  schedules : 'DD.MM.YYYY HH:MM:SS'
Both are parsed to timestamp[s] without timezone so they are directly comparable.
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "cache"

VIEW_FILE = "dataset11-30.csv"
SCHEDULE_FILES = {
    "11-20": "export_arh_11-20-final.csv",
    "21-30": "export_arh_21-30-final.csv",
    "31-42": "export_arh_31-42-final.csv",
}
SUBMISSION_FILE = "submission.csv"

# raw name -> canonical name
VIEW_RENAME = {"stop_time": "end_time", "duraton": "duration"}

VIEW_TYPES = {
    "user_id": pa.uint64(),  # ids exceed int64 max
    "vsetv_id": pa.int32(),
    "start_time": pa.timestamp("s"),
    "stop_time": pa.timestamp("s"),
    "duraton": pa.int32(),
}

SCHEDULE_TYPES = {
    "channel_id": pa.int32(),
    "channel_title": pa.string(),
    "start_time": pa.timestamp("s"),
    "duration": pa.int32(),
    "tv_show_title": pa.string(),
    "tv_show_id": pa.int64(),
    "tv_show_category": pa.string(),
    "tv_show_genre_1": pa.string(),
    "tv_show_genre_2": pa.string(),
    "tv_show_genre_3": pa.string(),
    "year_of_production": pa.string(),  # contains values like '1996-'
    "director": pa.string(),
    "actors": pa.string(),
}


def _cached(name: str, build) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.parquet"
    if not path.exists():
        pq.write_table(build(), path)
    return pq.read_table(path).to_pandas()


def load_views() -> pd.DataFrame:
    """View log with canonical columns: user_id, vsetv_id, start_time, end_time, duration."""
    def build():
        t = pacsv.read_csv(
            DATA / VIEW_FILE,
            convert_options=pacsv.ConvertOptions(
                column_types=VIEW_TYPES, timestamp_parsers=["%Y-%m-%d %H:%M:%S"]
            ),
        )
        return t.rename_columns([VIEW_RENAME.get(c, c) for c in t.column_names])
    return _cached("views_11-30", build)


def load_schedule(period: str) -> pd.DataFrame:
    """Schedule for period '11-20', '21-30' or '31-42'."""
    def build():
        return pacsv.read_csv(
            DATA / SCHEDULE_FILES[period],
            convert_options=pacsv.ConvertOptions(
                column_types=SCHEDULE_TYPES, timestamp_parsers=["%d.%m.%Y %H:%M:%S"],
                strings_can_be_null=True,  # empty fields -> null, not ''
            ),
        )
    return _cached(f"schedule_{period}", build)


def load_submission() -> pd.DataFrame:
    return pd.read_csv(DATA / SUBMISSION_FILE, dtype={"user_id": "uint64", "tv_show_id": "string"})
