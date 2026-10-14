"""Filesystem locations for stage 1 data. Everything under data/ is git-ignored."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
DB = DATA / "db"
