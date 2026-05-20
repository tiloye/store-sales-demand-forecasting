from __future__ import annotations

from pathlib import Path

import pandas as pd
from upath.implementations.cloud import S3Path
from ssdf.config import STORAGE_OPTIONS


def read_data_from_storage(path: Path | S3Path, **kwargs) -> pd.DataFrame:
    """Read csv or parquet files from storage."""

    file_ext = str(path).split(".")[-1]
    if isinstance(path, Path):
        return eval(f"pd.read_{file_ext}")(path, **kwargs)

    if isinstance(path, S3Path):
        return eval(f"pd.read_{file_ext}")(
            path.as_posix(), storage_options=STORAGE_OPTIONS, **kwargs
        )
    raise TypeError(f"path must be either a Path or S3Path, got: {type(path)}")


def write_data_to_storage(df: pd.DataFrame, path: Path | S3Path, **kwargs) -> None:
    """Save data to storage as csv or parquet files."""
    file_ext = str(path).split(".")[-1]
    if isinstance(path, Path):
        eval(f"df.to_{file_ext}")(path, **kwargs)
    elif isinstance(path, S3Path):
        eval(f"df.to_{file_ext}")(
            path.as_posix(), storage_options=STORAGE_OPTIONS, **kwargs
        )
    else:
        raise TypeError(f"path must be either a Path or S3Path, got: {type(path)}")
