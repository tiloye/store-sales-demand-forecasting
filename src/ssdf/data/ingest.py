from __future__ import annotations

import tempfile
from pathlib import Path

import kagglehub
import s3fs
from upath.implementations.cloud import S3Path

from ssdf.config import RAW_DATA_DIR, STORAGE_OPTIONS

KAGGLE_COMPETITION = "store-sales-time-series-forecasting"


def get_source_data(
    path: str | None = None,
    force_download: bool = False,
) -> str:
    """Download the Kaggle competition data to local raw storage or S3."""
    if isinstance(RAW_DATA_DIR, S3Path):
        return _download_to_s3(path=path, force_download=force_download)

    return kagglehub.competition_download(
        KAGGLE_COMPETITION,
        path=path,
        output_dir=RAW_DATA_DIR.as_posix(),
        force_download=force_download,
    )


def _download_to_s3(path: str | None, force_download: bool) -> str:
    """Download competition data to a temp directory, then upload to S3."""
    fs = s3fs.S3FileSystem(**STORAGE_OPTIONS)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            download_path = kagglehub.competition_download(
                KAGGLE_COMPETITION,
                path=path,
                output_dir=tmp_dir,
                force_download=force_download,
            )
            _upload_directory_to_s3(download_path, fs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download or upload source data to S3: {RAW_DATA_DIR.as_posix()}"
        ) from exc

    return RAW_DATA_DIR.as_posix()


def _upload_directory_to_s3(source_dir: str | Path, fs: s3fs.S3FileSystem) -> None:
    """Recursively upload a local directory to the configured S3 raw bucket."""
    source = Path(source_dir)
    s3_prefix = RAW_DATA_DIR.as_posix()

    for file_path in source.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(source)
            s3_dest = f"{s3_prefix}/{rel_path.as_posix()}"
            fs.put_file(str(file_path), s3_dest)
