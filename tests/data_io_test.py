import os
import pytest
import pandas as pd
from upath import UPath

from ssdf.data_io import read_data_from_storage, write_data_to_storage


@pytest.mark.parametrize("ext", ["parquet", "csv"])
def test_read_write_data_local_path(training_data, tmp_path, ext):
    path = tmp_path / f"data.{ext}"
    if ext == "csv":
        write_data_to_storage(training_data, path, index=False)
        df = read_data_from_storage(path, parse_dates=["date"])
    else:
        write_data_to_storage(training_data, path)
        df = read_data_from_storage(path)

    pd.testing.assert_frame_equal(training_data, df)


@pytest.mark.parametrize("ext", ["parquet", "csv"])
def test_read_write_data_s3(training_data, ext):
    bucket_name = os.getenv("S3_BUCKET_NAME")
    assert bucket_name is not None, "S3_BUCKET_NAME env var must be set for this test"

    path = UPath(f"s3://{bucket_name}/data.{ext}")
    # Test write
    if ext == "csv":
        write_data_to_storage(training_data, path, index=False)
    else:
        write_data_to_storage(training_data, path)

    # Test read
    if ext == "csv":
        df = read_data_from_storage(path, parse_dates=["date"])
    else:
        df = read_data_from_storage(path)

    # Verify content
    pd.testing.assert_frame_equal(training_data, df)
