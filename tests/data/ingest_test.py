import os

import s3fs
from upath import UPath

from ssdf.config import STORAGE_OPTIONS
from ssdf.data.ingest import get_source_data


def test_get_source_data_saves_to_s3(
    monkeypatch, tmp_path, dummy_train_data, dummy_test_data
):
    bucket_name = os.getenv("S3_BUCKET_NAME")
    assert bucket_name is not None, "S3_BUCKET_NAME env var must be set for this test"

    mock_s3_path = UPath(f"s3://{bucket_name}/raw")
    monkeypatch.setattr("ssdf.data.ingest.RAW_DATA_DIR", mock_s3_path)

    def mock_download(competition, path, output_dir, force_download):
        dummy_train_data.to_csv(tmp_path / "train.csv", index=False)
        dummy_test_data.to_csv(tmp_path / "test.csv", index=False)
        return tmp_path

    monkeypatch.setattr("kagglehub.competition_download", mock_download)
    res = get_source_data()
    assert res == mock_s3_path.as_posix()

    fs = s3fs.S3FileSystem(**STORAGE_OPTIONS)
    assert fs.exists(f"s3://{bucket_name}/raw/train.csv")
    assert fs.exists(f"s3://{bucket_name}/raw/test.csv")
