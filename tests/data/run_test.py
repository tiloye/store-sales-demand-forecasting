import pandas as pd

from ssdf.data.run import run


def test_run(
    monkeypatch,
    tmp_path,
    subtests,
    dummy_train_data,
    dummy_test_data,
    cleaned_train_data,
    cleaned_test_data,
):
    monkeypatch.setattr("ssdf.data.run.RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr("ssdf.data.run.PROCESSED_DATA_DIR", tmp_path)

    def mock_get_source_data(path=None, force_download=False):
        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        dummy_train_data.to_csv(train_path, index=False)
        dummy_test_data.to_csv(test_path, index=False)
        return tmp_path.as_posix()

    monkeypatch.setattr("ssdf.data.run.get_source_data", mock_get_source_data)

    run()

    assert (tmp_path / "sales.parquet").exists()
    df = pd.read_parquet(tmp_path / "sales.parquet")
    pd.testing.assert_frame_equal(
        df.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True),
        cleaned_train_data.drop("onpromotion", axis=1)
        .sort_values(["date", "store_nbr", "family"])
        .reset_index(drop=True),
    )

    assert (tmp_path / "promotions.parquet").exists()
    expected_promotions = pd.concat(
        [cleaned_train_data.drop("sales", axis=1), cleaned_test_data],
        ignore_index=True,
    )
    df = pd.read_parquet(tmp_path / "promotions.parquet")
    pd.testing.assert_frame_equal(
        df.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True),
        expected_promotions.sort_values(["date", "store_nbr", "family"]).reset_index(
            drop=True
        ),
        check_dtype=False,
    )
