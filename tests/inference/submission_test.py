import pandas as pd

from ssdf.inference.submission import create_submission_file


def test_create_submission_file(tmp_path, monkeypatch):
    monkeypatch.setattr("ssdf.inference.submission.PREDICTIONS_DIR", tmp_path)
    monkeypatch.setattr("ssdf.inference.submission.RAW_DATA_DIR", tmp_path)

    forecasts = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"]
            ),
            "store_nbr": [1, 1, 2, 2],
            "family": ["a", "a", "b", "b"],
            "sales": [10.0, 20.0, 15.0, 25.0],
        }
    )
    forecasts.to_parquet(tmp_path / "sales_forecasts.parquet", index=False)
    test_data = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "date": pd.to_datetime(
                ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"]
            ),
            "store_nbr": [1, 1, 2, 2],
            "family": ["A", "A", "B", "B"],
            "onpromotion": [0, 0, 0, 0],
        }
    )
    test_data.to_csv(tmp_path / "test.csv", index=False)

    expected_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "sales": [10.0, 20.0, 15.0, 25.0],
        }
    )

    create_submission_file()

    saved_df = pd.read_csv(tmp_path / "submission.csv")
    pd.testing.assert_frame_equal(saved_df, expected_df)
