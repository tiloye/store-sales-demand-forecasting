import pandas as pd
from ssdf.data import wrangle_train_test, run

dummy_train_data = pd.DataFrame(
    [
        [1, "2023-01-01", 1, "A", 10.0, 1],
        [2, "2023-01-01", 1, "B", 10.0, 1],
        [3, "2023-01-01", 2, "A", 20.0, 0],
        [4, "2023-01-01", 2, "B", 20.0, 0],
        [5, "2023-01-03", 1, "A", 15.0, 3],
        [6, "2023-01-03", 1, "B", 30.0, 5],
        [7, "2023-01-03", 2, "A", 20.0, 0],
        [8, "2023-01-03", 2, "B", 20.0, 0],
    ],
    columns=["id", "date", "store_nbr", "family", "sales", "onpromotion"],
)  # train.csv
cleaned_train_data = pd.DataFrame(
    [
        ["2023-01-01", 1, "a", 10.0, 1],
        ["2023-01-01", 1, "b", 10.0, 1],
        ["2023-01-01", 2, "a", 20.0, 0],
        ["2023-01-01", 2, "b", 20.0, 0],
        ["2023-01-02", 1, "a", 0.0, 0],
        ["2023-01-02", 1, "b", 0.0, 0],
        ["2023-01-02", 2, "a", 0.0, 0],
        ["2023-01-02", 2, "b", 0.0, 0],
        ["2023-01-03", 1, "a", 15.0, 3],
        ["2023-01-03", 1, "b", 30.0, 5],
        ["2023-01-03", 2, "a", 20.0, 0],
        ["2023-01-03", 2, "b", 20.0, 0],
    ],
    columns=["date", "store_nbr", "family", "sales", "onpromotion"],
)
cleaned_train_data["date"] = pd.to_datetime(cleaned_train_data["date"])


dummy_test_data = pd.DataFrame(
    [
        [1, "2023-01-03", 1, "A", 0],
        [2, "2023-01-03", 1, "B", 0],
        [3, "2023-01-03", 2, "A", 0],
        [4, "2023-01-03", 2, "B", 0],
        [5, "2023-01-04", 1, "A", 1],
        [6, "2023-01-04", 1, "B", 1],
        [7, "2023-01-04", 2, "A", 1],
        [8, "2023-01-04", 2, "B", 1],
    ],
    columns=["id", "date", "store_nbr", "family", "onpromotion"],
)  # test.csv
cleaned_test_data = pd.DataFrame(
    [
        ["2023-01-03", 1, "a", 0],
        ["2023-01-03", 1, "b", 0],
        ["2023-01-03", 2, "a", 0],
        ["2023-01-03", 2, "b", 0],
        ["2023-01-04", 1, "a", 1],
        ["2023-01-04", 1, "b", 1],
        ["2023-01-04", 2, "a", 1],
        ["2023-01-04", 2, "b", 1],
    ],
    columns=["date", "store_nbr", "family", "onpromotion"],
)
cleaned_test_data["date"] = pd.to_datetime(cleaned_test_data["date"])


def test_wrangle_train_test(subtests):
    with subtests.test("train data"):
        train_data = dummy_train_data.copy()
        wrangled_train = wrangle_train_test(train_data)
        expected_train_df = cleaned_train_data.copy()
        pd.testing.assert_frame_equal(
            wrangled_train,
            expected_train_df.sort_values(["store_nbr", "family", "date"]).reset_index(
                drop=True
            ),
            check_dtype=False,
        )
    with subtests.test("test data"):
        test_data = dummy_test_data.copy()
        wrangled_test = wrangle_train_test(test_data)
        expected_test_df = cleaned_test_data.copy()
        pd.testing.assert_frame_equal(
            wrangled_test,
            expected_test_df.sort_values(["store_nbr", "family", "date"]).reset_index(
                drop=True
            ),
            check_dtype=False,
        )


def test_run(monkeypatch, tmp_path, subtests):
    monkeypatch.setattr("ssdf.data.RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr("ssdf.data.PROCESSED_DATA_DIR", tmp_path)

    # Mock get_source_data to create a mock train.csv file and return its path
    def mock_get_source_data(path=None, force_download=False):
        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        dummy_train_data.to_csv(train_path, index=False)
        dummy_test_data.to_csv(test_path, index=False)
        return tmp_path.as_posix()

    monkeypatch.setattr("ssdf.data.get_source_data", mock_get_source_data)

    # Execute the run function
    run()

    # Validate the results
    assert (tmp_path / "sales.parquet").exists()
    df = pd.read_parquet(tmp_path / "sales.parquet")
    pd.testing.assert_frame_equal(
        df.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True),
        cleaned_train_data.drop("onpromotion", axis=1)
        .sort_values(["date", "store_nbr", "family"])
        .reset_index(drop=True),
    )

    # Verify the transformed promotions data has expected schema
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
