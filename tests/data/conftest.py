import pandas as pd
import pytest


@pytest.fixture
def dummy_train_data():
    return pd.DataFrame(
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
    )


@pytest.fixture
def cleaned_train_data():
    df = pd.DataFrame(
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
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def dummy_test_data():
    return pd.DataFrame(
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
    )


@pytest.fixture
def cleaned_test_data():
    df = pd.DataFrame(
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
    df["date"] = pd.to_datetime(df["date"])
    return df
