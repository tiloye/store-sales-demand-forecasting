import pandas as pd

from ssdf.data.transform import wrangle_train_test


def test_wrangle_train_test(
    subtests,
    dummy_train_data,
    dummy_test_data,
    cleaned_train_data,
    cleaned_test_data,
):
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
