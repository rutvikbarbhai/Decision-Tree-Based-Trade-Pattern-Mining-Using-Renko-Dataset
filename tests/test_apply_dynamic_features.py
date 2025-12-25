import pandas as pd
# import numpy as np
from module2_tsfresh_features_v3 import apply_dynamic_features


# Test apply_dynamic_features
def test_apply_dynamic_features_success():
    """Test successful application of base feature formulas."""
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40]})
    # df['a'] = df['a'].astype(float)
    formulas = [
        ('ratio_ab', 'a / b'),
        ('sum_ab', 'a + b'),
        ('log_a', 'np.log(a)')
    ]

    result_df = apply_dynamic_features(df.copy(), formulas)
    assert 'ratio_ab' in result_df.columns
    assert 'sum_ab' in result_df.columns
    # assert 'log_a' in result_df.columns

    pd.testing.assert_series_equal(result_df['ratio_ab'], pd.Series([0.1, 0.1, 0.1, 0.1], name='ratio_ab'))
    # pd.testing.assert_series_equal(result_df['log_a'], pd.Series([0.0, 0.693147, 1.098612, 1.386294], name='log_a', dtype=float), atol=1e-6)


def test_apply_dynamic_features_empty_formulas():
    """Test case with an empty formula list."""
    df = pd.DataFrame({'a': [1, 2]})
    result_df = apply_dynamic_features(df.copy(), [])
    pd.testing.assert_frame_equal(result_df, df)
