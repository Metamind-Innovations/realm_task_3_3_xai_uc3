import numpy as np
import hashlib

def clean_infinite_values(df):
    """
    Replace infinities and extremely large values in dataframe with NaN or capped values.

    Helps prevent numerical instability in calculations by:
    1. Replacing infinity values with NaN
    2. Capping extreme values (beyond 1e10) to reasonable maximum

    :param df: Input dataframe with potential infinite or extreme values
    :type df: pandas.DataFrame
    :returns: Cleaned dataframe with handled values
    :rtype: pandas.DataFrame
    """
    if df is None:
        return None

    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()

    inf_count = np.isinf(df_cleaned.select_dtypes(include=[np.number])).sum().sum()

    if inf_count > 0:
        print(f"Warning: Found {inf_count} infinity values in the dataset. Replacing with NaN.")
        df_cleaned = df_cleaned.replace([np.inf, -np.inf], np.nan)

    for col in df_cleaned.select_dtypes(include=[np.number]).columns:
        extreme_mask = (df_cleaned[col].abs() > 1e10) & df_cleaned[col].notna()
        extreme_count = extreme_mask.sum()

        if extreme_count > 0:
            print(f"Warning: Found {extreme_count} extreme values in column '{col}'. Capping to reasonable values.")
            df_cleaned.loc[extreme_mask, col] = df_cleaned.loc[extreme_mask, col].apply(
                lambda x: 1e10 if x > 0 else -1e10
            )

    return df_cleaned


def encode_icd9_code(code):
    """
    Create a numeric representation of ICD9 code.

    Uses first 3 characters as category and hashes the rest for subcategory.
    This creates a consistent numerical representation usable for modeling.

    :param code: ICD-9 diagnosis code string
    :type code: str
    :returns: Tuple of (category, subcategory) as numerical values
    :rtype: tuple(float, float)
    """
    import pandas as pd
    
    if pd.isna(code) or not isinstance(code, str) or len(code) < 3:
        return 0.0, 0.0

    try:
        category = float(code[:3])
    except ValueError:
        category = 0.0

    if len(code) > 3:
        hash_val = int(hashlib.md5(code[3:].encode()).hexdigest(), 16)
        subcategory = float(hash_val % 1000) / 1000  # Normalize to 0-1
    else:
        subcategory = 0.0

    return category, subcategory
