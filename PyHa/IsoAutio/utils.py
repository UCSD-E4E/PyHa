import pandas as pd


def kaleidoscope_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Function that strips away Pandas Dataframe columns necessary for PyHa
    package that aren't compatible with Kaleidoscope software

    Args:
        df (pd.DataFrame): Dataframe compatible with PyHa package whether it be human labels
            or automated labels.

    Returns:
        pd.DataFrame: Pandas Dataframe compatible with Kaleidoscope.
    """

    # ensure appropriate columns exist in dataframe
    headers = ["FOLDER", "IN FILE", "CHANNEL", "OFFSET", "DURATION", "MANUAL ID"]
    assert set(headers).issubset(df.columns)

    # get required headers only
    return pd.concat(
        [
            df["FOLDER"].str.rstrip("/\\"),
            df["IN FILE"],
            df["CHANNEL"],
            df["OFFSET"],
            df["DURATION"],
            df["MANUAL ID"],
        ],
        axis=1,
        keys=headers,
    )
