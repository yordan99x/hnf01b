import pandas as pd

from config import *


def load_data():

    df = pd.read_excel(INPUT_FILE)

    df.columns = df.columns.str.strip()

    df = df[
        [
            PART_COLUMN,
            CALL_COLUMN,
            PRICE_COLUMN
        ]
    ].copy()

    df[CALL_COLUMN] = (
        pd.to_numeric(df[CALL_COLUMN], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    df[PRICE_COLUMN] = (
        pd.to_numeric(df[PRICE_COLUMN], errors="coerce")
        .fillna(0)
    )

    return df