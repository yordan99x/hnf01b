from config import *


def categorize_call(call):

    if call == 0:
        return "C0"

    elif call == 1:
        return "C1"

    elif call <= 3:
        return "LA"

    elif call <= A_MAX:
        return "A"

    elif call <= B_MAX:
        return "B"

    elif call <= C_MAX:
        return "C"

    else:
        return "D"


def categorize_price(price):

    if price <= PRICE1_MAX:
        return "low"

    elif price <= PRICE2_MAX:
        return "medium"

    elif price <= PRICE3_MAX:
        return "high"

    else:
        return "very high"


def categorize_dataframe(df):

    df["Call Category"] = df[CALL_COLUMN].apply(categorize_call)

    df["Price Category"] = df[PRICE_COLUMN].apply(categorize_price)

    return df