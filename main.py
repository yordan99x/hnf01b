# %% 
# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

import time
import sys
import requests
import logging
import os
import json

from IPython.display import display
from urllib.parse import quote
from dotenv import load_dotenv
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

# %% 
# Configurations


# create folder logs/forecast.log if not exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Set Logging
logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.FileHandler("logs/forecast.log"), logging.StreamHandler()],
)
logging.info("="*40)
logging.info("BEGIN PYTHON FORECAST PROGRAM FOR SPAREPARTS")


# %% 
# API Call

# Retrive data from API
#logging.info('BEGIN Retrieving API')

#max_retries=8
#delay=2

# Initialize Start and End Date
#start_date = (datetime.today().replace(day=1) - relativedelta(months=16)).strftime("%d-%m-%Y") 
#end_date = (datetime.today().replace(day=1) - relativedelta(months=1)).strftime("%d-%m-%Y")  

#logging.info(f"API Data From Start Date: {start_date} to End Date: {end_date}")

#load_dotenv()
#internal_key = os.getenv("INTERNAL_KEY")
#base_url = os.getenv("BASE_URL")

#params = {
#    "start-date": start_date,
#    "end-date": end_date,
#    "exclude-older": start_date,
#    "branch": "",
#    "agency": "",
#    "partno": "LF  670"
#}

#headers = {
#    "x-api-key": internal_key
#}

#url = base_url + "/bckground/precalc/get-demand-call?" + "&".join(
#    f"{quote(str(k))}={quote(str(v))}" for k, v in params.items()
#)

#for attempt in range(1, max_retries + 1):
#    try:
#        response = requests.get(url, headers=headers)
#        response.raise_for_status()
#        data = response.json()
#        if 'data' in data and 'data-count' in data:
#            logging.info(str(data['month-count']) + " Month Data Retrived")
#            logging.info(str(data['data-count']) + " Data retrived from API")
#            df = pd.DataFrame(data['data'])
#            break
#        else:
#            logging.info("Error: Unexpected API response format")
#            break
#    except requests.RequestException as e:
#        logging.info(f"Attempt {attempt}: API request failed - {e}")
#        if attempt < max_retries:
#            time.sleep(delay * (2 ** (attempt - 1)))  # Exponential backoff
#        else:
#            logging.info("Max retries reached. Exiting.")
#            sys.exit(1)

#display(df.head())
#display(df.tail())


# %%


# # USING DUMMY 1 DATA INSTEAD OF API DATA
# with open("data/dummy2.json", "r") as f:
#     dummy_data = pd.read_json(f)
# df = pd.DataFrame(dummy_data["data"].tolist())
# display(df.head())


# %%


# df = pd.read_excel("data/dummy2.xlsx", sheet_name="noBTM", skiprows=4, usecols="A,B,C,M:AB")
df = pd.read_excel("data/agc 23 aug 25.xlsx", sheet_name="NoBTM", skiprows=4)
df = df.rename(columns={"Brc": "branch", "Agc": "agency", "P/N": "partno"})
# # Combine D-1 to D-16 columns into a single 'd' column as an array
d_cols = [f"D-{i}" for i in range(1, 17)]
df["d"] = df[d_cols].values.tolist()
df = df.drop(columns=d_cols)
#display(df.head(10))


# %%


# Contruct All Branch Data and Concat It To DF
logging.info("BEGIN Constructing All Branch Data and Combine It to DF")

# uppercase partno for consistency
df["partno"] = df["partno"].str.upper()

# create new df containing sum of demand for each agency and partno
df_all = df.groupby(["agency", "partno"], as_index=False)["d"].apply(
    lambda x: np.sum(np.array(x.tolist()), axis=0).tolist()
)

# insert a new column "branch" with value "ALL" and add to last data
df_all.insert(0, "branch", "ALL")
df = pd.concat([df, df_all], ignore_index=True)

logging.info(f"All Branch Data Constructed And Merged With DF With Total Data {len(df)}")

# display(df.head())
# display(df.tail())


# %%


logging.info("BEGIN Mean, Std, UB Calculation, and Construct Clipping Data")

# Get mean and standard deviation of 12 periods before the last one
df["d"] = df["d"].apply(lambda x: x if isinstance(x, list) else [])  # Ensure d is a list
df['mean_12'] = df['d'].apply(lambda x: np.mean(x[-13:-1]))  # Use 12 periods before the last one
df['std_12'] = df['d'].apply(lambda x: np.std(x[-13:-1]))    # Use 12 periods before the last one

# Get upper bound from mean and std
df['ub'] = df['mean_12'] + 1.5 * df['std_12']

# Limit the original df to upper bound (using the 12 periods before the last one)
df['clipped_d'] = df.apply( lambda row: np.clip(row['d'][-13:-1], 0, row['ub']).tolist(), axis=1)

# Display the updated DataFrame
# display(df.tail())


# %%


logging.info("BEGIN Moving Average Calculation")

# Calculate Simple Moving Average
df['clipped_d_15'] = df.apply(lambda row: np.clip(row['d'][:15], 0, row['ub']).tolist(), axis=1)

# Function to compute SMA forecasts for D-13 to D-1 using 3-point averages
def sma_forecast(data):
    sma_values = []
    for i in range(13):  # We want 13 forecast points: D-13 to D-1
        window = data[i:i+3]
        forecast = np.mean(window)  # Equal weights
        sma_values.append(forecast)
    return sma_values

# Apply SMA forecasting logic
df['ma'] = df['clipped_d_15'].apply(sma_forecast)

# Extract the last forecast (for D-1)
df['ma_result'] = df['ma'].apply(lambda x: x[-1])

# display(df.tail())


# %%


logging.info("BEGIN Weighted Moving Average Calculation")

# Function to compute WMA forecasts for D-13 to D-1
def wma_forecast_with_weights(data, weights):
    wma_values = []
    for i in range(13):  # Forecasting D-13 to D-1 using D-16 to D-2
        window = data[i:i+3]
        forecast = np.sum(np.array(window) * weights) / sum(weights)
        wma_values.append(forecast)
    return wma_values

# Define step size
step = 0.05

# Initialize columns to store best weights, forecasts, and WMA results
df['wma_best_w1'] = np.nan
df['wma_best_w2'] = np.nan
df['wma_best_w3'] = np.nan
df['wma_result'] = np.nan
df['wma_forecast'] = df.apply(lambda _: [], axis=1)  # Initialize as empty lists

# Optimize weights for each row
for idx, row in df.iterrows():
    best_rmse = float('inf')
    best_weights = (0.15, 0.25, 0.6)  # Initial weight assumption
    best_forecast = None
    best_full_forecast = None  # Store full forecast array

    # Iterate over valid w1 values
    for w1 in np.round(np.arange(0.15, 0.81, step), 2):  # w1 ≥ 0.15
        for w2 in np.round(np.arange(0.25, 0.86 - w1, step), 2):  # w2 ≥ 0.25 and w1 + w2 ≤ 0.85
            w3 = 1 - (w1 + w2)  # Ensure sum is exactly 1

            # Ensure w3 > w2 > w1
            if w3 > w2 > w1:
                weights = (w1, w2, w3)

                # Compute WMA forecast for this row
                wma_forecast = wma_forecast_with_weights(row['clipped_d_15'], weights)

                # Extract the D-1 prediction (last forecast)
                wma_result = wma_forecast[-1]

                # Extract actual last value of 'd' (D-1)
                d_last = row['d'][-1]

                # Compute RMSE for this row
                rmse = np.sqrt((d_last - wma_result) ** 2)

                # Store best weights if RMSE improves
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = weights
                    best_forecast = wma_result
                    best_full_forecast = wma_forecast  # Store full forecast

    # Store the best weights and forecast for this row
    df.at[idx, 'wma_best_w1'] = best_weights[0]
    df.at[idx, 'wma_best_w2'] = best_weights[1]
    df.at[idx, 'wma_best_w3'] = best_weights[2]
    df.at[idx, 'wma_result'] = best_forecast
    df.at[idx, 'wma_forecast'] = best_full_forecast  # Store full WMA forecast
    
# display(df.tail())


# %%


logging.info("BEGIN Exponential Weighted Moving Average Calculation")

alpha_ewma = 0.4
def custom_exponential_weighted_moving_average(values, alpha=alpha_ewma):
    ewma_values = [values[0]]  # Start with the first value

    # Apply EWMA formula up to D-2 (i.e., index 11 if length = 12)
    for t in range(1, len(values)):
        if np.isnan(values[t]):
            ewma_t = alpha * 0 + (1 - alpha) * ewma_values[-1]
        else:
            ewma_t = alpha * values[t] + (1 - alpha) * ewma_values[-1]
        ewma_values.append(ewma_t)

    return ewma_values  # This gives you EWMA from D-13 to D-2


def ewma_forecast(data, alpha=alpha_ewma):
    # Calculate EWMA up to D-2
    ewma_up_to_d2 = custom_exponential_weighted_moving_average(data, alpha)

    # Forecast D-1 as same as EWMA at D-2
    ewma_d1 = ewma_up_to_d2[-1]

    # Append D-1 forecast to the EWMA list
    ewma_with_d1 = ewma_up_to_d2 + [ewma_d1]

    # Return full EWMA list (D-13 to D-1) and D-1 forecast
    return ewma_with_d1, ewma_d1

df['ewma'], df['ewma_result'] = zip(*df['clipped_d'].apply(lambda x: ewma_forecast(x[-12:], alpha_ewma)))
# display(df.tail())


# %%


logging.info("BEGIN Linear Reggression Calculation")

#LINEAR REGRESSION
#  Calculate Linear Regression
def lr(x):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)
    model =  LinearRegression()
    model.fit(df[['x']], df['y'])
    df.loc[len(df), 'x'] = len(df) + 1
    return model.predict(df[['x']])

df['lr'] = df['clipped_d'].apply(lambda x: lr(x).tolist())
df['lr_result'] = df['lr'].apply(lambda x: x[-1:])
# display(df.tail())


# %%


logging.info("BEGIN Polynomial Reggression Calculation")

#POLYNOMIAL 2ND AND 3RD
# Calculate Polynomial Regression
def pr(x, pr_degree):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)

    X = df[['x']]  # Independent variable (reshape to 2D array)
    y = df['y']    # Dependent variable

    poly = PolynomialFeatures(degree=pr_degree)  # Create polynomial features
    X_poly = poly.fit_transform(X)  # Transform input features
    poly_model = LinearRegression()  # Initialize linear regression model
    poly_model.fit(X_poly, y)  # Fit polynomial model

    df.loc[len(df), 'x'] = len(df) + 1
    X_all_poly = poly.transform(df[['x']])
    return poly_model.predict(X_all_poly)  

df['pr2'] = df['clipped_d'].apply(lambda x: pr(x, 2).tolist())
df['pr2_result'] = df['pr2'].apply(lambda x: x[-1:])
df['pr3'] = df['clipped_d'].apply(lambda x: pr(x, 3).tolist())
df['pr3_result'] = df['pr3'].apply(lambda x: x[-1:])
# display(df.tail())


# %%


logging.info("BEGIN Simple Exponential Smoothing Calculation")

alpha_ses = 0.8  # ubah nilai alpha (semakin besar semakin berat ke data terbaru)

#SES
def ses(x, alpha = alpha_ses):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)
    df.loc[len(df), 'x'] = len(df) + 1

    new_data = SimpleExpSmoothing(df['y']).fit(smoothing_level=alpha, optimized=False).fittedvalues
    return new_data.tolist()

df['ses'] = df['clipped_d'].apply(lambda x: ses(x, alpha_ses))
df['ses_result'] = df['ses'].apply(lambda x: x[-1:])
# display(df.tail())


# %%


logging.info("BEGIN Double Exponential Smoothing Calculation")

# Define Grid Search Ranges
alpha_values = np.arange(0.1, 1.0, 0.1)  # Alpha range from 0.1 to 0.9
beta_values = np.arange(0.1, 1.0, 0.1)   # Beta range from 0.1 to 0.9

# Double Exponential Smoothing function
def des(x, alpha, beta):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)
    df.loc[len(df), 'x'] = len(df) + 1

    model = ExponentialSmoothing(df['y'], trend='add', seasonal=None)
    fitted_model = model.fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False)
    
    return fitted_model.fittedvalues.tolist()

# Function to find the best alpha & beta using Grid Search with RMSE
def optimize_des(series):
    best_alpha, best_beta, best_rmse = None, None, float("inf")

    for alpha, beta in product(alpha_values, beta_values):
        try:
            predictions = des(series, alpha, beta)
            rmse = np.sqrt(mean_squared_error(series, predictions[:len(series)]))  # RMSE calculation

            if rmse < best_rmse:
                best_alpha, best_beta, best_rmse = alpha, beta, rmse

        except Exception as e:
            continue  # Skip if model fails for some values

    return best_alpha, best_beta

# Apply Grid Search Optimization
df[['best_alpha', 'best_beta']] = df['clipped_d'].apply(lambda x: pd.Series(optimize_des(x)))
df['des'] = df['clipped_d'].apply(lambda x: des(x, *optimize_des(x)))
df['des_result'] = df['des'].apply(lambda x: x[-1:])  # Get last predicted value
# display(df.tail())



# %%


logging.info("BEGIN Metric Calculation")

# Calculate metrics including MASE, MAPE, and SMAPE
def metric(x):
    period_length = len(x['clipped_d'])
    df = pd.DataFrame()
    df['qty'] = x['clipped_d'][:period_length]  # Ground truth values
    
    # Naive forecast (previous period's value)
    df['naive'] = df['qty'].shift(1)

    models = ['ma', 'wma_forecast', 'ewma', 'lr', 'pr2', 'pr3', 'ses', 'des']
    for model in models:
        df[model] = x[model][:period_length]

    # Compute MASE scaling factor (denominator)
    naive_diff = np.abs(df['qty'].diff()).dropna()
    naive_mae = naive_diff.mean() if not naive_diff.empty else np.nan

    result = []
    for model in models:
        y_true = df['qty'].dropna()
        y_pred = df[model].dropna()
        y_naive = df['naive'].dropna()

        # Standard error metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Relative errors for MdRAE and GMRAE
        relative_errors = np.abs(y_true - y_pred) / np.abs(y_true - y_naive)
        relative_errors = relative_errors.replace([np.inf, -np.inf], np.nan).dropna()

        # Compute MASE
        mase = mae / naive_mae if naive_mae > 0 else np.nan

        # Compute MAPE (bounded between 0% - 100%)
        mape_values = np.abs((y_true - y_pred) / y_true)
        mape_values = mape_values.replace([np.inf, -np.inf], np.nan).dropna()
        mape = 100 * mape_values.mean() if not mape_values.empty else np.nan

        # Compute SMAPE (bounded between 0% - 100%)
        smape_values = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)  # Avoid div by zero
        smape_values = smape_values.replace([np.inf, -np.inf], np.nan).dropna()
        smape = 100 * smape_values.mean() if not smape_values.empty else np.nan

        result.append({
            'model': model, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MASE': mase, 'MAPE': mape, 'SMAPE': smape
        })

    metrics_df = pd.DataFrame(result)

    # Select the best model based on MAE
    best_model_row = metrics_df.loc[metrics_df['MAE'].idxmin()]
    best_model = best_model_row['model']

    return {'best_model': best_model, 'metrics': metrics_df.to_dict(orient='records')}

# Apply metric function
df['metric'] = df.apply(lambda x: metric(x), axis=1)

# Extract best model and metrics
df['best_model'] = df['metric'].apply(lambda x: x['best_model'])
df['metrics'] = df['metric'].apply(lambda x: x['metrics'])
df = df.drop(columns=['metric'])
# Define the number of months
num_months = 13

# Create new columns dynamically for each month
for i in range(num_months, 0, -1):
    df[f'pred_{i}'] = df.apply(
        lambda x: x[x['best_model']][num_months - i] if pd.notna(x['best_model']) else np.nan, axis=1
    )

# Extract R² of the best model into a new column
def get_best_model_r2(row):
    best_model = row['best_model']
    for m in row['metrics']:
        if m['model'] == best_model:
            return m['R2']
    return np.nan

df['best_r2'] = df.apply(get_best_model_r2, axis=1)
# Mark R2 performance
df['note'] = np.where(df['best_r2'] < 0.25, "R2 < 0.25", "Good")
# display(df.tail())



# %%


#kalkulasi semua model D-0
logging.info("BEGIN Data Selection Calculation")

# Select the best model for each row
df['mean_12_FD'] = df['d'].apply(lambda x: np.mean(x[-12:]))
df['std_12_FD'] = df['d'].apply(lambda x: np.std(x[-12:]))
df['ub_FD'] = df['mean_12_FD'] + 1.5 * df['std_12_FD']
df['clipped_d_FD'] = df.apply(lambda row: np.clip(row['d'][-12:], 0, row['ub_FD']).tolist(), axis=1)
# display(df.tail())


# %%


logging.info("BEGIN Moving Average Calculation")

# Calculate Simple Moving Average
df['clipped_d_15_FD'] = df.apply(lambda row: np.clip(row['d'][-15:], 0, row['ub_FD']).tolist(), axis=1)

# Function to compute SMA forecasts for D-13 to D-1 using 3-point averages
def sma_forecast(data):
    sma_values = []
    for i in range(13):  # We want 13 forecast points: D-13 to D-1
        window = data[i:i+3]
        forecast = np.mean(window)  # Equal weights
        sma_values.append(forecast)
    return sma_values

# Apply SMA forecasting logic
df['ma_FD'] = df['clipped_d_15_FD'].apply(sma_forecast)

# Extract the last forecast (for D-1)
df['ma_result_FD'] = df['ma_FD'].apply(lambda x: x[-1])
# display(df.tail())


# %%


logging.info("BEGIN Weighted Moving Average Calculation for FD")

# Function to compute WMA forecasts for D-13 to D-1
def wma_forecast_with_weights_FD(data, weights):
    wma_values_FD = []
    for i in range(13):  # Forecasting D-13 to D-1 using D-16 to D-2
        window_FD = data[i:i+3]
        forecast_FD = np.sum(np.array(window_FD) * weights) / sum(weights)
        wma_values_FD.append(forecast_FD)
    return wma_values_FD

# Define step size
step_FD = 0.05

# Initialize columns to store best weights, forecasts, and WMA results
df['wma_best_w1_FD'] = np.nan
df['wma_best_w2_FD'] = np.nan
df['wma_best_w3_FD'] = np.nan
df['wma_result_FD'] = np.nan
df['wma_forecast_FD'] = df.apply(lambda _: [], axis=1)  # Initialize as empty lists

# Optimize weights for each row
for idx, row in df.iterrows():
    best_rmse_FD = float('inf')
    best_weights_FD = (0.15, 0.25, 0.6)  # Initial weight assumption
    best_forecast_FD = None
    best_full_forecast_FD = None  # Store full forecast array

    # Iterate over valid w1_FD values
    for w1_FD in np.round(np.arange(0.15, 0.81, step_FD), 2):  # w1_FD ≥ 0.15
        for w2_FD in np.round(np.arange(0.25, 0.86 - w1_FD, step_FD), 2):  # w2_FD ≥ 0.25 and w1_FD + w2_FD ≤ 0.85
            w3_FD = 1 - (w1_FD + w2_FD)  # Ensure sum is exactly 1

            # Ensure w3_FD > w2_FD > w1_FD
            if w3_FD > w2_FD > w1_FD:
                weights_FD = (w1_FD, w2_FD, w3_FD)

                # Compute WMA forecast for this row
                wma_forecast_FD = wma_forecast_with_weights_FD(row['clipped_d_15_FD'], weights_FD)

                # Extract the D-1 prediction (last forecast)
                wma_result_FD = wma_forecast_FD[-1]

                # Extract actual last value of 'd' (D-1)
                d_last_FD = row['d'][-1]

                # Compute RMSE for this row
                rmse_FD = np.sqrt((d_last_FD - wma_result_FD) ** 2)

                # Store best weights if RMSE improves
                if rmse_FD < best_rmse_FD:
                    best_rmse_FD = rmse_FD
                    best_weights_FD = weights_FD
                    best_forecast_FD = wma_result_FD
                    best_full_forecast_FD = wma_forecast_FD  # Store full forecast

    # Store the best weights and forecast for this row
    df.at[idx, 'wma_best_w1_FD'] = best_weights_FD[0]
    df.at[idx, 'wma_best_w2_FD'] = best_weights_FD[1]
    df.at[idx, 'wma_best_w3_FD'] = best_weights_FD[2]
    df.at[idx, 'wma_result_FD'] = best_forecast_FD
    df.at[idx, 'wma_forecast_FD'] = best_full_forecast_FD  # Store full WMA forecast
    
# display(df.tail())  


# %%


# EWMA
alpha_ewma = 0.4

# Custom Exponential Weighted Moving Average Function
def custom_exponential_weighted_moving_average(values, alpha=alpha_ewma):
    ewma_values = [values[0]]  # Start with the first value (D-12)

    # Apply the EWMA formula for D-11 to D-1 (i.e., 11 more steps)
    for t in range(1, len(values)):  # len(values) = 12
        if np.isnan(values[t]):
            ewma_t = alpha * 0 + (1 - alpha) * ewma_values[-1]
        else:
            ewma_t = alpha * values[t] + (1 - alpha) * ewma_values[-1]
        ewma_values.append(ewma_t)
    
    return ewma_values  # EWMA from D-12 to D-1

# Forecast Function Using the Custom EWMA
def ewma_forecast(data, alpha=alpha_ewma):
    # Compute EWMA values for D-12 to D-1
    ewma_values = custom_exponential_weighted_moving_average(data, alpha)

    # Forecast D-0 as the same as EWMA at D-1
    forecast_d0 = ewma_values[-1]

    # Full series includes D-12 to D-0 (13 values total)
    ewma_full = ewma_values + [forecast_d0]

    return ewma_full, forecast_d0

# Apply the EWMA forecast to the dataset
df['ewma_FD'], df['ewma_result_FD'] = zip(*df['clipped_d_FD'].apply(lambda x: ewma_forecast(x[-12:], alpha_ewma)))

# display(df.tail())


# %%


#LR
def lr(x):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)
    model =  LinearRegression()
    model.fit(df[['x']], df['y'])
    df.loc[len(df), 'x'] = len(df) + 1
    return model.predict(df[['x']])
df['lr_FD'] = df['clipped_d_FD'].apply(lambda x: lr(x).tolist())
df['lr_result_FD'] = df['lr_FD'].apply(lambda x: x[-1:])

# display(df.tail())


# %%


#PR2&3
def pr(x, pr_degree):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)
    X = df[['x']]  # Independent variable (reshape to 2D array)
    y = df['y']    # Dependent variable
    poly = PolynomialFeatures(degree=pr_degree)  # Create polynomial features
    X_poly = poly.fit_transform(X)  # Transform input features
    poly_model = LinearRegression()  # Initialize linear regression model
    poly_model.fit(X_poly, y)  # Fit polynomial model
    df.loc[len(df), 'x'] = len(df) + 1
    X_all_poly = poly.transform(df[['x']])
    return poly_model.predict(X_all_poly)  
df['pr2_FD'] = df['clipped_d_FD'].apply(lambda x: pr(x, 2).tolist())
df['pr2_result_FD'] = df['pr2_FD'].apply(lambda x: x[-1:])
df['pr3_FD'] = df['clipped_d_FD'].apply(lambda x: pr(x, 3).tolist())
df['pr3_result_FD'] = df['pr3_FD'].apply(lambda x: x[-1:])

# display(df.tail())


# %%


#SES
def ses(x, alpha = alpha_ses):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)
    df.loc[len(df), 'x'] = len(df) + 1
    new_data = SimpleExpSmoothing(df['y']).fit(smoothing_level=alpha, optimized=False).fittedvalues
    return new_data.tolist()
df['ses_FD'] = df['clipped_d_FD'].apply(lambda x: ses(x, alpha_ses))
df['ses_result_FD'] = df['ses_FD'].apply(lambda x: x[-1:])

# display(df.tail())


# %%


#DES
# Define Grid Search Ranges
alpha_values_FD = np.arange(0.1, 1.0, 0.1)  # Alpha range from 0.1 to 0.9
beta_values_FD = np.arange(0.1, 1.0, 0.1)   # Beta range from 0.1 to 0.9

# Double Exponential Smoothing function for FD
def des_FD(x, alpha_FD, beta_FD):
    df_FD = pd.DataFrame()
    df_FD['y'] = x
    df_FD['x'] = range(1, len(df_FD) + 1)
    df_FD.loc[len(df_FD), 'x'] = len(df_FD) + 1

    model_FD = ExponentialSmoothing(df_FD['y'], trend='add', seasonal=None)
    fitted_model_FD = model_FD.fit(smoothing_level=alpha_FD, smoothing_trend=beta_FD, optimized=False)
    
    return fitted_model_FD.fittedvalues.tolist()

# Function to find the best alpha & beta using Grid Search with RMSE for FD
def optimize_des_FD(series_FD):
    best_alpha_FD, best_beta_FD, best_rmse_FD = None, None, float("inf")

    for alpha_FD, beta_FD in product(alpha_values_FD, beta_values_FD):
        try:
            predictions_FD = des_FD(series_FD, alpha_FD, beta_FD)
            rmse_FD = np.sqrt(mean_squared_error(series_FD, predictions_FD[:len(series_FD)]))  # RMSE calculation

            if rmse_FD < best_rmse_FD:
                best_alpha_FD, best_beta_FD, best_rmse_FD = alpha_FD, beta_FD, rmse_FD

        except Exception as e:
            continue  # Skip if model fails for some values

    return best_alpha_FD, best_beta_FD

# Apply Grid Search Optimization for FD
df[['best_alpha_FD', 'best_beta_FD']] = df['clipped_d_FD'].apply(lambda x: pd.Series(optimize_des_FD(x)))
df['des_FD'] = df['clipped_d_FD'].apply(lambda x: des_FD(x, *optimize_des_FD(x)))
df['des_result_FD'] = df['des_FD'].apply(lambda x: x[-1:])  # Get last predicted value

# display(df.tail())


# %%


logging.info("BEGIN Metric Calculation for _FD")

# Calculate metrics including MdRAE, GMRAE, MASE, MAPE, and SMAPE
def metric_FD(x):
    period_length = len(x['clipped_d_FD'])
    df = pd.DataFrame()
    df['qty'] = x['clipped_d_FD'][:period_length]  # Ground truth values

    # Naive forecast (previous period's value)
    df['naive'] = df['qty'].shift(1)

    models = ['ma_FD', 'wma_forecast_FD', 'ewma_FD', 'lr_FD', 'pr2_FD', 'pr3_FD', 'ses_FD', 'des_FD']
    for model in models:
        df[model] = x[model][:period_length]

    # Compute MASE scaling factor (denominator)
    naive_diff = np.abs(df['qty'].diff()).dropna()
    naive_mae = naive_diff.mean() if not naive_diff.empty else np.nan

    result = []
    for model in models:
        y_true = df['qty'].dropna()
        y_pred = df[model].dropna()
        y_naive = df['naive'].dropna()

        # Standard error metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Relative errors for MdRAE and GMRAE
        relative_errors = np.abs(y_true - y_pred) / np.abs(y_true - y_naive)
        relative_errors = relative_errors.replace([np.inf, -np.inf], np.nan).dropna()

        # Compute MASE
        mase = mae / naive_mae if naive_mae > 0 else np.nan

        # Compute MAPE (bounded between 0% - 100%)
        mape_values = np.abs((y_true - y_pred) / y_true)
        mape_values = mape_values.replace([np.inf, -np.inf], np.nan).dropna()
        mape = 100 * mape_values.mean() if not mape_values.empty else np.nan

        # Compute SMAPE (bounded between 0% - 100%)
        smape_values = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)  # Avoid div by zero
        smape_values = smape_values.replace([np.inf, -np.inf], np.nan).dropna()
        smape = 100 * smape_values.mean() if not smape_values.empty else np.nan

        result.append({
            'model': model, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MASE': mase, 'MAPE': mape, 'SMAPE': smape
        })

    return result  # Returning the metrics list

# Apply the metric function
df['metrics_FD'] = df.apply(lambda x: metric_FD(x), axis=1)

def get_best_r2_FD(row):
    best_model = row['best_model']
    metrics_fd = row.get('metrics_FD', [])
    for m in metrics_fd:
        if m['model'] == best_model + '_FD':
            return m['R2']
    return np.nan
df['best_r2_FD'] = df.apply(get_best_r2_FD, axis=1)
df['r2_status_FD'] = np.where(df['best_r2_FD'] < 0.25, "R2 < 0.25", "Good")

# display(df.tail())


# %%


def apply_best_model_forecast(row):
    best_model = row['best_model']
    if best_model == 'ma':
        return row['ma_result_FD']
    elif best_model == 'wma':
        return row['wma_result']
    elif best_model == 'ewma':
        return row['ewma_result_FD']
    elif best_model == 'lr':
        return row['lr_result_FD'][-1] if isinstance(row['lr_result_FD'], list) else row['lr_result_FD']
    elif best_model == 'pr2':
        return row['pr2_result_FD'][-1] if isinstance(row['pr2_result_FD'], list) else row['pr2_result_FD']
    elif best_model == 'pr3':
        return row['pr3_result_FD'][-1] if isinstance(row['pr3_result_FD'], list) else row['pr3_result_FD']
    elif best_model == 'ses':
        return row['ses_result_FD'][-1] if isinstance(row['ses_result_FD'], list) else row['ses_result_FD']
    elif best_model == 'des':
        return row['des_result_FD'][-1] if isinstance(row['des_result_FD'], list) else row['des_result_FD']
    else:
        return np.nan
    
df['FD_forecast'] = df.apply(apply_best_model_forecast, axis=1)
# Define the number of months (from 12 to 1, excluding 0)
num_months = 13  # Total months (D-12 to D-0), but we exclude D-0

# Map best model to the correct forecast series (excluding pred_0_FD)
def extract_forecast_values(row, month_idx):
    best_model = row['best_model']
    forecast_column = f"{best_model}_FD"  # Example: 'ma_result_FD', 'wma_result_FD'
    
    if forecast_column in row and isinstance(row[forecast_column], list):
        if len(row[forecast_column]) >= (13 - month_idx):
            return row[forecast_column][12 - month_idx]  # Extract the correct past forecast
    return np.nan  # Return NaN if data is missing or not a list

# Create columns for pred_12_FD to pred_1_FD
for i in range(num_months - 1, 0, -1):  # From 12 to 1
    df[f'pred_{i}_FD'] = df.apply(lambda x: extract_forecast_values(x, i), axis=1)

# Ensure FD_forecast contains only numeric values
df_all['FD_final'] = np.maximum(0, df_all['FD_forecast'].round().astype(int))

# Get all columns except the last four we want to reorder
columns_to_keep = [col for col in df.columns if col not in ['best_model', 'metrics', 'FD_forecast', 'FD_final']]

# Define the new order with the last four columns at the end
column_order = columns_to_keep + ['best_model', 'metrics', 'FD_forecast', 'FD_final']

# Reorder DataFrame
df = df[column_order]

# display(df.tail())


# %%


logging.info("Forecast Calculation Completed")


# %%


logging.info("Begin Creating Excel For DataFrame")

# if output folder not exist, create it
if not os.path.exists("output"):
    os.makedirs("output")

# Create Excel File, filename with date
filename = "output/forecast_" + time.strftime("%Y-%m-%d") + ".xlsx"

# Save DataFrame to Excel
df.to_excel(filename, index=False)

# Get the file size in MB
file_size = os.path.getsize(filename) / (1024 * 1024)

logging.info(f"Excel File Created: {filename}, Size: {file_size:.2f} MB")



# %%


# # Send Data Back To API
# logging.info("BEGIN Constructing Final Data and send it back to API")

# url = base_url + "/bckground/precalc/post-demand-call"

# # construct result with branch, agency, partno
# result = df[['branch', 'agency', 'partno', 'FD_final', 'std_12_FD', 'mean_12_FD', 'ub_FD']]

# # change column name
# result.columns = ['branch', 'agency', 'partno', 'fd', 'std', 'mean', 'ub']

# # result = df.drop('d', axis=1)
# result_json = result.to_dict(orient='records')

# # Save result_json to output folder as JSON file
# json_filename = "output/forecast_result_" + time.strftime("%Y-%m-%d") + ".json"
# with open(json_filename, "w", encoding="utf-8") as json_file:
#     json.dump(result_json, json_file, ensure_ascii=False, indent=2)
# logging.info(f"Result JSON saved to: {json_filename}")


# logging.info("Start Sending " + str(len(result)) + " Row To API")

# for attempt in range(1, max_retries + 1):
#     try:
#         response = requests.post(url, json=result_json, headers=headers)
#         response.raise_for_status() 
#         logging.info("Send API Complete")
#         logging.info(f"Status Code: {response.status_code}")

#         if response.status_code == 200:
#             logging.info(f"Response Body: {response.text}")
#         else:
#             logging.info("Send Failed")

#         break
#     except requests.RequestException as e:
#         logging.info(f"Attempt {attempt}: API request failed - {e} - {response.text}")
#         if attempt < max_retries:
#             time.sleep(delay * (2 ** (attempt - 1)))  # Exponential backoff
#         else:
#             logging.info("Max retries reached. Exiting.")
#             sys.exit(1)  # Stop execution after max retries


# %%
