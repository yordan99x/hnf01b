import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
import tempfile
import warnings

warnings.filterwarnings("ignore")

# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config("FPFL by Detail", layout="wide")
st.title("📊 FPFL by Detail Processor")

# =========================
# HIDDEN PM FILE (SERVER SIDE)
# =========================
PM_FILE = "pm_master.xlsx"  # <-- your AGC file (NOT visible to users)

@st.cache_data
def load_pm():
    return pd.read_excel(
        PM_FILE,
        sheet_name="no BTM",
        skiprows=4
    )

df_fd_master = load_pm()

# =========================
# USER INPUT
# =========================
uploaded_fpfl = st.file_uploader(
    "📂 Upload FPFL file",
    type=["xlsx"]
)

branch = st.number_input(
    "🏢 Target Branch",
    value=27,
    step=1
)

run = st.button("🚀 RUN PROCESS")

# =========================
# MAIN PROCESS FUNCTION
# =========================
def run_process_streamlit(df_fd, df1, TARGET_BRANCH):

    # === YOUR ORIGINAL LOGIC STARTS HERE ===
    # I ONLY removed Drive paths and widget calls

    df = df_fd.copy()
    df_fd.columns = df_fd.columns.str.lower()

    demand_columns = sorted(
        [col for col in df_fd.columns if col.startswith("d-")],
        key=lambda x: int(x.split("-")[1]),
        reverse=True
    )

    df_fd["p/n"] = df_fd["p/n"].str.upper()
    df_fd_branch = df_fd[df_fd["brc"] == TARGET_BRANCH]

    df_all = (
        df_fd_branch
        .groupby(["agc", "p/n"], as_index=False)[demand_columns]
        .sum()
    )

    df_all["d"] = df_all[demand_columns].values.tolist()
    df_all = df_all[["agc", "p/n", "d"]]
    df_all.insert(0, "branch", TARGET_BRANCH)

    df_fd = pd.concat([df_fd, df_all], ignore_index=True)
    df_all["d"] = df_all["d"].apply(lambda x: x if isinstance(x, list) else [])
    df_all['mean_12'] = df_all['d'].apply(lambda x: np.mean(x[-13:-1]))
    df_all['std_12'] = df_all['d'].apply(lambda x: np.std(x[-13:-1]))
    df_all['ub'] = df_all['mean_12'] + 1.5 * df_all['std_12']
    df_all['clipped_d'] = df_all.apply(lambda row: np.clip(row['d'][-13:-1], 0, row['ub']).tolist(), axis=1)
    df_all['clipped_d_15'] = df_all.apply(lambda row: np.clip(row['d'][:15], 0, row['ub']).tolist(), axis=1)
    def sma_forecast(data):
        sma_values = []
        for i in range(13):
            window = data[i:i+3]
            forecast = np.mean(window)
            sma_values.append(forecast)
        return sma_values
    df_all['ma'] = df_all['clipped_d_15'].apply(sma_forecast)
    df_all['ma_result'] = df_all['ma'].apply(lambda x: x[-1])
    def wma_forecast_with_weights(data, weights):
        wma_values = []
        for i in range(13):
            window = data[i:i+3]
            forecast = np.sum(np.array(window) * weights) / sum(weights)
            wma_values.append(forecast)
        return wma_values
    step = 0.05
    df_all['wma_best_w1'] = np.nan
    df_all['wma_best_w2'] = np.nan
    df_all['wma_best_w3'] = np.nan
    df_all['wma_result'] = np.nan
    df_all['wma_forecast'] = df_all.apply(lambda _: [], axis=1)
    for idx, row in df_all.iterrows():
        best_rmse = float('inf')
        best_weights = (0.15, 0.25, 0.6)
        best_forecast = None
        best_full_forecast = None
        for w1 in np.round(np.arange(0.15, 0.81, step), 2):
            for w2 in np.round(np.arange(0.25, 0.86 - w1, step), 2):
                w3 = 1 - (w1 + w2)
                if w3 > w2 > w1:
                    weights = (w1, w2, w3)
                    wma_forecast = wma_forecast_with_weights(row['clipped_d_15'], weights)
                    wma_result = wma_forecast[-1]
                    d_last = row['d'][-1]
                    rmse = np.sqrt((d_last - wma_result) ** 2)
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_weights = weights
                        best_forecast = wma_result
                        best_full_forecast = wma_forecast
        df_all.at[idx, 'wma_best_w1'] = best_weights[0]
        df_all.at[idx, 'wma_best_w2'] = best_weights[1]
        df_all.at[idx, 'wma_best_w3'] = best_weights[2]
        df_all.at[idx, 'wma_result'] = best_forecast
        df_all.at[idx, 'wma_forecast'] = best_full_forecast
    alpha_ewma = 0.4
    def custom_exponential_weighted_moving_average(values, alpha=alpha_ewma):
        ewma_values = [values[0]]
        for t in range(1, len(values)):
            if np.isnan(values[t]):
                ewma_t = alpha * 0 + (1 - alpha) * ewma_values[-1]
            else:
                ewma_t = alpha * values[t] + (1 - alpha) * ewma_values[-1]
            ewma_values.append(ewma_t)
        return ewma_values
    def ewma_forecast(data, alpha=alpha_ewma):
        ewma_up_to_d2 = custom_exponential_weighted_moving_average(data, alpha)
        ewma_d1 = ewma_up_to_d2[-1]
        ewma_with_d1 = ewma_up_to_d2 + [ewma_d1]
        return ewma_with_d1, ewma_d1
    df_all['ewma'], df_all['ewma_result'] = zip(*df_all['clipped_d'].apply(lambda x: ewma_forecast(x[-12:], alpha_ewma)))
    def lr(x):
        df_all = pd.DataFrame()
        df_all['y'] = x
        df_all['x'] = range(1, len(df_all) + 1)
        model =  LinearRegression()
        model.fit(df_all[['x']], df_all['y'])
        df_all.loc[len(df_all), 'x'] = len(df_all) + 1
        return model.predict(df_all[['x']])
    df_all['lr'] = df_all['clipped_d'].apply(lambda x: lr(x).tolist())
    df_all['lr_result'] = df_all['lr'].apply(lambda x: x[-1:])
    def pr(x, pr_degree):
        df_all = pd.DataFrame()
        df_all['y'] = x
        df_all['x'] = range(1, len(df_all) + 1)
        X = df_all[['x']]
        y = df_all['y']
        poly = PolynomialFeatures(degree=pr_degree)
        X_poly = poly.fit_transform(X)
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y)
        df_all.loc[len(df_all), 'x'] = len(df_all) + 1
        X_all_poly = poly.transform(df_all[['x']])
        return poly_model.predict(X_all_poly)
    df_all['pr2'] = df_all['clipped_d'].apply(lambda x: pr(x, 2).tolist())
    df_all['pr2_result'] = df_all['pr2'].apply(lambda x: x[-1:])
    df_all['pr3'] = df_all['clipped_d'].apply(lambda x: pr(x, 3).tolist())
    df_all['pr3_result'] = df_all['pr3'].apply(lambda x: x[-1:])
    alpha_ses = 0.8
    def ses(x, alpha = alpha_ses):
        df_all = pd.DataFrame()
        df_all['y'] = x
        df_all['x'] = range(1, len(df_all) + 1)
        df_all.loc[len(df_all), 'x'] = len(df_all) + 1
        new_data = SimpleExpSmoothing(df_all['y']).fit(smoothing_level=alpha, optimized=False).fittedvalues
        return new_data.tolist()
    df_all['ses'] = df_all['clipped_d'].apply(lambda x: ses(x, alpha_ses))
    df_all['ses_result'] = df_all['ses'].apply(lambda x: x[-1:])
    ALPHA = 0.1
    BETA = 0.1
    def des(x, alpha=ALPHA, beta=BETA):
        df_tmp = pd.DataFrame()
        df_tmp['y'] = x
        df_tmp['x'] = range(1, len(df_tmp) + 1)
        df_tmp.loc[len(df_tmp), 'x'] = len(df_tmp) + 1
        model = ExponentialSmoothing(
            df_tmp['y'],
            trend='add',
            seasonal=None
        )
        fitted_model = model.fit(
            smoothing_level=alpha,
            smoothing_trend=beta,
            optimized=False
        )
        return fitted_model.fittedvalues.tolist()
    df_all['des'] = df_all['clipped_d'].apply(lambda x: des(x))
    df_all['des_result'] = df_all['des'].apply(lambda x: x[-1])
    def metric(x):
        period_length = len(x['clipped_d'])
        df_all = pd.DataFrame()
        df_all['qty'] = x['clipped_d'][:period_length]
        df_all['naive'] = df_all['qty'].shift(1)
        models = ['ma', 'wma_forecast', 'ewma', 'lr', 'pr2', 'pr3', 'ses', 'des']
        for model in models:
            df_all[model] = x[model][:period_length]
        naive_diff = np.abs(df_all['qty'].diff()).dropna()
        naive_mae = naive_diff.mean() if not naive_diff.empty else np.nan
        result = []
        for model in models:
            y_true = df_all['qty'].dropna()
            y_pred = df_all[model].dropna()
            y_naive = df_all['naive'].dropna()
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            relative_errors = np.abs(y_true - y_pred) / np.abs(y_true - y_naive)
            relative_errors = relative_errors.replace([np.inf, -np.inf], np.nan).dropna()
            if not relative_errors.empty:
                mdrae = np.median(relative_errors)
                gmrae = np.exp(np.mean(np.log(relative_errors)))
            else:
                mdrae, gmrae = np.nan, np.nan
            mase = mae / naive_mae if naive_mae > 0 else np.nan
            mape_values = np.abs((y_true - y_pred) / y_true)
            mape_values = mape_values.replace([np.inf, -np.inf], np.nan).dropna()
            mape = 100 * mape_values.mean() if not mape_values.empty else np.nan
            smape_values = np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
            smape_values = smape_values.replace([np.inf, -np.inf], np.nan).dropna()
            smape = 100 * smape_values.mean() if not smape_values.empty else np.nan
            result.append({
                'model': model, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
                'MdRAE': mdrae, 'GMRAE': gmrae, 'MASE': mase, 'MAPE': mape, 'SMAPE': smape
            })
        metrics_df_all = pd.DataFrame(result)
        best_model_row = metrics_df_all.loc[metrics_df_all['MAE'].idxmin()]
        best_model = best_model_row['model']
        return {'best_model': best_model, 'metrics': metrics_df_all.to_dict(orient='records')}
    df_all['metric'] = df_all.apply(lambda x: metric(x), axis=1)
    df_all['best_model'] = df_all['metric'].apply(lambda x: x['best_model'])
    df_all['metrics'] = df_all['metric'].apply(lambda x: x['metrics'])
    df_all = df_all.drop(columns=['metric'])
    num_months = 13
    for i in range(num_months, 0, -1):
        df_all[f'pred_{i}'] = df_all.apply(
            lambda x: x[x['best_model']][num_months - i] if pd.notna(x['best_model']) else np.nan, axis=1
        )
    def get_best_model_r2(row):
        best_model = row['best_model']
        for m in row['metrics']:
            if m['model'] == best_model:
                return m['R2']
        return np.nan
    df_all['best_r2'] = df_all.apply(get_best_model_r2, axis=1)
    df_all['note'] = np.where(df_all['best_r2'] < 0.25, "R2 < 0.25", "Good")
    #kalkulasi semua model D-0
    # Select the best model for each row
    df_all['mean_12_FD'] = df_all['d'].apply(lambda x: np.mean(x[-12:]))
    df_all['std_12_FD'] = df_all['d'].apply(lambda x: np.std(x[-12:]))
    df_all['ub_FD'] = df_all['mean_12_FD'] + 1.5 * df_all['std_12_FD']
    df_all['clipped_d_FD'] = df_all.apply(lambda row: np.clip(row['d'][-12:], 0, row['ub_FD']).tolist(), axis=1)
    # Calculate Simple Moving Average
    df_all['clipped_d_15_FD'] = df_all.apply(lambda row: np.clip(row['d'][-15:], 0, row['ub_FD']).tolist(), axis=1)

    # Function to compute SMA forecasts for D-13 to D-1 using 3-point averages
    def sma_forecast(data):
        sma_values = []
        for i in range(13):  # We want 13 forecast points: D-13 to D-1
            window = data[i:i+3]
            forecast = np.mean(window)  # Equal weights
            sma_values.append(forecast)
        return sma_values
        # Apply SMA forecasting logic
    df_all['ma_FD'] = df_all['clipped_d_15_FD'].apply(sma_forecast)
        # Extract the last forecast (for D-1)
    df_all['ma_result_FD'] = df_all['ma_FD'].apply(lambda x: x[-1])
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
    df_all['wma_best_w1_FD'] = np.nan
    df_all['wma_best_w2_FD'] = np.nan
    df_all['wma_best_w3_FD'] = np.nan
    df_all['wma_result_FD'] = np.nan
    df_all['wma_forecast_FD'] = df_all.apply(lambda _: [], axis=1)  # Initialize as empty lists

        # Optimize weights for each row
    for idx, row in df_all.iterrows():
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
        df_all.at[idx, 'wma_best_w1_FD'] = best_weights_FD[0]
        df_all.at[idx, 'wma_best_w2_FD'] = best_weights_FD[1]
        df_all.at[idx, 'wma_best_w3_FD'] = best_weights_FD[2]
        df_all.at[idx, 'wma_result_FD'] = best_forecast_FD
        df_all.at[idx, 'wma_forecast_FD'] = best_full_forecast_FD  # Store full WMA forecast
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
    df_all['ewma_FD'], df_all['ewma_result_FD'] = zip(*df_all['clipped_d_FD'].apply(lambda x: ewma_forecast(x[-12:], alpha_ewma)))
        #LR
    def lr(x):
        df_all = pd.DataFrame()
        df_all['y'] = x
        df_all['x'] = range(1, len(df_all) + 1)
        model =  LinearRegression()
        model.fit(df_all[['x']], df_all['y'])
        df_all.loc[len(df_all), 'x'] = len(df_all) + 1
        return model.predict(df_all[['x']])
    df_all['lr_FD'] = df_all['clipped_d_FD'].apply(lambda x: lr(x).tolist())
    df_all['lr_result_FD'] = df_all['lr_FD'].apply(lambda x: x[-1:])
    #PR2&3
    def pr(x, pr_degree):
        df_all = pd.DataFrame()
        df_all['y'] = x
        df_all['x'] = range(1, len(df_all) + 1)
        X = df_all[['x']]  # Independent variable (reshape to 2D array)
        y = df_all['y']    # Dependent variable
        poly = PolynomialFeatures(degree=pr_degree)  # Create polynomial features
        X_poly = poly.fit_transform(X)  # Transform input features
        poly_model = LinearRegression()  # Initialize linear regression model
        poly_model.fit(X_poly, y)  # Fit polynomial model
        df_all.loc[len(df_all), 'x'] = len(df_all) + 1
        X_all_poly = poly.transform(df_all[['x']])
        return poly_model.predict(X_all_poly)
    df_all['pr2_FD'] = df_all['clipped_d_FD'].apply(lambda x: pr(x, 2).tolist())
    df_all['pr2_result_FD'] = df_all['pr2_FD'].apply(lambda x: x[-1:])
    df_all['pr3_FD'] = df_all['clipped_d_FD'].apply(lambda x: pr(x, 3).tolist())
    df_all['pr3_result_FD'] = df_all['pr3_FD'].apply(lambda x: x[-1:])
    #SES
    def ses(x, alpha = alpha_ses):
        df_all = pd.DataFrame()
        df_all['y'] = x
        df_all['x'] = range(1, len(df_all) + 1)
        df_all.loc[len(df_all), 'x'] = len(df_all) + 1
        new_data = SimpleExpSmoothing(df_all['y']).fit(smoothing_level=alpha, optimized=False).fittedvalues
        return new_data.tolist()
    df_all['ses_FD'] = df_all['clipped_d_FD'].apply(lambda x: ses(x, alpha_ses))
    df_all['ses_result_FD'] = df_all['ses_FD'].apply(lambda x: x[-1:])
        # DES - FD
    ALPHA_FD = 0.1
    BETA_FD = 0.1

        # Double Exponential Smoothing function for FD
    def des_FD(x, alpha=ALPHA_FD, beta=BETA_FD):
        df_tmp = pd.DataFrame()
        df_tmp['y'] = x
        df_tmp['x'] = range(1, len(df_tmp) + 1)
        df_tmp.loc[len(df_tmp), 'x'] = len(df_tmp) + 1

        model = ExponentialSmoothing(
            df_tmp['y'],
            trend='add',
            seasonal=None
        )

        fitted_model = model.fit(
            smoothing_level=alpha,
            smoothing_trend=beta,
            optimized=False
        )

        return fitted_model.fittedvalues.tolist()
    df_all['des_FD'] = df_all['clipped_d_FD'].apply(lambda x: des_FD(x))
    df_all['des_result_FD'] = df_all['des_FD'].apply(lambda x: x[-1])
    # Calculate metrics including MdRAE, GMRAE, MASE, MAPE, and SMAPE
    def metric_FD(x):
        period_length = len(x['clipped_d_FD'])
        df_all = pd.DataFrame()
        df_all['qty'] = x['clipped_d_FD'][:period_length]  # Ground truth values

            # Naive forecast (previous period's value)
        df_all['naive'] = df_all['qty'].shift(1)

        models = ['ma_FD', 'wma_forecast_FD', 'ewma_FD', 'lr_FD', 'pr2_FD', 'pr3_FD', 'ses_FD', 'des_FD']
        for model in models:
            df_all[model] = x[model][:period_length]

            # Compute MASE scaling factor (denominator)
        naive_diff = np.abs(df_all['qty'].diff()).dropna()
        naive_mae = naive_diff.mean() if not naive_diff.empty else np.nan

        result = []
        for model in models:
            y_true = df_all['qty'].dropna()
            y_pred = df_all[model].dropna()
            y_naive = df_all['naive'].dropna()

                # Standard error metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

                # Relative errors for MdRAE and GMRAE
            relative_errors = np.abs(y_true - y_pred) / np.abs(y_true - y_naive)
            relative_errors = relative_errors.replace([np.inf, -np.inf], np.nan).dropna()

                # Compute MdRAE and GMRAE
            if not relative_errors.empty:
                mdrae = np.median(relative_errors)
                gmrae = np.exp(np.mean(np.log(relative_errors)))
            else:
                mdrae, gmrae = np.nan, np.nan

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
                'model': model, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
                'MdRAE': mdrae, 'GMRAE': gmrae, 'MASE': mase, 'MAPE': mape, 'SMAPE': smape
            })

        return result  # Returning the metrics list

        # Apply the metric function
    df_all['metrics_FD'] = df_all.apply(lambda x: metric_FD(x), axis=1)

    def get_best_r2_FD(row):
        best_model = row['best_model']
        metrics_fd = row.get('metrics_FD', [])
        for m in metrics_fd:
            if m['model'] == best_model + '_FD':
                return m['R2']
        return np.nan
    df_all['best_r2_FD'] = df_all.apply(get_best_r2_FD, axis=1)
    df_all['r2_status_FD'] = np.where(df_all['best_r2_FD'] < 0.25, "R2 < 0.25", "Good")
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

    df_all['FD_forecast'] = df_all.apply(apply_best_model_forecast, axis=1)
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
        df_all[f'pred_{i}_FD'] = df_all.apply(lambda x: extract_forecast_values(x, i), axis=1)

        # Ensure FD_forecast contains only numeric values
    df_all['FD_final'] = np.maximum(0, df_all['FD_forecast'].round().astype(int))
        # Get all columns except the last four we want to reorder
    columns_to_keep = [col for col in df_all.columns if col not in ['best_model', 'metrics', 'FD_forecast', 'FD_final']]
        # Define the new order with the last four columns at the end
    column_order = columns_to_keep + ['best_model', 'metrics', 'FD_forecast', 'FD_final']
        # Reorder DataFrame
    df_all = df_all[column_order]
        # --- Normalize column names ---
    df.columns = df.columns.str.lower()
        # --- Normalize P/N ---
    df["p/n"] = df["p/n"].astype(str).str.upper()
        # --- Filter ONLY target branch ---
    df_branch = df[df["brc"] == TARGET_BRANCH]
        # --- Identify all C- columns ---
    c_columns = sorted(
        [col for col in df_branch.columns if isinstance(col, str) and col.startswith("c-")],
        key=lambda x: int(x.split("-")[1]),
        reverse=True
    )

        # Required columns
    oh_col = "oh"
    oo_col = "oo"
    dn_price_col = "dn price"
    agc_col = "agc"

        # --- Aggregation rules ---
    aggregation_dict = {col: "sum" for col in c_columns + [oh_col, oo_col]}
    aggregation_dict[dn_price_col] = "first"

        # --- GROUP BY P/N ---
    df_sum = (
        df_branch
        .groupby(["agc", "p/n"], as_index=False)
        .agg(aggregation_dict)
    )
        # --- Total Calls ---
    df_sum["Total Calls"] = df_sum[c_columns].sum(axis=1)

        # --- Drop individual C- columns ---
    df_sum = df_sum.drop(columns=c_columns)

        # --- Add identifiers ---
        # df_sum.insert(0, "Agc", "All")
    df_sum.insert(0, "Brc", TARGET_BRANCH)

        # --- Reorder columns ---
    df_sum = df_sum[
        ["Brc", "agc", "p/n", "Total Calls", "dn price", "oh", "oo"]
    ]
        # ===============================
        # Perhitungan RC (Rank Call)
        # ===============================

    df_sum = (
        df_sum
        .sort_values(
            by=["Total Calls", "p/n"],
            ascending=[False, True]
        )
        .reset_index(drop=True)
    )

        # Add cumulative sum
    df_sum["Accum."] = df_sum["Total Calls"].cumsum()

        # Percentage cumulative
    total_calls = df_sum["Accum."].iloc[-1]
    df_sum["%Accum."] = (df_sum["Accum."] / total_calls) * 100
    df_sum["%Accum."] = df_sum["%Accum."].round(2)

        # Klasifikasi RC (ABCD) with edge-case fix
    def assign_rc(pct, total_calls):
        if total_calls > 0 and pct >= 100:
            return "C"

        if 0 <= pct <= 50:
            return "A"
        elif 50 < pct <= 80:
            return "B"
        elif 80 < pct < 100:
            return "C"
        else:
            return "D"

    df_sum["RC"] = df_sum.apply(
        lambda row: assign_rc(row["%Accum."], row["Total Calls"]),
        axis=1
    )
        # ===============================
        # MM06 (per TARGET_BRANCH)
        # ===============================

    c06_cols = [f"c-{i}" for i in range(1, 7)]
    c06_cols = [col for col in c06_cols if col in df.columns]

    temp06 = (
        df[df["brc"] == TARGET_BRANCH]
        .groupby("p/n", as_index=False)[c06_cols]
        .sum()
    )

        # Count months with calls > 0
    temp06["MM06"] = temp06[c06_cols].gt(0).sum(axis=1)

    mm06_df = temp06[["p/n", "MM06"]]

    df_sum = df_sum.merge(mm06_df, on="p/n", how="left")

        # Insert after Total Calls
    insert_pos = df_sum.columns.get_loc("RC") + 1
    mm06_series = df_sum.pop("MM06")
    df_sum.insert(insert_pos, "MM06", mm06_series)
        # ===============================
        # MM12 (per TARGET_BRANCH)
        # ===============================

    c12_cols = [f"c-{i}" for i in range(1, 13)]
    c12_cols = [col for col in c12_cols if col in df.columns]

    temp12 = (
        df[df["brc"] == TARGET_BRANCH]
        .groupby("p/n", as_index=False)[c12_cols]
        .sum()
    )

        # Count months with calls > 0
    temp12["MM12"] = temp12[c12_cols].gt(0).sum(axis=1)

    mm12_df = temp12[["p/n", "MM12"]]

    df_sum = df_sum.merge(mm12_df, on="p/n", how="left")

        # Insert after MM06
    insert_pos = df_sum.columns.get_loc("MM06") + 1
    mm12_series = df_sum.pop("MM12")
    df_sum.insert(insert_pos, "MM12", mm12_series)
        # ===============================
        # Calls12 = Total Calls
        # ===============================

    df_sum["Calls12"] = df_sum["Total Calls"]

        # Insert Calls12 right after MM12
    insert_pos = df_sum.columns.get_loc("MM12") + 1
    calls12_series = df_sum.pop("Calls12")
    df_sum.insert(insert_pos, "Calls12", calls12_series)
    df_final = (
        df_all[["branch", "agc", "p/n", "FD_final"]]
        .merge(
            df_sum[["p/n", "RC", "MM06", "MM12", "Calls12"]],
            on="p/n",
            how="left"
        )
    )
        # Create the new column logic
    df1["Rtn=Ord"] = df1.apply(
        lambda x: "1" if x["RtnQty"] == x["Order"] else "0",
        axis=1
    )

        # Insert the column right after 'RtnQty'
    rtnqty_index = df1.columns.get_loc("RtnQty")
    df1.insert(rtnqty_index + 1, "Rtn=Ord", df1.pop("Rtn=Ord"))
        # Create the new column logic
    df1["Rtn+Cncl=Ord"] = df1.apply(
        lambda x: "1" if x["RtnQty"] + x["Cancl"] == x["Order"] else "0",
        axis=1
    )

        # Insert the column right after 'Rtn=Ord'
    rtnqty_index = df1.columns.get_loc("Rtn=Ord")
    df1.insert(rtnqty_index + 1, "Rtn+Cncl=Ord", df1.pop("Rtn+Cncl=Ord"))
        # Create the new column logic
    df1["Ord=0"] = df1.apply(
        lambda x: "1" if x["Order"] == 0 else "0",
        axis=1
    )

        # Insert the column right after 'Rtn=Ord'
    rtnqty_index = df1.columns.get_loc("Rtn+Cncl=Ord")
    df1.insert(rtnqty_index + 1, "Ord=0", df1.pop("Ord=0"))
        # Create the new column logic
    df1["Ord=Cancl+Supply+Rtn"] = df1.apply(
        lambda x: "1" if x["Order"] == x["Cancl"]+x["Supply"]+x["RtnQty"] else "0",
        axis=1
    )

        # Insert the column right after 'Rtn=Ord'
    rtnqty_index = df1.columns.get_loc("Ord=0")
    df1.insert(rtnqty_index + 1, "Ord=Cancl+Supply+Rtn", df1.pop("Ord=Cancl+Supply+Rtn"))
        # Create "Doc Type" from PSO No (5th–6th characters)
    df1["Doc Type"] = (
        df1["PSO No"]
        .astype(str)
        .str[4:6]   # Python is 0-based → 5th char = index 4
    )

    pso_index = df1.columns.get_loc("Ord=Cancl+Supply+Rtn")
    df1.insert(pso_index + 1, "Doc Type", df1.pop("Doc Type"))
        # --- Create Release column ---
    df1["Release"] = df1["PSO No"].astype(str).str[-1].apply(
        lambda x: "No" if x.isdigit() else "Release"
    )

        # Insert after "Doc Type"
    doc_index = df1.columns.get_loc("Doc Type")
    df1.insert(doc_index + 1, "Release", df1.pop("Release"))
        # --- Normalize part numbers (CRITICAL) ---
    df1["Partno"] = df1["Partno"].astype(str).str.strip().str.upper()
    df_final["p/n"] = df_final["p/n"].astype(str).str.strip().str.upper()

        # --- Select only needed columns from df2 ---
    df_final_merge = df_final[["p/n", "FD_final", "RC", "MM06","MM12","Calls12"]].drop_duplicates()

        # --- Merge ---
    df_fpfl = df1.merge(
        df_final_merge,
        left_on="Partno",
        right_on="p/n",
        how="left"
    )

        # --- Drop duplicate key column ---
    df_fpfl.drop(columns=["p/n"], inplace=True)

    # =========================
    # RETURN FINAL RESULT
    # =========================
    return df_fpfl


# =========================
# RUN BUTTON
# =========================
if run:
    if uploaded_fpfl is None:
        st.error("❌ Please upload FPFL file")
        st.stop()

    with st.spinner("⏳ Processing... please wait"):
        df_fpfl_input = pd.read_excel(uploaded_fpfl, skiprows=1)

        df_result = run_process_streamlit(
            df_fd_master,
            df_fpfl_input,
            branch
        )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            output_path = tmp.name

        df_result.to_excel(output_path, index=False)

    st.success("✅ PROCESS COMPLETED")

    with open(output_path, "rb") as f:
        st.download_button(
            "📥 Download Result",
            f,
            file_name=f"FPFL_by_Detail_Branch_{branch}.xlsx"
        )
