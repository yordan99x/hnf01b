#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import time
import sys
import requests
import logging
import os

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from scipy import stats
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
import base64
from dotenv import load_dotenv


# In[2]:

# Get Environment Variables
load_dotenv()
CREDENTIAL = os.getenv('CREDENTIAL')

# Set Display Width Longer
pd.options.display.max_colwidth = 200  # 100 for long width

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


# In[6]:

# Begin API Authentication

logging.info('BEGIN API Authentication')
url = "http://localhost:8080/web/auth/login"
auth_string = CREDENTIAL
encoded_auth = base64.b64encode(auth_string.encode()).decode()
headers = {
    "Auth": f"Basic {encoded_auth}"
}

try:
    response = requests.post(url, headers=headers)
    response.raise_for_status()
    tokens = response.json()
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")
    if access_token and refresh_token:
        logging.info("Authentication successful. Tokens retrieved.")
    else:
        logging.info("Authentication failed. Tokens not found in response.")
        sys.exit(1)
except requests.RequestException as e:
    logging.info(f"API Authentication failed - {e}")
    sys.exit(1)


# Retrive data from API
logging.info('BEGIN Retrieving API')

max_retries=8
delay=2

# Initialize Start and End Date
start_date = (datetime.today().replace(day=1) - relativedelta(months=16)).strftime("%Y-%m-%d") 
end_date = (datetime.today().replace(day=1) - relativedelta(months=1)).strftime("%Y-%m-%d")  

logging.info(f"API Data From Start Date: {start_date} to End Date: {end_date}")

params = {
    "start-date": start_date,
    "end-date": end_date,
    "exclude-older": start_date,
    "branch": "",
    "agency": "",
    "partno": ""
}

headers = {
    "Auth": f"Bearer {access_token}"
}

url = "http://localhost:8080/web/gdmdcall"
    
for attempt in range(1, max_retries + 1):
    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        if 'data' in data and 'data-count' in data:
            logging.info(str(data['month-count']) + " Month Data Retrived")
            logging.info(str(data['data-count']) + " Data retrived from API")
            df = pd.DataFrame(data['data'])
            break
        else:
            logging.info("Error: Unexpected API response format")
            break
    except requests.RequestException as e:
        logging.info(f"Attempt {attempt}: API request failed - {e}")
        if attempt < max_retries:
            time.sleep(delay * (2 ** (attempt - 1)))  # Exponential backoff
        else:
            logging.info("Max retries reached. Exiting.")
            sys.exit(1)

# display(df.head())


# In[7]:


# Contruct All Branch Data and Concat It To DF
logging.info("BEGIN Constructing All Branch Data and Combine It to DF")

df_all = df.groupby(["agency", "partno"], as_index=False)["d"].apply(
    lambda x: np.sum(np.array(x.tolist()), axis=0).tolist()
)
df_all.insert(0, "branch", "ALL")
df = pd.concat([df, df_all], ignore_index=True)

logging.info(
    f"All Branch Data Constructed And Merged With DF With Total Data {len(df)}"
)


# In[8]:


# Calculate Forecast
logging.info("BEGIN Forecast Calculation")
# display(df)


# In[9]:


logging.info("BEGIN Mean, Std, UB Calculation, and Construct Clipping Data")

# Get mean and standard deviation of 12 periods before the last one
df['mean_12'] = df['d'].apply(lambda x: np.mean(x[-13:-1]))  # Use 12 periods before the last one
df['std_12'] = df['d'].apply(lambda x: np.std(x[-13:-1]))    # Use 12 periods before the last one

# Get upper bound from mean and std
df['ub'] = df['mean_12'] + 1.5 * df['std_12']

# Limit the original df to upper bound (using the 12 periods before the last one)
df['clipped_d'] = df.apply(lambda row: np.clip(row['d'][-13:-1], 0, row['ub']).tolist(), axis=1)

# Display the updated DataFrame
# display(df.head())


# In[10]:


logging.info("BEGIN Simple Moving Average Calculation")

# Calculate Simple Moving Average
df['ma'] = df['clipped_d'].apply(lambda x: pd.Series(x).rolling(window=len(x), min_periods=1).mean().tolist())
df['ma_result'] = df['ma'].apply(lambda x: x[-1:])

# Display the updated DataFrame
# display(df.head())


# In[11]:


logging.info("BEGIN Weighted Moving Average Calculation")

df['wma_clipped_d'] = df.apply(lambda row: np.clip(row['d'][-16:-1], 0, row['ub']).tolist(), axis=1)

def wma_forecast_with_weights(df, weights):
    wma_values = [None] * 3
    for i in range(3, len(df)):
        forecast = np.sum(np.array(df[i-3:i]) * weights) / sum(weights)
        wma_values.append(forecast)
    return wma_values

def generate_weights(step=0.05):
    weights = []
    for w1 in np.arange(0.01, 1, step):
        for w2 in np.arange(w1 + 0.01, 1 - w1, step):
            w3 = 1 - w1 - w2
            if w3 > w2 > w1 > 0 and abs(w1 + w2 + w3 - 1) < 1e-6:
                weights.append((w1, w2, w3))
    return weights

best_weights_list = []
best_maes = []

for row in df['wma_clipped_d']:
    best_mae = float('inf')
    best_weights = None
    for weights in generate_weights(step=0.05):
        wma_values = wma_forecast_with_weights(row, weights)
        mae = mean_absolute_error(row[-12:], wma_values[-12:])
        if mae < best_mae:
            best_mae = mae
            best_weights = weights
    best_weights_list.append(best_weights)
    best_maes.append(best_mae)

df['best_weights'] = best_weights_list
df['best_mae'] = best_maes

df['wma'], df['wma_result'] = zip(*df.apply(lambda row: (
    wma_forecast_with_weights(row['wma_clipped_d'], row['best_weights'])[3:][-12:],
    wma_forecast_with_weights(row['wma_clipped_d'], row['best_weights'])[-1:]
), axis=1))

# display(df)


# In[12]:


logging.info("BEGIN Exponential Weighted Moving Average Calculation")

# Calculate Exponential Weighted Moving Average (EWMA)
alpha_ewma = 0.4

def ewma(list, alpha = alpha_ewma):
    df = pd.DataFrame(list)
    df['ewma'] = df.ewm(alpha=alpha_ewma, adjust=False).mean()
    return df['ewma'].tolist()

def ewma_forecast(list, alpha):
    ewma_values = ewma(list, alpha)
    if len(ewma_values) > 0:
        # Prediction for the next period
        next_forecast = (1 - alpha) * ewma_values[-1]
    else:
        next_forecast = None
    return ewma_values, next_forecast

df['ewma'], df['ewma_result'] = zip(*df['clipped_d'].apply(lambda x: ewma_forecast(x, alpha_ewma)))

# display(df)


# In[13]:


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

df['lr'] = df['clipped_d'].apply(lambda x: lr(x))
df['lr_result'] = df['lr'].apply(lambda x: x[-1:])
# display(df)


# In[14]:


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

df['pr2'] = df['clipped_d'].apply(lambda x: pr(x, 2))
df['pr2_result'] = df['pr2'].apply(lambda x: x[-1:])
df['pr3'] = df['clipped_d'].apply(lambda x: pr(x, 3))
df['pr3_result'] = df['pr3'].apply(lambda x: x[-1:])
# display(df)


# In[15]:


logging.info("BEGIN Simple Exponential Smoothing Calculation")

alpha_ses = 0.65  # ubah nilai alpha (semakin besar semakin berat ke data terbaru)

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

# display(df)


# In[16]:


logging.info("BEGIN Double Exponential Smoothing Calculation")

beta_des = 0.45

#DES
def des(x, alpha = alpha_ses, beta = beta_des):
    df = pd.DataFrame()
    df['y'] = x
    df['x'] = range(1, len(df) + 1)
    df.loc[len(df), 'x'] = len(df) + 1

    new_data = ExponentialSmoothing(df['y'], trend='add', seasonal=None).fit(smoothing_level=alpha, smoothing_trend=beta, optimized=False).fittedvalues
    return new_data.tolist()

df['des'] = df['clipped_d'].apply(lambda x: des(x,alpha_ses, beta_des))
df['des_result'] = df['des'].apply(lambda x: x[-1:])
# display(df)


# In[17]:


logging.info("BEGIN Metric Calculation")

# Calculate metrics for each model
def metric(x):
    period_length = len(x['clipped_d'])
    df = pd.DataFrame()
    df['period'] = range(1, period_length + 1)
    df['qty'] = x['clipped_d'][:period_length]  # Ground truth values
    df['ma'] = x['ma'][:period_length]
    df['wma'] = x['wma'][:period_length]
    df['ewma'] = x['ewma'][:period_length]
    df['lr'] = x['lr'][:period_length]
    df['pr2'] = x['pr2'][:period_length]
    df['pr3'] = x['pr3'][:period_length]
    df['ses'] = x['ses'][:period_length]
    df['des'] = x['des'][:period_length]

    # Calculate metrics for each model
    result = []
    for model in df.columns[2:]:  # Loop through model columns (ma, ewma, etc.)
        rmse = np.sqrt(mean_squared_error(df['qty'], df[model]))  # Calculate RMSE
        r2 = r2_score(df['qty'], df[model])  # Calculate R²
        mae = mean_absolute_error(df['qty'], df[model])  # Calculate MAE
        result.append({'model': model, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
    
    # Convert result to a DataFrame
    metrics_df = pd.DataFrame(result)
    
    # Select the best model (e.g., based on RMSE)
    best_model_row = metrics_df.loc[metrics_df['MAE'].idxmin()]  # Row with the lowest RMSE
    best_model = best_model_row['model']
    
    # Add the best model and metrics to the result
    return {'best_model': best_model, 'metrics': metrics_df.to_dict(orient='records')}

# Apply the metric function
df['metric'] = df.apply(lambda x: metric(x), axis=1)

# Extract the best model and metrics for each row
df['best_model'] = df['metric'].apply(lambda x: x['best_model'])
df['metrics'] = df['metric'].apply(lambda x: x['metrics'])

# Display the DataFrame
# display(df[['wma']])
# display(df[['best_model', 'metrics']])


# In[18]:


logging.info("BEGIN Data Selection Calculation")

# Select the best model for each row
df['mean_12_FD'] = df['d'].apply(lambda x: np.mean(x[-12:]))
df['std_12_FD'] = df['d'].apply(lambda x: np.std(x[-12:]))

df['ub_FD'] = df['mean_12_FD'] + 1.5 * df['std_12_FD']

df['clipped_d_FD'] = df.apply(lambda row: np.clip(row['d'][-12:], 0, row['ub_FD']).tolist(), axis=1)
def apply_best_model_forecast(row):
    best_model = row['best_model']
    
    data = row['d'][-15:] if best_model == 'wma' else row['d'][-12:]
    
    ub = row['ub_FD']
    clipped_data = np.clip(data, 0, ub).tolist()
    # print(f"Clipped data for model {best_model}: {clipped_data}")
    
    if best_model == 'ma':
        ma_values = pd.Series(clipped_data).rolling(window=len(clipped_data), min_periods=1).mean().tolist()
        forecast = ma_values[-1]
        # print('ma')
    elif best_model == 'ewma':
        alpha = 0.4
        weights = np.array([(1 - alpha) ** i for i in range(len(clipped_data))][::-1])
        forecast = np.sum(weights * clipped_data) / np.sum(weights)
        # print('ewma')
    elif best_model == 'wma':
        weights = [0.2, 0.3, 0.5]
        if len(clipped_data) >= len(weights):
            forecast = np.sum(np.array(clipped_data[-3:]) * weights)
        else:
            forecast = np.nan
        # print('wma')
    elif best_model == 'lr':
        X = np.arange(len(clipped_data)).reshape(-1, 1)
        y = np.array(clipped_data)
        coef = np.polyfit(X.flatten(), y, 1)
        forecast = coef[0] * len(clipped_data) + coef[1]
        # print('lr')
    elif best_model == 'pr2':
        X = np.arange(len(clipped_data)).reshape(-1, 1)
        y = np.array(clipped_data)
        coef = np.polyfit(X.flatten(), y, 2)
        forecast = coef[0] * (len(clipped_data) ** 2) + coef[1] * len(clipped_data) + coef[2]
        # print('pr2')
    elif best_model == 'pr3':
        X = np.arange(len(clipped_data)).reshape(-1, 1)
        y = np.array(clipped_data)
        coef = np.polyfit(X.flatten(), y, 3)
        forecast = (
            coef[0] * (len(clipped_data) ** 3)
            + coef[1] * (len(clipped_data) ** 2)
            + coef[2] * len(clipped_data)
            + coef[3]
        )
    elif best_model == 'ses':
        model = SimpleExpSmoothing(clipped_data).fit(smoothing_level=0.65, optimized=False)
        forecast = model.forecast(1)[0]
        # print('ses')
    elif best_model == 'des':
        model = Holt(clipped_data).fit(smoothing_level=0.65, smoothing_slope=0.45, optimized=False)
        forecast = model.forecast(1)[0]
        # print('des')
    else:
        forecast = np.nan
    
    return forecast

df['FD_forecast'] = df.apply(apply_best_model_forecast, axis=1)

df['FD_final'] = df['FD_forecast'].apply(np.ceil)

# display(df[['best_model', 'FD_forecast', 'FD_final']])
# display(df)


# In[19]:


logging.info("Forecast Calculation Completed")


# In[20]:


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



# In[21]:


# Send Data Back To API
logging.info("BEGIN Constructing Final Data and send it back to API")

# Refreshing The Access Token
logging.info("BEGIN Refreshing The Access Token")
access_token = ''

url = "http://localhost:8080/web/auth/refresh"
headers = {
    "Auth": f"Bearer {refresh_token}"
}
for attempt in range(1, max_retries + 1):
    try:
        response = requests.post(url, headers=headers)
        response.raise_for_status()
        tokens = response.json()
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        if access_token:
            logging.info("Token Refreshed")
        else:
            logging.info("Token Refresh Failed")
            
        break
    except requests.RequestException as e:
        logging.info(f"Attempt {attempt}: API refresh failed - {e}")
        if attempt < max_retries:
            time.sleep(delay * (2 ** (attempt - 1)))  # Exponential backoff
        else:
            logging.info("Max retries reached. Exiting.")
            sys.exit(1)  # Stop execution after max retries
    
    
url = "http://localhost:8080/web/postdmdcall"

# construct result with branch, agency, partno
result = df[['branch', 'agency', 'partno', 'FD_final', 'std_12_FD', 'mean_12_FD', 'ub_FD']]

# change column name
result.columns = ['branch', 'agency', 'partno', 'fd', 'std', 'mean', 'ub']

# result = df.drop('d', axis=1)
result_json = result.to_dict(orient='records')

logging.info("Start Sending " + str(len(result)) + " Row To API")

for attempt in range(1, max_retries + 1):
    try:
        response = requests.post(url, json=result_json, headers={"Auth": f"Bearer {access_token}"})
        response.raise_for_status() 
        logging.info("Send API Complete")
        logging.info(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            logging.info(f"Response Body: {response.text}")
        else:
            logging.info("Send Failed")

        break
    except requests.RequestException as e:
        logging.info(f"Attempt {attempt}: API request failed - {e}")
        if attempt < max_retries:
            time.sleep(delay * (2 ** (attempt - 1)))  # Exponential backoff
        else:
            logging.info("Max retries reached. Exiting.")
            sys.exit(1)  # Stop execution after max retries

