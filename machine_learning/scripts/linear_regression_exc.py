# ### Task 1: Train a Linear Regression with Single Variable

# 1. Import the data from the "weather_data.csv" file and store it in a variable df.
# 2. Display the number of rows and columns in the dataset.
# 3. Display the first few rows of the dataset to get an overview.
# 4. Create a Linear Regression model and fit it using only the `hours_sunlight` variable to predict `daily_temperature`.
# 5. Print the model's coefficient and intercept.
# 6. Predict the daily temperature with the following hours of sunlight:
#    - 5 hours
#    - 8 hours
#    - 12 hours

from xml.parsers.expat import model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def codebasic_linear_regression_workflow():
    df = pd.read_csv('datas/weather_data.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    model = LinearRegression()
    model.fit(df[['hours_sunlight']], df['daily_temperature'])
    hours_sunlight = [[5], [8], [12]]
    predicted_temperatures = model.predict(hours_sunlight)
    # Print the predicted temperatures
    for hours, temp in zip(hours_sunlight, predicted_temperatures):
        print(f"Predicted daily temperature for {hours[0]} hours of sunlight: {temp:.2f}°C")
    print("Slope (m):", model.coef_)
    print("Intercept (b):", model.intercept_)
    print("Equation: daily_temperature = m * hours_sunlight + b")
    print(f"Equation: daily_temperature = {model.coef_[0]} * hours_sunlight + {model.intercept_}: {model.coef_[0] * 5 + model.intercept_}")


def codebasic_multiple_linear_regression_workflow():
    # ### Task 2: Train a Linear Regression with Multiple Variable

    # - Create a Linear Regression model and fit it using both `hours_sunlight` and `humidity_level` variables to predict `daily_temperature`.
    # - Print the model's coefficients and intercept.
    # - Predict the daily temperature for the following conditions:
    #     - Hours of sunlight: 5 hours, Humidity level: 60%
    #     - Hours of sunlight: 8 hours, Humidity level: 75%
    #     - Hours of sunlight: 12 hours, Humidity level: 50%

    df = pd.read_csv('datas/weather_data.csv')
    model = LinearRegression()
    model.fit(df[['hours_sunlight', 'humidity_level']], df['daily_temperature'])
    conditions = [[5, 60], [8, 75], [12, 50]]
    predicted_temperatures = model.predict(conditions)
    # Print the predicted temperatures
    for condition, temp in zip(conditions, predicted_temperatures):
        print(f"Predicted daily temperature for {condition[0]} hours of sunlight and {condition[1]}% humidity: {temp:.2f}°C")
    print("Coefficients (m):", model.coef_)
    print("Intercept (b):", model.intercept_)
    print("Equation: daily_temperature = m1 * hours_sunlight + m2 * humidity_level + b")
    print(f"Equation: daily_temperature = {model.coef_[0]} * hours_sunlight + {model.coef_[1]} * humidity_level + {model.intercept_}: {model.coef_[0] * 5 + model.coef_[1] * 60 + model.intercept_}")

def main():
    print("="*70)
    print("SIMPLE LINEAR REGRESSION WITH SCIKIT-LEARN")
    print("="*70)
    print("\nStarting the simple linear regression workflow...\n")
    codebasic_linear_regression_workflow()
    print("\n" + "="*70)
    print("MULTIPLE LINEAR REGRESSION WITH SCIKIT-LEARN")
    print("="*70)
    print("\nStarting the multiple linear regression workflow...\n")
    codebasic_multiple_linear_regression_workflow()


if __name__ == "__main__":
    main()