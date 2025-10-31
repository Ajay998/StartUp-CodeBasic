

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



def codebasic_workflow():
    df = pd.read_csv('datas/home_prices2.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    model = LinearRegression()
    model.fit(df[['area_sqr_ft', 'bedrooms']], df['price_lakhs'])
    print("Predicted price for 1500 sq ft and 2 bedrooms:", model.predict([[1500, 2]]))
    print("Slope (m):", model.coef_)
    print("Intercept (b):", model.intercept_)
    print("Equation: price = m * area + n * bedrooms + b")
    print(f"Equation: price = {model.coef_[0]} * area + {model.coef_[1]} * bedrooms + {model.intercept_}: {model.coef_[0] * 1500 + model.coef_[1] * 2 + model.intercept_}")



def main():
    print("="*70)
    print("MULTIPLE LINEAR REGRESSION WITH SCIKIT-LEARN")
    print("="*70)
    print("\nStarting the multiple linear regression workflow...\n")
    codebasic_workflow()
    
  

if __name__ == "__main__":
    main()