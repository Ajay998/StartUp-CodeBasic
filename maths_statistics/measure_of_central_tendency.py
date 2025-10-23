"""
Shoe Sales Analytics
Analysis of Nike and Adidas shoe sales data including mean and median calculations
"""

import pandas as pd
from matplotlib import pyplot as plt

def load_and_explore_data():
    """Load the shoe sales data and perform initial exploration"""
    # Load Data
    df = pd.read_csv("shoe_sales.csv")
    
    print("First 5 rows of the dataset:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"\nDataset description:")
    print(df.describe())
    
    # Analysis of 25th percentile
    print("\nAnalyzing 25th percentile (12.25):")
    values_below_25th = df.sold_qty[df.sold_qty < 12.25].shape[0]
    print(f"Values below 25th percentile: {values_below_25th}")
    print("Total values = 60")
    print("25% of total values = 60 * 0.25 = 15")
    
    return df

def analyze_nike_sales(df):
    """Analyze Nike sales data"""
    print("\n" + "="*50)
    print("NIKE SALES ANALYSIS")
    print("="*50)
    
    # Filter Nike data
    df_nike = df[df.brand == "Nike"].copy()
    print(f"Nike data shape: {df_nike.shape}")
    print(f"\nFirst 5 rows of Nike data:")
    print(df_nike.head())
    print(f"\nNike sales description:")
    print(df_nike.describe())
    
    # Handle NA values
    print(f"\nNull values in Nike data:")
    print(df_nike.isnull().sum())
    
    if df_nike.sold_qty.isnull().sum() > 0:
        print(f"\nRows with null sold_qty:")
        print(df_nike[df_nike.sold_qty.isnull()])
        
        # Fill NA values with median
        median_val = round(df_nike.sold_qty.median())
        print(f"\nMedian value for filling NA: {median_val}")
        df_nike.sold_qty.fillna(median_val, inplace=True)
        
        print(f"Null values after filling:")
        print(df_nike.isnull().sum())
    
    print(f"\nFinal Nike sales description:")
    print(df_nike.describe())
    print(f"Total Nike shoes sold in September: {df_nike.sold_qty.sum()}")
    
    print("\nNike Shoe Sales Insights:")
    print("1. On average we sell 20 nike shoes per day")
    print("2. The daily sales range is 14 to 25")
    print("3. In september month we sold 590 nike shoes")
    
    return df_nike

def analyze_adidas_sales(df):
    """Analyze Adidas sales data"""
    print("\n" + "="*50)
    print("ADIDAS SALES ANALYSIS")
    print("="*50)
    
    # Filter Adidas data
    df_adidas = df[df.brand == "Adidas"].copy()
    print(f"Adidas data shape: {df_adidas.shape}")
    print(f"\nFirst 5 rows of Adidas data:")
    print(df_adidas.head())
    print(f"\nAdidas sales description:")
    print(df_adidas.describe())
    
    # Identify outliers
    print(f"\nAnalyzing outliers (values above 75th percentile):")
    outliers = df_adidas[df_adidas.sold_qty > 15]
    print(outliers)
    
    print("\nOutlier detected on 9/12/2023 with sold_qty = 689")
    print("Replacing outlier with median value (12)")
    
    # Handle outlier
    df_adidas.sold_qty.replace(689, 12, inplace=True)
    
    print(f"\nAdidas sales description after outlier treatment:")
    print(df_adidas.describe())
    print(f"Total Adidas shoes sold in September: {df_adidas.sold_qty.sum()}")
    
    print("\nAdidas Shoe Sales Insights:")
    print("1. On average we sell 12 adidas shoes per day")
    print("2. The daily sales range is 7 to 19") 
    print("3. In september month we sold 367 adidas shoes")
    
    return df_adidas

def plot_qty(df_nike, df_adidas):
    """Plot daily sales quantities for Nike and Adidas"""
    plt.figure(figsize=(15, 6))
    
    dates = df_nike['date']
    
    plt.plot(dates, df_nike['sold_qty'], marker='o', label='Nike', color='blue')
    plt.plot(dates, df_adidas['sold_qty'], marker='o', label='Adidas', color='red')
    
    plt.xlabel('Date')
    plt.ylabel('Total Qty Sold')
    plt.title('Daily Sales Qty for Nike and Adidas in September 2023')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def main():
    """Main function to run the complete analysis"""
    print("SHOE SALES ANALYTICS")
    print("="*50)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Analyze Nike sales
    df_nike = analyze_nike_sales(df)
    
    # Analyze Adidas sales  
    df_adidas = analyze_adidas_sales(df)
    
    # Plot before outlier treatment
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    print("Plotting daily sales quantities...")
    plot_qty(df_nike, df_adidas)
    
    print("\n" + "="*50)
    print("OVERALL INSIGHTS")
    print("="*50)
    print("Sales of Nike shoes are higher than Adidas on any given date")

if __name__ == "__main__":
    main()