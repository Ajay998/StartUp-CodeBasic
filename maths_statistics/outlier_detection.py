"""
Outlier Treatment Using IQR and Box Plot
Analysis of outliers in heights and sales data using statistical methods and visualization
"""

import pandas as pd
import matplotlib.pyplot as plt

def analyze_heights_outliers():
    """Find outliers in heights data using IQR method"""
    print("OUTLIER DETECTION IN HEIGHTS DATA")
    print("="*50)
    
    # Load heights data
    df = pd.read_csv("heights.csv")
    print("First 5 rows of heights data:")
    print(df.head())
    
    # Calculate Q1 and Q3
    Q1, Q3 = df.height.quantile([0.25, 0.75])
    print(f"\nQ1: {Q1}, Q3: {Q3}")
    
    # Calculate IQR
    IQR = Q3 - Q1
    print(f"IQR: {IQR}")
    
    # Find lower and upper boundaries for outlier detection
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    print(f"Lower boundary: {lower}, Upper boundary: {upper}")
    
    # Find outliers
    outliers = df[(df.height < lower) | (df.height > upper)]
    print(f"\nOutliers found:")
    print(outliers)
    print(f"Number of outliers: {len(outliers)}")
    
    # Create new dataframe with outliers removed
    df_new = df[(df.height > lower) & (df.height < upper)]
    print(f"\nDataset after removing outliers:")
    print(f"Original size: {len(df)}, New size: {len(df_new)}")
    print(df_new.head())
    
    return df, df_new

def get_lower_upper(data):
    """Helper function to calculate lower and upper boundaries for outlier detection"""
    Q1, Q3 = data.Sales.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper

def analyze_sales_outliers():
    """Find outliers in sales data using IQR method"""
    print("\n" + "="*50)
    print("OUTLIER DETECTION IN REGIONAL SALES DATA")
    print("="*50)
    
    # Load sales data (make sure to install openpyxl using "pip install openpyxl")
    df = pd.read_excel("region_wise_sales.xlsx")
    print("First 5 rows of sales data:")
    print(df.head())
    
    # Check unique regions
    print(f"\nUnique regions: {df.Region.unique()}")
    
    # Split data by region
    df_apac = df[df.Region == "APAC"]
    df_europe = df[df.Region == "Europe"]
    df_americas = df[df.Region == "Americas"]
    
    # Analyze APAC region
    print(f"\nAPAC Region Analysis:")
    lower, upper = get_lower_upper(df_apac)
    print(f"Lower boundary: {lower:.2f}, Upper boundary: {upper:.2f}")
    print("APAC Sales description:")
    print(df_apac.Sales.describe())
    
    # Analyze Europe region
    print(f"\nEurope Region Analysis:")
    lower, upper = get_lower_upper(df_europe)
    print(f"Lower boundary: {lower:.2f}, Upper boundary: {upper:.2f}")
    print("Europe Sales description:")
    print(df_europe.Sales.describe())
    
    # Find outliers in Europe
    europe_outliers = df_europe[(df_europe.Sales < lower) | (df_europe.Sales > upper)]
    print(f"\nOutliers in Europe region:")
    print(europe_outliers)
    
    # Analyze Americas region
    print(f"\nAmericas Region Analysis:")
    lower, upper = get_lower_upper(df_americas)
    print(f"Lower boundary: {lower:.2f}, Upper boundary: {upper:.2f}")
    
    print("\nFor Europe we see one outlier. For other regions there are no outliers")
    
    return df

def create_box_plot(df):
    """Create box plot for outlier detection using visualization"""
    print("\n" + "="*50)
    print("OUTLIER DETECTION USING BOX PLOT")
    print("="*50)
    
    # Get unique labels
    labels = df['Region'].unique()
    print(f"Regions for box plot: {labels}")
    
    # Prepare data for box plot
    plot_data = [df['Sales'][df['Region'] == label].values for label in labels]
    print("Data prepared for box plot")
    
    # Create box plot
    plt.figure(figsize=(12, 8))
    plt.boxplot(plot_data, labels=labels, vert=True, patch_artist=True)
    plt.title('Box plot of Sales by Region and Year')
    plt.ylabel('Sales')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.show()
    
    print("Box plot created successfully!")
    print("You can clearly see an outlier point above Europe box plot.")
    print("This shows how one can use a box plot to spot outliers")

def main():
    """Main function to run the complete outlier detection analysis"""
    print("OUTLIER TREATMENT USING IQR AND BOX PLOT")
    print("="*60)
    
    # Analyze heights data outliers
    df_heights, df_heights_clean = analyze_heights_outliers()
    
    # Analyze sales data outliers
    df_sales = analyze_sales_outliers()
    
    # Create box plot visualization
    create_box_plot(df_sales)
    
    print("\n" + "="*60)
    print("OUTLIER ANALYSIS COMPLETE")
    print("="*60)
    print("Summary:")
    print("1. Heights data: Found and removed outliers using IQR method")
    print("2. Sales data: Identified outliers in Europe region")
    print("3. Box plot: Visual confirmation of outliers in sales data")

if __name__ == "__main__":
    main()