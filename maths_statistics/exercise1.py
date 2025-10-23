"""
Telecom Customer Churn Case Study
Analysis of telecom customer data including measures of central tendency
"""

import pandas as pd
import numpy as np

def load_and_explore_data():
    """Load the telecom customer churn dataset and perform initial exploration"""
    print("TELECOM CUSTOMER CHURN CASE STUDY")
    print("="*50)
    
    # 1. Import the provided dataset to dataframe (telecom_customer_churn.csv)
    # 2. Change the settings to display all the columns  
    # 3. Check the number of rows and columns
    # 4. Check the top 5 rows
    customer_df = pd.read_csv("telecom_customer_churn.csv")
    pd.set_option('display.max_columns', None)
    
    print(f"Dataset shape: {customer_df.shape}")
    print(f"\nFirst 5 rows:")
    print(customer_df.head())
    
    # Display all the column names
    print(f"\nColumn names:")
    print(customer_df.columns.tolist())
    
    # Check if the dataset contains nulls
    print(f"\nNull values in dataset:")
    print(customer_df.isnull().sum())
    
    # Check the datatype of all columns
    print(f"\nData types:")
    print(customer_df.dtypes)
    
    return customer_df

def fix_data_types(customer_df):
    """Fix the datatype of numeric columns"""
    print("\n" + "="*50)
    print("FIXING DATA TYPES")
    print("="*50)
    
    # Convert the datatype of 'monthly_charges', 'total_charges', 'tenure' to numeric datatype
    cols_to_num = ['monthly_charges', 'total_charges', 'tenure']
    customer_df[cols_to_num] = customer_df[cols_to_num].apply(pd.to_numeric, errors='coerce')
    
    print("Converted columns to numeric:", cols_to_num)
    print(f"\nUpdated data types:")
    print(customer_df[cols_to_num].dtypes)
    
    return customer_df

def q1_monthly_charges_statistics(customer_df):
    """Q1 - Calculate the mean, median, and mode of the monthly_charges column"""
    print("\n" + "="*50)
    print("Q1 - MONTHLY CHARGES STATISTICS")
    print("="*50)
    
    mean = customer_df['monthly_charges'].mean()
    median = customer_df['monthly_charges'].median()
    mode = customer_df['monthly_charges'].mode()
    
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Mode: {mode[0]:.2f}")

def q2_total_charges_percentiles(customer_df):
    """Q2 - Calculate the 25th, 50th, and 75th percentiles of the total_charges column"""
    print("\n" + "="*50)
    print("Q2 - TOTAL CHARGES PERCENTILES")
    print("="*50)
    
    percentiles = customer_df['total_charges'].quantile([0.25, 0.5, 0.75])
    
    print(f"25th Percentile: {percentiles[0.25]:.2f}")
    print(f"50th Percentile: {percentiles[0.50]:.2f}")
    print(f"75th Percentile: {percentiles[0.75]:.2f}")

def q3_monthly_charges_range(customer_df):
    """Q3 - Calculate the range of monthly_charges column"""
    print("\n" + "="*50)
    print("Q3 - MONTHLY CHARGES RANGE")
    print("="*50)
    
    range_monthly_charges = customer_df['monthly_charges'].max() - customer_df['monthly_charges'].min()
    print(f'Range of monthly charges column: {range_monthly_charges:.2f}')

def q4_first_quartile_not_churned(customer_df):
    """Q4 - What is the first quartile of the monthly_charges column for customers who have not churned?"""
    print("\n" + "="*50)
    print("Q4 - FIRST QUARTILE FOR NON-CHURNED CUSTOMERS")
    print("="*50)
    
    first_quartile_monthly_charges_not_churned = customer_df.loc[customer_df['churn'] == 'No', 'monthly_charges'].quantile(0.25)
    print(f'First quartile of monthly charges column for customers who have not churned: {first_quartile_monthly_charges_not_churned:.2f}')

def q5_third_quartile_churned(customer_df):
    """Q5 - What is the third quartile of the total_charges column for customers who have churned?"""
    print("\n" + "="*50)
    print("Q5 - THIRD QUARTILE FOR CHURNED CUSTOMERS")
    print("="*50)
    
    third_quartile_total_charges_churned = customer_df.loc[customer_df['churn'] == 'Yes', 'total_charges'].quantile(0.75)
    print(f'Third quartile of total charges column for customers who have churned: {third_quartile_total_charges_churned:.2f}')

def q6_payment_method_mode_churned(customer_df):
    """Q6 - What is the mode of the payment method column for customers who have churned?"""
    print("\n" + "="*50)
    print("Q6 - PAYMENT METHOD MODE FOR CHURNED CUSTOMERS")
    print("="*50)
    
    mode_payment_method_churned = customer_df.loc[customer_df['churn'] == 'Yes', 'payment_method'].mode()[0]
    print(f'Mode of payment method column for customers who have churned: {mode_payment_method_churned}')

def q7_mean_total_charges_churned_monthly(customer_df):
    """Q7 - What is the mean of the total charges column for customers who have churned and have a month-to-month contract?"""
    print("\n" + "="*50)
    print("Q7 - MEAN TOTAL CHARGES FOR CHURNED MONTH-TO-MONTH CUSTOMERS")
    print("="*50)
    
    # Filter the rows based on the churn status and contract type
    filtered_df = customer_df.loc[(customer_df['churn'] == 'Yes') & (customer_df['contract'] == 'Month-to-month')]
    
    # Calculate the mean of the total charges column
    mean_total_charges = filtered_df['total_charges'].mean()
    
    # Print the result
    print(f'Mean of total charges column for customers who have churned and have a month-to-month contract: {mean_total_charges:.2f}')

def q8_median_tenure_not_churned_two_year(customer_df):
    """Q8 - What is the median of the tenure column for customers who have not churned and have a two-year contract?"""
    print("\n" + "="*50)
    print("Q8 - MEDIAN TENURE FOR NON-CHURNED TWO-YEAR CONTRACT CUSTOMERS")
    print("="*50)
    
    # Filter the rows based on the churn status and contract type
    filtered_df = customer_df.loc[(customer_df['churn'] == 'No') & (customer_df['contract'] == 'Two year')]
    
    # Calculate the median of the tenure column
    median_tenure = filtered_df['tenure'].median()
    
    # Print the result
    print(f'Median of tenure column for customers who have not churned and have a two-year contract: {median_tenure:.2f}')

def main():
    """Main function to run the complete telecom customer churn analysis"""
    # Load and explore data
    customer_df = load_and_explore_data()
    
    # Fix data types
    customer_df = fix_data_types(customer_df)
    
    # Answer all questions
    q1_monthly_charges_statistics(customer_df)
    q2_total_charges_percentiles(customer_df)
    q3_monthly_charges_range(customer_df)
    q4_first_quartile_not_churned(customer_df)
    q5_third_quartile_churned(customer_df)
    q6_payment_method_mode_churned(customer_df)
    q7_mean_total_charges_churned_monthly(customer_df)
    q8_median_tenure_not_churned_two_year(customer_df)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()