"""
Fitness Data Analysis
Analysis of fitness tracker data including steps, calories, sleep, and water intake
with range, IQR calculations, box plots, and outlier detection
"""

import pandas as pd
import matplotlib.pyplot as plt

def load_and_explore_data():
    """Load fitness data and perform initial exploration"""
    print("FITNESS DATA ANALYSIS")
    print("="*50)
    
    # Import the data from the "fitness_data.xlsx" Excel file
    df = pd.read_excel("fitness_data.xlsx")
    
    # Display dataset shape and first few rows
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 5 rows of the dataset:")
    print(df.head())
    
    # Calculate and display basic statistics for each column
    print(f"\nBasic statistics for all columns:")
    print(df.describe())
    
    return df

def calculate_range_and_iqr(df):
    """Calculate range and IQR for specified columns"""
    print("\n" + "="*50)
    print("TASK 2: RANGE AND IQR CALCULATIONS")
    print("="*50)
    
    # Calculate the range of "steps_taken"
    steps_taken_range = df["steps_taken"].max() - df["steps_taken"].min()
    
    # Calculate the range of "calories_burned"
    calories_burned_range = df["calories_burned"].max() - df["calories_burned"].min()
    
    # Calculate the Interquartile Range (IQR) for "sleep_duration(hours)"
    sleep_duration_iqr = df["sleep_duration(hours)"].quantile(0.75) - df["sleep_duration(hours)"].quantile(0.25)
    
    # Calculate the IQR for "water_intake(ounces)"
    water_intake_iqr = df["water_intake(ounces)"].quantile(0.75) - df["water_intake(ounces)"].quantile(0.25)
    
    # Print the results
    print("Range of Steps Taken:", steps_taken_range)
    print("Range of Calories Burned:", calories_burned_range)
    print("IQR of Sleep Duration:", sleep_duration_iqr)
    print("IQR of Water Intake:", water_intake_iqr)

def create_steps_box_plot(df):
    """Create box plot for steps taken to visualize distribution and identify outliers"""
    print("\n" + "="*50)
    print("TASK 3: BOX PLOT FOR STEPS TAKEN")
    print("="*50)
    
    # Set the figure size
    plt.figure(figsize=(5, 5))
    
    # Create a box plot for "Steps Taken"
    plt.boxplot(df["steps_taken"], vert=True, patch_artist=True)
    
    # Set the title and labels
    plt.title('Box Plot of Steps Taken')
    plt.xlabel('Data')
    plt.ylabel('Steps Taken')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Ensure proper layout and display the plot
    plt.tight_layout()
    plt.show()
    
    print("\nObservations:")
    print("- Most individuals appear to have a median daily step count around 10,000")
    print("- The presence of an outlier at 15,000 indicates at least one individual")
    print("  took an exceptionally high number of steps")
    print("- This could be due to an unusually active day or measurement error")

def get_lower_upper(data):
    """Helper function to calculate lower and upper boundaries for outlier detection"""
    Q1, Q3 = data["sleep_duration(hours)"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper

def identify_sleep_outliers(df):
    """Use IQR method to identify and label outliers in sleep duration"""
    print("\n" + "="*50)
    print("TASK 4: OUTLIER DETECTION IN SLEEP DURATION")
    print("="*50)
    
    # Get the lower and upper limits
    lower, upper = get_lower_upper(df)
    print(f"Lower boundary: {lower:.2f}")
    print(f"Upper boundary: {upper:.2f}")
    
    # Identify and label outliers
    outliers = df[(df["sleep_duration(hours)"] < lower) | (df["sleep_duration(hours)"] > upper)]
    
    # Display the outliers
    print(f"\nOutliers found in sleep duration:")
    if len(outliers) > 0:
        print(outliers)
        print(f"\nNumber of outliers: {len(outliers)}")
    else:
        print("No outliers found in sleep duration data")
    
    return outliers

def main():
    """Main function to run the complete fitness data analysis"""
    print("FITNESS TRACKER DATA ANALYSIS")
    print("="*60)
    print("Dataset includes:")
    print("- name: Name of the person")
    print("- steps_taken: Number of steps taken by individuals")
    print("- calories_burned: Estimated calories burned by individuals")
    print("- sleep_duration(hours): Hours of sleep individuals got")
    print("- water_intake(ounces): Amount of water individuals consumed")
    
    # Task 1: Load and explore data
    df = load_and_explore_data()
    
    # Task 2: Calculate range and IQR
    calculate_range_and_iqr(df)
    
    # Task 3: Create box plot for steps taken
    create_steps_box_plot(df)
    
    # Task 4: Identify outliers in sleep duration
    outliers = identify_sleep_outliers(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Summary of tasks completed:")
    print("✓ Data import and basic statistics")
    print("✓ Range and IQR calculations")
    print("✓ Box plot visualization for steps taken")
    print("✓ Outlier detection in sleep duration using IQR method")

if __name__ == "__main__":
    main()