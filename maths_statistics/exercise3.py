"""
Climate Stability Analysis for Agricultural Investment Decisions
Comparing weather temperature variability between two cities to determine
the best location for agricultural investment based on temperature stability.
"""

import pandas as pd

def load_climate_data():
    """Load climate data and perform initial exploration"""
    print("CLIMATE STABILITY ANALYSIS FOR AGRICULTURAL INVESTMENT")
    print("="*60)
    print("Context: An agricultural investment firm is evaluating two cities")
    print("based on temperature stability for a new farming project.")
    print("\nDataset includes:")
    print("- month: Indicates the month of the year")
    print("- Avg Temp City A (°C): Average daily temperature for City A")
    print("- Avg Temp City B (°C): Average daily temperature for City B")
    
    print("\n" + "="*50)
    print("TASK 1: DATA IMPORT")
    print("="*50)
    
    # Import Excel file to dataframe
    df = pd.read_excel("climate_stability_comparison.xlsx")
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 12 rows of climate data:")
    print(df.head(12))
    
    return df

def calculate_standard_deviations(df):
    """Calculate standard deviations for both cities"""
    print("\n" + "="*50)
    print("TASK 2: CALCULATE STANDARD DEVIATIONS")
    print("="*50)
    
    # Calculate standard deviation for City A
    cityA_std_dev = round(df['Avg Temp City A (°C)'].std(), 2)
    print("Standard Deviation of Avg Temp in City A:", cityA_std_dev)
    
    # Calculate standard deviation for City B
    cityB_std_dev = round(df['Avg Temp City B (°C)'].std(), 2)
    print("Standard Deviation of Avg Temp in City B:", cityB_std_dev)
    
    return cityA_std_dev, cityB_std_dev

def compare_cities(cityA_std_dev, cityB_std_dev):
    """Compare the cities and make investment decision"""
    print("\n" + "="*50)
    print("TASK 3: COMPARISON")
    print("="*50)
    
    # Decision Making based on standard deviation
    if cityA_std_dev < cityB_std_dev:
        recommendation = "City A"
        print("Invest in City A due to more stable temperature conditions.")
    else:
        recommendation = "City B"
        print("Invest in City B due to more stable temperature conditions.")
    
    return recommendation

def provide_detailed_analysis(df, cityA_std_dev, cityB_std_dev, recommendation):
    """Provide detailed observations and final decision"""
    print("\n" + "="*50)
    print("TASK 4: OBSERVATION AND DECISION")
    print("="*50)
    
    # Calculate temperature ranges for better analysis
    cityA_min = df['Avg Temp City A (°C)'].min()
    cityA_max = df['Avg Temp City A (°C)'].max()
    cityB_min = df['Avg Temp City B (°C)'].min()
    cityB_max = df['Avg Temp City B (°C)'].max()
    
    print("Detailed Analysis:")
    print(f"- City A temperature range: {cityA_min}°C to {cityA_max}°C")
    print(f"- City B temperature range: {cityB_min}°C to {cityB_max}°C")
    print(f"- City A standard deviation: {cityA_std_dev}°C")
    print(f"- City B standard deviation: {cityB_std_dev}°C")
    
    print("\nObservations:")
    if recommendation == "City A":
        print("- City A demonstrated a notably lower standard deviation in its average")
        print("  daily temperatures compared to City B. This indicates that City A")
        print("  experiences less fluctuation in temperature throughout the year.")
        print("- The temperature range in City A was found to be more consistent and")
        print(f"  narrow, fluctuating moderately between {cityA_min}°C and {cityA_max}°C.")
        print("- Conversely, City B displayed higher variability with temperatures")
        print(f"  ranging more broadly from {cityB_min}°C to {cityB_max}°C.")
        print("- Based on above observations, we recommend choosing City A for the")
        print("  agricultural investment.")
    else:
        print("- City B demonstrated a lower standard deviation in its average")
        print("  daily temperatures compared to City A. This indicates that City B")
        print("  experiences less fluctuation in temperature throughout the year.")
        print("- The temperature range in City B was found to be more consistent and")
        print(f"  narrow, fluctuating moderately between {cityB_min}°C and {cityB_max}°C.")
        print("- Conversely, City A displayed higher variability with temperatures")
        print(f"  ranging more broadly from {cityA_min}°C to {cityA_max}°C.")
        print("- Based on above observations, we recommend choosing City B for the")
        print("  agricultural investment.")

def main():
    """Main function to run the complete climate stability analysis"""
    print("AGRICULTURAL INVESTMENT CLIMATE ANALYSIS")
    print("="*70)
    
    # Task Objectives
    print("Task Objectives:")
    print("- Calculate the standard deviation for average temperatures in both cities")
    print("- Determine which city has a higher temperature variability")
    print("- Decide on the city with more stable temperatures for agricultural investment")
    
    # Task 1: Load climate data
    df = load_climate_data()
    
    # Task 2: Calculate standard deviations
    cityA_std_dev, cityB_std_dev = calculate_standard_deviations(df)
    
    # Task 3: Compare cities and make decision
    recommendation = compare_cities(cityA_std_dev, cityB_std_dev)
    
    # Task 4: Provide detailed analysis
    provide_detailed_analysis(df, cityA_std_dev, cityB_std_dev, recommendation)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"FINAL RECOMMENDATION: Invest in {recommendation}")
    print("Rationale: Lower temperature variability provides more predictable")
    print("conditions for successful agricultural operations and crop planning.")

if __name__ == "__main__":
    main()