

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_data(file_path):
    """Load dataset from CSV file"""
    print("="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nBasic statistics:")
    print(df.describe())
    
    return df

def prepare_features(df, feature_cols, target_col):
    """Prepare features (X) and target (y) variables"""
    print("\n" + "="*60)
    print("STEP 2: PREPARING FEATURES AND TARGET")
    print("="*60)
    
    # Features (independent variables)
    X = df[feature_cols]
    print(f"Features (X) shape: {X.shape}")
    print(f"Feature columns: {feature_cols}")
    
    # Target (dependent variable)
    y = df[target_col]
    print(f"\nTarget (y) shape: {y.shape}")
    print(f"Target column: {target_col}")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    print("\n" + "="*60)
    print("STEP 3: SPLITTING DATA")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    print(f"Test size ratio: {test_size*100}%")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train linear regression model"""
    print("\n" + "="*60)
    print("STEP 4: TRAINING LINEAR REGRESSION MODEL")
    print("="*60)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("✓ Model training complete!")
    
    # Display model parameters
    print(f"\nModel coefficients: {model.coef_}")
    print(f"Model intercept: {model.intercept_}")
    
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance on training and testing sets"""
    print("\n" + "="*60)
    print("STEP 5: MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Training set metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    print("Training Set Performance:")
    print(f"  - MSE (Mean Squared Error): {train_mse:.2f}")
    print(f"  - RMSE (Root Mean Squared Error): {train_rmse:.2f}")
    print(f"  - MAE (Mean Absolute Error): {train_mae:.2f}")
    print(f"  - R² Score: {train_r2:.4f}")
    
    # Testing set metrics
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\nTesting Set Performance:")
    print(f"  - MSE (Mean Squared Error): {test_mse:.2f}")
    print(f"  - RMSE (Root Mean Squared Error): {test_rmse:.2f}")
    print(f"  - MAE (Mean Absolute Error): {test_mae:.2f}")
    print(f"  - R² Score: {test_r2:.4f}")
    
    # Check for overfitting/underfitting
    print("\nModel Diagnosis:")
    if train_r2 > 0.9 and test_r2 < 0.7:
        print("  ⚠ Warning: Possible overfitting (high train R², low test R²)")
    elif train_r2 < 0.5 and test_r2 < 0.5:
        print("  ⚠ Warning: Possible underfitting (low R² on both sets)")
    else:
        print("  ✓ Model performance looks reasonable")
    
    return y_train_pred, y_test_pred

def visualize_results(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred, feature_name):
    """Visualize model predictions vs actual values"""
    print("\n" + "="*60)
    print("STEP 6: VISUALIZATION")
    print("="*60)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training set plot
    axes[0].scatter(X_train, y_train, color='blue', alpha=0.5, label='Actual')
    axes[0].plot(X_train, y_train_pred, color='red', linewidth=2, label='Predicted')
    axes[0].set_xlabel(feature_name)
    axes[0].set_ylabel('Target')
    axes[0].set_title('Training Set: Actual vs Predicted')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Testing set plot
    axes[1].scatter(X_test, y_test, color='green', alpha=0.5, label='Actual')
    axes[1].plot(X_test, y_test_pred, color='red', linewidth=2, label='Predicted')
    axes[1].set_xlabel(feature_name)
    axes[1].set_ylabel('Target')
    axes[1].set_title('Testing Set: Actual vs Predicted')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Visualization complete!")

def make_predictions(model, new_data):
    """Make predictions on new data"""
    print("\n" + "="*60)
    print("STEP 7: MAKING PREDICTIONS ON NEW DATA")
    print("="*60)
    
    predictions = model.predict(new_data)
    
    print(f"Input data:")
    print(new_data)
    print(f"\nPredictions:")
    print(predictions)
    
    return predictions

def codebasic_workflow():
    df = pd.read_csv('datas/home_prices.csv')
    model = LinearRegression()
    model.fit(df[['area_sqr_ft']], df['price_lakhs'])
    print("Predicted price for 1500 sq ft:", model.predict([[1500]]))
    print("Slope (m):", model.coef_)
    print("Intercept (b):", model.intercept_)
    print("Equation: price = m * area + b")
    print(f"Equation: price = {model.coef_[0]} * area + {model.intercept_}: {model.coef_[0] * 1500 + model.intercept_}")


def main():
    """Main function to run the complete linear regression workflow"""
    print("="*70)
    print("SIMPLE LINEAR REGRESSION WITH SCIKIT-LEARN")
    print("="*70)
    print("\nStarting the linear regression workflow...\n")
    codebasic_workflow()
    
    # Configuration
    FILE_PATH = 'datas/home_prices.csv'
    FEATURE_COLS = ['area_sqr_ft']  # Change based on your dataset
    TARGET_COL = 'price_lakhs'      # Change based on your dataset
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Step 1: Load data
    df = load_data(FILE_PATH)
    
    # Step 2: Prepare features and target
    X, y = prepare_features(df, FEATURE_COLS, TARGET_COL)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = split_data(X, y, TEST_SIZE, RANDOM_STATE)
    
    # Step 4: Train model
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate model
    y_train_pred, y_test_pred = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Step 6: Visualize results
    visualize_results(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred, FEATURE_COLS[0])
    
    # Step 7: Make predictions on new data (example)
    new_data = pd.DataFrame({FEATURE_COLS[0]: [1000, 1500, 2000]})
    predictions = make_predictions(model, new_data)
    
    print("\n" + "="*70)
    print("LINEAR REGRESSION WORKFLOW COMPLETE")
    print("="*70)
    print(f"\nModel Summary:")
    print(f"  - Coefficients: {model.coef_}")
    print(f"  - Intercept: {model.intercept_}")
    print(f"  - Features used: {FEATURE_COLS}")
    print(f"  - Target variable: {TARGET_COL}")
    
    return model

if __name__ == "__main__":
    model = main()