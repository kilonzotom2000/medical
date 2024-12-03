# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to compute compound interest
def future_value(current_savings, annual_rate, time_years, compounding_periods=12):
    return current_savings * (1 + annual_rate / compounding_periods) ** (compounding_periods * time_years)

# Function to adjust target for inflation
def adjust_for_inflation(target_amount, inflation_rate, time_years):
    return target_amount * (1 + inflation_rate) ** time_years

# Load dataset (replace with your dataset path)
data = pd.read_csv("financial_target_data.csv")

# Feature engineering
data['future_savings'] = data.apply(lambda row: future_value(
    row['current_savings'], row['interest_rate'], row['time_to_target'] / 12), axis=1)

data['adjusted_target'] = data.apply(lambda row: adjust_for_inflation(
    row['target_amount'], row['inflation_rate'], row['time_to_target'] / 12), axis=1)

# New features to include expenses and debt
data['available_income'] = data['income'] - data['monthly_expenses'] - data['monthly_debt']

# Final input features
features = ['future_savings', 'adjusted_target', 'time_to_target', 'interest_rate', 'available_income']
X = data[features]
y = data['monthly_contribution']  # Target variable

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Example prediction for a new individual
new_data = pd.DataFrame({
    "current_savings": [5000],
    "target_amount": [50000],
    "time_to_target": [24],  # 2 years
    "interest_rate": [0.05],  # 5% annual
    "inflation_rate": [0.02],  # 2% annual
    "income": [60000],  # Annual income
    "monthly_expenses": [2000],
    "monthly_debt": [500]
})

# Feature engineering for new data
new_data['future_savings'] = new_data.apply(lambda row: future_value(
    row['current_savings'], row['interest_rate'], row['time_to_target'] / 12), axis=1)

new_data['adjusted_target'] = new_data.apply(lambda row: adjust_for_inflation(
    row['target_amount'], row['inflation_rate'], row['time_to_target'] / 12), axis=1)

new_data['available_income'] = new_data['income'] - new_data['monthly_expenses'] - new_data['monthly_debt']

# Select features for prediction
new_features = ['future_savings', 'adjusted_target', 'time_to_target', 'interest_rate', 'available_income']
predicted_contribution = model.predict(new_data[new_features])

print(f"Predicted Monthly Contribution: ${predicted_contribution[0]:.2f}")
