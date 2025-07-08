import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
df = pd.read_csv("C:\\Users\\saivi\\OneDrive\\Desktop\\tem_rel.csv")

# Keep only YEAR and ANN columns
df = df[['YEAR', 'ANN']]

# Convert to numeric and clean garbage values
df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
df['ANN'] = pd.to_numeric(df['ANN'], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Filter out invalid values
df = df[(df['ANN'] > 0) & (df['ANN'] < 100)]  # realistic temperature range
df = df[(df['YEAR'] >= 1900) & (df['YEAR'] <= 2024)]  # realistic years

# Final check before training
print(df.describe())

# Features and target
X = df[['YEAR']]
y = df['ANN']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict future years
future_years = pd.DataFrame({'YEAR': [2025, 2026, 2027, 2028, 2029]})
predicted_temps = model.predict(future_years)

# Print predictions
for year, temp in zip(future_years['YEAR'], predicted_temps):
    print(f"Year: {year}, Predicted Temperature: {temp:.2f}")

# Plot
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.scatter(future_years, predicted_temps, color='green', label='Future Predictions')
plt.xlabel('Year')
plt.ylabel('Avg Temperature')
plt.title('Temperature Prediction using Linear Regression')
plt.legend()
plt.grid(True)
plt.show()
