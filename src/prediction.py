import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


df = pd.read_csv("C:\\Users\\saivi\\OneDrive\\Desktop\\tem_rel.csv")


df = df[['YEAR', 'ANN']].dropna()
df = df[df['ANN'] != -999]  
X = df[['YEAR']]  
y = df['ANN']


model = LinearRegression()
model.fit(X, y)


future_years = np.array(range(2025, 2300)).reshape(-1, 1)  # Reshape to 2D
predicted_temps = model.predict(future_years)


for year, temp in zip(future_years.ravel(), predicted_temps):
    print(f"Year: {year}, Predicted Temperature: {temp:.2f}")


plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.plot(future_years, predicted_temps, color='green', label='Future Predictions')
plt.xlabel('Year')
plt.ylabel('Avg Temperature')
plt.title('Linear Regression for Temperature Prediction')
plt.legend()
plt.grid(True)
plt.show()
