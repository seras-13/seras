# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dataset
data = {
    "Date": ["2024-1-1", "2024-1-2", "2024-1-3", "2024-1-4", "2024-1-5"],
    "Temperature": [5, 7, 10, 8, 11],
    "Sales": [15, 21, 30, 20, 33]
}

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Features (Temperature) and Target (Sales)
X = np.array(df["Temperature"]).reshape(-1, 1)  # Convert to a NumPy array
y = np.array(df["Sales"]).reshape(-1, 1)

# Build a simple neural network model
model = Sequential([
    Dense(8, input_dim=1, activation='relu'),  # Input layer with 8 neurons
    Dense(1)  # Output layer with 1 neuron
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=200, verbose=0)  # Train for 200 epochs (adjust if needed)

# Predict sales for temperature = 9
temp_to_predict = np.array([[9]])
predicted_sales = model.predict(temp_to_predict)

# Output the prediction
print(f"Predicted sales for a temperature of 9 degrees: {predicted_sales[0][0]:.2f}")
