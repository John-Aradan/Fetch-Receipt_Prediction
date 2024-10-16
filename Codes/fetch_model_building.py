import numpy as np
import pandas as pd
import pickle
import os

save_folder = os.path.join(os.path.dirname(__file__), 'Model-Params')
data_folder = os.path.join(os.path.dirname(__file__), 'Data')

df = pd.read_csv(os.path.join(data_folder, "data_months.csv"))

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

X = pd.get_dummies(X, columns=['Month', 'Quarter']).astype(int)

continuous_columns = ['SMA_1', 'SMA_3', 'SMA_6', 'Lag_1', 'Lag_3', 'Lag_6']
X[continuous_columns] = X[continuous_columns].astype('float64')
X[continuous_columns] = (X[continuous_columns] - X[continuous_columns].mean()) / X[continuous_columns].std()


def gradient_descent(theta_now, b_now, X, y, L):
    # Initialize gradients for weights (theta) and bias (b)
    theta_gradient = np.zeros(theta_now.shape)
    b_gradient = 0
    n = float(len(X))  # Number of data points

    # Compute gradients for weights (theta) and bias (b)
    for i in range(len(X)):
        x = X.iloc[i, :].values  # Extract feature values for the i-th sample
        y_true = y.iloc[i]       # True value of y for the i-th sample

        # Compute the predicted value using current weights (theta) and bias (b)
        y_pred = np.dot(x, theta_now) + b_now

        # Calculate gradients
        theta_gradient += -(2/n) * x * (y_true - y_pred)  # Gradient for weights
        b_gradient += -(2/n) * (y_true - y_pred)          # Gradient for bias

    # Update weights (theta) and bias (b) using the gradients
    theta_next = theta_now - L * theta_gradient
    b_next = b_now - L * b_gradient

    return theta_next, b_next

# Initial values for weights (theta) and bias (b)
theta = np.zeros(X.shape[1])  # Initialize weights (theta) with zeros, size = number of features
b = Y.mean()  # Initialize bias (b) with mean of receipt_counts
L = 0.001  # Learning rate
epochs = 100000

# Run gradient descent for the specified number of epochs
for epoch in range(epochs):
    theta, b = gradient_descent(theta, b, X, Y, L)

    # Print Mean Squared Error (MSE) every 100 epochs for tracking performance
    if epoch % 1000 == 0:
        predictions = X.dot(theta) + b  # Predicted values
        cost = np.mean((Y - predictions) ** 2)  # Compute MSE
        print(f"Epoch {epoch}: MSE = {cost}")

print(f"Final weights: {theta}")
print(f"Final bias: {b}")
print(f"Final MSE: {cost}")

theta_all = theta.copy()
b_all = b.copy()

with open(os.path.join(save_folder, "model_parameters_all.pkl"), "wb") as f:
    pickle.dump((theta_all, b_all), f)

import numpy as np

def gradient_descent(theta_now, b_now, X, y, L):
    # Initialize gradients for weights (theta) and bias (b)
    theta_gradient = np.zeros(theta_now.shape)
    b_gradient = 0
    n = float(len(X))  # Number of data points

    # Compute gradients for weights (theta) and bias (b)
    for i in range(len(X)):
        x = X.iloc[i, :].values  # Extract feature values for the i-th sample
        y_true = y.iloc[i]       # True value of y for the i-th sample

        # Compute the predicted value using current weights (theta) and bias (b)
        y_pred = np.dot(x, theta_now) + b_now

        # Calculate gradients
        theta_gradient += -(2/n) * x * (y_true - y_pred)  # Gradient for weights
        b_gradient += -(2/n) * (y_true - y_pred)          # Gradient for bias

    # Update weights (theta) and bias (b) using the gradients
    theta_next = theta_now - L * theta_gradient
    b_next = b_now - L * b_gradient

    return theta_next, b_next

# Initial values for weights (theta) and bias (b)
theta = np.zeros(X.shape[1])  # Initialize weights (theta) with zeros, size = number of features
b = Y.mean()  # Initialize bias (b) with mean of receipt_counts
L = 0.001  # Learning rate
epochs = 100000

# Early stopping threshold
mse_threshold = 10000   # accuracy of 0.0001 (0.01%)

# Run gradient descent for the specified number of epochs
for epoch in range(epochs):
    theta, b = gradient_descent(theta, b, X, Y, L)

    # Calculate predictions and compute MSE
    predictions = X.dot(theta) + b  # Predicted values
    mse = np.mean((Y - predictions) ** 2)  # Compute MSE

    # Early stopping check based on MSE threshold
    if mse < mse_threshold:
        print(f"Stopping early at epoch {epoch} with MSE: {mse}")
        break

    # Print Mean Squared Error (MSE) every 1000 epochs for tracking performance
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: MSE = {mse}")

print(f"Final weights: {theta}")
print(f"Final bias: {b}")
print(f"Final MSE: {mse}")

theta_early = theta.copy()
b_early = b.copy()

with open(os.path.join(save_folder, "model_parameters_early.pkl"), "wb") as f:
    pickle.dump((theta_early, b_early), f)