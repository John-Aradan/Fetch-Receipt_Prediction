# Importing necessary libraries
import numpy as np  # Used for numerical computations
import pandas as pd  # Used for handling and analyzing data
import pickle  # Used to save and load model parameters to/from files
import os  # Used for interacting with the file system

# Defining folders where the data and model parameters will be stored
save_folder = os.path.join(os.path.dirname(__file__), 'Model-Params')  # Folder to save model parameters
data_folder = os.path.join(os.path.dirname(__file__), 'Data')  # Folder where data files are stored

# Reading the 'data_months.csv' file into a pandas DataFrame
df = pd.read_csv(os.path.join(data_folder, "data_months.csv"))  # Load the data for monthly trends

# Splitting the data into input features (X) and the target variable (Y)
X = df.iloc[:, :-1]  # All columns except the last one are features (input)
Y = df.iloc[:, -1]   # The last column is the target (output)

# Convert categorical columns ('Month', 'Quarter') into one-hot encoded (binary) columns
X = pd.get_dummies(X, columns=['Month', 'Quarter']).astype(int)  # Convert 'Month' and 'Quarter' to binary format

# List of continuous columns that need to be normalized (scaled)
continuous_columns = ['SMA_1', 'SMA_3', 'SMA_6', 'Lag_1', 'Lag_3', 'Lag_6']

# Convert these columns to float type and normalize them (mean 0, std deviation 1)
X[continuous_columns] = X[continuous_columns].astype('float64')  # Ensure these columns are floats
X[continuous_columns] = (X[continuous_columns] - X[continuous_columns].mean()) / X[continuous_columns].std()

# Function to perform gradient descent for linear regression
def gradient_descent(theta_now, b_now, X, y, L):
    # Initialize gradients for weights (theta) and bias (b)
    theta_gradient = np.zeros(theta_now.shape)  # Array of zeros, same size as theta (weights)
    b_gradient = 0  # Initialize bias gradient as zero
    n = float(len(X))  # Number of data points

    # Loop over each data point to calculate gradients
    for i in range(len(X)):
        x = X.iloc[i, :].values  # Get the features (input) of the i-th data point
        y_true = y.iloc[i]       # Get the actual target (output) value for the i-th data point

        # Calculate the predicted value using current weights (theta) and bias (b)
        y_pred = np.dot(x, theta_now) + b_now

        # Compute gradients for weights (theta) and bias (b)
        theta_gradient += -(2/n) * x * (y_true - y_pred)  # Update the gradient for theta (weights)
        b_gradient += -(2/n) * (y_true - y_pred)  # Update the gradient for b (bias)

    # Update weights (theta) and bias (b) using the gradients and learning rate (L)
    theta_next = theta_now - L * theta_gradient
    b_next = b_now - L * b_gradient

    return theta_next, b_next  # Return updated weights and bias

# Initialize weights (theta) and bias (b) for the model
theta = np.zeros(X.shape[1])  # Set all weights to zero initially, size matches number of features
b = Y.mean()  # Set the bias (b) as the mean of the target variable
L = 0.001  # Learning rate, controls how big the updates are during each step
epochs = 100000  # Number of iterations for gradient descent

# Run the gradient descent algorithm for the specified number of epochs (iterations)
for epoch in range(epochs):
    theta, b = gradient_descent(theta, b, X, Y, L)  # Update theta (weights) and b (bias)

    # Every 1000 epochs, calculate and print the Mean Squared Error (MSE)
    if epoch % 1000 == 0:
        predictions = X.dot(theta) + b  # Calculate predictions
        cost = np.mean((Y - predictions) ** 2)  # Compute MSE (how far predictions are from actual values)
        print(f"Epoch {epoch}: MSE = {cost}")  # Print the epoch number and MSE

# After training is complete, print the final model parameters
print(f"Final weights: {theta}")
print(f"Final bias: {b}")
print(f"Final MSE: {cost}")

# Save the trained weights (theta) and bias (b) to a file using pickle
theta_all = theta.copy()  # Copy the final weights
b_all = b.copy()  # Copy the final bias

with open(os.path.join(save_folder, "model_parameters_all.pkl"), "wb") as f:
    pickle.dump((theta_all, b_all), f)  # Save weights and bias to 'model_parameters_all.pkl' file

### Early Stopping Implementation ###

# Early stopping is a method to stop training when the error is low enough, to avoid overfitting

# Define an accuracy threshold for early stopping
mse_threshold = 10000  # Stop training if Mean Squared Error (MSE) falls below this value

# Re-run gradient descent with early stopping logic
for epoch in range(epochs):
    theta, b = gradient_descent(theta, b, X, Y, L)  # Update weights (theta) and bias (b)

    # Calculate predictions and compute MSE
    predictions = X.dot(theta) + b  # Predicted values
    mse = np.mean((Y - predictions) ** 2)  # Compute MSE

    # Stop training early if MSE is lower than the threshold
    if mse < mse_threshold:
        print(f"Stopping early at epoch {epoch} with MSE: {mse}")
        break  # Exit the loop if MSE is low enough

    # Print MSE every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: MSE = {mse}")

# Print final results after early stopping
print(f"Final weights: {theta}")
print(f"Final bias: {b}")
print(f"Final MSE: {mse}")

# Save the weights and bias obtained from early stopping to a separate file
theta_early = theta.copy()  # Copy weights after early stopping
b_early = b.copy()  # Copy bias after early stopping

with open(os.path.join(save_folder, "model_parameters_early.pkl"), "wb") as f:
    pickle.dump((theta_early, b_early), f)  # Save the parameters to 'model_parameters_early.pkl' file
