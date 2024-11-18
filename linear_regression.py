import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv("carseats_train.csv")
test_data = pd.read_csv("carseats_test.csv")

# Binary encoding for 'Urban' and 'US'
def binary_encode(df, columns):
    for col in columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    return df

train_data = binary_encode(train_data, ['Urban', 'US'])
test_data = binary_encode(test_data, ['Urban', 'US'])

# Correct one-hot encoding to ensure integer (0/1) values
train_data = pd.get_dummies(train_data, columns=['ShelveLoc'], drop_first=False).astype(int)
test_data = pd.get_dummies(test_data, columns=['ShelveLoc'], drop_first=False).astype(int)

# Feature standardization
def standardize_features(train, test, features):
    scaler = StandardScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test

continuous_features = ['CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age', 'Education']
train_data, test_data = standardize_features(train_data, test_data, continuous_features)

# Check the processed data
print("Train Data:")
print(train_data.head())
print("\nTest Data:")
print(test_data.head())

# Step 1: Forward pass to compute predictions
def predict(X, weights, bias):
    """
    Compute predictions using the linear regression model: ŷ = X @ weights + bias
    """
    return np.dot(X, weights) + bias

# Step 2: Loss computation
def compute_loss(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) loss: L = (1/n) * Σ(ŷ - y)^2
    """
    return np.mean((y_pred - y_true) ** 2)

def sgd_update_l1(X, y_true, weights, bias, learning_rate, lambda_reg):
    """
    Perform a single step of SGD with L1 regularization (LASSO).
    """
    n_samples = X.shape[0]
    
    # Predictions
    y_pred = predict(X, weights, bias)
    
    # Gradients
    dw = -(2 / n_samples) * np.dot(X.T, (y_true - y_pred)) + lambda_reg * np.sign(weights)
    db = -(2 / n_samples) * np.sum(y_true - y_pred)
    
    # Update weights and bias
    weights -= learning_rate * dw
    bias -= learning_rate * db
    
    return weights, bias


def train_sgd_l1(X, y, epochs, learning_rate, lambda_reg):
    """
    Train the linear regression model using SGD with L1 regularization.
    """
    n_features = X.shape[1]
    weights = np.zeros(n_features)  # Initialize weights to 0
    bias = 0  # Initialize bias to 0
    
    loss_history = []
    
    for epoch in range(epochs):
        # Perform SGD update with L1 regularization
        weights, bias = sgd_update_l1(X, y, weights, bias, learning_rate, lambda_reg)
        
        # Compute loss for this epoch
        y_pred = predict(X, weights, bias)
        # Loss includes regularization
        loss = compute_loss(y, y_pred) + lambda_reg * np.sum(np.abs(weights))
        loss_history.append(loss)
        
        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.5f}")
    
    return weights, bias, loss_history


# Step 5: Visualization
def plot_loss(loss_history):
    """
    Plot the loss curve.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Split train_data into features (X) and target (y)
    X_train = train_data.drop(columns=['Sales']).values
    y_train = train_data['Sales'].values
    
    # Set hyperparameters
    learning_rate = 0.001
    epochs = 50
    regularization_params = [1, 0.1, 0.001]  # L1 regularization strengths
    
    # Train models with different regularization parameters
    loss_histories = []
    for lambda_reg in regularization_params:
        print(f"Training with L1 regularization (lambda = {lambda_reg})...")
        weights, bias, loss_history = train_sgd_l1(X_train, y_train, epochs, learning_rate, lambda_reg)
        loss_histories.append((lambda_reg, loss_history))
    
    # Plot the loss curves
    plt.figure(figsize=(10, 6))
    for lambda_reg, loss_history in loss_histories:
        plt.plot(loss_history, label=f"L1 Regularization (λ={lambda_reg})")
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss with L1 Regularization')
    plt.legend()
    plt.show()
