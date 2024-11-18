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

def sgd_update(X, y_true, weights, bias, learning_rate, regularization=None, lambda_reg=0):
    n_samples = X.shape[0]
    
    # Predictions
    y_pred = predict(X, weights, bias)
    
    # Gradients
    dw = -(2 / n_samples) * np.dot(X.T, (y_true - y_pred))
    db = -(2 / n_samples) * np.sum(y_true - y_pred)
    
    # Add L1 regularization term to gradients if applicable
    if regularization == "l1":
        dw += lambda_reg * np.sign(weights)
    
    # Update weights and bias
    weights -= learning_rate * dw
    bias -= learning_rate * db
    
    return weights, bias, y_pred



def train_sgd(X, y, epochs, learning_rate, regularization=None, lambda_reg=0):
    """
    Train the linear regression model using SGD, with optional L1 regularization.
    """
    n_features = X.shape[1]
    weights = np.zeros(n_features)  # Initialize weights to 0
    bias = 0  # Initialize bias to 0
    
    loss_history = []
    
    for epoch in range(epochs):
        # Call sgd_update for a single step of SGD
        weights, bias, y_pred = sgd_update(X, y, weights, bias, learning_rate, regularization, lambda_reg)
        
        loss = compute_loss(y, y_pred)
        if regularization == "l1":
            loss += lambda_reg * np.sum(np.abs(weights))
        
        loss_history.append(loss)
        
        # Optional: Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.5f}")
    
    return weights, bias, loss_history




# Step 5: Visualization
def plot_loss(loss_histories):
    # Plot all training loss curves
    plt.figure(figsize=(10, 6))
    for label, loss_history in loss_histories:
        plt.plot(loss_history, label=label)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Configurations')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Split train_data into features (X) and target (y)
    X_train = train_data.drop(columns=['Sales']).values
    y_train = train_data['Sales'].values

    # Set hyperparameters and configurations
    epochs = 50
    configurations = [
        {"regularization": None, "learning_rate": 0.01, "lambda_reg": 0, "label": "No Regularization (η=0.01)"},
        {"regularization": None, "learning_rate": 0.001, "lambda_reg": 0, "label": "No Regularization (η=0.001)"},
        {"regularization": "l1", "learning_rate": 0.001, "lambda_reg": 1, "label": "L1 Regularization (λ=1, η=0.001)"},
        {"regularization": "l1", "learning_rate": 0.001, "lambda_reg": 0.1, "label": "L1 Regularization (λ=0.1, η=0.001)"},
        {"regularization": "l1", "learning_rate": 0.001, "lambda_reg": 0.001, "label": "L1 Regularization (λ=0.001, η=0.001)"}
    ]

    # Train models and store loss histories
    loss_histories = []

    for config in configurations:
        print(f"Training: {config['label']}")
        weights, bias, loss_history = train_sgd(
            X_train,
            y_train,
            epochs,
            config["learning_rate"],
            regularization=config["regularization"],
            lambda_reg=config["lambda_reg"]
        )
        loss_histories.append((config["label"], loss_history))

    plot_loss(loss_histories)

