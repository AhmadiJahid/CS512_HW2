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
    Tracks the training loss for each step (data point) during training.
    
    Parameters:
        X (ndarray): Training data features (n_samples, n_features).
        y (ndarray): True target values (n_samples,).
        epochs (int): Number of epochs.
        learning_rate (float): Learning rate.
        regularization (str): Type of regularization ("l1" or None).
        lambda_reg (float): Regularization strength (λ).
    
    Returns:
        weights (ndarray): Final weights after training.
        bias (float): Final bias after training.
        step_losses (list): Step-wise training losses during training.
    """
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)  # Initialize weights
    bias = 0  # Initialize bias
    step_losses = []  # Track step-wise losses

    for epoch in range(epochs):
        for i in range(n_samples):
            # Get a single data point
            X_i = X[i:i+1]
            y_i = y[i:i+1]
            
            # Call sgd_update for a single step
            weights, bias, y_pred = sgd_update(X_i, y_i, weights, bias, learning_rate, regularization, lambda_reg)
            
            # Compute loss for this step and append to step_losses
            step_loss = compute_loss(y_i, y_pred)
            if regularization == "l1":
                step_loss += lambda_reg * np.sum(np.abs(weights))
            step_losses.append(step_loss)

        # Print progress every epoch
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {step_loss:.5f}")
    
    return weights, bias, step_losses
# Step 5: Visualization
def smooth_curve(values, window_size=100):
        return np.convolve(values, np.ones(window_size)/window_size, mode='valid')

# Plotting Function for Coefficient Magnitudes
def plot_coefficients(log_lambdas, lasso_weights, feature_names):
    """
    Plots coefficient magnitudes for LASSO regularized models.
    
    Parameters:
        log_lambdas (list): Logarithmic values of lambda (regularization strengths).
        lasso_weights (ndarray): Coefficient weights for each lambda.
        feature_names (list): Names of features.
    """
    plt.figure(figsize=(10, 6))
    for i, feature in enumerate(feature_names):
        plt.plot(log_lambdas, lasso_weights[:, i], label=feature)

    plt.xlabel("log(λ)")
    plt.ylabel("Coefficient Magnitude")
    plt.title("Coefficient Magnitudes for LASSO Regularized Models")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Plotting Function for Training Loss
def plot_loss(loss_histories, window_size=100):
    plt.figure(figsize=(10, 6))
    for label, step_losses in loss_histories:
        smoothed_losses = smooth_curve(step_losses, window_size)
        plt.plot(range(len(smoothed_losses)), smoothed_losses, label=label)

    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Step-wise Training Loss for Different Configurations")
    plt.legend()
    plt.grid(True)
    plt.show()


# Evaluate Test Loss
def evaluate_test_loss(X_test, y_test, weights, bias):
    y_pred = predict(X_test, weights, bias)
    test_loss = compute_loss(y_test, y_pred)
    return test_loss

def display_test_results(test_results):
    # Convert the results to a DataFrame
    results_df = pd.DataFrame(test_results)
    results_df = results_df.sort_values(by="Test Loss", ascending=True)


    # Print a formatted table
    print("\nTest Loss Results (Neatly Structured):\n")
    print(results_df.to_markdown(index=False, tablefmt="grid"))

if __name__ == "__main__":
    import pandas as pd

    # Split train_data into features (X) and target (y)
    X_train = train_data.drop(columns=['Sales']).values
    y_train = train_data['Sales'].values
    feature_names = train_data.drop(columns=['Sales']).columns.tolist()

    # Split test_data into features (X) and target (y)
    X_test = test_data.drop(columns=['Sales']).values
    y_test = test_data['Sales'].values

    # Set hyperparameters and configurations
    epochs = 50
    configurations = [
        {"regularization": None, "learning_rate": 0.01, "lambda_reg": 0, "label": "No Regularization (η=0.01)"},
        {"regularization": None, "learning_rate": 0.001, "lambda_reg": 0, "label": "No Regularization (η=0.001)"},
        {"regularization": "l1", "learning_rate": 0.001, "lambda_reg": 1, "label": "L1 Regularization (λ=1, η=0.001)"},
        {"regularization": "l1", "learning_rate": 0.001, "lambda_reg": 0.1, "label": "L1 Regularization (λ=0.1, η=0.001)"},
        {"regularization": "l1", "learning_rate": 0.001, "lambda_reg": 0.001, "label": "L1 Regularization (λ=0.001, η=0.001)"}
    ]

    # Train models and store loss histories, weights, and test losses
    loss_histories = []
    lasso_lambdas = []  # Regularization strengths for LASSO
    lasso_weights = []  # To store weights for LASSO configurations
    test_results = []   # To store test loss results

    for config in configurations:
        print(f"Training: {config['label']}")
        # Train the model
        weights, bias, loss_history = train_sgd(
            X_train,
            y_train,
            epochs,
            config["learning_rate"],
            regularization=config["regularization"],
            lambda_reg=config["lambda_reg"]
        )
        # Store training loss for visualization
        loss_histories.append((config["label"], loss_history))

        # Evaluate and store test loss
        test_loss = evaluate_test_loss(X_test, y_test, weights, bias)
        test_results.append({
            "Configuration": config["label"],
            "Test Loss": round(test_loss, 5)
        })

        # For LASSO configurations, store weights for coefficient magnitude plotting
        if config["regularization"] == "l1":
            lasso_lambdas.append(config["lambda_reg"])
            lasso_weights.append(weights)

    # Plot training loss for all configurations
    plot_loss(loss_histories)

    # Plot coefficient magnitudes for LASSO models (Part b)
    if lasso_lambdas:
        log_lambdas = np.log(lasso_lambdas)  # Use log(λ) for the x-axis
        lasso_weights = np.array(lasso_weights)  # Convert weights to numpy array
        plot_coefficients(log_lambdas, lasso_weights, feature_names)

    # Display test results in a neat table
    display_test_results(test_results)



