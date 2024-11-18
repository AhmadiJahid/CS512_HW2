import pandas as pd
from sklearn.preprocessing import StandardScaler

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

# One-hot encoding for 'ShelveLoc'
train_data = pd.get_dummies(train_data, columns=['ShelveLoc'], drop_first=False)
test_data = pd.get_dummies(test_data, columns=['ShelveLoc'], drop_first=False)

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
