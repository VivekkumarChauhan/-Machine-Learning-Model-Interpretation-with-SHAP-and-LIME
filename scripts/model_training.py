import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print accuracy."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.2f}')
    return accuracy

if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv('data/heart_disease.csv')  # Adjust the path as needed

    # Check for NaN values and handle them
    if data.isnull().values.any():
        print("Data contains NaN values. Here's a summary of missing values:")
        print(data.isnull().sum())

        # Handle NaN values in the target column
        if 'target' in data.columns:
            # Drop rows with NaN in the target column
            data = data.dropna(subset=['target'])  # Drop rows where target is NaN
            print(f'Dropped {data.isnull().sum()["target"]} rows with missing target values.')

    # Split the dataset into features and target variable
    X = data.drop(columns=['target'])  # Features
    y = data['target']  # Target variable

    # Check if the dataset is empty after dropping NaN values
    if X.empty or y.empty:
        print("No data available after dropping NaN values. Please check the dataset.")
    else:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        accuracy = evaluate_model(model, X_test, y_test)

        # Save the trained model
        joblib.dump(model, 'model/random_forest_model.pkl')
        print("Model saved successfully.")
