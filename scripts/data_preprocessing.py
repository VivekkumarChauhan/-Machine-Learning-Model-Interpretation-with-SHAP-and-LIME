import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Convert categorical variables, e.g., `sex`, `cp`
    data = pd.get_dummies(data, columns=['sex', 'cp', 'restecg', 'slope', 'thal'], drop_first=True)
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
