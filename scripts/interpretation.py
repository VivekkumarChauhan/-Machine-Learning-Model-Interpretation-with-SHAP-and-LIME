import shap
import lime
import lime.lime_tabular
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
data = pd.read_csv('data/heart_disease.csv')  # Replace with the actual path to your dataset

# Check for NaN values in the dataset
if data.isnull().values.any():
    print("Data contains NaN values. Here's a summary of missing values:")
    print(data.isnull().sum())
    
    # Handle NaN values in the target column
    if 'target' in data.columns:
        # Print rows where target is NaN for inspection
        missing_rows = data[data['target'].isnull()]
        if not missing_rows.empty:
            print("Rows with missing target values:")
            print(missing_rows)

        # Option 1: Drop rows with NaN in the target column
        data = data.dropna(subset=['target'])
        print(f"Remaining rows after dropping NaNs in 'target': {data.shape[0]}")

    else:
        print("Target column 'target' not found in the dataset.")

# Check for NaN values again after handling
if data.isnull().values.any():
    print("There are still NaN values in the dataset after handling.")
else:
    print("All NaN values have been successfully addressed.")

# Proceed only if no NaN values are found
if not data.isnull().values.any():
    # Split into features and target
    X = data.drop(columns=['target'])  # Correctly referencing the target column
    y = data['target']  # Use the correct name of the target column

    # Check for NaN values in y
    if y.isnull().any():
        print("Target variable 'y' contains NaN values.")
        print(y.isnull().sum())
    else:
        # Proceed only if no NaN values are found
        print(f"Training on {X.shape[0]} samples.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train and save the model if not already done
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        joblib.dump(model, 'model/random_forest_model.pkl')

        def shap_interpretation(model, X_train, X_test):
            # Initialize SHAP TreeExplainer for the model
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            # Summary plot for SHAP values
            shap.summary_plot(shap_values[1], X_test)  # For binary classification

        def lime_interpretation(model, X_test, instance_idx=0):
            # Initialize LIME TabularExplainer for the model
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_test.values,
                feature_names=X_test.columns,
                class_names=['No Disease', 'Disease'],  # Update as appropriate
                discretize_continuous=True
            )
            # Explain a single instance
            exp = explainer.explain_instance(X_test.values[instance_idx], model.predict_proba)
            exp.show_in_notebook(show_table=True)

        # Run SHAP and LIME interpretation
        shap_interpretation(model, X_train, X_test)
        lime_interpretation(model, X_test, instance_idx=1)
else:
    print("Please address the NaN values before proceeding.")
