import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create artifacts directory if it doesn't exist
if not os.path.exists('artifacts'):
    os.makedirs('artifacts')

# Load and preprocess data
filename = "data/Telco-Customer-Churn.csv"
df = pd.read_csv(filename)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Scale numerical features
scaler = StandardScaler()
df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Save the scaler
joblib.dump(scaler, 'artifacts/scaler.pkl')

# Prepare features and target
X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grids
logreg_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 300]
}
rf_params = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Train Logistic Regression with RandomizedSearchCV
logreg_model = LogisticRegression(random_state=42)
logreg_search = RandomizedSearchCV(logreg_model, logreg_params, cv=5, n_iter=10, random_state=42, n_jobs=-1)
logreg_search.fit(X_train, y_train)
best_logreg = logreg_search.best_estimator_

# Train Random Forest with RandomizedSearchCV
rf_model = RandomForestClassifier(random_state=42)
rf_search = RandomizedSearchCV(rf_model, rf_params, cv=5, n_iter=10, random_state=42, n_jobs=-1)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

# Save the trained models
joblib.dump(best_logreg, 'artifacts/best_logreg.pkl')
joblib.dump(best_rf, 'artifacts/best_rf.pkl')

print("Models and scaler saved to 'artifacts' folder:")
print("- artifacts/best_logreg.pkl")
print("- artifacts/best_rf.pkl")
print("- artifacts/scaler.pkl")