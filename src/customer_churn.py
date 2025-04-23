import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

filename = "data/Telco-Customer-Churn.csv"
df = pd.read_csv(filename)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())

df_encoded = pd.get_dummies(df, drop_first=True)

scaler = StandardScaler()
df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']])

X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 300]
}
logreg_model = LogisticRegression(random_state=42)
logreg_search = RandomizedSearchCV(logreg_model, logreg_params, cv=5, n_iter=10, random_state=42, n_jobs=-1)

rf_params = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_model = RandomForestClassifier(random_state=42)
rf_search = RandomizedSearchCV(rf_model, rf_params, cv=5, n_iter=10, random_state=42, n_jobs=-1)

logreg_search.fit(X_train, y_train)
rf_search.fit(X_train, y_train)

best_logreg = logreg_search.best_estimator_
best_rf = rf_search.best_estimator_

voting_clf = VotingClassifier(estimators=[('logreg', best_logreg), ('rf', best_rf)], voting='hard')
voting_clf.fit(X_train, y_train)

joblib.dump(best_logreg, 'best_logreg.pkl')
joblib.dump(best_rf, 'best_rf.pkl')

voting_preds = voting_clf.predict(X_test)
voting_accuracy = accuracy_score(y_test, voting_preds)
voting_report = classification_report(y_test, voting_preds)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Customer Churn Prediction'),
    
    html.Div([
        html.Label('Enter Customer Data:'),
        dcc.Input(id='input-tenure', type='number', placeholder='Tenure (e.g., 12)', style={'margin': '5px'}),
        dcc.Input(id='input-monthlycharges', type='number', placeholder='Monthly Charges (e.g., 60)', style={'margin': '5px'}),
        dcc.Input(id='input-totalcharges', type='number', placeholder='Total Charges (e.g., 500)', style={'margin': '5px'}),
        html.Button('Predict', id='submit-button', n_clicks=0, style={'margin': '5px'}),
    ], style={'padding': '20px'}),
    
    html.Div(id='output-prediction', style={'padding': '20px'}),
    
    html.Div([
        html.Label('Upload CSV of Customer Data for Batch Prediction:'),
        dcc.Upload(id='upload-data', children=html.Button('Upload CSV'), multiple=False),
        html.Div(id='output-upload')
    ], style={'padding': '20px'})
])

@app.callback(
    Output('output-prediction', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('input-tenure', 'value'),
    Input('input-monthlycharges', 'value'),
    Input('input-totalcharges', 'value')
)
def update_prediction(n_clicks, tenure, monthlycharges, totalcharges):
    if n_clicks > 0:
        input_data = pd.DataFrame([[tenure, monthlycharges, totalcharges]], columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
        input_data_scaled = scaler.transform(input_data)

        logreg_prediction = best_logreg.predict(input_data_scaled)
        rf_prediction = best_rf.predict(input_data_scaled)
        
        return f'Logistic Regression Prediction: {"Churn" if logreg_prediction[0] else "No Churn"}\nRandom Forest Prediction: {"Churn" if rf_prediction[0] else "No Churn"}'

import base64
import io

@app.callback(
    Output('output-upload', 'children'),
    Input('upload-data', 'contents')
)
def update_output(content):
    if content is not None:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        try:

            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df_encoded_batch = pd.get_dummies(df, drop_first=True)
            df_encoded_batch[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(df_encoded_batch[['tenure', 'MonthlyCharges', 'TotalCharges']])
            predictions = voting_clf.predict(df_encoded_batch)
            df['Churn Prediction'] = ['Churn' if p == 1 else 'No Churn' for p in predictions]

            return html.Div([
                html.H5('Predictions for uploaded customers:'),
                dcc.Graph(
                    figure={
                        'data': [
                            {'x': df['customerID'], 'y': predictions, 'type': 'bar', 'name': 'Prediction'},
                        ],
                        'layout': {
                            'title': 'Customer Churn Predictions'
                        }
                    }
                ),
                html.Table([
                    html.Tr([html.Th(col) for col in df.columns])]+
                    [html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(min(10, len(df)))]
                )
            ])
        except Exception as e:
            return f'Error processing file: {e}'

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run_server(debug=False, host="0.0.0.0", port=port)
