import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import joblib
import base64
import io
import os

best_logreg = joblib.load('artifacts/best_logreg.pkl')
best_rf = joblib.load('artifacts/best_rf.pkl')
scaler = joblib.load('artifacts/scaler.pkl')

app = dash.Dash(__name__, assets_folder='src/assets') 
server = app.server  

app.layout = html.Div([
    html.H1('Customer Churn Prediction', className='header'),
    
    html.Div([
        html.Label('Enter Customer Data:', className='label'),
        dcc.Input(id='input-tenure', type='number', placeholder='Tenure (e.g., 12)', style={'margin': '5px'}),
        dcc.Input(id='input-monthlycharges', type='number', placeholder='Monthly Charges (e.g., 60)', style={'margin': '5px'}),
        dcc.Input(id='input-totalcharges', type='number', placeholder='Total Charges (e.g., 500)', style={'margin': '5px'}),
        html.Button('Predict', id='submit-button', n_clicks=0, className='button'),
    ], className='input-container'),
    
    html.Div(id='output-prediction', className='output'),
    
    html.Div([
        html.Label('Upload CSV of Customer Data for Batch Prediction:', className='label'),
        dcc.Upload(id='upload-data', children=html.Button('Upload CSV', className='button'), multiple=False),
        html.Div(id='output-upload', className='output')
    ], className='upload-container')
])

@app.callback(
    Output('output-prediction', 'children'),
    Input('submit-button', 'n_clicks'),
    Input('input-tenure', 'value'),
    Input('input-monthlycharges', 'value'),
    Input('input-totalcharges', 'value')
)
def update_prediction(n_clicks, tenure, monthlycharges, totalcharges):
    if n_clicks > 0 and all([tenure, monthlycharges, totalcharges]):
        input_data = pd.DataFrame([[tenure, monthlycharges, totalcharges]], columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
        input_data_scaled = scaler.transform(input_data)

        logreg_prediction = best_logreg.predict(input_data_scaled)
        rf_prediction = best_rf.predict(input_data_scaled)
        
        return f'Logistic Regression Prediction: {"Churn" if logreg_prediction[0] else "No Churn"}\nRandom Forest Prediction: {"Churn" if rf_prediction[0] else "No Churn"}'
    return "Please enter all customer data."

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
            
            # Ensure all required columns are present
            required_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            for col in required_cols:
                if col not in df_encoded_batch:
                    df_encoded_batch[col] = 0
            
            df_encoded_batch[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(df_encoded_batch[['tenure', 'MonthlyCharges', 'TotalCharges']])
            
            voting_clf = VotingClassifier(estimators=[('logreg', best_logreg), ('rf', best_rf)], voting='hard')
            voting_clf.fit(df_encoded_batch, [0]*len(df_encoded_batch))  # Dummy fit to initialize
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
                    html.Tr([html.Th(col) for col in df.columns])] +
                    [html.Tr([html.Td(df.iloc[i][col]) for col in df.columns]) for i in range(min(10, len(df)))]
                )
            ])
        except Exception as e:
            return f'Error processing file: {e}'
    return "Please upload a CSV file."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run_server(debug=False, host="0.0.0.0", port=port)