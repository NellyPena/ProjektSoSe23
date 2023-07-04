from dash import Dash, html, dcc, callback, Output, Input
import numpy as np
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

company_config = [
    {
        "company_name": "Tesla", #ticker originalmente
        "company_header": "Tesla (TSLA) - 1 Year",
        "input_file": "TSLA1Y.csv",
        "predictions_lstm": "Prediction_LSTM_TSLA(1Y).csv",
        "predictions_gru": "Prediction_GRU_TSLA(1Y).csv", #change to GRU
        "predictions_LR": "Prediction_LR_TSLA(1Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "mse_lstm" : "MSE_LSTM_TSLA(1Y).csv",########
        "mse_gru" : "MSE_GRU_TSLA(1Y).csv",
        "accuracy_LR" : "Accuracy_LR_MSFT(1Y).csv"
    },
    {
        "company_name": "Tesla", #prueba de segunda empresa
        "company_header": "Tesla (TSLA) - 5 Years",
        "input_file": "TSLA5Y.csv",
        "predictions_lstm": "Prediction_LSTM_TSLA(5Y).csv",
        "predictions_gru": "Prediction_GRU_TSLA(5Y).csv", #change to GRU
        "predictions_LR": "Prediction_LR_TSLA(5Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "mse_lstm" : "MSE_LSTM_TSLA(1Y).csv",#######
        "mse_gru" : "MSE_GRU_TSLA(1Y).csv",
        "accuracy_LR" : "Accuracy_LR_MSFT(1Y).csv"
    }
]

current_company = company_config[0]

app = Dash(__name__)

ticker = current_company["company_name"]
input_df = pd.read_csv(current_company["input_file"]) #modified .csv
predictions_gru_df = pd.read_csv(current_company["predictions_gru"])
predictions_lstm_df = pd.read_csv(current_company["predictions_lstm"])
fig = px.line(predictions_gru_df, y='Prediction_GRU')
fig1 = px.line(input_df, x='Date', y="Open")
fig2 = px.line(predictions_lstm_df, y='Prediction_LSTM')

company_options = [{"label": company_config[n]["company_header"], "value": n} for n in range(len(company_config))]

app.layout = html.Div([
    html.H1(children=ticker, style={'textAlign':'center'}),
    html.Hr(),
    html.H1(children = 'Analyse:', style={'textAlign':'left'}),
    dcc.Dropdown(options=company_options, value=0, id='company-control'),
#    html.P(children= "Hier kann man zwischen die unterschiedlichen Aktienkurse auswählen."
#                    "Alle Diagramme werden mit der Auswahl aktualisiert."),
    html.Div([
        html.H1(children=current_company["company_header"], id="company-header"),
        dcc.RadioItems(options=current_company["columns_options"], value=current_company["columns_options"][0], id='controls-and-radio-item'),
        html.Br(),
        html.Div([
            html.Span(children="Min: Loading...", id="historic-min"),
            html.Br(),
            html.Span(children="Max: Loading...", id="historic-max"),
        ]),
        html.Br(),
        dcc.Graph(id='graph-history', figure = fig1),
    ]),
    html.Hr(),
    html.Div([
        html.Hr(),
        html.H1(children = 'Vergleich: '),
        dcc.Graph(id='graph-comparison', figure = fig)
    ])
    
])


#### callback for options and headings
@callback(
    Output('company-header', 'children'),
    Output('controls-and-radio-item', 'options'),
    Output('controls-and-radio-item', 'value'),
    Input('company-control', 'value')
)
def update_company(company_index):
    current_company = company_config[company_index]
    return current_company["company_header"], current_company["columns_options"], current_company["columns_options"][0]
    

#### callback for the second graph
@callback(
    Output('graph-history', 'figure'),
    Output('historic-min', 'children'),
    Output('historic-max', 'children'),
    Input('controls-and-radio-item', 'value'),
    Input('company-control', 'value')
)
def update_graph(col_chosen, company_index):
    filename = company_config[company_index]["input_file"]
    input_df = pd.read_csv(filename) #modified .csv
    min_value = '{:.2f}'.format(min(input_df[col_chosen]))
    max_value = '{:.2f}'.format(max(input_df[col_chosen]))
    fig1 = px.line(input_df, x='Date', y=col_chosen)
    return fig1, f'Min: {min_value}', f'Max: {max_value}'

def convert_index_to_date(index, base_date):
    converted_date = datetime.strptime(base_date, '%Y-%m-%d').date()
    end_date = converted_date + timedelta(days=index)
    return end_date

#### callback for the first graph
@callback(
    Output('graph-comparison', 'figure'),
    Input('company-control', 'value')
)
def update_comparison(company_index):
    current_company = company_config[company_index]
    history_filename = current_company["input_file"] #debe ser y_training, pero y_training no tiene fechas
    lstm_filename = current_company["predictions_lstm"]
    gru_filename = current_company["predictions_gru"]
    accuracy_filename = current_company["accuracy_LR"]

    accuracy_file = open(accuracy_filename, "r")
    accuracy = float(accuracy_file.read())
    history_df = pd.read_csv(history_filename)
    dataset_train = history_df[:int(history_df.shape[0]*0.7)]
    dataset_test = history_df[int(history_df.shape[0]*0.7):]
    lstm_df = pd.read_csv(lstm_filename) #debe empezar antes, al finalizar y_training = historico * 0.7
    gru_df = pd.read_csv(gru_filename) #debe empezar antes, al finalizar y_training = historico * 0.7
    max_date = max(dataset_train["Date"]) #debe ser el 70%

    # Create figure
    fig = px.line(dataset_train, x="Date", y="Close")
    lstm_df["Date"] = lstm_df.index.to_series().map(lambda index : convert_index_to_date(index, max_date))
    gru_df["Date"] = gru_df.index.to_series().map(lambda index : convert_index_to_date(index, max_date))

    fig.add_traces(list(px.line(lstm_df, y="Prediction_LSTM", x="Date").update_traces(line_color="darkred", name="LSTM", showlegend=True).select_traces()))
    fig.add_traces(list(px.line(gru_df, y="Prediction_GRU", x="Date").update_traces(line_color="teal", name="GRU", showlegend=True).select_traces()))
    fig.add_traces(list(px.line(dataset_test, y="Close", x="Date").update_traces(line_color="darkgreen", name="Test data", showlegend=True).select_traces()))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

    #multiple outputs
