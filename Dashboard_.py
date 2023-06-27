from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import Constants

company_config = [
    {
        "company_name": "TSLA",
        "company_header": "Tesla (TSLA)",
        "input_file": "TSLA.csv",
        "predictions_lstm": "Prediction_LSTM.csv",
        "predictions_gru": "Prediction_GRU.csv",
        "predictions_LR": "Prediction_LR.csv",
        "columns_options": ['Open', 'High', 'Low','Close']
    },
    {
        "company_name": "MSFT",
        "company_header": "Microsoft (MSFT)",
        "input_file": "MSFT.csv",
        "predictions_lstm": "Prediction_LSTM_MSFT.csv",
        "predictions_gru": "Prediction_GRU_MSFT.csv",
        "predictions_LR": "Prediction_LR_MSFT.csv",
        "columns_options": ['Open', 'High', 'Low','Close']
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
    html.H2(children = 'Analyse', style={'textAlign':'center'}),
    html.P(children= "Man kann zwischen Aktienkurse wählen."
                    " Alle Diagramme werden mit der Auswahl aktualisiert."),
    html.Hr(),
    dcc.Dropdown(options=company_options, value=0, id='company-control'),
    html.H1(children=current_company["company_header"], id="company-header"),
    dcc.RadioItems(options=current_company["columns_options"], value=current_company["columns_options"][0], id='controls-and-radio-item'),
    dcc.Graph(id='graph-content1', figure = fig1),
    html.H1(children = 'Prediction Neural'),
    dcc.Graph(id='graph-content', figure = fig),
    html.H1(children='Prediction_LSTM_test'),
    dcc.Graph(id='graph-content2', figure = fig2),
    html.Div([
        html.Hr(),
        html.H1(children = 'Comparison'),
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
    

#### callback for the first graph
@callback(
    Output(component_id='graph-content', component_property='figure'),
    Input('company-control', 'value')
)
def update_graph(company_index):
    filename = company_config[company_index]["predictions_gru"]
    predictions_gru_df = pd.read_csv(filename)
    fig = px.line(predictions_gru_df)
    return fig

#### callback for the second graph
@callback(
    Output('graph-content1', 'figure'),
    Input('controls-and-radio-item', 'value'),
    Input('company-control', 'value')
)
def update_graph(col_chosen, company_index):
    filename = company_config[company_index]["input_file"]
    input_df = pd.read_csv(filename) #modified .csv
    fig1 = px.line(input_df, x='Date', y=col_chosen)
    return fig1

#### callback for the third graph
@callback(
    Output('graph-content2', 'figure'),
    Input('company-control', 'value')
)
def update_graph(company_index):
    filename = company_config[company_index]["predictions_lstm"]
    input_df = pd.read_csv(filename) #modified .csv
    fig2 = px.line(input_df)
    return fig2


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
    history_filename = company_config[company_index]["input_file"]
    lstm_filename = company_config[company_index]["predictions_lstm"]
    gru_filename = company_config[company_index]["predictions_gru"]
    history_df = pd.read_csv(history_filename)
    lstm_df = pd.read_csv(lstm_filename)
    gru_df = pd.read_csv(gru_filename)

    max_date = max(history_df["Date"])

    # Create figure
    fig = px.line(history_df, x="Date", y="Close")
    lstm_df["Date"] = lstm_df.index.to_series().map(lambda index : convert_index_to_date(index, max_date))
    gru_df["Date"] = gru_df.index.to_series().map(lambda index : convert_index_to_date(index, max_date))

    fig.add_traces(list(px.line(lstm_df, y="Prediction_LSTM", x="Date").update_traces(line_color="darkred", name="LSTM", showlegend=True).select_traces()))
    fig.add_traces(list(px.line(gru_df, y="Prediction_GRU", x="Date").update_traces(line_color="teal", name="GRU", showlegend=True).select_traces()))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

    #multiple outputs
