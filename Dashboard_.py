from dash import Dash, html, dcc, callback, Output, Input
import numpy as np
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

company_config = [
    {
        "company_name": "Tesla", #ticker originalmente
        "company_header": "Tesla (TSLA) - 1 Year",
        "input_file": "TSLA(1Y).csv",
        "future_predictions": "FuturePrediction_LSTM_TSLA(1Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "metadata": [{
            "name": "Hauptsitz",
            "value": "Austin"
        }, {
            "name": "Gründung",
            "value": "1994"
        }, {
            "name": "Sektor",
            "value": "Consumer Cyclical"
        }, {
            "name": "Branche",
            "value": "Auto Manufacturers"
        }, {
            "name": "Vollzeitmitarbeiter",
            "value": "127.855"
        }]
    },
    {
        "company_name": "Tesla", #prueba de segunda empresa
        "company_header": "Tesla (TSLA) - 5 Years",
        "input_file": "TSLA(5Y).csv",
        "future_predictions": "FuturePrediction_LSTM_TSLA(5Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "metadata": [{
            "name": "Hauptsitz",
            "value": "Austin"
        }, {
            "name": "Gründung",
            "value": "1994"
        }, {
            "name": "Sektor",
            "value": "Consumer Cyclical"
        }, {
            "name": "Branche",
            "value": "Auto Manufacturers"
        }, {
            "name": "Vollzeitmitarbeiter",
            "value": "127.855"
        }]
    },
    {
        "company_name": "Amazon", #prueba de segunda empresa
        "company_header": "Amazon (AMZN) - 1 Year",
        "input_file": "AMZN(1Y).csv",
        "future_predictions": "FuturePrediction_LSTM_AMZN(1Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "metadata": [{
            "name": "Hauptsitz",
            "value": "Seattle"
        }, {
            "name": "Gründung",
            "value": "1994"
        }, {
            "name": "Sektor",
            "value": "Consumer Cyclical"
        }, {
            "name": "Branche",
            "value": "Internet Retail"
        }, {
            "name": "Vollzeitmitarbeiter",
            "value": "1.541.000"
        }]
    },
    {
        "company_name": "Amazon", #prueba de segunda empresa
        "company_header": "Amazon (AMZN) - 5 Years",
        "input_file": "AMZN(5Y).csv",
        "future_predictions": "FuturePrediction_LSTM_AMZN(5Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "metadata": [{
            "name": "Hauptsitz",
            "value": "Seattle"
        }, {
            "name": "Gründung",
            "value": "1994"
        }, {
            "name": "Sektor",
            "value": "Consumer Cyclical"
        }, {
            "name": "Branche",
            "value": "Internet Retail"
        }, {
            "name": "Vollzeitmitarbeiter",
            "value": "1.541.000"
        }]
    },
    {
        "company_name": "Allianz", #prueba de segunda empresa
        "company_header": "Allianz (ALV.DE) - 1 Year",
        "input_file": "ALV.DE(1Y).csv",
        "future_predictions": "FuturePrediction_LSTM_ALV.DE(1Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "metadata": [{
            "name": "Hauptsitz",
            "value": "München"
        }, {
            "name": "Gründung",
            "value": "1890"
        }, {
            "name": "Sektor",
            "value": "Financial Services"
        }, {
            "name": "Branche",
            "value": "Insurance—Diversified"
        }, {
            "name": "Vollzeitmitarbeiter",
            "value": "159.253"
        }]
    },
    {
        "company_name": "Allianz", #prueba de segunda empresa
        "company_header": "Allianz (ALV.DE) - 5 Years",
        "input_file": "ALV.DE(5Y).csv",
        "future_predictions": "FuturePrediction_LSTM_ALV.DE(5Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "metadata": [{
            "name": "Hauptsitz",
            "value": "München"
        }, {
            "name": "Gründung",
            "value": "1890"
        }, {
            "name": "Sektor",
            "value": "Financial Services"
        }, {
            "name": "Branche",
            "value": "Insurance—Diversified"
        }, {
            "name": "Vollzeitmitarbeiter",
            "value": "159.253"
        }]
    },
    {
        "company_name": "^MDAXI", #prueba de segunda empresa
        "company_header": "^MDAXI - 1 Year",
        "input_file": "^MDAXI(1Y).csv",
        "future_predictions": "FuturePrediction_LSTM_^MDAXI(1Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "metadata": []
    },
    {
        "company_name": "^MDAXI", #prueba de segunda empresa
        "company_header": "^MDAXI - 5 Years",
        "input_file": "^MDAXI(5Y).csv",
        "future_predictions": "FuturePrediction_LSTM_^MDAXI(5Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "metadata": []
    },
    {
        "company_name": "NVIDIA Corp", #prueba de segunda empresa
        "company_header": "NVIDIA Corp (NVD.DE) - 1 Year",
        "input_file": "NVD.DE(1Y).csv",
        "future_predictions": "FuturePrediction_LSTM_NVD.DE(1Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "metadata": [{
            "name": "Hauptsitz",
            "value": "Santa Clara"
        }, {
            "name": "Gründung",
            "value": "1993"
        }, {
            "name": "Sektor",
            "value": "Technology"
        }, {
            "name": "Branche",
            "value": "Semiconductors"
        }, {
            "name": "Vollzeitmitarbeiter",
            "value": "26.196"
        }]
    },
    {
        "company_name": "NVIDIA Corp", #prueba de segunda empresa
        "company_header": "NVIDIA Corp (NVD.DE) - 5 Years",
        "input_file": "NVD.DE(5Y).csv",
        "future_predictions": "FuturePrediction_LSTM_NVD.DE(5Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "metadata": [{
            "name": "Hauptsitz",
            "value": "Santa Clara"
        }, {
            "name": "Gründung",
            "value": "1993"
        }, {
            "name": "Sektor",
            "value": "Technology"
        }, {
            "name": "Branche",
            "value": "Semiconductors"
        }, {
            "name": "Vollzeitmitarbeiter",
            "value": "26.196"
        }]
    },
    {
        "company_name": "Deutsche Post AG", #prueba de segunda empresa
        "company_header": "Deutsche Post AG (DPW.DE) - 1 Year",
        "input_file": "DPW.DE(1Y).csv",
        "future_predictions": "FuturePrediction_LSTM_DPW.DE(1Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "metadata": [{
            "name": "Hauptsitz",
            "value": "Bonn"
        }, {
            "name": "Gründung",
            "value": "1490"
        }, {
            "name": "Sektor",
            "value": "Industrials"
        }, {
            "name": "Branche",
            "value": "Integrated Freight & Logistics"
        }, {
            "name": "Vollzeitmitarbeiter",
            "value": "590.605"
        }]
    },
    {
        "company_name": "Deutsche Post AG", #prueba de segunda empresa
        "company_header": "Deutsche Post AG (DPW.DE) - 5 Years",
        "input_file": "DPW.DE(5Y).csv",
        "future_predictions": "FuturePrediction_LSTM_DPW.DE(5Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "metadata": [{
            "name": "Hauptsitz",
            "value": "Bonn"
        }, {
            "name": "Gründung",
            "value": "1490"
        }, {
            "name": "Sektor",
            "value": "Industrials"
        }, {
            "name": "Branche",
            "value": "Integrated Freight & Logistics"
        }, {
            "name": "Vollzeitmitarbeiter",
            "value": "590.605"
        }]
    },      
    {
        "company_name": "McDonald's Corporation", #prueba de segunda empresa
        "company_header": "McDonald's Corporation (MDO.DE) - 1 Year",
        "input_file": "MDO.DE(1Y).csv",
        "future_predictions": "FuturePrediction_LSTM_MDO.DE(1Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "metadata": [{
            "name": "Hauptsitz",
            "value": "Chicago"
        }, {
            "name": "Gründung",
            "value": "1940"
        }, {
            "name": "Sektor",
            "value": "Consumer Cyclical"
        }, {
            "name": "Branche",
            "value": "Restaurants"
        }, {
            "name": "Vollzeitmitarbeiter",
            "value": "100.000"
        }]
    },
    {
        "company_name": "McDonald's Corporation", #prueba de segunda empresa
        "company_header": "McDonald's Corporation (MDO.DE) - 5 Years",
        "input_file": "MDO.DE(5Y).csv",
        "future_predictions": "FuturePrediction_LSTM_MDO.DE(5Y).csv", #change to LR
        "columns_options": ['Open', 'High', 'Low','Close'],
        "metadata": [{
            "name": "Hauptsitz",
            "value": "Chicago"
        }, {
            "name": "Gründung",
            "value": "1940"
        }, {
            "name": "Sektor",
            "value": "Consumer Cyclical"
        }, {
            "name": "Branche",
            "value": "Restaurants"
        }, {
            "name": "Vollzeitmitarbeiter",
            "value": "100.000"
        }]
    },       
]

current_company = company_config[0]

app = Dash(__name__)

ticker = current_company["company_name"]
input_df = pd.read_csv(current_company["input_file"]) #modified .csv
fig1 = px.line(input_df, x='Date', y="Open")

company_options = [{"label": company_config[n]["company_header"], "value": n} for n in range(len(company_config))]

app.layout = html.Div([
    html.H1(children=ticker, style={'textAlign':'center'}),
    html.Hr(),
    html.H1(children = 'Analyse:', style={'textAlign':'left'}),
    dcc.Dropdown(options=company_options, value=1, id='company-control'),
#    html.P(children= "Hier kann man zwischen die unterschiedlichen Aktienkurse auswählen."
#                    "Alle Diagramme werden mit der Auswahl aktualisiert."),
    html.Div([
        html.H1(children=current_company["company_header"], id="company-header"),
        html.Div(id="company-metadata"),
        dcc.RadioItems(options=current_company["columns_options"], value=current_company["columns_options"][0], id='controls-and-radio-item'),
        html.Br(),
        html.Div([
            html.Span(children="Min: Loading...", id="historic-min"),
            html.Br(),
            html.Span(children="Max: Loading...", id="historic-max"),
            html.Br(),
            html.Span(children="Durchschnitt: Loading...", id="historic-avg"),
        ]),
        html.Br(),
        dcc.Graph(id='graph-history', figure = fig1),
    ]),
    html.Hr(),
    html.Div([
        html.Hr(),
        html.H1(children = 'Prognose (LSTM): '),
        dcc.Graph(id='graph-comparison')
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
    

@callback(
    Output('company-metadata', 'children'),
    Input('company-control', 'value')
)
def update_metadata(company_index):
    current_company = company_config[company_index]
    metadata_values = current_company["metadata"]
    metadata_structure = []
    for value in metadata_values:
        parent = html.Div([
            html.Strong(children=f"{value['name']}: "),
            html.Span(children=value["value"])
        ])
        metadata_structure.append(parent)
    if len(metadata_structure) > 0:
        metadata_structure.append(html.Br())
    return metadata_structure

#### callback for the second graph
@callback(
    Output('graph-history', 'figure'),
    Output('historic-min', 'children'),
    Output('historic-max', 'children'),
    Output('historic-avg', 'children'),
    Input('controls-and-radio-item', 'value'),
    Input('company-control', 'value')
)
def update_graph(col_chosen, company_index):
    filename = company_config[company_index]["input_file"]
    input_df = pd.read_csv(filename) #modified .csv
    min_value = '{:.2f}'.format(min(input_df[col_chosen]))
    avg_value = '{:.2f}'.format(sum(input_df[col_chosen]) / len(input_df[col_chosen]))
    max_value = '{:.2f}'.format(max(input_df[col_chosen]))
    fig1 = px.line(input_df, x='Date', y=col_chosen)
    return fig1, f'Min: {min_value}', f'Max: {max_value}', f'Durchschnitt: {avg_value}'

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
    future_predictions_filename = current_company["future_predictions"] #debe ser y_training, pero y_training no tiene fechas
    

    history_df = pd.read_csv(history_filename)
    future_df = pd.read_csv(future_predictions_filename)
    dataset_train = history_df[:int(history_df.shape[0]*0.8)]
    dataset_test = history_df[int(history_df.shape[0]*0.8):]

    max_date = max(history_df["Date"]) #debe ser el 70%
    future_df["Date"] = future_df.index.to_series().map(lambda index : convert_index_to_date(index, max_date))
    # Create figure
    fig = px.line(dataset_train, x="Date", y="Close")
    
    fig.add_traces(list(px.line(dataset_test, y="Close", x="Date").update_traces(line_color="darkgreen", name="Test data", showlegend=True).select_traces()))
    fig.add_traces(list(px.line(future_df, y="Prediction_LSTM", x="Date").update_traces(line_color="teal", name="Future predictions", showlegend=True).select_traces()))

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)

    #multiple outputs
