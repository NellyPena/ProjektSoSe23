from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

df = pd.read_csv('Prediction_neural.csv')
df1 = pd.read_csv('TSLA.csv') #modified .csv
df2 = pd.read_csv('Prediction_LSTM_test.csv')

app = Dash(__name__)

fig = px.line(df, y = 'Prediction_Neural')
fig1 = px.line(df1, x='Date', y="Open")
fig2 = px.line(df2, y='Prediction_LSTM')


app.layout = html.Div([
    html.H1(children='Aktienkurse', style={'textAlign':'center'}),
    html.Hr(),
    html.P(children= "Man kann zwischen Aktienkurse wählen."
                    " Alle Diagramme werden mit der Auswahl aktualisiert."),
    html.Hr(),
    dcc.RadioItems(options=['Open', 'High', 'Low','Close'], value='Open', id='controls-and-radio-item'),
    html.H1(children= 'Tesla (TSL)'),
    dcc.Graph(id='graph-content1', figure = fig1),
    html.H1(children = 'Prediction Neural'),
    dcc.Graph(id='graph-content', figure = fig),
    html.H1(children='Prediction_LSTM_test'),
    dcc.Graph(id='graph-content2', figure = fig2)
])

#### callback for the first graph
@callback(
    Output(component_id='graph-content', component_property='figure'),
    Input(component_id='controls-and-radio-item', component_property='value')
)
def update_graph(col_chosen):
    fig = px.line(df)
    return fig

#### callback for the second graph
@callback(
    Output(component_id='graph-content1', component_property='figure'),
    Input(component_id='controls-and-radio-item', component_property='value')
)
def update_graph(col_chosen):
    fig1 = px.line(df1,x='Date',y=col_chosen)
    return fig1

#### callback for the third graph
@callback(
    Output(component_id='graph-content2', component_property='figure'),
    Input(component_id='controls-and-radio-item', component_property='value')
)
def update_graph(col_chosen):
    fig2 = px.line(df2,x='Date',y=col_chosen)
    return fig2


if __name__ == '__main__':
    app.run_server(debug=True)

    #multiple outputs
