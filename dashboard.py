import pandas as pd
import dash_core_components as dcc
import dash_html_components as html
import dash

df = pd.read_csv('MSFT.csv')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(f'Stock Price for MSFT'),
    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': df['Date'], 'y': df['Close'], 'type': 'line', 'name': 'Data'},
            ],
            'layout': {
                'title': 'Close Prices from 18.04.2022 until 17.04.2023'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
