# Import required libraries
import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs

# Function to generate additional synthetic datasets
def generate_datasets():
    n_samples = 300
    noisy_moons = make_moons(n_samples=n_samples, noise=0.1)
    noisy_circles = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    varied_blobs = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5])
    random_blobs = make_blobs(n_samples=n_samples, centers=5)
    aniso_data, _ = make_blobs(n_samples=n_samples, cluster_std=0.5)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    aniso = np.dot(aniso_data, transformation)
    no_structure = np.random.rand(n_samples, 2), None
    return [
        ('Noisy Moons', noisy_moons),
        ('Noisy Circles', noisy_circles),
        ('Varied Blobs', varied_blobs),
        ('Random Blobs', random_blobs),
        ('Anisotropic', (aniso, _)),
        ('No Structure', no_structure)
    ]

# Load sklearn datasets and add synthetic datasets
available_datasets = {
    'Iris': datasets.load_iris(),
    'Wine': datasets.load_wine(),
    'Digits': datasets.load_digits(n_class=10),
    'Breast Cancer': datasets.load_breast_cancer(),
    **{name: data for name, data in generate_datasets()}
}

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "DBSCAN Clustering Visualization"

# Define the layout of the app
app.layout = html.Div([
    html.H1("DBSCAN Clustering Visualization", style={'text-align': 'center'}),
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[{'label': name, 'value': name} for name in available_datasets.keys()],
        value='Iris',
        style={'width': '50%', 'padding': '0px 20px 20px 20px', 'margin': 'auto'}
    ),
    html.Div([
        html.Label('Epsilon (eps) - Neighborhood Distance:', style={'padding': '5px'}),
        dcc.Slider(
            id='dbscan-eps-slider',
            min=0.1,
            max=2.0,
            step=0.1,
            value=0.5,
            marks={i/10: str(i/10) for i in range(1, 21)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Label('Minimum Samples:', style={'padding': '5px'}),
        dcc.Slider(
            id='dbscan-min-samples-slider',
            min=2,
            max=20,
            step=1,
            value=5,
            marks={i: str(i) for i in range(2, 21)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Button('Run Clustering', id='run-clustering-button', n_clicks=0, style={'margin': '10px'})
    ], style={'width': '50%', 'margin': 'auto'}),
    dcc.Graph(id='dbscan-graph', style={'height': '600px'})
])

# Define callback to update DBSCAN graph when the button is clicked
@app.callback(
    Output('dbscan-graph', 'figure'),
    [Input('run-clustering-button', 'n_clicks')],
    [State('dataset-dropdown', 'value'),
     State('dbscan-eps-slider', 'value'),
     State('dbscan-min-samples-slider', 'value')]
)
def update_graph(n_clicks, selected_dataset, eps, min_samples):
    if n_clicks > 0:  # Only run clustering when the button is clicked
        # Load the dataset
        data = available_datasets[selected_dataset]
        X, y = data if isinstance(data, tuple) else (data.data, data.target)
        X = X[:, :2]  # Use only the first two features for simplicity, adjust if the dataset has fewer features

        # Standardize the features
        X = StandardScaler().fit_transform(X)

        # Initialize DBSCAN and fit data
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(X)

        # Plot
        fig = go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                         marker=dict(color=labels, size=10, line=dict(color='black', width=1)),
                                         text=[f'Label: {label}' for label in labels])])
        fig.update_layout(title='DBSCAN Clustering Results', xaxis_title='Feature 1', yaxis_title='Feature 2')

        return fig
    return go.Figure()  # Return an empty figure for initial load

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
