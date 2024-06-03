# Import required libraries
import dash
from dash import html, dcc, Input, Output
import plotly.graph_objects as go
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# Load sklearn datasets
available_datasets = {
    'Iris': datasets.load_iris(),
    'Wine': datasets.load_wine()
}

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "KNN Visualization"

# Define the layout of the app
app.layout = html.Div([
    html.H1("K-Nearest Neighbors (KNN) Classifier Visualization", style={'text-align': 'center'}),
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[{'label': name, 'value': name} for name in available_datasets.keys()],
        value='Iris',
        style={'width': '50%', 'padding': '0px 20px 20px 20px', 'margin': 'auto'}
    ),
    html.Div([
        html.Label('Number of Neighbors:', style={'padding': '5px'}),
        dcc.Slider(
            id='knn-neighbors-slider',
            min=1,
            max=20,
            step=1,
            value=5,
            marks={i: str(i) for i in range(1, 21)},
            tooltip={"placement": "bottom", "always_visible": True}
        ),
        html.Label('Weight Points by:', style={'padding': '5px'}),
        dcc.RadioItems(
            id='knn-weight-radio',
            options=[
                {'label': 'Uniform', 'value': 'uniform'},
                {'label': 'Distance', 'value': 'distance'}
            ],
            value='uniform',
            style={'margin': '10px'}
        ),
        html.Label('Scaling Method:', style={'padding': '5px'}),
        dcc.RadioItems(
            id='knn-scaling-radio',
            options=[
                {'label': 'No Scaling', 'value': 'none'},
                {'label': 'Standard Scaling', 'value': 'standard'},
                {'label': 'Min-Max Normalization', 'value': 'minmax'}
            ],
            value='none',
            style={'margin': '10px'}
        )
    ], style={'width': '50%', 'margin': 'auto'}),
    dcc.Graph(id='knn-graph', style={'height': '600px'})
])

# Define callback to update KNN graph based on user selections
@app.callback(
    Output('knn-graph', 'figure'),
    [Input('dataset-dropdown', 'value'),
     Input('knn-neighbors-slider', 'value'),
     Input('knn-weight-radio', 'value'),
     Input('knn-scaling-radio', 'value')]
)
def update_graph(selected_dataset, n_neighbors, weight, scaling):
    # Load the dataset
    dataset = available_datasets[selected_dataset]
    X = dataset.data[:, :2]  # Use only the first two features for simplicity
    y = dataset.target

    # Scale or normalize features based on user input
    if scaling == 'standard':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaling == 'minmax':
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    # Initialize KNN and fit data
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight)
    knn.fit(X, y)

    # Create a meshgrid for plotting decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    # Predict classifications over the grid for visualization
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    fig = go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, size=10, line=dict(color='black', width=1)), text=[f'Label: {label}' for label in y])])
    fig.add_trace(go.Contour(x=np.linspace(x_min, x_max, 300), y=np.linspace(y_min, y_max, 300), z=Z, showscale=False, contours_coloring='fill', line_width=0))
    fig.update_layout(title='KNN Decision Boundary', xaxis_title='Feature 1', yaxis_title='Feature 2')

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
