# Import required libraries
import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS
from sklearn import datasets
from sklearn.svm import SVC
import pandas as pd
import numpy as np

# Load sklearn datasets
available_datasets = {
    'Iris': datasets.load_iris(),
    'Breast Cancer': datasets.load_breast_cancer(),
    'Wine': datasets.load_wine(),
    'Digits': datasets.load_digits()
}

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Interactive Dimensionality Reduction"),
    html.Div([
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[{'label': name, 'value': name} for name in available_datasets.keys()],
            value='Iris'
        ),
        dcc.Dropdown(
            id='method-dropdown',
            options=[
                {'label': 'PCA', 'value': 'PCA'},
                {'label': 't-SNE', 'value': 'TSNE'},
                {'label': 'MDS', 'value': 'MDS'}
            ],
            value='PCA'
        ),
        html.Div([
            html.Label('SVM Kernel:', style={'display': 'block'}),
            dcc.Dropdown(
                id='svm-kernel-dropdown',
                options=[
                    {'label': 'Linear', 'value': 'linear'},
                    {'label': 'RBF', 'value': 'rbf'},
                    {'label': 'Polynomial', 'value': 'poly'},
                    {'label': 'Sigmoid', 'value': 'sigmoid'}
                ],
                value='rbf'
            )
        ]),
        html.Div([
            html.Label('Number of PCA Components (if PCA selected):', style={'display': 'block'}),
            dcc.Slider(
                id='pca-slider',
                min=2,
                max=4,
                step=1,
                marks={i: str(i) for i in range(2, 5)},
                value=3
            )
        ], id='pca-slider-container', style={'display': 'none'}),
        dcc.RadioItems(
            id='plot-type',
            options=[{'label': '2D Plot', 'value': '2D'}, {'label': '3D Plot', 'value': '3D'}],
            value='2D',
            labelStyle={'display': 'block'}
        ),
    ], style={'width': '20%', 'float': 'left', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(id='dimension-graph')
    ], style={'width': '80%', 'display': 'inline-block', 'float': 'right'})
])

# Update slider visibility based on the selected method
@app.callback(
    Output('pca-slider-container', 'style'),
    Input('method-dropdown', 'value')
)
def toggle_slider_visibility(selected_method):
    if selected_method == 'PCA':
        return {'display': 'block'}
    return {'display': 'none'}

# Define callback to update the dimensionality reduction graph based on selections
@app.callback(
    Output('dimension-graph', 'figure'),
    [Input('dataset-dropdown', 'value'),
     Input('method-dropdown', 'value'),
     Input('svm-kernel-dropdown', 'value'),
     Input('pca-slider', 'value'),
     Input('plot-type', 'value')]
)
def update_graph(selected_dataset, selected_method, kernel, n_components, plot_type):
    # Load the dataset
    dataset = available_datasets[selected_dataset]
    X = dataset.data
    y = dataset.target

    # Standardize the features
    X = StandardScaler().fit_transform(X)

    # Dimensionality reduction
    if selected_method == 'PCA':
        reducer = PCA(n_components=min(n_components, X.shape[1]))
    elif selected_method == 'TSNE':
        reducer = TSNE(n_components=2)
    elif selected_method == 'MDS':
        reducer = MDS(n_components=2)

    X_reduced = reducer.fit_transform(X)

    # SVM classifier
    svm = SVC(kernel=kernel, C=1.0)
    svm.fit(X_reduced, y)

    # Meshgrid for decision boundary
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # Make predictions
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create plot
    fig = go.Figure(data=[go.Scatter(x=X_reduced[:, 0], y=X_reduced[:, 1], mode='markers', marker=dict(color=y))])
    fig.update_layout(title='SVM Decision Boundary', xaxis_title='PC1', yaxis_title='PC2')
    fig.add_trace(go.Contour(x=np.linspace(x_min, x_max, 500), y=np.linspace(y_min, y_max, 500), z=Z, showscale=False, contours_coloring='fill', line_width=0))

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
