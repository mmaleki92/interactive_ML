# Import required libraries
import dash
from dash import html, dcc, Input, Output
import plotly.graph_objects as go
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Load sklearn datasets
available_datasets = {
    'Iris': datasets.load_iris(),
    'Wine': datasets.load_wine()
}

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("SVM Classifier Visualization"),
    dcc.Dropdown(
        id='dataset-dropdown',
        options=[{'label': name, 'value': name} for name in available_datasets.keys()],
        value='Iris'
    ),
    dcc.Dropdown(
        id='svm-kernel-dropdown',
        options=[
            {'label': 'Linear', 'value': 'linear'},
            {'label': 'RBF', 'value': 'rbf'},
            {'label': 'Polynomial', 'value': 'poly'},
            {'label': 'Sigmoid', 'value': 'sigmoid'}
        ],
        value='rbf'
    ),
    dcc.Graph(id='svm-graph')
])

# Define callback to update SVM graph based on user selections
@app.callback(
    Output('svm-graph', 'figure'),
    [Input('dataset-dropdown', 'value'),
     Input('svm-kernel-dropdown', 'value')]
)
def update_graph(selected_dataset, kernel):
    # Load the dataset
    dataset = available_datasets[selected_dataset]
    X = dataset.data[:, :2]  # Take only the first two features for simplicity
    y = dataset.target

    # Standardize the features
    X = StandardScaler().fit_transform(X)

    # Create a meshgrid for plotting decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # Initialize SVM and fit data
    svm = SVC(kernel=kernel, C=1.0, probability=True)
    svm.fit(X, y)

    # Predict classifications over the grid for visualization
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    fig = go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, line=dict(color='black', width=1)), text=dataset.target_names[y])])
    fig.add_trace(go.Contour(x=np.linspace(x_min, x_max, 500), y=np.linspace(y_min, y_max, 500), z=Z, showscale=False, contours_coloring='fill', line_width=0))
    fig.update_layout(title=f'SVM Decision Boundary using {kernel.capitalize()} Kernel', xaxis_title='Feature 1', yaxis_title='Feature 2')

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
