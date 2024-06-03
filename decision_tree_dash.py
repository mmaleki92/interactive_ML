# Import required libraries
import dash
from dash import html, dcc, Input, Output
import plotly.graph_objects as go
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load sklearn datasets
available_datasets = {
    'Iris': datasets.load_iris(),
    'Wine': datasets.load_wine()
}

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Decision Tree Visualization"

# Add custom CSS styles
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1("Decision Tree Classifier Visualization", style={'text-align': 'center'}),
    ], className='row'),
    html.Div([
        html.Div([
            html.Label('Select Dataset:'),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=[{'label': name, 'value': name} for name in available_datasets.keys()],
                value='Iris'
            ),
            html.Label('Select Tree Criterion (Gini Impurity or Entropy):'),
            dcc.Dropdown(
                id='tree-criterion-dropdown',
                options=[
                    {'label': 'Gini Impurity', 'value': 'gini'},
                    {'label': 'Entropy', 'value': 'entropy'}
                ],
                value='gini'
            ),
            html.Label('Maximum Depth of the Tree:'),
            dcc.Slider(
                id='tree-depth-slider',
                min=1,
                max=10,
                step=1,
                marks={i: str(i) for i in range(1, 11)},
                value=3
            ),
            html.Label('Minimum Samples to Split a Node:'),
            dcc.Slider(
                id='min-samples-split-slider',
                min=2,
                max=20,
                step=1,
                marks={i: str(i) for i in range(2, 21)},
                value=2
            ),
            html.Label('Minimum Samples at a Leaf Node:'),
            dcc.Slider(
                id='min-samples-leaf-slider',
                min=1,
                max=10,
                step=1,
                marks={i: str(i) for i in range(1, 11)},
                value=1
            )
        ], className='pretty_container four columns'),

        html.Div([
            dcc.Graph(id='decision-tree-graph')
        ], className='pretty_container eight columns'),
    ], className='row')
])

# Define callback to update Decision Tree graph based on user selections
@app.callback(
    Output('decision-tree-graph', 'figure'),
    [Input('dataset-dropdown', 'value'),
     Input('tree-criterion-dropdown', 'value'),
     Input('tree-depth-slider', 'value'),
     Input('min-samples-split-slider', 'value'),
     Input('min-samples-leaf-slider', 'value')]
)
def update_graph(selected_dataset, criterion, max_depth, min_samples_split, min_samples_leaf):
    # Load the dataset
    dataset = available_datasets[selected_dataset]
    X = dataset.data[:, :2]  # Use only the first two features for simplicity
    y = dataset.target

    # Standardize the features
    X = StandardScaler().fit_transform(X)

    # Create a meshgrid for plotting decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # Initialize Decision Tree and fit data
    dtree = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf)
    dtree.fit(X, y)

    # Predict classifications over the grid for visualization
    Z = dtree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    fig = go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', marker=dict(color=y, line=dict(color='black', width=1)), text=dataset.target_names[y])])
    fig.add_trace(go.Contour(x=np.linspace(x_min, x_max, 500), y=np.linspace(y_min, y_max, 500), z=Z, showscale=False, contours_coloring='fill', line_width=0))
    fig.update_layout(title='Decision Tree Decision Boundary', xaxis_title='Feature 1', yaxis_title='Feature 2')

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
