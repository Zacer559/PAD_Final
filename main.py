import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import dash
from dash import dcc
from dash import html
import plotly.graph_objects as go


class DataProcessor:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.num_features = self.data.select_dtypes(include=[np.number]).drop('SalePrice', axis=1)
        self.cat_features = self.data.select_dtypes(include=[object])

    def preprocess_data(self):
        num_pipeline = Pipeline(steps=[('num_imputer', SimpleImputer(strategy='mean'))])
        cat_pipeline = Pipeline(steps=[('cat_imputer', SimpleImputer(strategy='most_frequent')),
                                       ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipeline, self.num_features.columns),
            ('cat', cat_pipeline, self.cat_features.columns)])
        self.X = self.data.drop('SalePrice', axis=1)
        self.y = self.data['SalePrice']
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2,
                                                                              random_state=42)
        self.X_train_preprocessed = preprocessor.fit_transform(self.X_train)
        self.X_val_preprocessed = preprocessor.transform(self.X_val)


class ModelTrainer:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.model = LinearRegression()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def train_model(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_train_pred = self.model.predict(self.X_train)
        y_val_pred = self.model.predict(self.X_val)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        val_mse = mean_squared_error(self.y_val, y_val_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        val_r2 = r2_score(self.y_val, y_val_pred)
        return train_mse, val_mse, train_r2, val_r2, y_train_pred, y_val_pred


class Plotter:
    def __init__(self, actual, predicted):
        self.df = pd.DataFrame({'Actual': actual, 'Predicted': predicted}).sort_index()
        self.df['Percentage Difference'] = ((self.df['Predicted'] - self.df['Actual']) / self.df['Actual']) * 100

    def plot_actual_vs_predicted(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Actual'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Predicted'], mode='lines', name='Predicted'))
        fig.update_layout(xaxis={'title': 'Index'}, yaxis={'title': 'Sale Price'},
                          margin={'l': 40, 'b': 40, 't': 10, 'r': 10}, legend={'x': 0, 'y': 1}, hovermode='closest')
        return fig

    def plot_percentage_difference(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Percentage Difference'], mode='lines',
                                 name='Percentage Difference'))
        fig.update_layout(xaxis={'title': 'Index'}, yaxis={'title': 'Percentage Difference (%)'},
                          margin={'l': 40, 'b': 40, 't': 10, 'r': 10}, legend={'x': 0, 'y': 1}, hovermode='closest')
        return fig


def calculate_correlations(data):
    corr_matrix = data.corr()
    corr_values = (corr_matrix['SalePrice'].drop('SalePrice') * 100).rename('Points')
    fig = go.Figure(data=go.Bar(y=corr_values.values, x=corr_values.index))
    fig.update_layout(title='Correlation between Independent Factors and House Prices',
                      xaxis={'title': 'Independent Factors'}, yaxis={'title': 'Points'},
                      margin={'l': 40, 'b': 40, 't': 40, 'r': 10})
    return fig




if __name__ == '__main__':
    data_processor = DataProcessor('data/train.csv')
    data_processor.preprocess_data()

    model_trainer = ModelTrainer(data_processor.X_train_preprocessed, data_processor.y_train,
                                 data_processor.X_val_preprocessed, data_processor.y_val)
    model_trainer.train_model()
    train_mse, val_mse, train_r2, val_r2, y_train_pred, y_val_pred = model_trainer.evaluate_model()
    print("Training MSE:", train_mse)
    print("Validation MSE:", val_mse)
    print("Training R2:", train_r2)
    print("Validation R2:", val_r2)

    plotter = Plotter(data_processor.y_train, y_train_pred)
    actual_vs_predicted_fig = plotter.plot_actual_vs_predicted()
    percentage_difference_fig = plotter.plot_percentage_difference()

    correlation_fig = calculate_correlations(data_processor.data)

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("Choose a Diagram"),
        dcc.Dropdown(
            id='diagram-selector',
            options=[
                {'label': 'Actual vs Predicted', 'value': 'actual_vs_predicted'},
                {'label': 'Percentage Difference', 'value': 'percentage_difference'},
                {'label': 'Correlation Matrix', 'value': 'correlation_matrix'}
            ],
            value='actual_vs_predicted'
        ),
        dcc.Graph(id='selected-diagram'),
    ])


    @app.callback(
        dash.dependencies.Output('selected-diagram', 'figure'),
        [dash.dependencies.Input('diagram-selector', 'value')]
    )
    def update_selected_diagram(value):
        if value == 'actual_vs_predicted':
            return actual_vs_predicted_fig
        elif value == 'percentage_difference':
            return percentage_difference_fig
        elif value == 'correlation_matrix':
            return correlation_fig


    app.run_server(debug=True)
