import dash

from DataProcessor import DataProcessor
from dash import Dash, dcc, html

from ModelEvaluator import ModelEvaluator
from ModelTrainer import ModelTrainer
from Plotter import Plotter


def preprocess_data():
    data_processor = DataProcessor()
    data_processor.preprocess_data()
    return data_processor


def train_model(data_processor):
    model_trainer = ModelTrainer(data_processor.X_train_preprocessed, data_processor.y_train,
                                 data_processor.X_val_preprocessed, data_processor.y_val)
    model_trainer.train_model()
    return model_trainer


def evaluate_model(model_trainer):
    model_metrics = model_trainer.evaluate_model()
    model_evaluator = ModelEvaluator(model_metrics)
    model_evaluation_table = model_evaluator.create_evaluation_table()
    return model_metrics, model_evaluation_table


def create_plots(data_processor, model_metrics):
    plotter = Plotter(data_processor.y_train, model_metrics[4], data_processor.data)
    actual_vs_predicted_fig = plotter.plot_actual_vs_predicted()
    percentage_difference_fig = plotter.plot_percentage_difference()
    patterns_and_trends_fig = plotter.plot_patterns_and_trends()
    price_by_location_fig = plotter.plot_price_by_location()
    price_by_location_and_time_fig = plotter.plot_price_by_location_and_time()
    return (
        actual_vs_predicted_fig,
        percentage_difference_fig,
        patterns_and_trends_fig,
        price_by_location_fig,
        price_by_location_and_time_fig
    )


def create_app_layout():
    return html.Div([
        html.H1("Choose a Diagram"),
        dcc.Dropdown(
            id='diagram-selector',
            options=[
                {'label': 'Actual vs Predicted - Regression Model', 'value': 'actual_vs_predicted'},
                {'label': 'Percentage Difference between predicted and actual - Regression Model',
                 'value': 'percentage_difference'},
                {'label': 'Patterns and Trends - Average Sale Price over Time', 'value': 'patterns_and_trends'},
                {'label': 'Average Sale Price by Location', 'value': 'price_by_location'},
                {'label': 'Average Sale Price by Location and Time', 'value': 'price_by_location_and_time'},
                {'label': 'Model Evaluation Metrics', 'value': 'model_evaluation'}
            ],
            value='actual_vs_predicted'
        ),
        html.Div(id='output-diagram')
    ])


def create_callback(actual_vs_predicted_fig, percentage_difference_fig, patterns_and_trends_fig,
                    price_by_location_fig, price_by_location_and_time_fig, model_evaluation_table):
    def update_output(value):
        if value == 'actual_vs_predicted':
            return dcc.Graph(figure=actual_vs_predicted_fig)
        elif value == 'percentage_difference':
            return dcc.Graph(figure=percentage_difference_fig)
        elif value == 'patterns_and_trends':
            return dcc.Graph(figure=patterns_and_trends_fig)
        elif value == 'price_by_location':
            return dcc.Graph(figure=price_by_location_fig)
        elif value == 'price_by_location_and_time':
            return dcc.Graph(figure=price_by_location_and_time_fig)
        elif value == 'model_evaluation':
            return model_evaluation_table

    return update_output


if __name__ == '__main__':
    data_processor = preprocess_data()
    model_trainer = train_model(data_processor)
    model_metrics, model_evaluation_table = evaluate_model(model_trainer)
    (
        actual_vs_predicted_fig,
        percentage_difference_fig,
        patterns_and_trends_fig,
        price_by_location_fig,
        price_by_location_and_time_fig
    ) = create_plots(data_processor, model_metrics)

    app = Dash(__name__)
    app.layout = create_app_layout()

    app.callback(
        dash.dependencies.Output('output-diagram', 'children'),
        [dash.dependencies.Input('diagram-selector', 'value')]
    )(create_callback(actual_vs_predicted_fig, percentage_difference_fig, patterns_and_trends_fig,
                      price_by_location_fig, price_by_location_and_time_fig, model_evaluation_table))

    app.run_server(debug=True)
