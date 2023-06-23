import pandas as pd
from dash import dash_table
import plotly.graph_objs as go
class Metrics:
    """
    Encapsulates the metrics used in the ModelEvaluator.
    """
    def __init__(self, metrics):
        self.metrics = metrics
        self.names = ["train_mse", "val_mse", "train_r2", "val_r2",
                      "train_mae", "val_mae", "train_medae", "val_medae"]

    def to_dataframe(self):
        """
        Converts the metrics to a pandas dataframe.

        :return: A pandas dataframe of the metrics.
        """
        metrics_dict = {name: [str(metric)] for name, metric in zip(self.names, self.metrics)}
        return pd.DataFrame(metrics_dict, index=[0])





class ModelEvaluator:
    """
    Evaluates a model and presents its metrics.
    """
    def __init__(self, model_metrics):
        self.model_metrics = Metrics(model_metrics)

    def create_evaluation_table(self):
        """
        Creates an evaluation table presenting the metrics of the evaluated model.

        :return: A dash_table.DataTable of the model metrics.
        """
        metrics_df = self.model_metrics.to_dataframe()
        return self.create_dash_table(metrics_df)

    @staticmethod
    def create_dash_table(df):
        """
        Creates a dash_table.DataTable from a given dataframe.

        :param df: A pandas dataframe.
        :return: A dash_table.DataTable of the dataframe.
        """
        return dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            style_cell={
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white',
                'textAlign': 'center',
                'fontFamily': 'Helvetica',
                'fontSize': '1.2em',
                'border': '1px solid grey'
            },
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'fontWeight': 'bold',
                'color': 'white'
            },
            style_table={
                'maxHeight': '300px',
                'overflowY': 'scroll',
                'border': 'thin lightgrey solid'
            }
        )