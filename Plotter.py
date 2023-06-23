import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class Plotter:
    def __init__(self, actual, predicted, data):
        self.df = pd.DataFrame({'Actual': actual, 'Predicted': predicted}).sort_index()
        self.df['Percentage Difference'] = ((self.df['Predicted'] - self.df['Actual']) / self.df['Actual']) * 100
        self.data = data

    def configure_layout(self, title, xaxis_title, yaxis_title, fig):
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#2e2e2e"  # darker font color
            ),
            autosize=True,
            width=None,
            height=600,
            margin=dict(
                l=50,
                r=50,
                b=100,
                t=100,
                pad=4
            ),
            paper_bgcolor="White",  # lighter background color
        )
        return fig

    def plot_actual_vs_predicted(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Actual'], mode='lines', name='Actual',
                                 line=dict(color='darkviolet')))
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Predicted'], mode='lines', name='Predicted',
                                 line=dict(color='deepskyblue')))
        fig = self.configure_layout("Actual vs Predicted", "Index", "Sale Price", fig)
        return fig

    def plot_percentage_difference(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Percentage Difference'], mode='lines',
                                 name='Percentage Difference', line=dict(color='orangered')))
        fig = self.configure_layout("Percentage Difference", "Index", "Percentage Difference (%)", fig)
        return fig

    def plot_patterns_and_trends(self):
        # Combine 'Year Sold' and 'Month Sold' into a datetime format
        self.data['Date Sold'] = pd.to_datetime(
            self.data['Year Sold'].astype(str) + '-' + self.data['Month Sold'].astype(str) + '-01')

        avg_price_by_month = self.data.groupby('Date Sold')['SalePrice'].mean().reset_index()

        fig = px.line(avg_price_by_month, x='Date Sold', y='SalePrice', title='Average Sale Price per Month',
                      color_discrete_sequence=px.colors.sequential.Plasma)
        fig = self.configure_layout("Average Sale Price per Month", "Date", "Average Sale Price", fig)

        return fig

    def plot_price_by_location(self):
        avg_price_by_neighborhood = self.data.groupby('Neighborhood')['SalePrice'].mean()
        fig = px.bar(x=avg_price_by_neighborhood.index, y=avg_price_by_neighborhood.values,
                     labels={'x': 'Neighborhood', 'y': 'Average Sale Price'},
                     title='Average Sale Price by Neighborhood',
                     color=avg_price_by_neighborhood.values, color_continuous_scale='Viridis')
        fig = self.configure_layout("Average Sale Price by Neighborhood", "Neighborhood", "Average Sale Price", fig)
        fig.update_xaxes(tickfont=dict(size=10))  # smaller font size for x-axis labels
        return fig

    def plot_price_by_location_and_time(self):
        # Combine 'Year Sold' and 'Month Sold' into a datetime format
        self.data['Date Sold'] = pd.to_datetime(
            self.data['Year Sold'].astype(str) + '-' + self.data['Month Sold'].astype(str) + '-01')

        # Compute the average sale price by month
        avg_price_by_month = self.data.groupby(['Date Sold', 'Neighborhood'])['SalePrice'].mean().reset_index()

        fig = px.line(avg_price_by_month, x='Date Sold', y='SalePrice', color='Neighborhood',
                      title='Average Sale Price by Location and Time',
                      color_discrete_sequence=px.colors.cyclical.IceFire)
        fig = self.configure_layout("Average Sale Price by Location and Time", "Date", "Average Sale Price", fig)

        return fig


