import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from data.column_mappings import column_mappings
from data.value_mappings import mappings

class DataProcessor:
    """
    Class for processing data for machine learning.

    :param data_path: path to the data file.
    :param data: the data loaded from the file.
    :param X_train_preprocessed: preprocessed training data.
    :param X_val_preprocessed: preprocessed validation data.
    :param y_train: target values for the training data.
    :param y_val: target values for the validation data.
    """

    def __init__(self, data_path='data/train.csv'):
        """
        Initialize DataProcessor with path to the data file.

        :param data_path: path to the data file.
        """
        self.data_path = data_path
        self.data = self.load_data()
        self.X_train_preprocessed = None
        self.X_val_preprocessed = None
        self.y_train = None
        self.y_val = None

    def load_data(self):
        """
        Load and preprocess the data from the file.

        :return: The loaded and preprocessed data.
        """
        data = pd.read_csv(self.data_path)
        data.rename(columns=column_mappings, inplace=True)
        data.replace(mappings, inplace=True)
        return data

    def preprocess_data(self):
        """
        Preprocess the data, creating preprocessed training and validation data and targets.
        """
        preprocessor = self.create_preprocessor()
        self.split_data(preprocessor)

    def create_preprocessor(self):
        """
        Create a preprocessor for the data.

        :return: preprocessor for the data.
        """
        num_features = self.data.select_dtypes(include=[np.number]).drop('SalePrice', axis=1).columns
        cat_features = self.data.select_dtypes(include=[object]).columns

        num_pipeline = Pipeline(steps=[
            ('num_imputer', SimpleImputer(strategy='mean'))
        ])

        cat_pipeline = Pipeline(steps=[
            ('cat_imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        transformers = [
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ]

        return ColumnTransformer(transformers=transformers)

    def split_data(self, preprocessor):
        """
        Split the data into training and validation sets, and preprocess them.

        :param preprocessor: preprocessor for the data.
        """
        X = self.data.drop('SalePrice', axis=1)
        y = self.data['SalePrice']

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        self.X_train_preprocessed = preprocessor.fit_transform(X_train)
        self.X_val_preprocessed = preprocessor.transform(X_val)
        self.y_train = y_train
        self.y_val = y_val
