import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import cross_val_score


class ModelTrainer:
    def __init__(self, X_train, y_train, X_val, y_val):
        """
        Initialize the ModelTrainer class.

        :param X_train: Training features.
        :param y_train: Training target.
        :param X_val: Validation features.
        :param y_val: Validation target.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.model = self.initialize_model()

    def initialize_model(self):
        """
        Initialize the model.

        :return: The initialized model.
        """
        return LinearRegression()

    def train_model(self):
        """
        Train the model and perform cross-validation.
        """
        self.model.fit(self.X_train, self.y_train)
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)
        print('Cross-validation scores (5-fold):', cv_scores)
        print('Mean cross-validation score (5-fold): {:.3f}'.format(np.mean(cv_scores)))

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate the evaluation metrics.

        :param y_true: Actual target values.
        :param y_pred: Predicted target values.
        :return: The evaluation metrics.
        """
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        return mse, r2, mae, medae

    def evaluate_model(self):
        """
        Evaluate the model on training and validation data.

        :return: Evaluation metrics for both training and validation data.
        """
        y_train_pred = self.model.predict(self.X_train)
        y_val_pred = self.model.predict(self.X_val)

        train_metrics = self.calculate_metrics(self.y_train, y_train_pred)
        val_metrics = self.calculate_metrics(self.y_val, y_val_pred)

        # Expanding the returned tuples
        return (*train_metrics, y_train_pred, *val_metrics, y_val_pred)
