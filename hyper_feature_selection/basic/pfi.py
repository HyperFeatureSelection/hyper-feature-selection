import numpy as np
import pandas as pd
from hyper_feature_selection.utils.decorators import check_empty_dataframe

from sklearn.base import TransformerMixin


class PFI(TransformerMixin):

    def __init__(self, model, metric, score_lost=0.0, seed=42):
        """
        Initializes the PFI class with the specified model, metric, and loss ratio.

        Args:
            model: The machine learning model to use for feature selection.
            metric: The evaluation metric to use for feature importance calculation.
            score_lost: The ratio of loss to apply during feature selection (default is 0.0).

        """

        self.model = model
        self.metric = metric
        self.score_lost = score_lost
        self.perm_importances = {}
        self.keep_columns = []

        np.random.seed(seed=seed)

    @check_empty_dataframe
    def fit(self, X, y=None):
        """Calculates the importance of features by permutation based on the model predictions and a given metric.

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
            y : array-like of shape (n_samples,) or (n_samples, n_targets)
        """
        y_pred = self.model.predict(X)
        metric_before = self.metric(y, y_pred)

        for column in X.columns:
            X_perm = X.copy()

            X_perm[column] = np.random.permutation(X_perm[column].values)  # type: ignore
            y_pred_perm = self.model.predict(X_perm)

            metric_after = self.metric(y, y_pred_perm)
            perm_importance = metric_before - metric_after
            self.perm_importances[column] = perm_importance

            if perm_importance > self.score_lost:
                self.keep_columns.append(column)

        if not self.keep_columns:
            self.keep_columns = X.columns


    def transform(self, X):
        """
        Transform the input data by keeping only the specified columns.

        Args:
            X: Input data to transform.

        Returns:
            Transformed data with only the specified columns.
        """

        return X[self.keep_columns]
