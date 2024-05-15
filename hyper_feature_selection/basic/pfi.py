import numpy as np
import pandas as pd
from sklearn.base import clone
from hyper_feature_selection.utils.decorators import check_empty_dataframe

from sklearn.base import TransformerMixin


class PFI(TransformerMixin):

    def __init__(self, model, metric, score_lost=0.0, seed=42, cross_validation=None):
        """
        Initializes the PFI class with the specified model, metric, and loss ratio.

        Args:
            model: The machine learning model to use for feature selection.
            metric: The evaluation metric to use for feature importance calculation.
            score_lost: The ratio of loss to apply during feature selection (default is 0.0).

        """
        np.random.seed(seed=seed)

        self.model = model
        self.metric = metric
        self.score_lost = score_lost
        self.perm_importances = {}
        self.keep_columns = []
        self.cross_validation = cross_validation

    @check_empty_dataframe
    def fit(self, X, y):
        """Calculates the importance of features by permutation based on the model predictions and a given metric.

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
            y : array-like of shape (n_samples,) or (n_samples, n_targets)
        """
        cloned_classifier = clone(self.model)

        if self.cross_validation:
            for train_index, test_index in self.cross_validation.split(X):
                X_train = X.iloc[train_index]
                y_train = y[train_index]
                X_test = X.iloc[test_index]
                y_test = y[test_index]

                cloned_classifier.fit(X_train, y_train)

                self._fit(cloned_classifier, X_test, y_test)
        else:
            cloned_classifier.fit(X, y)
            self._fit(cloned_classifier, X, y)

        for col, scores in self.perm_importances.items():
            print(col, (sum(scores) / len(scores)), self.score_lost)
            if (sum(scores) / len(scores)) >= self.score_lost:
                print('IN')
                self.keep_columns.append(col)

        # TODO: Check if this is necesary
        # if not self.keep_columns:
        #     self.keep_columns = X.columns

        return self

    def _fit(self, model, X, y):
        y_pred = model.predict(X)
        metric_before = self.metric(y, y_pred)

        for column in X.columns:
            X_perm = X.copy()

            X_perm[column] = np.random.permutation(X_perm[column].values)  # type: ignore
            y_pred_perm = self.model.predict(X_perm)

            metric_after = self.metric(y, y_pred_perm)
            perm_importance = metric_before - metric_after
            if column in self.perm_importances:
                self.perm_importances[column].append(perm_importance)
            else:
                self.perm_importances[column] = [perm_importance]

    def transform(self, X):
        """
        Transform the input data by keeping only the specified columns.

        Args:
            X: Input data to transform.

        Returns:
            Transformed data with only the specified columns.
        """

        return X[self.keep_columns]

    def get_importance(self):
        return pd.DataFrame.from_dict(
            self.perm_importances, orient="index", columns=["score"]
        )
