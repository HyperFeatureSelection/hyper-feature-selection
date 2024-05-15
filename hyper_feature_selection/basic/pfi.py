import numpy as np
import pandas as pd
from sklearn.base import clone
from hyper_feature_selection.utils.decorators import check_empty_dataframe
from hyper_feature_selection.utils.scorers import create_scorer


from sklearn.base import TransformerMixin


class PFI(TransformerMixin):

    def __init__(self, model, metric, score_lost=0.0, n_permutations=5, cross_validation=None, seed=42):
        """
        Initializes the PFI class with the specified model, metric, and loss ratio.

        Args:
            model: The machine learning model to use for feature selection.
            metric: The evaluation metric to use for feature importance calculation.
            n_permutations: The number of permutations to calculate feature importance.
            score_lost: The ratio of loss to apply during feature selection (default is 0.0).

        """
        np.random.seed(seed=seed)

        self.model = model
        self.metric = metric
        self.n_permutations = n_permutations
        self.score_lost = score_lost
        self.perm_importances = {}
        self.keep_columns = []
        self.cross_validation = cross_validation

    @check_empty_dataframe
    def fit(self, X, y, sample_weights=None):
        """Calculates the importance of features by permutation based on the model predictions and a given metric.

        Args:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
            y : array-like of shape (n_samples,) or (n_samples, n_targets)
            sample_weights: : array-like of shape (n_samples,) including the loss weight for each sample
        """
        cloned_classifier = clone(self.model)

        if self.cross_validation:
            for train_index, test_index in self.cross_validation.split(X):
                X_train = X.iloc[train_index]
                y_train = y[train_index]
                X_test = X.iloc[test_index]
                y_test = y[test_index]

                cloned_classifier.fit(X_train, y_train)

                self._fit(cloned_classifier, X_test, y_test, sample_weights)
        else:
            cloned_classifier.fit(X, y)
            self._fit(cloned_classifier, X, y, sample_weights)

        for col, scores in self.perm_importances.items():
            if (sum(scores) / len(scores)) >= self.score_lost:
                self.keep_columns.append(col)

        # TODO: Check if this is necesary
        # if not self.keep_columns:
        #     self.keep_columns = X.columns

        return self

    def _fit(self, model, X, y, sample_weights=None):
        y_pred = model.predict(X)
        scorer = create_scorer(self.metric)
        metric_before = scorer(
            X=X,
            y_true=y,
            model=self.model,
            sample_weights=sample_weights
        )

        for column in X.columns:
            perm_importances = []
            for _ in range(self.n_permutations):
                X_perm = X.copy()

                X_perm[column] = np.random.permutation(X_perm[column].values)  # type: ignore
                y_pred_perm = self.model.predict(X_perm)

                metric_after = scorer(
                    X=X_perm,
                    y_true=y_pred_perm,
                    model=self.model,
                    sample_weights=sample_weights
                )
                perm_importance = metric_before - metric_after

                if column in self.perm_importances:
                    self.perm_importances[column].append(perm_importance)
                else:
                    perm_importances.append(metric_before - metric_after)
            
            self.perm_importances[column] = np.mean(perm_importances)

            if self.perm_importances[column] > self.score_lost:
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

    def get_importance(self):
        return pd.DataFrame.from_dict(
            self.perm_importances, orient="index", columns=["score"]
        )
