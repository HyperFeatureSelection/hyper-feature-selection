import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from hyper_feature_selection.utils.decorators import check_empty_dataframe


class RFE:

    def __init__(
        self,
        model,
        metric,
        params,
        cross_validate,
        global_lost=True,
        min_feature=1,
        score_lost=0.0,
    ):
        """
        Initializes the RFE class with the specified model, metric, and loss ratio.

        Args:
            model: The machine learning model to use for feature selection.
            metric: The evaluation metric to use for feature importance calculation.
            score_lost: The ratio of loss to apply during feature selection (default is 0.0).

        """

        self.model = model
        self.metric = metric
        self.score_lost = score_lost
        self.params = params
        self.min_features = min_feature
        self.cross_validation = cross_validate
        self.global_lost = global_lost
        self.perm_importances = {}
        self.keep_columns = []

    @check_empty_dataframe
    def run(self, X, y):
        """
        Performs Recursive Feature Elimination (RFE) to select relevant features.

        Args:
            X (pd.DataFrame): The training dataset.
            y (pd.Series): The target variable of the training dataset.

        Returns:
            List[str]: The list of selected features.
        """
        # Initialize survivors, ranks, scores, indexes
        survivors = list(X.columns)
        base_score = 0.0
        indexes = []
        X_tmp = pd.DataFrame()

        for i in range(len(X.columns), self.min_features - 1, -1):
            # Get only the surviving features
            X_tmp = X[survivors]
            estimator = self.model(**self.params)

            # Train and get the scores
            cr_val = cross_validate(
                estimator,
                X_tmp,
                y,
                cv=self.cross_validation,
                scoring=self.metric,
                return_estimator=True,
            )
            mean_val_score = np.mean(cr_val["test_score"])

            # Get squared feature weights
            weights = []
            for estimator in cr_val["estimator"]:
                if len(weights) != 0:
                    weights += estimator.feature_importances_
                else:
                    weights = estimator.feature_importances_

            weights = weights / len(cr_val["estimator"])  # type: ignore

            if base_score - mean_val_score > self.score_lost:
                break

            if base_score == 0.0:
                base_score = mean_val_score

            if not self.global_lost:
                base_score = mean_val_score

            self.keep_columns = X_tmp.columns

            # Find the feature with the smallest weight
            idx = np.argmin(weights)
            indexes.append(i)
            self.perm_importances[survivors[idx]] = base_score - mean_val_score
            del survivors[idx]
