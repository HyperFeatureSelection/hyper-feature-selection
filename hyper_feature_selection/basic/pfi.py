import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, clone

from hyper_feature_selection.utils.decorators import check_empty_dataframe
from hyper_feature_selection.utils.scorers import create_scorer


class PFI(TransformerMixin):
    """
    PFI (Permutation Feature Importance) object for calculating feature importances.

    - `__init__`: Initializes the PFI object with specified parameters.
    - `fit`: Fits the PFI object to the provided data.
    - `_computer_importance`: Fits the PFI object to the provided data.
    - `transform`: Transforms input features by keeping only important columns.
    - `get_importance`: Returns the feature importances calculated by the PFI object.
    """

    def __init__(
        self,
        model,
        metric="roc_auc",
        score_lost=0.0,
        n_permutations=5,
        cross_validation=None,
        sample_weights=None,
        direction="maximize",
        seed=42,
    ):
        """
        Initializes the object with specified parameters.

        Args:
            model: The machine learning model for which feature importance will be calculated.
            metric: The metric used to evaluate the model performance (default is 'roc_auc').
            score_lost: The threshold for score loss to consider a feature important (default is 0.0).
            n_permutations: The number of permutations to generate (default is 5).
            cross_validation: The cross-validation strategy to use (default is None).
            sample_weights: The weights for each sample (default is None).
            direction: The direction to optimize for, either 'maximize' or 'minimize' (default is 'maximize').
            seed: The random seed for reproducibility (default is 42).

        Returns:
            None
        """
        np.random.seed(seed=seed)

        self.model = model
        self.metric = metric
        self.n_permutations = n_permutations
        self.score_lost = score_lost
        self.perm_importances = {}
        self.keep_columns = []
        self.cross_validation = cross_validation
        self.sample_weights = sample_weights
        self.direction = direction

    @check_empty_dataframe
    def fit(self, X, y):
        """
        Fits the PFI (Permutation Feature Importance) object to the provided data.

        Args:
            X: The input features for training.
            y: The target variable for training.

        Returns:
            self
        """
        cloned_classifier = clone(self.model)

        if self.cross_validation:
            for train_index, test_index in self.cross_validation.split(X):
                X_train = X.iloc[train_index]
                y_train = y[train_index]
                X_test = X.iloc[test_index]
                y_test = y[test_index]

                cloned_classifier.fit(X_train, y_train)

                self._computer_importance(cloned_classifier, X_test, y_test, self.sample_weights)
        else:
            cloned_classifier.fit(X, y)
            self._computer_importance(cloned_classifier, X, y, self.sample_weights)

        for col, scores in self.perm_importances.items():
            if np.mean(scores) >= self.score_lost:
                self.keep_columns.append(col)

        # TODO: Check if this is necesary
        # if not self.keep_columns:
        #     self.keep_columns = X.columns

        return self

    def _computer_importance(self, model, X, y, sample_weights=None):
        """
        Fits the PFI (Permutation Feature Importance) object to the provided data.

        Args:
            model: The machine learning model used for prediction.
            X: The input features for training.
            y: The target variable for training.
            sample_weights: The weights for each sample (default is None).

        Returns:
            None
        """
        scorer = create_scorer(self.metric, self.direction)
        metric_before = scorer(
            X=X,
            y_true=y,
            estimator=model,
            # sample_weights=sample_weights
        )

        for column in X.columns:
            perm_importances = []
            for _ in range(self.n_permutations):
                X_perm = X.copy()

                X_perm[column] = np.random.permutation(X_perm[column].values)  # type: ignore

                metric_after = scorer(
                    X=X_perm,
                    y_true=y,
                    estimator=model,
                    # sample_weights=sample_weights,
                )
                perm_importance = metric_before - metric_after
                perm_importances.append(perm_importance)

            self.perm_importances[column] = perm_importances

    def transform(self, X):
        """
        Transforms the input features by keeping only the columns identified as important by the PFI (Permutation Feature Importance) object.

        Args:
            X: The input features to be transformed.

        Returns:
            DataFrame: Transformed DataFrame with only the important columns.
        """

        return X[self.keep_columns]

    def get_importance(self):
        """
        Returns the feature importances calculated by the PFI (Permutation Feature Importance) object.

        Returns:
            DataFrame: A DataFrame containing the feature importances.
        """
        return pd.DataFrame.from_dict(
            self.perm_importances, orient="index", columns=["score"]
        )
