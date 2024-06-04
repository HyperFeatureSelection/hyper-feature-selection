import numpy as np
import pandas as pd
import random
import math
from sklearn.base import TransformerMixin, clone
from sklearn.model_selection import cross_validate
from hyper_feature_selection.utils.decorators import check_empty_dataframe
from hyper_feature_selection.utils.scorers import create_scorer
from hyper_feature_selection.utils.utils import reset_estimator


class SFE(TransformerMixin):

    def __init__(
        self,
        model,
        metric: str,
        direction: str,
        cv=None,
        candidates=None,
        num_rounds: int=10,
        drop_perc: float=0.5,
        score_loss_threshold: float=0.0,
        seed: int=42
    ):
        """
        Initialize the Search Feature Elimination (SFE) object with the specified parameters.

        Args:
            model: The estimator model to use for feature selection.
            metric: The metric to use for evaluating feature importance.
            direction: The direction of improvement for the metric (e.g., 'max' or 'min').
            cv: The cross-validation strategy to use.
            candidates: The candidate features for selection (default is None).
            num_rounds: The number of rounds for the RFE process (default is 10).
            drop_perc: The percentage of features to drop in each round (default is 0.5).
            score_loss_threshold: The threshold for stopping the RFE process based on score loss (default is 0.0).
            seed: The random seed (default is 42).

        Returns:
            None
        """
        self.model = model
        self.metric = metric
        self.direction = direction
        self.init_params = model.get_params()
        self.cv = cv
        self.candidates = candidates
        self.num_rounds = num_rounds
        self.drop_perc = drop_perc
        self.score_loss_threshold = score_loss_threshold
        self.seed = seed
        self.keep_columns = []

    @staticmethod
    def compute_number_of_combinations(list_size: int, subset_size: int):
        return math.comb(list_size, subset_size)

    @staticmethod
    def random_subset(full_list: list, used_subsets: list, subset_size: int, seed: int):
        """
        Generate a random subset of a given size from a full list.

        Args:
            full_list: The full list from which to generate the subset.
            used_subsets: The previously sampled subsets.
            subset_size: The size of the subset to generate.
            seed: Random seed.

        Returns:
            A random subset of the specified size from the full list.
        Raises:
            ValueError: If subset_size is greater than the size of the full list.
        """
        repeated = True
        used_subset_tuples = [tuple(sorted(c)) for c in used_subsets]
        if subset_size > len(full_list):
            raise ValueError("Subset size cannot be greater than the size of the full list.")
        random.seed(seed)

        while repeated:
            subset = random.sample(full_list, subset_size)
            if tuple(sorted(subset)) not in used_subset_tuples:
                repeated = False
        return subset


    @check_empty_dataframe
    def fit(self, X: pd.DataFrame, y: pd.Series, fit_params=None):
        """
        Fit the Recursive Feature Elimination (RFE) object to the input data.

        Args:
            X: The input features to fit the RFE model.
            y: The target variable for fitting the RFE model.
            fit_params: Additional parameters for fitting the model (default is None).

        Returns:
            The fitted RFE object.
        """
        fit_params = {} if fit_params is None else fit_params
        cloned_classifier = clone(self.model)

        candidates = list(X.columns) if self.candidates is None else self.candidates
        scorer = create_scorer(self.metric, self.direction)
        cv_info_baseline = cross_validate(
            estimator=cloned_classifier,
            X=X,
            y=y,
            cv=self.cv,
            scoring=scorer,
            params=fit_params,
        )
        base_score = np.mean(cv_info_baseline["test_score"])
        improvement = False
        self.round_iter = 0
        val_scores, diff_scores, candidate_subsets = [], [], []

        n_combinations = self.compute_number_of_combinations(len(candidates), round(self.drop_perc*len(candidates)))
        self.num_rounds = min(self.num_rounds, n_combinations)
        # print(base_score)

        while (not improvement) and (self.round_iter < self.num_rounds):
            drop_candidates = self.random_subset(
                candidates, candidate_subsets, round(self.drop_perc*len(candidates)), self.seed+self.round_iter
            )
            print(self.round_iter, drop_candidates)
            X_tmp = X.drop(columns=drop_candidates)
            cv_info_iter = cross_validate(
                estimator=cloned_classifier,
                X=X_tmp,
                y=y,
                cv=self.cv,
                scoring=scorer,
                params=fit_params,
            )
            mean_val_score = np.mean(cv_info_iter["test_score"])
            diff_score = mean_val_score - base_score if self.direction == "maximize" else base_score - mean_val_score
            val_scores.append(mean_val_score)
            diff_scores.append(diff_score)
            candidate_subsets.append(drop_candidates)
            improvement = diff_score >= self.score_loss_threshold
            self.round_iter += 1
            # print(mean_val_score, round_iter, drop_candidates)

        if improvement:
            self.worst_features = drop_candidates
            self.score_improvemt = diff_score
            self.val_score = mean_val_score
        else:
            worst_iter_idx = np.argmax(diff_scores)
            self.worst_features = candidate_subsets[worst_iter_idx]
            self.score_improvemt = diff_scores[worst_iter_idx]
            self.val_score = val_scores[worst_iter_idx]

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform the input data by dropping the worst features.

        Args:
            X: The input data to transform.

        Returns:
            The transformed data with the worst features dropped.
        """
        return X.drop(columns=self.worst_features)