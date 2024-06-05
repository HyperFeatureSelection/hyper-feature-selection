import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.base import TransformerMixin
from hyper_feature_selection.utils.logger_config import configure_logger
from hyper_feature_selection.basic.pfi import PFI
from hyper_feature_selection.basic.sfe import SFE


class HFS(TransformerMixin):

    def __init__(
        self,
        model,
        metric: str,
        direction: str = "",
        n_permutations: int =5,
        cv=None,
        prune=1,
        score_lost=0.1,
        iter_drop_perc=0.4,
        min_column: int = 0,
        force_columns = [],
        verbose: int = 1,
        seed: int = 42,
    ):
        self.model = model
        self.metric = metric
        self.direction = direction
        self.n_permutations = n_permutations
        self.init_params = model.get_params()
        self.cv = cv
        self.prune = prune
        self.score_lost = score_lost
        self.iter_drop_perc = iter_drop_perc
        self.min_columns = 1
        self.seed = seed
        self.keep_columns = force_columns
        self.logger = configure_logger(self.__class__.__name__, verbose, "hfs.log")
        self.history = pd.DataFrame(
            columns=["iteration", "features_removed", "score", "sfe_rounds"]
        )

    def fit(self, X, y):
        self.selected_features = list(X.columns)
        X_hfs = X.copy()
        iter_hfs = 0
        while self.min_columns < len(self.selected_features):
            pfi = PFI(
                model=self.model,
                metric=self.metric,
                score_lost=self.score_lost,
                n_permutations=self.n_permutations,
                cross_validation=self.cv,
                direction=self.direction,
                seed=self.seed,
            )
            pfi.fit(X_hfs, y)

            perm_importances = {
                col: np.mean(lost_score) for col, lost_score in pfi.perm_importances.items()
            }
            perm_importances = OrderedDict(
                sorted(perm_importances.items(), key=lambda x: x[1])
            )
            lost_score = 0
            candidates = []
            for column, lost in perm_importances.items():
                if (
                    (lost_score <= self.score_lost)
                    and (len(candidates) <= len(X_hfs.columns) * self.prune)
                ):
                    candidates.append(column)
            
            # print(candidates)
            sfe = SFE(
                self.model,
                metric=self.metric,
                direction=self.direction,
                cv=self.cv,
                candidates=candidates,
                score_loss_threshold=self.score_lost,
                drop_perc=self.iter_drop_perc,
                num_rounds=50
            )
            sfe.fit(X_hfs, y)
            self.selected_features = list(set(X_hfs.columns) - set(sfe.worst_features))
            X_hfs = X_hfs.drop(columns=sfe.worst_features)
            iter_hfs += 1
            self.logger.info(
                f"Iteration: {iter_hfs} - " 
                f"Selected features: {len(self.selected_features)} - "
                f"Removed features: {len(sfe.worst_features)} - "
                f"Metric improvement: {sfe.score_improvemt:.4f} - "
                f"SFE rounds: {sfe.round_iter}." 
            )
            iter_data = {
                "iteration": iter_hfs,
                "features_removed": sfe.worst_features,
                "score": sfe.val_score,
                "sfe_rounds": sfe.round_iter
            }
            if self.history.shape[0] == 0:
                self.history = pd.DataFrame([iter_data]).copy()
            else:
                self.history = pd.concat([self.history, pd.DataFrame([iter_data])], ignore_index=True)
        return self

    def transform(
            self, 
            X, 
            max_score_loss=0.01,
        ):
        features = list(X.columns)

        # Selection
        best_score_limit = (
            self.history["score"].max() - max_score_loss if self.direction== "maximize" else self.history["score"].min() + max_score_loss
        )
        filter_condition = (
            self.history["score"] >= best_score_limit if self.direction== "maximize" else self.history["score"] <= best_score_limit
        )
        history_best_rounds = self.history[(filter_condition) & (~self.history["score"].isnull())]
        best_round = history_best_rounds["iteration"].max()
        feature_remove = list(
            self.history.loc[self.history["iteration"] <= best_round, "features_removed"]
        )

        self.feature_remove = [
            c for sub_list in feature_remove for c in sub_list
        ]
        self.features_selected = [c for c in features if c not in self.feature_remove]
        keep_cols_add = [c for c in self.keep_columns if c not in self.features_selected]
        self.features_selected += keep_cols_add

        self.logger.info(
            f"Features selected: {len(self.features_selected)}: {self.features_selected}" 
        )

        return X[self.features_selected]
