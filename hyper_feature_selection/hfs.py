import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.base import TransformerMixin
from utils.logger_config import  configure_logger
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
        self.keep_columns = []
        self.logger = configure_logger(self.__class__.__name__, verbose, "hfs.log")

    def fit(self, X, y):
        self.selected_features = list(X.columns)
        X_hfs = X.copy()
        while self.min_columns < self.selected_features:
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
                    and (len(candidates) < self.min_columns)
                ):
                    candidates.append(column)
                    lost_score += lost
                else:
                    break

            print(len(candidates))

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
        return self

    def transform(self):
        pass
