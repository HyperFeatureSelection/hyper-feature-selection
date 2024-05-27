import numpy as np
import pandas as pd
import random
from collections import OrderedDict
from sklearn.base import TransformerMixin
from sklearn.model_selection import cross_validate
from hyper_feature_selection.basic.pfi import PFI
from hyper_feature_selection.basic.sfe import SFE


class HFS(TransformerMixin):

    def __init__(
        self,
        model,
        metric: str,
        direction: str = "",
        cv=None,
        prune=1,
        score_lost=0.1,
        min_columns=0,
        seed: int = 42,
    ):
        self.model = model
        self.metric = metric
        self.direction = direction
        self.init_params = model.get_params()
        self.cv = cv
        self.prune = prune
        self.score_lost = score_lost
        self.min_columns = 1000 if min_columns == 0 else min_columns
        self.seed = seed
        self.keep_columns = []

    def fit(self, X, y):
        pfi = PFI(self.model, score_lost=0, seed=self.seed)
        pfi.fit(X, y)

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
                and (len(candidates) <= len(X.columns) * self.prune)
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
            num_rounds=50
        )
        sfe.fit(X, y)

        self.keep_columns = list(set(X.columns) - set(sfe.worst_features))

        return self

    def transform(self):
        pass
