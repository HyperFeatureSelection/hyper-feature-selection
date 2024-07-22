import pandas as pd
import numpy as np

from hyper_feature_selection.basic.pfi import PFI


class PermutationImportance:

    def __init__(self, model, n_permutations=5, seed=1):
        self.model = model
        self.n_permutations = n_permutations
        self.seed = seed

        self.__df_importance = None

    def calculate(self, X, y):
        data = {"column": [], "importance": [], "std": []}

        pfi = PFI(self.model, n_permutations=self.n_permutations, seed=self.seed)
        pfi._computer_importance(self.model, X, y)
        
        for col, importance in pfi.perm_importances.items():
            data["column"].append(col)
            data["importance"].append(sum(importance) / len(importance))
            data["std"].append(np.std(importance))

        importance = np.array(data["importance"])
        data["importance"] = (importance - importance.min()) / (importance - importance.min()).sum()

        self.__df_importance = pd.DataFrame(data)

    def get_importance(self):
        return self.__df_importance
