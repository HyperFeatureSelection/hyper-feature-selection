import pandas as pd
import numpy as np
from sklego.datasets import load_abalone
from sklego.datasets import load_arrests
from sklego.datasets import load_chicken
from sklego.datasets import load_heroes
from sklego.datasets import load_hearts
from sklego.datasets import load_penguins
from sklego.datasets import fetch_creditcard
from sklego.datasets import make_simpleseries

from sklearn.preprocessing import LabelEncoder


class DataSKLego:

    def __init__(self):
        self.datasets = {
            "abalone": {'data': load_abalone, 'type': 'multi'},
            "arrests": {'data': load_arrests, 'type': 'classi'},
            "chicken": {'data': load_chicken, 'type': 'regre'},
            "heroes": {'data': load_heroes, 'type': 'classi'},
            "hearts": {'data': load_hearts, 'type': 'classi'},
            "penguins": {'data': load_penguins, 'type': 'multi'},
        }

    def get_dataset_names(self):
        return self.datasets.keys()

    def get_type_data(self, name):
        return self.datasets[name]['type']

    def get_dataset(self, name):
        X, y = self.datasets[name]['data'](return_X_y=True)
        X = pd.DataFrame(X)
        num_cols = X._get_numeric_data().columns
        cat_cols = list(set(X.columns) - set(num_cols))

        for c in cat_cols:
            label_encoder = LabelEncoder()
            X[c] = label_encoder.fit_transform(X[c])

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        return X, pd.Series(y)
