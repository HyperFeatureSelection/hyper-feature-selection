import seaborn as sns
from sklearn.preprocessing import LabelEncoder


class DataSNS:

    def __init__(self):
        self.datasets = {
            # "anagrams": {"target": "", "type": ""},
            "anscombe": {"target": "dataset", "type": "multi"},
            "attention": {"target": "score", "type": "regre"},
            # "brain_networks": {"target": "network", "type": "multi"},
            "car_crashes": {"target": "total", "type": "regre"},
            "diamonds": {"target": "cut", "type": "multi"},
            "dots": {"target": "firing_rate", "type": "regre"},
            # "dowjones": {"target": "", "type": ""},
            # "exercise": {"target": "", "type": ""},
            # "flights": {"target": "passengers", "type": "regre"},
            # "fmri": {"target": "", "type": ""},
            # "geyser": {"target": "", "type": ""},
            # "glue": {"target": "", "type": ""},
            # "healthexp": {"target": "", "type": ""},
            # "iris": {"target": "", "type": ""},
            # "mpg": {"target": "", "type": ""},
            # "penguins": {"target": "", "type": ""},
            # "planets": {"target": "", "type": ""},
            # "seaice": {"target": "", "type": ""},
            # "taxis": {"target": "", "type": ""},
            # "tips": {"target": "", "type": ""},
            # "titanic": {"target": "", "type": ""},
        }

    def get_dataset_names(self):
        return self.datasets.keys()

    def get_type_data(self, name):
        return self.datasets[name]['type']

    def get_dataset(self, name):
        df = sns.load_dataset(name)
        target = self.datasets[name]["target"]
        num_cols = df._get_numeric_data().columns
        cat_cols = list(set(df.columns) - set(num_cols))

        for c in cat_cols:
            label_encoder = LabelEncoder()
            df[c] = label_encoder.fit_transform(df[c])

        X = df.drop(target, axis=1)
        y = df[target]

        return X, y
