from hyper_feature_selection.datasets import seaborn_data
from hyper_feature_selection.datasets import sklego_data


class Dataset:

    def __init__(self):
        self.source = {
            # "seaborn": seaborn_data.DataSNS(),
            "skleago": sklego_data.DataSKLego(),
        }

        self.datasets = {}
        for name, data in self.source.items():
            datasets = {n: name for n in data.get_dataset_names()}
            self.datasets = {**self.datasets, **datasets}

    def get_dataset_names(self):
        return self.datasets.keys()

    def get_type_data(self, name):
        source = self.datasets[name]
        return self.source[source].get_type_data(name)

    def get_dataset(self, name):
        source = self.datasets[name]
        return self.source[source].get_dataset(name)
