import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

from hyper_feature_selection.hfs import HFS
from hyper_feature_selection.utils.scorers import create_scorer
from hyper_feature_selection.datasets import Dataset

dataset_urls = [
    "https://people.sc.fsu.edu/~jburkardt/data/csv/hw_200.csv",  # Ejemplo de dataset de alturas y pesos
    "https://people.sc.fsu.edu/~jburkardt/data/csv/airtravel.csv",  # Ejemplo de dataset de viajes aéreos
]


DATASETS = [
    # {"name": "anagrams", "target": "", "type": ""},
    {"name": "anscombe", "target": "dataset", "type": "multi"},
    {"name": "attention", "target": "score", "type": "regre"},
    # {"name": "brain_networks", "target": "network", "type": "multi"},
    {"name": "car_crashes", "target": "total", "type": "regre"},
    {"name": "diamonds", "target": "cut", "type": "multi"},
    {"name": "dots", "target": "firing_rate", "type": "regre"},
    {"name": "dowjones", "target": "", "type": ""},
    {"name": "exercise", "target": "", "type": ""},
    {"name": "flights", "target": "passengers", "type": "regre"},
    {"name": "fmri", "target": "", "type": ""},
    {"name": "geyser", "target": "", "type": ""},
    {"name": "glue", "target": "", "type": ""},
    {"name": "healthexp", "target": "", "type": ""},
    {"name": "iris", "target": "", "type": ""},
    {"name": "mpg", "target": "", "type": ""},
    {"name": "penguins", "target": "", "type": ""},
    {"name": "planets", "target": "", "type": ""},
    {"name": "seaice", "target": "", "type": ""},
    {"name": "taxis", "target": "", "type": ""},
    {"name": "tips", "target": "", "type": ""},
    {"name": "titanic", "target": "", "type": ""},
]


def get_dataset_sns(name, target):
    df = sns.load_dataset(name)
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(df.columns) - set(num_cols))
    print(df)
    for c in cat_cols:
        label_encoder = LabelEncoder()
        df[c] = label_encoder.fit_transform(df[c])

    X = df.drop(target, axis=1)
    y = df[target]

    return X, y


def get_titanic_dataset():
    df = sns.load_dataset("titanic")
    X = df.drop("survived", axis=1).drop("alive", axis=1)
    y = df["survived"]
    # Manejar datos faltantes
    X["age"].fillna(X["age"].median(), inplace=True)
    # Codificar variables categóricas
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns
    X = pd.get_dummies(X, columns=categorical_columns)  # type: ignore
    return X, y


print('Start')
dataset_class = Dataset()
for dataset_name in dataset_class.get_dataset_names():
    X, y = dataset_class.get_dataset(dataset_name)
    type_data = dataset_class.get_type_data(dataset_name)
    print(dataset_name, X.shape)
    print(y.value_counts())
    # print(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    if type_data == "classi":
        metric = "roc_auc"
        model = RandomForestClassifier
    elif type_data == "cluster":
        metric = "v_measure_score"
        model = None
    elif type_data == "multi":
        metric = "roc_auc_ovo"
        model = RandomForestClassifier
    elif type_data == "regre":
        metric = "r2"
        model = RandomForestRegressor

    rf_classifier = model(n_estimators=50, max_depth=4, random_state=42)
    selector = HFS(
        model=rf_classifier,
        metric=metric,
        direction="maximize",
        n_permutations=10,
        cv=KFold(n_splits=3),
        prune=0.5,
        score_lost=0.001,
        iter_drop_perc=0.5,
        min_columns=1,
        verbose=1,
        seed=42,
    )
    selector.fit(X_train, y_train)
    selector.history

    X_train_selected = selector.transform(X_train, max_score_loss=0.01)
    X_test_selected = selector.transform(X_test, max_score_loss=0.01)

    # Metrics
    clf_all_features = model(n_estimators=50, max_depth=4, random_state=42)
    clf_all_features.fit(X_train, y_train)

    clf_selected_features = model(n_estimators=50, max_depth=4, random_state=42)
    clf_selected_features.fit(X_train_selected, y_train)

    scorer = create_scorer(metric, "maximize")

    print(f"Number of features in model 1: {X_train.shape[1]}")
    print(f"AUC train all features: {scorer(clf_all_features, X_train, y_train)}")
    print(f"AUC test all features: {scorer(clf_all_features, X_test, y_test)}")

    print(f"Number of features in model 2: {X_train_selected.shape[1]}")
    print( f"AUC train all features: {scorer(clf_selected_features, X_train_selected, y_train)}")
    print(f"AUC test all features: {scorer(clf_selected_features, X_test_selected, y_test)}")
