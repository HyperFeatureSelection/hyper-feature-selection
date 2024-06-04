import pandas as pd
import pytest
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from hyper_feature_selection.hfs import HFS


def get_titanic_dataset():
    df = sns.load_dataset("titanic")
    X = df.drop("survived", axis=1).drop("alive", axis=1)
    y = df["survived"]
    # Manejar datos faltantes
    X["age"].fillna(X["age"].median(), inplace=True)
    # Codificar variables categóricas
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns
    X = pd.get_dummies(X, columns=categorical_columns) # type: ignore
    return X, y

X, y = get_titanic_dataset()
rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
selector = HFS(
    model=rf_classifier,
    metric="roc_auc",
    direction="maximize",
    n_permutations=10,
    cv=KFold(n_splits=3),
    prune=0.5,
    score_lost=0.001,
    iter_drop_perc=0.5,
    min_column=10,
    verbose=1,
    seed=42,
)
selector.fit(X, y)
selector.history
selector.transform(X, max_score_loss=0.01)

