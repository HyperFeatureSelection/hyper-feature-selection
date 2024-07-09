import pandas as pd
import pytest
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


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
    min_columns=1,
    verbose=1,
    seed=42,
)
selector.fit(X_train, y_train)
selector.history

X_train_selected = selector.transform(X_train, max_score_loss=0.01)
X_test_selected = selector.transform(X_test, max_score_loss=0.01)

# Metrics
clf_all_features = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
clf_all_features.fit(X_train, y_train)
preds_train_all = clf_all_features.predict_proba(X_train)[:, 1]
preds_test_all = clf_all_features.predict_proba(X_test)[:, 1]

clf_selected_features = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
clf_selected_features.fit(X_train_selected, y_train)

preds_train_selected = clf_selected_features.predict_proba(X_train_selected)[:, 1]
preds_test_selected = clf_selected_features.predict_proba(X_test_selected)[:, 1]

print(f"Number of features in model 1: {X_train.shape[1]}")
print(f"AUC train all features: {roc_auc_score(y_train, preds_train_all)}")
print(f"AUC test all features: {roc_auc_score(y_test, preds_test_all)}")

print(f"Number of features in model 2: {X_train_selected.shape[1]}")
print(f"AUC train all features: {roc_auc_score(y_train, preds_train_selected)}")
print(f"AUC test all features: {roc_auc_score(y_test, preds_test_selected)}")
