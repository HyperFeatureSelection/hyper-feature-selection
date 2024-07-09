
import pandas as pd
import pytest
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from hyper_feature_selection.basic.pfi import PFI


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


@pytest.mark.parametrize(
    "loss_ratio, expected_keep_columns, seed",
    [
        (0.0, ['pclass', 'age', 'sibsp', 'parch', 'fare', 'adult_male', 'alone', 'sex_female', 'sex_male', 'embarked_C', 'embarked_S', 'class_First', 'class_Second', 'class_Third', 'who_child', 'who_man', 'who_woman', 'deck_A', 'deck_B', 'deck_C', 'deck_D', 'deck_E', 'deck_F', 'deck_G', 'embark_town_Cherbourg', 'embark_town_Southampton'] , 0),  # ID: happy-path-no-loss
        (0.01, ['age', 'fare'], 0),  # ID: happy-path-with-loss
        (1.0, [], 0),  # ID: edge-case-max-loss
        (-0.5, ['pclass', 'age', 'sibsp', 'parch', 'fare', 'adult_male', 'alone', 'sex_female', 'sex_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'class_First', 'class_Second', 'class_Third', 'who_child', 'who_man', 'who_woman', 'deck_A', 'deck_B', 'deck_C', 'deck_D', 'deck_E', 'deck_F', 'deck_G', 'embark_town_Cherbourg', 'embark_town_Queenstown', 'embark_town_Southampton'] , 0),  # ID: edge-case-negative-loss
    ],
    ids=[
        "happy-path-no-loss",
        "happy-path-with-loss",
        "edge-case-max-loss",
        "edge-case-negative-loss",
    ],
)
def test_run(loss_ratio, expected_keep_columns, seed):
    X, y = get_titanic_dataset()

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf_classifier.fit(X,y)

    pfi = PFI(rf_classifier, score_lost=loss_ratio, seed=seed)
    pfi.fit(X, y)

    # Assert
    # print(pfi.keep_columns)
    # print(expected_keep_columns)
    assert len(pfi.keep_columns) == len(expected_keep_columns)
    assert set(pfi.keep_columns) == set(expected_keep_columns)

