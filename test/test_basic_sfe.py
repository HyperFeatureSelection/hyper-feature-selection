
import pandas as pd
import pytest
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from hyper_feature_selection.basic.sfe import SFE


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
    "candidates, num_rounds, drop_perc, score_loss_threshold, seed, expected_remove_columns",
    [
        (
            ["sibsp", 'fare', 'adult_male', 'alone', 'sex_female', 'sex_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'class_First', 'class_Second', 'class_Third'], 
            20, 0.5, 0.01, 42, ['class_First', 'alone', 'embarked_Q', 'embarked_S', 'class_Third', 'sex_male', 'sex_female']
        ),
       (
            ['deck_A', 'deck_B', 'deck_C', 'deck_D', 'deck_E', 'deck_F'], 
            60, 0.3, 0.0, 10,  ['deck_E', 'deck_C', 'deck_B']
        ),
        (
            ['pclass', 'age', 'sibsp', 'parch', 'fare', 'adult_male', 'alone', 'sex_female', 'sex_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'class_First', 'class_Second', 'class_Third', 'who_child', 'who_man', 'who_woman', 'deck_A', 'deck_B', 'deck_C', 'deck_D', 'deck_E', 'deck_F', 'deck_G', 'embark_town_Cherbourg', 'embark_town_Queenstown', 'embark_town_Southampton'], 
            150, 0.1, 0.015, 1,  ['deck_A', 'age', 'deck_B', 'class_Third']
        ),
        (
            ['pclass', 'age', 'sibsp', 'parch', 'fare', 'adult_male', 'alone', 'sex_female', 'sex_male', 'embarked_C', 'embarked_Q', 'embarked_S', 'who_woman', 'deck_A', 'deck_B', 'deck_C', 'deck_D', 'deck_E', 'deck_F', 'deck_G', 'embark_town_Cherbourg', 'embark_town_Queenstown', 'embark_town_Southampton'], 
            10, 0.3, -0.001, 500,  ['deck_B', 'deck_F', 'embarked_S', 'pclass', 'alone', 'age', 'fare', 'adult_male']
        )

    ],
    ids=[
        "standard-caase",
        "few-features-case",
        "all-features-case",
        "negative-loss-case",
    ],
)
def test_run(candidates, num_rounds, drop_perc, score_loss_threshold, seed, expected_remove_columns):
    X, y = get_titanic_dataset()

    rf_classifier = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)

    sfe = SFE(
        model=rf_classifier,
        metric="roc_auc",
        direction="maximize",
        cv=KFold(n_splits=3),
        candidates=candidates,
        num_rounds=num_rounds,
        drop_perc=drop_perc,
        score_loss_threshold=score_loss_threshold,
        seed=seed
    )
    sfe.fit(X, y)

    # Assert
    assert len(sfe.worst_features) == len(expected_remove_columns)
    assert set(sfe.worst_features) == set(expected_remove_columns)
