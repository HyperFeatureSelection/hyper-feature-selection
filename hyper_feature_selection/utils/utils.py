from sklearn.base import clone


def get_categorical(df):
    return df.select_dtypes(include=["object", "category"]).columns


def get_numerical(df):
    return df.select_dtypes(exclude=["object", "category"]).columns


def reset_estimator(estimator):
    params = estimator.get_params()
    new_estimator = clone(estimator)
    new_estimator.set_params(**params)
    return new_estimator
