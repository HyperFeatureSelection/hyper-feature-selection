def get_categorical(df):
    return df.select_dtypes(include=["object", "category"]).columns


def get_numerical(df):
    return df.select_dtypes(exclude=["object", "category"]).columns
