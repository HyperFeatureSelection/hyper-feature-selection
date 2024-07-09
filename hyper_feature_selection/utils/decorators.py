import pandas as pd


def check_empty_dataframe(func):
    def wrapper(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, pd.DataFrame) and arg.shape[0] == 0:
                raise ValueError(f"DataFrame {arg} is empty.")
            elif isinstance(arg, pd.Series) and arg.count() == 0:
                raise ValueError(f"Serie {arg} está vacía.")

        for kwarg in kwargs.values():
            if isinstance(kwarg, pd.DataFrame) and kwarg.shape[0] == 0:
                raise ValueError(f"DataFrame {arg} is empty.")
            elif isinstance(kwarg, pd.Series) and arg.count() == 0:
                raise ValueError(f"Serie {arg} está vacía.")

        return func(self, *args, **kwargs)

    return wrapper
