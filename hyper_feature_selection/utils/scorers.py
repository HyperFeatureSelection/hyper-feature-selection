from sklearn.metrics import make_scorer, get_scorer
from sklearn.metrics._scorer import _SCORERS
import inspect


def create_scorer(metric, direction):
    """
    Create a custom scorer based on the provided metric, direction, and response method.

    Args:
        metric (str or callable): The metric to create a custom scorer for.
        direction (str): The direction to optimize the metric, either 'maximize' or 'minimize'.

    Returns:
        callable: A custom scorer function based on the input metric, direction, and response method.

    Raises:
        ValueError: If direction is not 'maximize' or 'minimize', metric is not a valid sklearn metric name,
            if metric callable does not have 'X', 'y', and 'model' as parameters or if the response_method
            is not appropriate for a string metric.
        TypeError: If metric is neither a string nor a callable.
    """

    if direction not in ["maximize", "minimize"]:
        raise ValueError("direction must be either 'maximize' or 'minimize'")

    # If metric is a string, attempt to create a scorer based on sklearn metric names
    if isinstance(metric, str):
        if metric not in list(_SCORERS.keys()):
            raise ValueError(f"{metric} is not a valid sklearn metric name.")
        return make_scorer(
            get_scorer(metric)._score_func,
            greater_is_better=direction == "maximize",
            response_method=_SCORERS[metric]._response_method,
        )
    elif callable(metric):
        print("call")
        params = inspect.signature(metric).parameters
        if any(
            param not in params for param in ["X", "y_true", "model", "sample_weights"]
        ):
            raise ValueError(
                "Metric callable must have 'X', 'y_true', 'sample_weights' and 'model' as parameters."
            )
        return metric

    else:
        raise TypeError("Metric must be either a string or a callable.")
