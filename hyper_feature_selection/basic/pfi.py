import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class PFI():
    
    def __init__(self, model, metric):
        self.model = model
        self.metric = metric
    
    def run(X, y):
       # Calcular el error en los datos de prueba antes de permutar
        y_pred = self.model.predict(X)
        metric_before = self.metric(y, y_pred)

        # Calcular la importancia de características por permutación
        perm_importances = {}
        for i, column in enumerate(X.columns):
            X_perm = X.copy()

            np.random.shuffle(X_perm[:, i])
            y_pred_perm = model.predict(X_perm)

            metric_after = self.metric(y_test, y_pred_perm)
            perm_importance = metric_before - metric_after
            perm_importances[column] = perm_importance

        sorted_indices = sorted(perm_importances, key=perm_importances.get, reverse=True)

        for column, value in sorted_indices.items():
            print(f'Feature {column}: {value}')