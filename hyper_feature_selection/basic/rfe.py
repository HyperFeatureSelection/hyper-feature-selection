def rfe(self, X, y):
        """
        Performs Recursive Feature Elimination (RFE) to select relevant features.

        Args:
            X (pd.DataFrame): The training dataset.
            y (pd.Series): The target variable of the training dataset.
            X_validation (pd.DataFrame): The validation dataset.
            y_validation (pd.Series): The target variable of the validation dataset.

        Returns:
            List[str]: The list of selected features.
        """
        params_iter = self.hyperparameters[self.mode_type]

        # Initialize survivors, ranks, scores, indexes
        survivors = list(X.columns)
        base_score = 0.0
        indexes = []
        X_tmp = pd.DataFrame()

        category_columns = get_category_columns(X)
        numeric_columns = get_numeric_columns(X)

        encoder = self.encoder(cols=category_columns)  # type: ignore
        encoder.fit(X, y)
        X_train = encoder.transform(X)

        imputer = self.imputer(cols=numeric_columns)  # type: ignore
        imputer.fit(X_train, y)
        X_train = imputer.transform(X_train)

        for i in range(len(X.columns), self.rfe_min_features_to_select - 1, -1):
            # Get only the surviving features
            X_tmp = X_train[survivors]
            estimator = Model.get_model_object(model_type=self.mode_type, params_dict=params_iter)  # type: ignore

            # Train and get the scores
            cr_val = cross_validate(estimator, X_tmp, y, cv=self.cross_validation, scoring=self.metric, return_estimator=True)
            mean_val_score = np.mean(cr_val["test_score"])

            print(f"The model with {len(survivors)} features scores Val: {mean_val_score:5f} points of {self.metric}")

            # Get squared feature weights
            weights = []
            for estimator in cr_val["estimator"]:
                if len(weights) != 0:
                    weights += estimator.feature_importances_
                else:
                    weights = estimator.feature_importances_

            weights = weights / len(cr_val["estimator"])  # type: ignore

            if base_score == 0.0:
                base_score = mean_val_score

            if base_score - mean_val_score > self.rfe_score_lost:
                break

            if not self.rfe_global_lost:
                base_score = mean_val_score

            # Find the feature with the smallest weight
            idx = np.argmin(weights)
            indexes.append(i)
            print(f"the feature removed is: {survivors[idx]}")
            del survivors[idx]

        return X_tmp.columns