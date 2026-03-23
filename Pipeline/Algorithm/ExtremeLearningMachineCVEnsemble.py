import numpy as np
from scipy.stats import mode

from Pipeline.Algorithm.ExtremeLearningMachine import ExtremeLearningMachine
from Pipeline.Methodology.CrossValidationDataSplit import CrossValidationDataSplit


class ExtremeLearningMachineCVEnsemble(ExtremeLearningMachine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.internal_folds  = None
        self.ensemble_models = []
        self.k_fold          = None

    def fit(self, x_train, y_train, cv_folds = 5, penalty_coefficient = None):
        self.k_fold = cv_folds
        self.ensemble_models = []

        splitter = CrossValidationDataSplit( k_fold = self.k_fold)
        self.internal_folds = splitter.k_fold_data_spiting(x_train, y_train)

        self.featureSize = np.asarray(x_train).shape[1]
        self.initialize_random_weights(random_seed=self.randomSeed)

        for fold_idx in range(self.k_fold):
            fold = self.internal_folds[fold_idx]
            x_tr = fold['X_train_fold']
            y_tr = fold['y_train_fold']
            fold_scaler = fold['scaler']

            fold_elm = ExtremeLearningMachine(
                features_size           = self.featureSize,
                hidden_size             = self.hiddenSize,
                activation_function     = self.activationFunction,
                regularization_lambda   = self.regularizationLambda
            )
            fold_elm.apply_hidden_weights(self.hiddenWeights)
            fold_elm.apply_hidden_bias(self.hiddenBias)

            # Fit calculates ONLY the output weights (beta) based on this fold's data variance
            fold_elm.fit(x_tr, y_tr)

            self.ensemble_models.append((fold_scaler, fold_elm))

    def predict(self, x_test):
        if not self.ensemble_models:
            raise ValueError("The ensemble must be fitted before predicting.")

        x_test_np = np.asarray(x_test)
        all_predictions = []

        for fold_scaler, fold_elm in self.ensemble_models:
            x_test_scaled = fold_scaler.transform(x_test_np)
            fold_predictions = fold_elm.predict(x_test_scaled)
            all_predictions.append(fold_predictions)

        all_predictions_matrix = np.array(all_predictions)
        majority_votes, _ = mode(all_predictions_matrix, axis=0, keepdims=False)

        return majority_votes.ravel()