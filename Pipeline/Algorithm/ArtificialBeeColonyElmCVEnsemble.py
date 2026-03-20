import numpy as np
from scipy.stats import mode


from Pipeline.Algorithm.ArtificialBeeColonyElmCV import ArtificialBeeColonyElmCV
from Pipeline.Algorithm.ExtremeLearningMachine import ExtremeLearningMachine


class ArtificialBeeColonyElmCVEnsemble(ArtificialBeeColonyElmCV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_models = []

    def train_best_model(self, x_train, y_train):

        weight_boundary = self.feature_size * self.hidden_size
        hidden_weight   = self.best_solution[:weight_boundary].reshape(self.feature_size, self.hidden_size)
        hidden_bias     = self.best_solution[weight_boundary:]

        self.ensemble_models = []

        # 2. Build the Ensemble
        for fold_idx in range(self.k_fold):
            fold = self.internal_folds[fold_idx]

            x_tr = fold['X_train_fold']
            y_tr = fold['y_train_fold']
            fold_scaler = fold['scaler']

            fold_elm = ExtremeLearningMachine(
                self.feature_size, self.hidden_size,
                self.activation_function, self.regularization_lambda
            )

            fold_elm.apply_hidden_weights(hidden_weight)
            fold_elm.apply_hidden_bias(hidden_bias)

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

    def get_validation_fitness(self, solution, x_train, y_train):

        all_val_predictions = []

        for fold_idx in range(self.k_fold):
            fold = self.internal_folds[fold_idx]
            x_tr, y_tr = fold['X_train_fold'], fold['y_train_fold']
            fold_scaler = fold['scaler']

            elm = self.build_elm_by_solution(solution, x_tr, y_tr)

            x_val_scaled = fold_scaler.transform(self.x_val)

            fold_predictions = elm.predict(x_val_scaled)
            all_val_predictions.append(fold_predictions)

        all_predictions_matrix = np.array(all_val_predictions)

        majority_votes, _ = mode(all_predictions_matrix, axis=0, keepdims=False)
        ensemble_predictions = majority_votes.ravel()

        return  self.get_fitness(self.y_val, ensemble_predictions)