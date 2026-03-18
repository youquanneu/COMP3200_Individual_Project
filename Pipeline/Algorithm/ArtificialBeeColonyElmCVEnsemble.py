import numpy as np
from scipy.stats import mode


from Pipeline.Algorithm.ArtificialBeeColonyElmCV import ArtificialBeeColonyElmCV
from Pipeline.Algorithm.ExtremeLearningMachine import ExtremeLearningMachine


class ArtificialBeeColonyElmCVEnsemble(ArtificialBeeColonyElmCV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_models = []

    def train_best_model(self, x_train, y_train):
        # 1. Extract the optimal hidden structure discovered by the ABC swarm
        weight_boundary = self.feature_size * self.hidden_size
        hidden_weight   = self.best_solution[:weight_boundary].reshape(self.feature_size, self.hidden_size)
        hidden_bias     = self.best_solution[weight_boundary:]

        self.ensemble_models = []

        # 2. Build the Ensemble
        for fold_idx in range(self.k_fold):
            fold = self.internal_folds[fold_idx]

            # Extract the fold-specific data and its isolated scaler
            # Note: The data in 'X_train_fold' is already correctly scaled
            x_tr = fold['X_train_fold']
            y_tr = fold['y_train_fold']
            fold_scaler = fold['scaler']

            # Instantiate a fold-specific ELM
            fold_elm = ExtremeLearningMachine(
                self.feature_size, self.hidden_size,
                self.activationFunction, self.regularizationLambda
            )

            # Lock in the swarm's optimized hidden weights
            fold_elm.apply_hidden_weights(hidden_weight)
            fold_elm.apply_hidden_bias(hidden_bias)

            # Calculate the output weights (\beta) specifically for this fold's variance
            fold_elm.fit(x_tr, y_tr)

            # Save the paired scaler and model to the ensemble list
            self.ensemble_models.append((fold_scaler, fold_elm))

    def predict(self, x_test):
        if not self.ensemble_models:
            raise ValueError("The ensemble must be fitted before predicting.")

        x_test_np = np.asarray(x_test)
        all_predictions = []

        # 1. Gather predictions from the entire ensemble
        for fold_scaler, fold_elm in self.ensemble_models:
            # Transform the test data using THIS specific fold's bounds
            # This prevents the domain shift bug
            x_test_scaled = fold_scaler.transform(x_test_np)

            # Get the predictions and store them
            fold_predictions = fold_elm.predict(x_test_scaled)
            all_predictions.append(fold_predictions)

        # Convert to a 2D matrix: Shape becomes (Num_Folds, Num_Samples)
        all_predictions_matrix = np.array(all_predictions)
        # 2. Apply Majority Voting across the Folds (axis=0 is the model axis)
        # keepdims=False ensures we return a flat 1D array matching the original y_train shape
        majority_votes, _ = mode(all_predictions_matrix, axis=0, keepdims=False)

        return majority_votes.ravel()