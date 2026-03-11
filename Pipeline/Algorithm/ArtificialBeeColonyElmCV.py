import numpy as np

from Pipeline.Algorithm.ArtificialBeeColonyElm import ArtificialBeeColonyElm
from Pipeline.Algorithm.CrossValidationDataSplit import CrossValidationDataSplit
from Pipeline.Algorithm.ExtremeLearningMachine import ExtremeLearningMachine


class ArtificialBeeColonyElmCV(ArtificialBeeColonyElm):
    def __init__(self, features_size, hidden_size,
                 activation_function, regularization_lambda=0.0,
                 random_state=None, fitness_function='Accuracy',
                 solution_size=10, trial_limit=10, max_iteration=100,
                 max_change=20.0, min_change=3.0, initial_sigma=0.8, final_sigma=0.1, nmi=3,
                 initial_probability=0.0, final_probability=1.0
                 ):

        super().__init__(features_size, hidden_size,
                         activation_function, regularization_lambda,
                         random_state, fitness_function,
                         solution_size, trial_limit, max_iteration,
                         max_change, min_change, initial_sigma, final_sigma, nmi,
                         initial_probability, final_probability
                         )


        self.full_x_train = None
        self.full_y_train = None

        self.k_fold         = None
        self.internal_folds = None

    def evaluation_fitness(self, solution, x_train, y_train):
        # Note: x_train and y_train are intentionally ignored.
        # Overridden to use self.internal_folds for CV evaluation.

        weight_boundary = self.feature_size * self.hidden_size
        hidden_weight = solution[:weight_boundary].reshape(self.feature_size, self.hidden_size)
        hidden_bias = solution[weight_boundary:]

        fold_fitness = []

        # 1. Evaluate generalization using K-Fold Validation
        for fold_idx in range(self.k_fold):
            fold = self.internal_folds[fold_idx]
            x_tr   , y_tr    = fold['X_train_fold'] , fold['y_train_fold']
            x_val  , y_val   = fold['X_val_fold']   , fold['y_val_fold']

            elm = ExtremeLearningMachine(self.feature_size, self.hidden_size,
                                         self.activationFunction, self.regularizationLambda)
            elm.apply_hidden_weights(hidden_weight)
            elm.apply_hidden_bias(hidden_bias)

            # Fit on internal fold train, predict on internal fold val
            elm.fit(x_tr, y_tr)
            y_pred = elm.predict(x_val.values)
            fold_fitness.append(self.get_fitness(y_val, y_pred))

        # The true fitness is the average validation score across all folds

        return np.mean(fold_fitness)

    def fit(self, x_train, y_train, data_seed = 42, cv_folds = 5):
        self.k_fold = cv_folds
        splitter = CrossValidationDataSplit(random_state = data_seed, k_fold = self.k_fold)
        self.internal_folds = splitter.k_fold_data_spiting(x_train, y_train)

        super().fit(x_train, y_train)
