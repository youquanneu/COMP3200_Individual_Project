import numpy as np

class ExtremeLearningMachine:

    def __init__(self, features_size, hidden_size, activation_function=np.tanh, regularization_lambda=0.0):

        self.featureSize = features_size
        self.hiddenSize  = hidden_size
        self.activationFunction   = activation_function
        self.regularizationLambda = regularization_lambda

        self.hiddenWeights  = None
        self.hiddenBias     = None

        self.hiddenLayerOutput  = None
        self.outputWeights      = None

    def initialize_random_weights(self, scale=1.0, random_seed=None):
        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
        else:
            rng = np.random
        self.hiddenWeights  = rng.randn(self.featureSize, self.hiddenSize) * scale
        self.hiddenBias     = rng.randn(self.hiddenSize) * scale

    def apply_activation_function(self, activation_function):
        self.activationFunction = activation_function

    def apply_hidden_weights(self, hidden_weights):
        self.hiddenWeights = hidden_weights

    def apply_hidden_bias(self, hidden_bias):
        self.hiddenBias = hidden_bias

    def fit(self, features_data, target_data):

        features_data = np.asarray(features_data)
        target_data = np.asarray(target_data)

        if self.hiddenWeights is None:
            self.featureSize = features_data.shape[1]
            self.initialize_random_weights()

        if target_data.ndim == 1:
            unique_classes = np.unique(target_data)
            if len(unique_classes) > 2:
                num_classes = len(unique_classes)
                one_hot = np.zeros((target_data.size, num_classes))
                one_hot[np.arange(target_data.size), target_data] = 1
                target_data = one_hot
            else:
                target_data = target_data.reshape(-1, 1)
        self.regularized_fit(features_data, target_data, self.regularizationLambda)

    def regularized_fit(self, features_data, target_data, regularization_lambda):
        self.regularizationLambda = regularization_lambda
        if target_data.ndim == 1:
            target_data = target_data.reshape(-1, 1)

        linear_output = features_data @ self.hiddenWeights + self.hiddenBias
        self.hiddenLayerOutput = self.activationFunction(linear_output)
        sample_size = features_data.shape[0]

        if regularization_lambda == 0:
            self.outputWeights = np.linalg.pinv(self.hiddenLayerOutput) @ target_data

        else :
            if sample_size > self.hiddenSize :
                gram_matrix     = self.hiddenLayerOutput.T @ self.hiddenLayerOutput
                penalize_matrix = self.regularizationLambda * np.eye(self.hiddenSize)
                ridge_matrix    = gram_matrix + penalize_matrix
                try:
                    inverse_ridge  = np.linalg.inv(ridge_matrix)
                except np.linalg.LinAlgError:
                    inverse_ridge  = np.linalg.pinv(ridge_matrix)
                self.outputWeights = inverse_ridge @ self.hiddenLayerOutput.T @ target_data
            else:
                gram_matrix     = self.hiddenLayerOutput @ self.hiddenLayerOutput.T
                penalize_matrix = self.regularizationLambda * np.eye(sample_size)
                ridge_matrix    = gram_matrix + penalize_matrix
                try:
                    inverse_ridge  = np.linalg.inv(ridge_matrix)
                except np.linalg.LinAlgError:
                    inverse_ridge  = np.linalg.pinv(ridge_matrix)
                self.outputWeights = self.hiddenLayerOutput.T @ inverse_ridge @ target_data

    def predict(self, features_data):
        linear_output = features_data @ self.hiddenWeights + self.hiddenBias
        hidden_layer_output = self.activationFunction(linear_output)
        raw_output = hidden_layer_output @ self.outputWeights
        if raw_output.shape[1] == 1:
            return (raw_output > 0.5).astype(int).ravel()
        else:
            return np.argmax(raw_output, axis=1)

    def get_output_weights(self):
        return self.outputWeights




    def set_evaluation_metrics(self, features_data, target_data):
        target_data = np.array(target_data).ravel()
        def safe_div(n, d):
            return n / d if d != 0 else 0.0

        prediction = self.predict(features_data).ravel()
        true_positive  = ((target_data == 1) & (prediction == 1)).sum()
        true_negative  = ((target_data == 0) & (prediction == 0)).sum()
        false_negative = ((target_data == 1) & (prediction == 0)).sum()
        false_positive = ((target_data == 0) & (prediction == 1)).sum()

        self.accuracy    = safe_div((true_positive + true_negative), features_data.shape[0])
        self.precision   = safe_div(true_positive , (true_positive + false_positive))
        self.recall      = safe_div(true_positive , (true_positive + false_negative))
        self.specificity = safe_div(true_negative , (true_negative + false_positive))
        self.f1_score    = safe_div(2 * (self.precision * self.recall) , (self.precision + self.recall))
    def get_evaluation_metrics(self):
        return self.accuracy, self.precision, self.recall, self.specificity, self.f1_score

    def get_accuracy(self):
        return self.accuracy
    def get_precision(self):
        return self.precision
    def get_recall(self):
        return self.recall
    def get_specificity(self):
        return self.specificity
    def get_f1_score(self):
        return self.f1_score