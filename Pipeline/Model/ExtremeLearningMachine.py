import numpy as np

class ExtremeLearningMachine:

    def __init__(self, input_size, hidden_size, activation_function=np.tanh, regularization_lambda=0.0):

        self.inputSize = input_size
        self.hiddenSize = hidden_size
        self.activationFunction = activation_function
        self.regularizationLambda = regularization_lambda

        self.hiddenWeights = None
        self.hiddenBias = None

        self.hidden_layer_output = None
        self.output_weights = None

        self.accuracy = None
        self.precision = None
        self.recall = None
        self.specificity = None
        self.f1_score = None

    def initialize_random_weights(self, scale=1.0, random_seed=None):
        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
        else:
            rng = np.random
        self.hiddenWeights = rng.randn(self.inputSize, self.hiddenSize) * scale
        self.hiddenBias = rng.randn(self.hiddenSize) * scale

    def apply_activation_function(self, activation_function):
        self.activationFunction = activation_function

    def apply_hidden_weights(self, hidden_weights):
        self.hiddenWeights = hidden_weights

    def apply_hidden_bias(self, hidden_bias):
        self.hiddenBias = hidden_bias

    def fit(self, input_features, target_features):
        if target_features.ndim == 1:
            target_features = target_features.reshape(-1, 1)
        linear_output = input_features @ self.hiddenWeights + self.hiddenBias
        self.hidden_layer_output = self.activationFunction(linear_output)
        self.output_weights = np.linalg.pinv(self.hidden_layer_output) @ target_features

    def get_output_weights(self):
        return self.output_weights

    def regularized_fit(self, input_features, target_features, regularization_lambda):

        self.regularizationLambda = regularization_lambda

        linear_output = input_features @ self.hiddenWeights + self.hiddenBias
        hidden_layer_output = self.activationFunction(linear_output)

        gram_matrix = hidden_layer_output.T @ hidden_layer_output

        identity_matrix = np.eye(self.hiddenSize)

        ridge_matrix = gram_matrix + (self.regularizationLambda * identity_matrix)

        try:
            inverse_ridge = np.linalg.inv(ridge_matrix)
        except np.linalg.LinAlgError:
            inverse_ridge = np.linalg.pinv(ridge_matrix)

        self.output_weights = inverse_ridge @ hidden_layer_output.T @ target_features

    def predict(self, input_features):
        linear_output = input_features @ self.hiddenWeights + self.hiddenBias
        hidden_layer_output = self.activationFunction(linear_output)
        raw_output = hidden_layer_output @ self.output_weights
        return (raw_output > 0.5).astype(int)
        # if raw_output.shape[1] == 1:
        #     return (raw_output > 0.5).astype(int)
        # else:
        #     return np.argmax(raw_output, axis=1)

    def set_evaluation_metrics(self, input_features, target_features):
        target_features = np.array(target_features).ravel()
        def safe_div(n, d):
            return n / d if d != 0 else 0.0

        prediction = self.predict(input_features).ravel()
        true_positive  = ((target_features==1) & (prediction == 1)).sum()
        true_negative  = ((target_features==0) & (prediction == 0)).sum()
        false_negative = ((target_features==1) & (prediction == 0)).sum()
        false_positive = ((target_features==0) & (prediction == 1)).sum()

        self.accuracy    = safe_div( (true_positive + true_negative) , input_features.shape[0])
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