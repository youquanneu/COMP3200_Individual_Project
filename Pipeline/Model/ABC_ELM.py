import time

import numpy as np
import copy
from ExtremeLearningMachine import ExtremeLearningMachine


class ABC_ELM:
    def __init__(self,
                 feature_size, hidden_size, activation_function, regularization_lambda=0.0,
                 SN=10, limit=10, iter_max=100,
                 P_initial=0.0, P_final=1.0):
        """
        Implementation of ABC(II)-ELM based on Alshamiri et al. (2018).
        """
        self.d = feature_size
        self.L = hidden_size
        self.D = (self.d + 1) * self.L  # Total parameters to optimize

        self.activation = activation_function
        self.reg_lambda = regularization_lambda

        self.SN = SN  # Number of food sources/employed bees [cite: 191]
        self.limit = limit  # Trials before abandoning solution [cite: 253]
        self.iter_max = iter_max  # Maximum iterations [cite: 235]
        self.P_initial = P_initial  # Initial copy probability [cite: 232]
        self.P_final = P_final  # Final copy probability [cite: 232]

        self.population = []
        self.fitness = np.zeros(self.SN)
        self.trials = np.zeros(self.SN)
        self.best_solution = None
        self.best_fitness = -1.0
        self.best_elm = None

    def _generate_random_solution(self):
        # Initial solutions are randomly generated over the range [-1, 1] [cite: 335]
        return np.random.uniform(-1.0, 1.0, self.D)

    def _evaluate_fitness(self, S, X, y):
        # Extract weights and biases from the flattened solution vector
        W = S[:self.d * self.L].reshape(self.d, self.L)
        b = S[self.d * self.L:]

        # Construct ELM and apply the candidate parameters
        elm = ExtremeLearningMachine(self.d, self.L, self.activation, self.reg_lambda)
        elm.apply_hidden_weights(W)
        elm.apply_hidden_bias(b)

        # Compute output weights using MP generalized inverse [cite: 189]
        elm.fit(X, y)

        # Calculate misclassifications
        y_pred = elm.predict(X)
        MC = np.sum(y_pred != np.asarray(y).ravel())

        # Fitness is inversely proportional to misclassifications [cite: 208, 210]
        fitness_val = 1.0 / (1.0 + MC)
        return fitness_val, elm

    def _algorithm_3_neighbor(self, i, current_iter):
        """
        Second neighborhood procedure based on difference between parameters. [cite: 223]
        """
        S_i = self.population[i]
        V_i = np.copy(S_i)

        # Calculate dynamic copy probability [cite: 233]
        P_copy = ((self.P_final - self.P_initial) / self.iter_max) * current_iter + self.P_initial

        # Choose another random solution k != i [cite: 229]
        k = np.random.choice([idx for idx in range(self.SN) if idx != i])
        S_k = self.population[k]

        for j in range(self.D):
            r = np.random.rand()
            if r >= P_copy:  # [cite: 238]
                phi = np.random.uniform(-1.0, 1.0)  # [cite: 229]
                V_i[j] = S_i[j] + phi * (S_i[j] - S_k[j])
                # Boundary constraint handling [cite: 239]
                V_i[j] = np.clip(V_i[j], -1.0, 1.0)

        return V_i

    def fit(self, X_train, y_train):
        X_train_np = np.asarray(X_train)
        y_train_np = np.asarray(y_train)

        # 1. Initialization Phase [cite: 241]
        for _ in range(self.SN):
            S = self._generate_random_solution()
            self.population.append(S)

        # Evaluate initial population [cite: 245]
        for i in range(self.SN):
            self.fitness[i], elm = self._evaluate_fitness(self.population[i], X_train_np, y_train_np)
            if self.fitness[i] > self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = np.copy(self.population[i])
                self.best_elm = copy.deepcopy(elm)

        # Main Iteration Loop [cite: 260]
        for current_iter in range(1, self.iter_max + 1):
            startt = time.time()

            # 2. Employed Bees Phase [cite: 246]
            for i in range(self.SN):
                V_i = self._algorithm_3_neighbor(i, current_iter)
                fit_v, elm_v = self._evaluate_fitness(V_i, X_train_np, y_train_np)

                if fit_v > self.fitness[i]:
                    self.population[i] = V_i
                    self.fitness[i] = fit_v
                    self.trials[i] = 0
                    if fit_v > self.best_fitness:
                        self.best_fitness = fit_v
                        self.best_solution = np.copy(V_i)
                        self.best_elm = copy.deepcopy(elm_v)
                else:
                    self.trials[i] += 1

            # 3. Onlooker Bees Phase [cite: 247, 248]
            prob = self.fitness / np.sum(self.fitness)  # [cite: 250]

            t = 0
            i = 0
            while t < self.SN:
                if np.random.rand() < prob[i]:
                    V_i = self._algorithm_3_neighbor(i, current_iter)
                    fit_v, elm_v = self._evaluate_fitness(V_i, X_train_np, y_train_np)

                    if fit_v > self.fitness[i]:
                        self.population[i] = V_i
                        self.fitness[i] = fit_v
                        self.trials[i] = 0
                        if fit_v > self.best_fitness:
                            self.best_fitness = fit_v
                            self.best_solution = np.copy(V_i)
                            self.best_elm = copy.deepcopy(elm_v)
                    else:
                        self.trials[i] += 1
                    t += 1
                i = (i + 1) % self.SN

            # 4. Scout Bees Phase [cite: 254]
            for i in range(self.SN):
                if self.trials[i] >= self.limit:
                    self.population[i] = self._generate_random_solution()  # [cite: 255]
                    self.fitness[i], _ = self._evaluate_fitness(self.population[i], X_train_np, y_train_np)
                    self.trials[i] = 0

            print(current_iter, " end : ", time.time() - startt)

    def predict(self, X_test):
        if self.best_elm is None:
            raise ValueError("The model must be fitted before predicting.")
        return self.best_elm.predict(np.asarray(X_test))