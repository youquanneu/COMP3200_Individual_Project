import time
import numpy as np
import copy

from Pipeline.Model.ExtremeLearningMachine import ExtremeLearningMachine


class ABC_ELM2:
    def __init__(self,
                 feature_size, hidden_size, activation_function, regularization_lambda=0.0,
                 algo_type='algo3', random_seed=None,
                 SN=10, limit=10, iter_max=100,
                 P_initial=0.0, P_final=1.0,
                 Chg_max=20.0, Chg_min=3.0, sigma_initial=0.8, sigma_final=0.1, nmi=3):

        self.d = feature_size
        self.L = hidden_size
        self.D = (self.d + 1) * self.L

        self.activation = activation_function
        self.reg_lambda = regularization_lambda

        # Core ABC logic routing
        self.algo_type = algo_type

        # Localized deterministic RNG (No global side effects)
        self.rng = np.random.RandomState(random_seed) if random_seed is not None else np.random.RandomState()

        self.SN = SN
        self.limit = limit
        self.iter_max = iter_max

        # Algorithm 3 Specifics
        self.P_initial = P_initial
        self.P_final = P_final

        # Algorithm 2 Specifics
        self.Chg_max = Chg_max
        self.Chg_min = Chg_min
        self.sigma_initial = sigma_initial
        self.sigma_final = sigma_final
        self.nmi = nmi

        self.population = []
        self.fitness = np.zeros(self.SN)
        self.trials = np.zeros(self.SN)
        self.best_solution = None
        self.best_fitness = -1.0
        self.best_elm = None

    def _generate_random_solution(self):
        # Initial solutions are randomly generated over the range [-1, 1]
        return self.rng.uniform(-1.0, 1.0, self.D)

    def _evaluate_fitness(self, S, X, y):
        W = S[:self.d * self.L].reshape(self.d, self.L)
        b = S[self.d * self.L:]

        elm = ExtremeLearningMachine(self.d, self.L, self.activation, self.reg_lambda)
        elm.apply_hidden_weights(W)
        elm.apply_hidden_bias(b)

        elm.fit(X, y)

        y_pred = elm.predict(X)
        MC = np.sum(y_pred != np.asarray(y).ravel())

        # Fitness is inversely proportional to misclassifications
        fitness_val = 1.0 / (1.0 + MC)
        return fitness_val, elm

    def _algorithm_2_neighbor(self, i, current_iter):
        """
        First neighborhood procedure inspired by spatial distribution with dynamic variance.
        """
        S_i = self.population[i]
        V_i = np.copy(S_i)

        fit_i = self.fitness[i]
        fit_h = np.max(self.fitness)

        # Intent: Safeguard against division by zero if all fitnesses are 0
        if fit_h == 0:
            ChgPerct = self.Chg_max
        else:
            ChgPerct = ((fit_h - fit_i) / fit_h) * (self.Chg_max - self.Chg_min) + self.Chg_min

        change_count = int(np.ceil((self.D * ChgPerct) / 100.0))
        # Ensure at least 1 mutation occurs, but don't exceed dimensions
        change_count = max(1, min(change_count, self.D))

        # Intent: Shift from exploration (high variance) to exploitation (low variance) over time
        decay_factor = ((self.iter_max - current_iter) / self.iter_max) ** self.nmi
        sigma_iter = decay_factor * (self.sigma_initial - self.sigma_final) + self.sigma_final

        indices = self.rng.choice(self.D, size=change_count, replace=False)
        for j in indices:
            V_i[j] = S_i[j] + self.rng.normal(0, sigma_iter)
            V_i[j] = np.clip(V_i[j], -1.0, 1.0)

        return V_i

    def _algorithm_3_neighbor(self, i, current_iter):
        """
        Second neighborhood procedure based on cross-solution parameter differences.
        """
        S_i = self.population[i]
        V_i = np.copy(S_i)

        P_copy = ((self.P_final - self.P_initial) / self.iter_max) * current_iter + self.P_initial

        k = self.rng.choice([idx for idx in range(self.SN) if idx != i])
        S_k = self.population[k]

        for j in range(self.D):
            r = self.rng.rand()
            if r >= P_copy:
                phi = self.rng.uniform(-1.0, 1.0)
                V_i[j] = S_i[j] + phi * (S_i[j] - S_k[j])
                V_i[j] = np.clip(V_i[j], -1.0, 1.0)

        return V_i

    def _generate_neighbor(self, i, current_iter):
        # Dynamic Router for Algorithms
        if self.algo_type == 'algo2':
            return self._algorithm_2_neighbor(i, current_iter)
        else:
            return self._algorithm_3_neighbor(i, current_iter)

    def fit(self, X_train, y_train):
        X_train_np = np.asarray(X_train)
        y_train_np = np.asarray(y_train)

        for _ in range(self.SN):
            S = self._generate_random_solution()
            self.population.append(S)

        for i in range(self.SN):
            self.fitness[i], elm = self._evaluate_fitness(self.population[i], X_train_np, y_train_np)
            if self.fitness[i] > self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = np.copy(self.population[i])
                self.best_elm = copy.deepcopy(elm)

        for current_iter in range(1, self.iter_max + 1):
            startt = time.time()

            # Employed Bees
            for i in range(self.SN):
                V_i = self._generate_neighbor(i, current_iter)
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

            # Onlooker Bees
            prob = self.fitness / np.sum(self.fitness)

            t = 0
            i = 0
            while t < self.SN:
                if self.rng.rand() < prob[i]:
                    V_i = self._generate_neighbor(i, current_iter)
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

            # Scout Bees
            for i in range(self.SN):
                if self.trials[i] >= self.limit:
                    self.population[i] = self._generate_random_solution()
                    self.fitness[i], _ = self._evaluate_fitness(self.population[i], X_train_np, y_train_np)
                    self.trials[i] = 0

            print(f"Iteration {current_iter} end : {time.time() - startt:.4f}s")

    def predict(self, X_test):
        if self.best_elm is None:
            raise ValueError("The model must be fitted before predicting.")
        return self.best_elm.predict(np.asarray(X_test))