import time

import numpy as np

from Pipeline.Global.GlobalSetting import GlobalSetting
from Pipeline.Methodology.EvaluationMatrix import EvaluationMatrix
from Pipeline.Algorithm.ExtremeLearningMachine import ExtremeLearningMachine

class ArtificialBeeColonyElm:
    def __init__(self, feature_size, hidden_size,
                 activation_function, regularization_lambda = 0.0,
                 random_state = None, fitness_function = None,
                 solution_size = 10, trial_limit = 10, max_iteration = 100,
                 max_change=20.0, min_change=3.0, initial_sigma=0.8, final_sigma=0.1, nmi=3,
                 initial_probability=0.0, final_probability=1.0
                 ):

        self.feature_size = feature_size
        self.hidden_size  = hidden_size
        self.solution_dimension = (self.feature_size + 1) * self.hidden_size

        self.activation_function     = activation_function
        self.regularization_lambda   = regularization_lambda

        self.preset_random_seed    = None
        self.random_state   = np.random.RandomState(random_state) if random_state is not None else np.random.RandomState()

        self.fitness_function = GlobalSetting.evaluation_function if fitness_function is None else fitness_function

        self.solution_size  = solution_size
        self.trial_limit    = trial_limit
        self.max_iteration  = max_iteration

        self.use_algo2_for_employed_bee = True
        self.use_algo2_for_onlooker_bee = True

        self.population = []
        self.fitness    = np.zeros(self.solution_size)
        self.trials     = np.zeros(self.solution_size)

        self.best_fitness   = -np.inf
        self.best_solution  = None
        self.best_elm       = None

        """ Algo 2 parameter """
        self.max_change     = max_change
        self.min_change     = min_change
        self.initial_sigma  = initial_sigma
        self.final_sigma    = final_sigma
        self.non_linear_modulation_index = nmi

        """ Algo 3 parameter """
        self.initial_probability = initial_probability
        self.final_probability   = final_probability

        self.convergence_curve = []
        self.val_fitness_curve = []
        self.scout_trigger_history = []

        self.x_val = None
        self.y_val = None

    def init_random_state(self,random_state):
        self.preset_random_seed = random_state
        self.random_state = np.random.RandomState(random_state)

    def init_algo2(self, max_change = 20.0, min_change = 3.0,
                   initial_sigma = 0.8, final_sigma = 0.1,
                   nmi = 3):
        self.max_change     = max_change
        self.min_change     = min_change
        self.initial_sigma  = initial_sigma
        self.final_sigma    = final_sigma
        self.non_linear_modulation_index = nmi

    def init_algo3(self,initial_probability = 0.0, final_probability = 1.0):
        self.initial_probability    = initial_probability
        self.final_probability      = final_probability

    def employed_bee_apply_algo2(self):
        self.use_algo2_for_employed_bee = True
    def employed_bee_apply_algo3(self):
        self.use_algo2_for_employed_bee = False
    def onlooker_bee_apply_algo2(self):
        self.use_algo2_for_onlooker_bee = True
    def onlooker_bee_apply_algo3(self):
        self.use_algo2_for_onlooker_bee = False
    def apply_validation_dataset(self, x_val, y_val):
        self.x_val = np.asarray(x_val)
        self.y_val = np.asarray(y_val)
    def generate_random_solution(self):
        return self.random_state.uniform(-1.0, 1.0, self.solution_dimension)

    def get_fitness(self, y_true, y_pred):
        evaluation = EvaluationMatrix(y_true, y_pred)

        metric_map = {
            "Accuracy"      : evaluation.get_accuracy,
            "Precision"     : evaluation.get_precision,
            "Recall"        : evaluation.get_recall,
            "NPV"           : evaluation.get_npv,
            "Specificity"   : evaluation.get_specificity,
            "F1-Score"      : evaluation.get_f1_score,
            "F2-Score"      : evaluation.get_f2_score,
            "Bal Accuracy"  : evaluation.get_bal_accuracy,
            "MCC"           : evaluation.get_mcc
        }

        fitness_function = metric_map.get(self.fitness_function)
        fitness_value = fitness_function()

        """ Apply function to make sure the MCC value always be positive when evaluating in ABC """
        if self.fitness_function == "MCC":
            return (fitness_value + 1.0) / 2.0

        return fitness_value

    def build_elm_by_solution(self, solution, x_train, y_train):
        weight_boundary = self.feature_size * self.hidden_size
        hidden_weight   = solution[:weight_boundary].reshape(self.feature_size, self.hidden_size)
        hidden_bias     = solution[weight_boundary:]

        elm = ExtremeLearningMachine(self.feature_size, self.hidden_size,
                                     self.activation_function, self.regularization_lambda)
        elm.apply_hidden_weights(hidden_weight)
        elm.apply_hidden_bias(hidden_bias)

        elm.fit(x_train, y_train)
        return elm
    def get_evaluation_fitness(self, solution, x_train, y_train):
        elm = self.build_elm_by_solution(solution, x_train, y_train)
        y_pred = elm.predict(x_train)
        return self.get_fitness(y_train, y_pred)

    def get_validation_fitness(self, solution, x_train, y_train):
        elm = self.build_elm_by_solution(solution, x_train, y_train)
        y_val_pred = elm.predict(self.x_val)
        return self.get_fitness(self.y_val, y_val_pred)
    def neighboring_s_algo_2(self, index, current_iteration):

        solution_s_idx = self.population[index]
        solution_v_idx = np.copy(solution_s_idx)

        fitness_idx  = self.fitness[index]
        fitness_best = np.max(self.fitness)

        if fitness_best == 0:
            change_percentage = self.max_change
        else:
            change_percentage = (((fitness_best - fitness_idx) / fitness_best)
                                 *(self.max_change - self.min_change) + self.min_change)

        change_count = int(np.ceil((self.solution_dimension * change_percentage) / 100))
        change_count = max(1, min(change_count, self.solution_dimension))

        decay_factor    = ((self.max_iteration - current_iteration) / self.max_iteration ) ** self.non_linear_modulation_index
        sigma_iteration = decay_factor * (self.initial_sigma - self.final_sigma) + self.final_sigma

        indexes = self.random_state.choice(self.solution_dimension, size = change_count, replace = False)

        """ Academic algorithm """
        # for j in indexes:
        #     solution_v_idx[j] = solution_s_idx[j] + self.random_state.normal(0,sigma_iteration)
        #     solution_v_idx[j] = np.clip(solution_v_idx[j] , -1.0 , 1.0)

        """ Speed optimization """
        noise = self.random_state.normal(0, sigma_iteration, size=change_count)
        solution_v_idx[indexes] = np.clip(solution_s_idx[indexes] + noise, -1.0, 1.0)

        return solution_v_idx

    def neighboring_s_algo_3(self, index, current_iteration):
        solution_s_idx = self.population[index]

        copy_probability  = (((self.final_probability - self.initial_probability) / self.max_iteration)
                             * current_iteration + self.initial_probability)

        """ Academic algorithm """
        # k = self.random_state.choice([idx for idx in range(self.solution_size) if idx != index])
        # solution_k_random = self.population[k]
        #
        # for j in range(self.D):
        #     r = self.random_state.rand()
        #     if r >= copy_probability :
        #         phi = self.random_state.uniform(-1.0,1.0)
        #         solution_v_idx[j] = solution_s_idx[j] + phi * (solution_s_idx[j] - solution_k_random[j])
        #         solution_v_idx[j] = np.clip(solution_v_idx[j], -1.0 , 1.0)

        """ Speed optimization """
        k = self.random_state.randint(0, self.solution_size - 1)
        if k >= index:
            k += 1
        solution_k_random = self.population[k]

        r_array = self.random_state.rand(self.solution_dimension)
        phi_array = self.random_state.uniform(-1.0, 1.0, size=self.solution_dimension)

        mutation_step = phi_array * (solution_s_idx - solution_k_random)

        solution_v_idx = np.where(r_array >= copy_probability,
                                  solution_s_idx + mutation_step,
                                  solution_s_idx)

        solution_v_idx = np.clip(solution_v_idx, -1.0, 1.0)

        return solution_v_idx

    def neighbour_iteration(self, index, solution_v_idx, x_train, y_train):
        v_idx_result = self.get_evaluation_fitness(solution_v_idx, x_train, y_train)
        if v_idx_result > self.fitness[index]:
            self.population[index] = solution_v_idx
            self.fitness[index] = v_idx_result
            self.trials[index] = 0
            if v_idx_result > self.best_fitness:
                self.best_fitness = v_idx_result
                self.best_solution = np.copy(solution_v_idx)
        else:
            self.trials[index] += 1

    def initialize_bee_colony(self, x_train , y_train):
        for _ in range(self.solution_size):
            self.population.append(self.generate_random_solution())

        for index in range(self.solution_size):
            self.fitness[index] = self.get_evaluation_fitness(self.population[index], x_train, y_train)
            if self.fitness[index] > self.best_fitness:
                self.best_fitness = self.fitness[index]
                self.best_solution = np.copy(self.population[index])
    def employed_bee(self, current_iteration, x_train, y_train):
        for index in range(self.solution_size):
            solution_v_idx = self.neighboring_s_algo_2(index, current_iteration) \
                if self.use_algo2_for_employed_bee \
                else self.neighboring_s_algo_3(index, current_iteration)
            self.neighbour_iteration(index, solution_v_idx, x_train, y_train)

    def onlooker_bee(self,current_iteration,x_train,y_train):
        probability = self.fitness/ np.sum(self.fitness + 1e-20)
        trial = 0
        index = 0
        while trial < self.solution_size:
            if self.random_state.rand() < probability[index]:
                solution_v_idx = self.neighboring_s_algo_2(index, current_iteration)\
                    if self.use_algo2_for_onlooker_bee \
                    else self.neighboring_s_algo_3(index, current_iteration)

                self.neighbour_iteration(index , solution_v_idx, x_train, y_train)

                trial += 1

            index = (index+1)%self.solution_size

    def scout_bee(self,x_train,y_train):
        trigger_count = 0
        for index in range(self.solution_size):
            if self.trials[index] >= self.trial_limit:
                self.population[index] = self.generate_random_solution()
                self.fitness[index] = self.get_evaluation_fitness(self.population[index], x_train, y_train)
                self.trials[index]  =   0
                trigger_count +=1
                if self.fitness[index]  > self.best_fitness:
                    self.best_fitness   = self.fitness[index]
                    self.best_solution  = np.copy(self.population[index])
        return trigger_count

    def fit(self,x_train , y_train):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        self.population = []
        self.fitness = np.zeros(self.solution_size)
        self.trials = np.zeros(self.solution_size)

        self.best_fitness   = -np.inf
        self.best_solution  = None
        self.best_elm       = None

        self.convergence_curve      = []
        self.scout_trigger_history  = []
        self.val_fitness_curve      = []

        self.initialize_bee_colony(x_train, y_train)

        for current_iteration in range(1, self.max_iteration + 1):
            start_time = time.time()

            self.employed_bee(current_iteration, x_train, y_train)
            self.onlooker_bee(current_iteration, x_train, y_train)

            scout_count = self.scout_bee(x_train, y_train)

            self.convergence_curve.append(self.best_fitness)
            self.scout_trigger_history.append(scout_count)

            val_print_str = ""
            if self.x_val is not None and self.best_solution is not None:
                current_val_fitness = self.get_validation_fitness(self.best_solution, x_train, y_train)
                self.val_fitness_curve.append(current_val_fitness)
                val_print_str = f" | Val Fitness: {current_val_fitness:.6f}"

            print(
                f"\rSeed {self.preset_random_seed}  | "
                f"Iteration {current_iteration:03d} complete | "
                f"Duration: {time.time() - start_time:.4f}s | "
                f"Scout Triggers: {scout_count} | "
                f"Best Fitness: {self.best_fitness:.6f}"
                f"{val_print_str}",
                end="", flush=True
            )

        """ Inverse function which revert the MCC record to be true value """
        if self.fitness_function == "MCC":
            self.best_fitness = (self.best_fitness * 2.0) - 1.0
            self.convergence_curve = np.array(self.convergence_curve) * 2.0 - 1.0
            self.val_fitness_curve = np.array(self.val_fitness_curve) * 2.0 - 1.0

        self.train_best_model(x_train, y_train)

    def train_best_model(self,x_train,y_train):
        weight_boundary = self.feature_size * self.hidden_size
        hidden_weight   = self.best_solution[:weight_boundary].reshape(self.feature_size, self.hidden_size)
        hidden_bias     = self.best_solution[weight_boundary:]

        self.best_elm = ExtremeLearningMachine(self.feature_size, self.hidden_size,
                                               self.activation_function, self.regularization_lambda)
        self.best_elm.apply_hidden_weights(hidden_weight)
        self.best_elm.apply_hidden_bias(hidden_bias)
        self.best_elm.fit(x_train, y_train)

    def predict(self, x_test):
        if self.best_elm is None:
            raise ValueError("The model must be fitted before predicting.")
        return self.best_elm.predict(np.asarray(x_test))