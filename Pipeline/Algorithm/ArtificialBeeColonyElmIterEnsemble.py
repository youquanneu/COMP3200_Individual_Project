import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler

from Pipeline.Algorithm.ArtificialBeeColonyElmCV import ArtificialBeeColonyElmCV


class ArtificialBeeColonyElmIterEnsemble(ArtificialBeeColonyElmCV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ensemble_models = []

    def train_best_model(self, x_train, y_train):
        ensemble_models = []

        x_train_np = np.asarray(x_train)
        self.global_scaler = MinMaxScaler()
        x_train_scaled = self.global_scaler.fit_transform(x_train_np)

        for solution in self.get_solutions(self.best_solution_list, self.max_iteration):
            iter_elm = self.build_elm_by_solution(solution, x_train_scaled, y_train)
            ensemble_models.append(iter_elm)

        self.ensemble_models = ensemble_models
        return ensemble_models

    def predict(self, x_test):
        if not self.ensemble_models:
            raise ValueError("The ensemble must be fitted before predicting.")

        x_test_np = np.asarray(x_test)
        x_test_scaled = self.global_scaler.transform(x_test_np)
        all_predictions = []

        for iter_elm in self.ensemble_models:
            iter_prediction = iter_elm.predict(x_test_scaled)
            all_predictions.append(iter_prediction)

        all_predictions_matrix = np.array(all_predictions)
        majority_votes, _ = mode(all_predictions_matrix, axis=0, keepdims=False)

        return majority_votes.ravel()

    def get_validation_fitness(self, solution, x_train, y_train):

        all_val_predictions = []
        best_models = self.train_best_model(x_train, y_train)

        x_val_np = np.asarray(self.x_val)
        x_val_scaled = self.global_scaler.transform(x_val_np)
        for iter_elm in best_models:
            iter_prediction = iter_elm.predict(x_val_scaled)
            all_val_predictions.append(iter_prediction)

        all_predictions_matrix = np.array(all_val_predictions)
        majority_votes, _ = mode(all_predictions_matrix, axis=0, keepdims=False)
        ensemble_predictions = majority_votes.ravel()

        return  self.get_fitness(self.y_val, ensemble_predictions)

    def get_solutions(self, best_solution_list, max_iteration):
        selected_solutions = [best_solution_list[-1][1]]

        best_solution_size = len(best_solution_list)

        # Drop the first solution and last solution(last solution already saved)
        candidate_list = best_solution_list[1:best_solution_size - 1]

        # Prevent solution as None
        def safe_append(candidate):
            if candidate is not None:
                selected_solutions.append(candidate)

        third_index = max_iteration / 3
        half_index  = max_iteration / 2

        first_third_solution  = [candidate[1] for candidate in candidate_list if candidate[0] < third_index]
        second_third_solution = [candidate[1] for candidate in candidate_list if
                                 third_index <= candidate[0] < third_index * 2]
        last_third_solution = [candidate[1] for candidate in candidate_list if candidate[0] >= third_index * 2]
        front_half_solution = [candidate[1] for candidate in candidate_list if candidate[0] < half_index]
        back_half_solution  = [candidate[1] for candidate in candidate_list if candidate[0] >= half_index]

        later_fit   = len(last_third_solution) >= 3
        middle_fit  = len(second_third_solution) >= 2
        back_fit    = len(back_half_solution) >= 2

        if later_fit and middle_fit and best_solution_size >= 11:
            safe_append(self.selected_furthest(last_third_solution, selected_solutions))
            safe_append(self.selected_furthest(last_third_solution, selected_solutions))
            safe_append(self.selected_furthest(second_third_solution, selected_solutions))
            safe_append(self.selected_furthest(first_third_solution, selected_solutions))

        elif back_fit and best_solution_size >= 7:
            safe_append(self.selected_furthest(back_half_solution, selected_solutions))
            safe_append(self.selected_furthest(front_half_solution, selected_solutions))

        return selected_solutions
    @staticmethod
    def selected_furthest(candidate_list, selected_best_solutions):
        selected_candidate = None
        min_similarity = float('inf')

        selected_l2_norm = [np.linalg.norm(solution) for solution in selected_best_solutions]

        for candidate in candidate_list:

            candidate_l2_norm = np.linalg.norm(candidate)
            if candidate_l2_norm == 0:
                continue
            similarities = []

            for selected_solution, selection_norm in zip(selected_best_solutions, selected_l2_norm):
                if selection_norm == 0:
                    similarity = 0
                else:
                    similarity = candidate @ selected_solution / (candidate_l2_norm * selection_norm)
                similarities.append(similarity)

            if similarities:
                max_sim_of_candidate = max(similarities)

                if max_sim_of_candidate < min_similarity:
                    min_similarity = max_sim_of_candidate
                    selected_candidate = candidate

        return selected_candidate