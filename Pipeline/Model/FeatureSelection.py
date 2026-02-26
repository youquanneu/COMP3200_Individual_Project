from collections import deque

import numpy as np

from Pipeline.Model.Metaheuristic import FeatureSelection


class BinaryGeneticAlgorithm(FeatureSelection):
    def __init__(self, pop_size: int, max_iter: int, mutation_rate: float = 0.05):
        super().__init__(pop_size, max_iter)
        self.mutation_rate = mutation_rate

    def select_features(self, X: np.ndarray, y: np.ndarray, evaluate_func: callable) -> np.ndarray:
        num_features = X.shape[1]

        # 1. Initialize binary population
        population = np.random.randint(2, size=(self.pop_size, num_features)).astype(bool)

        best_mask = None
        global_best_score = -float('inf')

        for iteration in range(self.max_iter):
            # Evaluate population
            scores = np.array([evaluate_func(X[:, ind], y) for ind in population])

            # Track global best
            current_best_idx = np.argmax(scores)
            if scores[current_best_idx] > global_best_score:
                global_best_score = scores[current_best_idx]
                best_mask = np.copy(population[current_best_idx])

            new_population = []

            # 2. Reproduction Loop
            for _ in range(self.pop_size // 2):
                # Tournament Selection (Size = 3)
                parents = []
                for _ in range(2):
                    tournament_idx = np.random.choice(self.pop_size, 3, replace=False)
                    winner_idx = tournament_idx[np.argmax(scores[tournament_idx])]
                    parents.append(population[winner_idx])

                # 3. Uniform Crossover
                cross_mask = np.random.rand(num_features) > 0.5
                child1 = np.where(cross_mask, parents[0], parents[1])
                child2 = np.where(cross_mask, parents[1], parents[0])

                # 4. Bit-flip Mutation
                for child in (child1, child2):
                    mut_mask = np.random.rand(num_features) < self.mutation_rate
                    child[mut_mask] = ~child[mut_mask]
                    new_population.append(child)

            population = np.array(new_population)

        return best_mask


class TabuSearch(FeatureSelection):
    def __init__(self, max_iter: int, tabu_tenure: int):
        # We don't need pop_size since Tabu Search is a single-agent algorithm
        super().__init__()
        self.max_iter = max_iter
        self.tabu_tenure = tabu_tenure
        # deque automatically deletes the oldest memory when it hits maxlen
        self.tabu_list = deque(maxlen=self.tabu_tenure)

    def select_features(self, X: np.ndarray, y: np.ndarray, evaluate_func: callable) -> np.ndarray:
        num_features = X.shape[1]

        # 1. Initialize: Start with ALL features selected for the gallstone dataset
        current_mask = np.ones(num_features, dtype=bool)
        best_mask = np.copy(current_mask)

        # Evaluate baseline
        best_score = evaluate_func(X[:, best_mask], y)
        self.tabu_list.append(tuple(current_mask))  # Store memory as a tuple

        for iteration in range(self.max_iter):
            best_neighbor_mask = None
            best_neighbor_score = -float('inf')

            # 2. Neighborhood Generation: Try flipping one feature bit at a time
            for i in range(num_features):
                neighbor = np.copy(current_mask)
                neighbor[i] = not neighbor[i]  # Flip True to False, or False to True

                # Engineering Reality Check: Prevent dropping 100% of the features
                if not np.any(neighbor):
                    continue

                neighbor_tuple = tuple(neighbor)

                # 3. The Tabu Check: Only evaluate if we haven't been here recently
                if neighbor_tuple not in self.tabu_list:
                    score = evaluate_func(X[:, neighbor], y)

                    if score > best_neighbor_score:
                        best_neighbor_score = score
                        best_neighbor_mask = neighbor

            # 4. State Transition: Move to the best valid neighbor
            if best_neighbor_mask is not None:
                current_mask = best_neighbor_mask
                self.tabu_list.append(tuple(current_mask))  # Update memory

                # Update global best if we hit a new peak
                if best_neighbor_score > best_score:
                    best_score = best_neighbor_score
                    best_mask = np.copy(current_mask)
            else:
                # Edge case: All neighbors are in the Tabu List.
                # The algorithm is boxed in, so we terminate early.
                break

        return best_mask