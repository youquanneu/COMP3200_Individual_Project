import numpy as np

from Pipeline.Model.Metaheuristic import Optimization


class ArtificialBeeColony(Optimization):
    def __init__(self, pop_size: int, max_iter: int, limit: int = 20):
        super().__init__()
        self.pop_size = pop_size
        self.num_sources = pop_size // 2
        self.max_iter = max_iter
        self.limit = limit

    def tune_parameters(self, X_reduced: np.ndarray, y: np.ndarray, evaluate_func: callable) -> tuple:
        bounds = [(10, 1000), (1e-5, 100)]

        # 1. Initialization
        sources = np.zeros((self.num_sources, 2))
        for j in range(2):
            sources[:, j] = np.random.uniform(bounds[j][0], bounds[j][1], self.num_sources)

        fitness = np.array([evaluate_func(X_reduced, y, int(s[0]), s[1]) for s in sources])
        trials = np.zeros(self.num_sources)

        for iteration in range(self.max_iter):
            # --- PHASE 1: Employed Bees ---
            for i in range(self.num_sources):
                # Pick a random neighbor (k) and random dimension (j)
                k = np.random.choice([idx for idx in range(self.num_sources) if idx != i])
                j = np.random.randint(2)
                phi = np.random.uniform(-1, 1)

                # Equation: v_ij = x_ij + phi * (x_ij - x_kj)
                new_pos = np.copy(sources[i])
                new_pos[j] = sources[i, j] + phi * (sources[i, j] - sources[k, j])
                new_pos[j] = np.clip(new_pos[j], bounds[j][0], bounds[j][1])

                new_fit = evaluate_func(X_reduced, y, int(new_pos[0]), new_pos[1])

                if new_fit > fitness[i]:
                    sources[i] = new_pos
                    fitness[i] = new_fit
                    trials[i] = 0
                else:
                    trials[i] += 1

            # --- PHASE 2: Onlooker Bees (Selection by Probability) ---
            probs = fitness / (np.sum(fitness) + 1e-10)  # Simple roulette wheel
            for _ in range(self.num_sources):
                i = np.random.choice(self.num_sources, p=probs)
                # Same neighbor update logic as Employed Phase
                k = np.random.choice([idx for idx in range(self.num_sources) if idx != i])
                j = np.random.randint(2)
                phi = np.random.uniform(-1, 1)

                new_pos = np.copy(sources[i])
                new_pos[j] = sources[i, j] + phi * (sources[i, j] - sources[k, j])
                new_pos[j] = np.clip(new_pos[j], bounds[j][0], bounds[j][1])

                new_fit = evaluate_func(X_reduced, y, int(new_pos[0]), new_pos[1])
                if new_fit > fitness[i]:
                    sources[i] = new_pos
                    fitness[i] = new_fit
                    trials[i] = 0
                else:
                    trials[i] += 1

            # --- PHASE 3: Scout Bees (Handling Stagnation) ---
            for i in range(self.num_sources):
                if trials[i] > self.limit:
                    for j in range(2):
                        sources[i, j] = np.random.uniform(bounds[j][0], bounds[j][1])
                    fitness[i] = evaluate_func(X_reduced, y, int(sources[i, 0]), sources[i, 1])
                    trials[i] = 0

        best_idx = np.argmax(fitness)
        return int(sources[best_idx][0]), sources[best_idx][1]


class DifferentialEvolution(Optimization):
    def __init__(self, pop_size: int, max_iter: int, F: float = 0.8, CR: float = 0.9):
        super().__init__()
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F  # Mutation scale factor [0, 2]
        self.CR = CR  # Crossover probability [0, 1]

    def tune_parameters(self, X_reduced: np.ndarray, y: np.ndarray, evaluate_func: callable) -> tuple:
        # Bounds for ELM: [Hidden Nodes L, Regularization Lambda]
        bounds = [(10, 1000), (1e-5, 100)]

        # 1. Initialize Population
        pop = np.zeros((self.pop_size, 2))
        for j in range(2):
            pop[:, j] = np.random.uniform(bounds[j][0], bounds[j][1], self.pop_size)

        # Initial Evaluation
        fitness = np.array([evaluate_func(X_reduced, y, int(ind[0]), ind[1]) for ind in pop])

        for iteration in range(self.max_iter):
            for i in range(self.pop_size):
                # 2. Mutation: Pick 3 random distinct agents (not i)
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

                # Formula: V = X_r1 + F * (X_r2 - X_r3)
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])

                # 3. Crossover (Binomial)
                trial = np.copy(pop[i])
                j_rand = np.random.randint(2)  # Ensure at least one parameter changes
                for j in range(2):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # Boundary Constraint Handling
                trial[0] = np.clip(trial[0], bounds[0][0], bounds[0][1])
                trial[1] = np.clip(trial[1], bounds[1][0], bounds[1][1])

                # 4. Selection (Greedy)
                trial_fit = evaluate_func(X_reduced, y, int(trial[0]), trial[1])
                if trial_fit >= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fit

        best_idx = np.argmax(fitness)
        return int(pop[best_idx][0]), pop[best_idx][1]