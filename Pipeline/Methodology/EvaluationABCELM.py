import logging

import pandas as pd

from Pipeline.Algorithm.ArtificialBeeColonyElmCV import ArtificialBeeColonyElmCV
from Pipeline.Global.GlobalSetting import GlobalSetting

logger = logging.getLogger(__name__)
class EvaluationABCELM:
    def __init__(self,x_train, y_train,
                 activation_function,
                 abc_init_seed_range = None,
                 solution_size_range = None,
                 trial_limit_range   = None,
                 max_iteration_range = None):

        self.x_train = x_train
        self.y_train = y_train
        self.activation_function = activation_function

        self.abc_init_seed_range = GlobalSetting.abc_elm_init_state_range \
            if abc_init_seed_range is None else abc_init_seed_range

        self.solution_size_range = GlobalSetting.solution_size_explore_range \
            if solution_size_range is None else solution_size_range

        self.trial_limit_range = GlobalSetting.trial_limit_explore_range \
            if trial_limit_range is None else trial_limit_range

        self.max_iteration_range = GlobalSetting.max_iteration_explore_range \
            if max_iteration_range is None else max_iteration_range


    def abc_parameter_explore(self,hidden_size, lambda_value ):
        results = []
        for max_iter in self.max_iteration_range:
            for trial_limit in self.trial_limit_range:
                for solution_size in self.solution_size_range:
                    for seed in self.abc_init_seed_range:
                        status = f"Swarm Search -> Max Iter: {max_iter} | Trial Limit: {trial_limit} | Pop Size: {solution_size} | Seed: {seed}      "
                        print(f"\r{status}", end="", flush=True)
                        abc_model_tested = ArtificialBeeColonyElmCV(
                            features_size       = self.x_train.shape[1]             ,
                            hidden_size         = hidden_size                       ,
                            regularization_lambda= lambda_value                     ,
                            activation_function = self.activation_function          ,
                            fitness_function    = GlobalSetting.evaluation_function ,
                            solution_size       = solution_size                     ,
                            trial_limit         = trial_limit                       ,
                            max_iteration       = max_iter
                        )
                        abc_model_tested.init_random_state(seed)
                        abc_model_tested.fit(self.x_train, self.y_train)

                        results.append({
                            'max_iter'      : max_iter,
                            'trial_limit'   : trial_limit,
                            'solution_size' : solution_size,
                            'seed'          : seed,
                            'best_fitness'  : abc_model_tested.best_fitness
                        })
        df_results = pd.DataFrame(results)
        df_agg_results = self.abc_seed_metrics(df_results)
        return df_results, df_agg_results

    @staticmethod
    def abc_seed_metrics(data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates convergence stability exclusively based on ABC random seeds.
        Groups by hyperparameter configuration to evaluate the meta-heuristic's structural robustness.
        """
        import math

        if data_frame.empty:
            return pd.DataFrame()

        # Define hyperparameter groupings and the target metric
        group_cols = ['max_iter', 'trial_limit', 'solution_size']
        metric_col = 'best_fitness'

        # STEP 1: Aggregate stats across all seeds for each hyperparameter configuration
        agg_df = data_frame.groupby(group_cols)[metric_col].agg(
            Mean='mean',
            Std='std',
            Min='min',
            Max='max',
            Count='count'
        ).reset_index()

        # STEP 2: Handle edge cases (n=1 seed results in NaN standard deviation)
        agg_df['Std'] = agg_df['Std'].fillna(0.0)

        # STEP 3: Calculate Standard Error of the Mean (SEM)
        agg_df['SEM'] = agg_df.apply(
            lambda row: row['Std'] / math.sqrt(row['Count']) if row['Count'] > 1 else 0.0,
            axis=1
        )

        # STEP 4: Standardize column names to match the global evaluation matrix
        rename_map = {
            'Mean': f'avg_{metric_col}_Seed_Mean',
            'Std': f'avg_{metric_col}_Seed_Std',
            'SEM': f'avg_{metric_col}_Seed_SEM',
            'Min': f'avg_{metric_col}_Seed_Min',
            'Max': f'avg_{metric_col}_Seed_Max'
        }

        agg_df = agg_df.rename(columns=rename_map).drop(columns=['Count'])

        return agg_df

    @staticmethod
    def extract_top_results(dataframe: pd.DataFrame,
                            punish_coefficient: float = None,
                            top_k: int = 5) -> pd.DataFrame:
        """
        Ranks ABC hyperparameter configurations using a robust Lower Confidence Bound (LCB).
        Penalizes swarm configurations that are highly sensitive to their initial random seeds.
        """

        # Fallback to GlobalSetting if no specific coefficient is provided
        if punish_coefficient is None:
            punish_coefficient = GlobalSetting.seed_punish_coefficient

        mean_col = f"avg_best_fitness_Seed_Mean"
        sem_col = f"avg_best_fitness_Seed_SEM"

        # Isolate the maximum standard error to penalize failed initializations
        max_sem = dataframe[sem_col].max()
        fallback_penalty = max_sem if pd.notna(max_sem) else 0.0

        # Compute the Lower Confidence Bound (LCB).
        # Score = Expected Fitness minus the structural uncertainty of the swarm
        adjusted_score = dataframe[mean_col] - (punish_coefficient * dataframe[sem_col].fillna(fallback_penalty))

        # Ranks the top K configurations.
        # (Change to nsmallest if best_fitness is an error metric like MSE)
        return dataframe.assign(rank_score=adjusted_score).nlargest(top_k, columns='rank_score')