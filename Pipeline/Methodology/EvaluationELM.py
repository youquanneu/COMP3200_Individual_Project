import logging
import math

import pandas as pd

from Pipeline.Global.GallstoneDataSet import GallstoneDataSet
from Pipeline.Methodology.EvaluationMatrix import EvaluationMatrix
from Pipeline.Algorithm.ExtremeLearningMachine import ExtremeLearningMachine
from Pipeline.Global.GlobalSetting import GlobalSetting

logger = logging.getLogger(__name__)
class EvaluationELM:
    def __init__(self,
                 activation_function,
                 use_raw_data: bool = False,
                 data_scaling: bool = False,
                 elm_init_seed_range = None,
                 k_fold = 5):

        self.activation_function = activation_function

        self.elm_init_seed_range = GlobalSetting.elm_initial_state_range \
            if elm_init_seed_range is None else elm_init_seed_range

        self.k_fold = GlobalSetting.data_cv_fold \
            if k_fold is None else k_fold

        self.use_raw_data = use_raw_data
        self.data_scaling = data_scaling
        self.gallstone_dataset = GallstoneDataSet()

        if self.use_raw_data:
            self.gallstone_dataset.fetch_raw_data_path()
        else:
            self.gallstone_dataset.fetch_cleaned_data_path()

        self.gallstone_dataset.cv_test_split(self.k_fold)

        self.feature_size = self.gallstone_dataset.x.shape[1]


    def ranged_seed_cross_validation(self,
                                     hidden_size = None,
                                     regularization_lambda = 0.0):

        if hidden_size is None:
            hidden_size = self.feature_size

        global_results_accumulator = []

        data_split = self.gallstone_dataset.val_scaled_fold_split \
            if self.data_scaling else self.gallstone_dataset.val_fold_split

        for fold_idx, (x_tr, y_tr, x_val, y_val) in enumerate(data_split):
            fold_result = self.single_fold_elm_result(x_tr, y_tr,
                                                      x_val, y_val,
                                                      hidden_size, regularization_lambda,
                                                      fold_idx)
            global_results_accumulator.extend(fold_result)

        raw_results_df = pd.DataFrame(global_results_accumulator)
        final_metrics = self.random_seed_metrics(raw_results_df)

        final_metrics.insert(0, 'Hidden_Nodes', hidden_size)
        final_metrics.insert(1, 'Activation', self.activation_function.__name__)
        final_metrics.insert(2, 'Lambda_Value', regularization_lambda)

        return raw_results_df, final_metrics

    def single_fold_elm_result(self, x_tr, y_tr, x_val, y_val,
                               hidden_size, regularization_lambda,
                               fold_idx):
        fold_results = []

        for elm_seed in self.elm_init_seed_range:
            try:
                elm = ExtremeLearningMachine(
                    feature_size            = x_tr.shape[1],
                    hidden_size             = hidden_size,
                    activation_function     = self.activation_function,
                    regularization_lambda   = regularization_lambda
                )
                elm.initialize_random_weights(random_seed=elm_seed)
                elm.fit(x_tr, y_tr)

                y_pred = elm.predict(x_val.values if isinstance(x_val, pd.DataFrame) else x_val)
                evaluation_metrics = EvaluationMatrix(y_val, y_pred).get_all_metrics()

                metrics = {
                    'Hidden_Nodes'          : hidden_size,
                    'Activation'            : self.activation_function.__name__,
                    'Lambda_Value'          : regularization_lambda,
                    'Fold'                  : fold_idx,
                    'ELM_Seed'              : elm_seed,
                    **evaluation_metrics
                }

                fold_results.append(metrics)

            except Exception as e:
                print(f"\n    [!] ERROR: Failed at Fold {fold_idx + 1}, ELM Seed {elm_seed}: {e}")
                continue
        return fold_results

    def grid_search_hidden_size(self, hidden_size_range):
        raw_results_list = []
        agg_results_list = []

        for hidden_size in hidden_size_range:
            raw_res, agg_res = self.ranged_seed_cross_validation(
                hidden_size  = hidden_size
            )
            raw_results_list.append(raw_res)
            agg_results_list.append(agg_res)

        return pd.concat(raw_results_list, ignore_index = True), pd.concat(agg_results_list, ignore_index = True)

    def grid_search_lambda(self,hidden_size,lambda_range):
        raw_results_list = []
        agg_results_list = []

        for lambda_value in lambda_range:
            raw_res, agg_res = self.ranged_seed_cross_validation(
                hidden_size = hidden_size,
                regularization_lambda   = lambda_value
            )

            raw_results_list.append(raw_res)
            agg_results_list.append(agg_res)

        return pd.concat(raw_results_list, ignore_index = True), pd.concat(agg_results_list, ignore_index = True)

    def grid_search_hidden_size_and_lambda(self,hidden_size_range,lambda_range):
        raw_results_list = []
        agg_results_list = []

        for hidden_size in hidden_size_range:
            raw_res, agg_res = self.grid_search_lambda(
                hidden_size  = hidden_size,
                lambda_range = lambda_range
            )
            raw_results_list.append(raw_res)
            agg_results_list.append(agg_res)

        return pd.concat(raw_results_list, ignore_index=True), pd.concat(agg_results_list, ignore_index=True)

    @staticmethod
    def random_seed_metrics(data_frame):
        ignore_cols = ['Hidden_Nodes', 'Activation', 'Lambda_Value', 'Fold', 'ELM_Seed']
        metric_cols = [col for col in data_frame.columns if col not in ignore_cols]

        grouped_by_seed = data_frame.groupby(['ELM_Seed'])[metric_cols]
        fold_means = grouped_by_seed.mean()
        fold_stds = grouped_by_seed.std().fillna(0.0)

        cv_punish_coe = getattr(GlobalSetting, 'cv_punish_coe', 1.0)
        seed_lcb = fold_means - (cv_punish_coe * fold_stds)

        n_seeds = len(seed_lcb)
        final_mean = seed_lcb.mean()
        final_std = seed_lcb.std().fillna(0.0)

        final_sem = final_std / math.sqrt(n_seeds) if n_seeds > 1 else 0.0

        flat_results = {}
        for metric in metric_cols:
            flat_results[f"lcb_{metric}_Seed_Mean"] = round(final_mean[metric], 8)
            flat_results[f"lcb_{metric}_Seed_Std"] = round(final_std[metric], 8)
            flat_results[f"lcb_{metric}_Seed_SEM"] = round(final_sem[metric] if n_seeds > 1 else 0.0, 8)

            flat_results[f"lcb_{metric}_Seed_Min"] = round(seed_lcb[metric].min(), 8)
            flat_results[f"lcb_{metric}_Seed_Max"] = round(seed_lcb[metric].max(), 8)

        return pd.DataFrame([flat_results])

    @staticmethod
    def extract_top_results(dataframe: pd.DataFrame,
                            base_metric_name    : str   = None,
                            punish_coefficient  : float = None,
                            top_k   : int = 5) -> pd.DataFrame:

        if base_metric_name is None:
            base_metric_name = f"lcb_{GlobalSetting.evaluation_function}_Seed"

        if punish_coefficient is None:
            punish_coefficient = GlobalSetting.seed_punish_coe

        mean_col = f"{base_metric_name}_Mean"
        sem_col = f"{base_metric_name}_SEM"

        if mean_col not in dataframe.columns:
            logging.error(f"Critical: Mean column '{mean_col}' missing from aggregated DataFrame.")
            return pd.DataFrame()

        if sem_col not in dataframe.columns:
            logging.warning(f"SEM column '{sem_col}' absent. Assuming zero structural variance.")
            dataframe[sem_col] = 0.0

        if dataframe.empty:
            return dataframe

        max_sem = dataframe[sem_col].max()
        fallback_penalty = max_sem if pd.notna(max_sem) else 0.0

        adjusted_score = dataframe[mean_col] - (punish_coefficient * dataframe[sem_col].fillna(fallback_penalty))

        return dataframe.assign(rank_score=adjusted_score).nlargest(top_k, columns='rank_score')