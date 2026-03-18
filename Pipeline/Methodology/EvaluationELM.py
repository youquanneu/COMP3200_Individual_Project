import logging
import math

import pandas as pd

from Pipeline.Methodology.CrossValidationDataSplit import CrossValidationDataSplit
from Pipeline.Methodology.EvaluationMatrix import EvaluationMatrix
from Pipeline.Algorithm.ExtremeLearningMachine import ExtremeLearningMachine
from Pipeline.Global.GlobalSetting import GlobalSetting

logger = logging.getLogger(__name__)
class EvaluationELM:
    def __init__(self, x_train, y_train,
                 activation_function,
                 elm_init_seed_range = None,
                 k_fold = 5):

        self.x_train = x_train
        self.y_train = y_train
        self.activation_function = activation_function

        self.elm_init_seed_range = GlobalSetting.initial_seed_range \
            if elm_init_seed_range is None else elm_init_seed_range

        self.k_fold = GlobalSetting.data_cv_fold \
            if k_fold is None else k_fold


    def ranged_seed_cross_validation(self,
                                     hidden_size = None,
                                     regularization_lambda = 0.0):

        if hidden_size is None:
            hidden_size = self.x_train.shape[1]

        global_results_accumulator = []


        splitter = CrossValidationDataSplit(k_fold = self.k_fold)
        folds = splitter.k_fold_data_spiting(self.x_train, self.y_train)

        for fold_idx in range(self.k_fold):
            fold = folds[fold_idx]
            x_tr    , y_tr      = fold['X_train_fold']  , fold['y_train_fold']
            x_val   , y_val     = fold['X_val_fold']    , fold['y_val_fold']
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
                    features_size           = x_tr.shape[1],
                    hidden_size             = hidden_size,
                    activation_function     = self.activation_function,
                    regularization_lambda   = regularization_lambda
                )
                elm.initialize_random_weights(random_seed=elm_seed)
                elm.fit(x_tr, y_tr)

                y_pred = elm.predict(x_val.values)
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
                # Add a newline (\n) before the error so it doesn't overwrite the progress text
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

        return pd.concat(raw_results_list, ignore_index=True), pd.concat(agg_results_list, ignore_index=True)

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

        return pd.concat(raw_results_list, ignore_index=True), pd.concat(agg_results_list, ignore_index=True)

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
    def extract_top_results(dataframe: pd.DataFrame,
                            base_metric_name: str = None,
                            punish_coefficient: float = None,
                            top_k: int = 5) -> pd.DataFrame:
        """
        Ranks results using robust Lower Confidence Bound (LCB).
        [Epistemological Shift] Variance (SEM) now strictly measures ELM structural
        stability across initializations, as data folds are deterministic.

        Formula: Mean - (Coefficient * Standard_Error)
        """
        if base_metric_name is None:
            base_metric_name = f"avg_{GlobalSetting.evaluation_function}_Seed"

        if punish_coefficient is None:
            punish_coefficient = GlobalSetting.seed_punish_coefficient

        mean_col = f"{base_metric_name}_Mean"
        sem_col = f"{base_metric_name}_SEM"

        # Validation: Ensure upstream deterministic aggregations exist
        if mean_col not in dataframe.columns:
            logging.error(f"Critical: Mean column '{mean_col}' missing from aggregated DataFrame.")
            return pd.DataFrame()

        # Fallback mechanism if SEM was strictly zeroed out or missing due to n=1 ELM seeds
        if sem_col not in dataframe.columns:
            logging.warning(f"SEM column '{sem_col}' absent. Assuming zero structural variance.")
            dataframe[sem_col] = 0.0

        if dataframe.empty:
            return dataframe

        # Isolate the maximum standard error to penalize NaNs (failed initializations)
        max_sem = dataframe[sem_col].max()
        fallback_penalty = max_sem if pd.notna(max_sem) else 0.0

        # Compute the Lower Confidence Bound.
        # High coefficient = Risk-averse (favors highly stable ELM architectures)
        # Low coefficient  = Risk-seeking (favors peak accuracy regardless of stability)
        adjusted_score = dataframe[mean_col] - (punish_coefficient * dataframe[sem_col].fillna(fallback_penalty))

        return dataframe.assign(rank_score=adjusted_score).nlargest(top_k, columns='rank_score')

    @staticmethod
    def random_seed_metrics(data_frame):
        """
        Calculates variance exclusively based on the K-Fold averages.
        This isolates the structural stability of the ELM initialization.
        """
        # Exclude metadata columns from the math
        ignore_cols = ['Hidden_Nodes', 'Activation', 'Lambda_Value', 'Fold', 'ELM_Seed']
        metric_cols = [col for col in data_frame.columns if col not in ignore_cols]

        # STEP 1: Calculate the K-Fold Average for each independent ELM_Seed
        # This collapses the fold variance and gives the expected performance of the network.
        k_fold_averages = data_frame.groupby(['ELM_Seed'])[metric_cols].mean().reset_index()

        # STEP 2: Calculate variance based ONLY on those K-Fold averages
        n_seeds = len(k_fold_averages)
        final_mean = k_fold_averages[metric_cols].mean()
        final_std = k_fold_averages[metric_cols].std()

        # Standard Error of the Mean (SEM)
        final_sem = final_std / math.sqrt(n_seeds) if n_seeds > 1 else 0.0

        flat_results = {}
        for metric in metric_cols:
            flat_results[f"avg_{metric}_Seed_Mean"] = round(final_mean[metric], 8)
            flat_results[f"avg_{metric}_Seed_Std"] = round(final_std[metric], 8)
            flat_results[f"avg_{metric}_Seed_SEM"] = round(final_sem[metric] if n_seeds > 1 else 0.0, 8)

            # Note: Min and Max are now based on the K-fold averages,
            # representing the "Worst Expected Initialization" and "Best Expected Initialization"
            flat_results[f"avg_{metric}_Seed_Min"] = round(k_fold_averages[metric].min(), 8)
            flat_results[f"avg_{metric}_Seed_Max"] = round(k_fold_averages[metric].max(), 8)

        return pd.DataFrame([flat_results])