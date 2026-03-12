import logging

import pandas as pd

from Pipeline.Algorithm.CrossValidationDataSplit import CrossValidationDataSplit
from Pipeline.Algorithm.EvaluationMatrix import EvaluationMatrix
from Pipeline.Algorithm.ExtremeLearningMachine import ExtremeLearningMachine
from Pipeline.Global.GlobalSetting import GlobalSetting

logger = logging.getLogger(__name__)
class EvaluationELM:
    def __init__(self, x_train , y_train, activation_function,
                 elm_initial_state_range=range(40, 61),
                 k_fold=5):

        self.x_train = x_train
        self.y_train = y_train
        self.activation_function = activation_function

        self.elm_initial_state_range = elm_initial_state_range

        self.k_fold = k_fold


    def ranged_seed_cross_validation(self,
                                     hidden_size=None,
                                     regularization_lambda=0.0):

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
        final_metrics = EvaluationMatrix.random_seed_metrics(raw_results_df)

        final_metrics.insert(0, 'Hidden_Nodes', hidden_size)
        final_metrics.insert(1, 'Activation', self.activation_function.__name__)
        final_metrics.insert(2, 'Lambda_Value', regularization_lambda)

        return raw_results_df, final_metrics

    def single_fold_elm_result(self, x_tr, y_tr, x_val, y_val,
                               hidden_size, regularization_lambda,
                               fold_idx):
        fold_results = []

        for elm_seed in self.elm_initial_state_range:
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
                logger.error(f"Failed at Fold {fold_idx}, ELM Seed {elm_seed}: {e}")
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
                            punish_coefficient: float = 1.0,
                            top_k: int = 5) -> pd.DataFrame:
        """
        Ranks results using robust Lower Confidence Bound (LCB).
        [Epistemological Shift] Variance (SEM) now strictly measures ELM structural
        stability across initializations, as data folds are deterministic.

        Formula: Mean - (Coefficient * Standard_Error)
        """
        if base_metric_name is None:
            base_metric_name = f"avg_{GlobalSetting.evaluation_function}_Seed"

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