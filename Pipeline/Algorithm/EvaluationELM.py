import logging

import pandas as pd

from Pipeline.Algorithm.CrossValidationDataSplit import CrossValidationDataSplit
from Pipeline.Algorithm.EvaluationMatrix import EvaluationMatrix
from Pipeline.Algorithm.ExtremeLearningMachine import ExtremeLearningMachine

logger = logging.getLogger(__name__)
class EvaluationELM:
    def __init__(self , x, y,activation_function,
                 elm_initial_state_range=range(40, 61),
                 data_split_state_range=range(40, 61),
                 k_fold=5):

        self.x = x
        self.y = y
        self.activation_function = activation_function

        self.elm_initial_state_range = elm_initial_state_range

        self.data_split_state_range  = data_split_state_range
        self.k_fold = k_fold


    def ranged_seed_cross_validation(self,
                                     hidden_size=None,
                                     regularization_lambda=0.0):

        if hidden_size is None:
            hidden_size = self.x.shape[1]

        global_results_accumulator = []

        for data_seed in self.data_split_state_range:

            splitter = CrossValidationDataSplit(random_state = data_seed, k_fold = self.k_fold)
            folds = splitter.k_fold_data_spiting(self.x, self.y)

            for fold_idx in range(self.k_fold):
                fold = folds[fold_idx]
                x_train , y_train   = fold['X_train_fold']  , fold['y_train_fold']
                x_val   , y_val     = fold['X_val_fold']    , fold['y_val_fold']
                fold_result = self.single_fold_elm_result(x_train, y_train,
                                                          x_val, y_val,
                                                          hidden_size, regularization_lambda,
                                                          data_seed, fold_idx)
                global_results_accumulator.extend(fold_result)

        raw_results_df = pd.DataFrame(global_results_accumulator)
        final_metrics = EvaluationMatrix.random_seed_metrics(raw_results_df)

        final_metrics.insert(0, 'Hidden_Nodes', hidden_size)
        final_metrics.insert(1, 'Activation', self.activation_function.__name__)
        final_metrics.insert(2, 'Lambda_Value', regularization_lambda)

        return raw_results_df, final_metrics

    def single_fold_elm_result(self, x_train, y_train, x_val, y_val,
                                    hidden_size, regularization_lambda,
                                    data_seed, fold_idx):
        fold_results = []

        for elm_seed in self.elm_initial_state_range:
            try:
                elm = ExtremeLearningMachine(
                    features_size           = x_train.shape[1],
                    hidden_size             = hidden_size,
                    activation_function     = self.activation_function,
                    regularization_lambda   = regularization_lambda
                )
                elm.initialize_random_weights(random_seed=elm_seed)
                elm.fit(x_train, y_train)

                y_pred = elm.predict(x_val.values)
                evaluation_metrics = EvaluationMatrix(y_val, y_pred).get_all_metrics()

                metrics = {
                    'Hidden_Nodes'          : hidden_size,
                    'Activation'            : self.activation_function.__name__,
                    'Lambda_Value'          : regularization_lambda,
                    'Data_Seed'             : data_seed,
                    'Fold'                  : fold_idx,
                    'ELM_Seed'              : elm_seed,
                    **evaluation_metrics
                }

                fold_results.append(metrics)

            except Exception as e:
                logger.error(f"Failed at Data Seed {data_seed}, Fold {fold_idx}, ELM Seed {elm_seed}: {e}")
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
                            base_metric_name: str = 'avg_F2-Score_Seed',
                            punish_coefficient: float = 1.0,
                            top_k: int = 5) -> pd.DataFrame:
        """
        Ranks results using robust Lower Confidence Bound (LCB) with Standard Error.
        Formula: Mean - (Coefficient * Standard_Error)
        """
        mean_col = f"{base_metric_name}_Mean"
        sem_col = f"{base_metric_name}_SEM"  # 使用我们刚刚加入的真实标准误

        # 如果旧数据没有 SEM 列，回退到 Std（为了兼容你之前的 CSV）
        if sem_col not in dataframe.columns:
            logging.warning(f"Column '{sem_col}' not found. Falling back to naive Standard Deviation.")
            sem_col = f"{base_metric_name}_Std"

        if mean_col not in dataframe.columns or sem_col not in dataframe.columns:
            logging.error(f"Missing Columns for ranking evaluation.")
            return pd.DataFrame()

        if dataframe.empty:
            return dataframe

        # 严谨的 LCB 计算：均值 - 惩罚系数 * 标准误
        adjusted_score = dataframe[mean_col] - (punish_coefficient * dataframe[sem_col].fillna(0))

        return dataframe.assign(rank_score=adjusted_score).nlargest(top_k, columns='rank_score')