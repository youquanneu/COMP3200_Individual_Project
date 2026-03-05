import pandas as pd

from Pipeline.Model.DataSplit import DataSplit
from Pipeline.Model.EvaluationMatrix import EvaluationMatrix
from Pipeline.Model.ExtremeLearningMachine import ExtremeLearningMachine


class EvaluationELM:
    def __init__(self , x, y,activation_function,
                 elm_initial_state_range=range(40, 61),
                 data_split_state_range=range(40, 61),
                 test_size=0.2,
                 k_fold=5):
        self.x = x
        self.y = y
        self.activation_function = activation_function

        self.elm_initial_state_range  = elm_initial_state_range
        self.data_split_state_range = data_split_state_range
        self.test_size = test_size
        self.k_fold = k_fold


    def ranged_seed_cross_validation(self,
                                     hidden_size=None,
                                     regularization_lambda=0.0):

        if hidden_size is None:
            hidden_size = self.x.shape[1]

        global_results_accumulator = []

        for data_seed in self.data_split_state_range:

            splitter = DataSplit(random_state=data_seed, test_size=self.test_size, k_fold=self.k_fold)
            x_test, y_test, folds = splitter.k_fold_data_spiting(self.x, self.y)

            for fold_idx in range(self.k_fold):
                fold = folds[fold_idx]
                x_train , y_train   = fold['X_train_fold']  , fold['y_train_fold']
                x_val   , y_val     = fold['X_val_fold']    , fold['y_val_fold']

                for elm_seed in self.elm_initial_state_range:
                    elm = ExtremeLearningMachine(features_size          = x_train.shape[1],
                                                 hidden_size            = hidden_size,
                                                 activation_function    = self.activation_function,
                                                 regularization_lambda  = regularization_lambda)
                    elm.initialize_random_weights(random_seed = elm_seed)
                    elm.fit(x_train, y_train)

                    y_pred = elm.predict(x_val.values)
                    metrics = EvaluationMatrix(y_val, y_pred).get_all_metrics()

                    metrics['Data_Seed'] = data_seed
                    metrics['Fold']      = fold_idx
                    metrics['ELM_Seed']  = elm_seed
                    global_results_accumulator.append(metrics)

        raw_results_df = pd.DataFrame(global_results_accumulator)

        final_metrics = EvaluationMatrix.random_seed_metrics(raw_results_df)

        raw_results_df.insert(0, 'Hidden_Nodes', hidden_size)
        raw_results_df.insert(1, 'Activation', self.activation_function.__name__)
        raw_results_df.insert(2, 'Lambda_Value', regularization_lambda)

        final_metrics.insert(0, 'Hidden_Nodes', hidden_size)
        final_metrics.insert(1, 'Activation', self.activation_function.__name__)
        final_metrics.insert(2, 'Lambda_Value', regularization_lambda)

        return raw_results_df, final_metrics

    def grid_search_hidden_size(self, hidden_size_range):
        raw_results_list = []
        agg_results_list = []

        for hidden_size in hidden_size_range:
            print(f"Hidden Node Size: {hidden_size}")

            # Stop discarding the raw data
            raw_res, agg_res = self.ranged_seed_cross_validation(
                hidden_size=hidden_size
            )
            raw_results_list.append(raw_res)
            agg_results_list.append(agg_res)

        return pd.concat(raw_results_list, ignore_index=True), pd.concat(agg_results_list, ignore_index=True)

    def grid_search_lambda(self,
                           hidden_size,
                           lambda_range):
        raw_results_list = []
        agg_results_list = []

        for lambda_value in lambda_range:
            print(f"Lambda Value: {lambda_value}")

            raw_res, agg_res = self.ranged_seed_cross_validation(
                hidden_size=hidden_size,
                regularization_lambda=lambda_value
            )

            raw_results_list.append(raw_res)
            agg_results_list.append(agg_res)

        return pd.concat(raw_results_list, ignore_index=True), pd.concat(agg_results_list, ignore_index=True)

    def grid_search_hidden_size_and_lambda(self,
                                           hidden_size_range,
                                           lambda_range):
        raw_results_list = []
        agg_results_list = []

        for hidden_size in hidden_size_range:
            print(f"Hidden Node Size: {hidden_size}")

            # Unpack the two dataframes returned by grid_search_lambda
            raw_res, agg_res = self.grid_search_lambda(
                hidden_size=hidden_size,
                lambda_range=lambda_range
            )
            raw_results_list.append(raw_res)
            agg_results_list.append(agg_res)

        return pd.concat(raw_results_list, ignore_index=True), pd.concat(agg_results_list, ignore_index=True)