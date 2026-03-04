import pandas as pd

from Pipeline.Model.DataSplit import DataSplit
from Pipeline.Model.EvaluationMatrix import EvaluationMatrix
from Pipeline.Model.ExtremeLearningMachine import ExtremeLearningMachine


class EvaluationELM:
    @staticmethod
    def cross_validation(x, y, activation_function,
                         hidden_size  = None,
                         elm_initial_state = 42,
                         data_random_state = 42,
                         test_size = 0.2,
                         k_fold    = 5,
                         regularization_lambda = 0.0
                         ):

        if hidden_size is None:
            hidden_size = x.shape[1]

        splitter = DataSplit(random_state = data_random_state, test_size = test_size, k_fold = k_fold)
        x_test, y_test, folds = splitter.k_fold_data_spiting(x, y)

        results_list = []
        for i in range(k_fold):
            fold = folds[i]

            x_train , y_train   = fold['X_train_fold'] , fold['y_train_fold']
            x_val   , y_val     = fold['X_val_fold']   , fold['y_val_fold']

            elm = ExtremeLearningMachine(features_size          = x_train.shape[1],
                                         hidden_size            = hidden_size,
                                         activation_function    = activation_function,
                                         regularization_lambda  = regularization_lambda)

            elm.initialize_random_weights(random_seed = elm_initial_state)
            elm.fit(x_train, y_train)

            y_pred  = elm.predict(x_val.values)
            metrics = EvaluationMatrix(y_val, y_pred)
            results_list.append(metrics.get_all_metrics())

        cross_fold_results = pd.DataFrame(results_list)

        return cross_fold_results , EvaluationMatrix.k_fold_metrics(results_list)


    @staticmethod
    def random_seed_cv_validation(x, y, activation_function,
                                  hidden_size = None,
                                  elm_initial_state_range = range(40, 61),
                                  data_random_state = 42,
                                  test_size = 0.2,
                                  k_fold = 5,
                                  regularization_lambda = 0.0
                                  ):
        if hidden_size is None:
            hidden_size = x.shape[1]

        results_list = []
        for i in elm_initial_state_range:
            _ , seed_result = EvaluationELM.cross_validation(
                x,
                y,
                activation_function,
                hidden_size         = hidden_size,
                elm_initial_state   = i,
                data_random_state   = data_random_state,
                test_size           = test_size,
                k_fold              = k_fold,
                regularization_lambda = regularization_lambda
            )
            results_list.append(seed_result.iloc[0].to_dict())

        random_seed_results = pd.DataFrame(results_list)

        return random_seed_results, EvaluationMatrix.random_seed_metrics(random_seed_results,
                                                                    activation_function,
                                                                    hidden_size,
                                                                    regularization_lambda)

    @staticmethod
    def grid_search_hidden_size(x, y, activation_function,
                                hidden_size_range,
                                elm_initial_state_range = range(40, 61),
                                data_random_state = 42,
                                test_size = 0.2,
                                k_fold = 5):
        results_list = []

        for hidden_size in hidden_size_range:
            print(f"Hidden Node Size: {hidden_size}")

            _, hidden_node_result = EvaluationELM.random_seed_cv_validation(
                x,
                y,
                activation_function     = activation_function,
                hidden_size             = hidden_size,
                elm_initial_state_range = elm_initial_state_range,
                data_random_state       = data_random_state,
                test_size               = test_size,
                k_fold                  = k_fold
            )
            results_list.append(hidden_node_result)

        hidden_node_results = pd.concat(results_list, ignore_index=True)

        return hidden_node_results

    @staticmethod
    def grid_search_lambda(x, y, activation_function,
                           hidden_size,
                           lambda_range,
                           elm_initial_state_range = range(40, 61),
                           data_random_state = 42,
                           test_size = 0.2,
                           k_fold = 5):
        results_list = []

        for lambda_value in lambda_range:
            print(f"Lambda Value: {lambda_value}")

            _, lambda_value_result = EvaluationELM.random_seed_cv_validation(
                x,
                y,
                activation_function     = activation_function,
                hidden_size             = hidden_size,
                elm_initial_state_range = elm_initial_state_range,
                data_random_state       = data_random_state,
                test_size               = test_size,
                k_fold                  = k_fold,
                regularization_lambda   = lambda_value
            )

            results_list.append(lambda_value_result)

        lambda_value_results = pd.concat(results_list, ignore_index=True)

        return lambda_value_results

    @staticmethod
    def grid_search_hidden_size_and_lambda(x, y, activation_function,
                                           hidden_size_range,
                                           lambda_range,
                                           elm_initial_state_range = range(40, 61),
                                           data_random_state = 42,
                                           test_size = 0.2,
                                           k_fold = 5):
        results_list = []
        for hidden_size in hidden_size_range:
            print(f"Hidden Node Size: {hidden_size}")
            results_list.append(
                EvaluationELM.grid_search_lambda(
                    x,
                    y,
                    activation_function     = activation_function,
                    hidden_size             = hidden_size,
                    lambda_range            = lambda_range,
                    elm_initial_state_range = elm_initial_state_range,
                    data_random_state       = data_random_state,
                    test_size               = test_size,
                    k_fold                  = k_fold
                )
            )

        final_grids_df = pd.concat(results_list, ignore_index=True)

        return final_grids_df