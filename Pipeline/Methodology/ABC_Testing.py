import pandas as pd

from Pipeline.Global.GallstoneDataSet import GallstoneDataSet
from Pipeline.Methodology.EvaluationMatrix import EvaluationMatrix
from Pipeline.Global.GlobalSetting import GlobalSetting


def cross_seed_testing(model_class,
                       expr_name    : str,
                       model_types  : str   = 'Grid_Optimization',
                       cv_folds     : int   = 5,
                       is_abc_opt   : bool  = True,
                       mod_cv_fold  : int   = None,
                       data_scaling : bool  = False,
                       force_h_size : int   = None,
                       force_lambda : float = None,
                       employed_bee_algo3   = False,
                       onlooker_bee_algo3   = False):

    gallstone_dataset = GallstoneDataSet()
    gallstone_dataset.fetch_data_path_0()
    gallstone_dataset.cross_validate_test(cv_folds)

    feature_size = gallstone_dataset.x.shape[1]

    model_configs = GlobalSetting.get_model_configs()
    config = next((item for item in model_configs if item.get('Model_Types') == model_types), model_configs[0])
    hidden_size = force_h_size if force_h_size is not None else config.get('Hidden_Nodes',feature_size)
    lambda_value = force_lambda if force_lambda is not None else config.get('Lambda_Value', 0.0)
    activation_func = config['Activation']

    data_split = gallstone_dataset.scaled_fold_split if data_scaling else gallstone_dataset.fold_split

    convergence_result, scout_history, testing_results = [], [], []

    for fold_idx, (x_train, y_train, x_test, y_test) in enumerate(data_split):
        for seed in GlobalSetting.seed_test_range:
            model = generate_model(model_class,
                                   feature_size,
                                   hidden_size,
                                   lambda_value,
                                   is_abc_opt,
                                   activation_func,
                                   seed,
                                   employed_bee_algo3,
                                   onlooker_bee_algo3)

            if mod_cv_fold is None:
                model.fit(x_train, y_train)
            else:
                model.fit(x_train, y_train, cv_folds = mod_cv_fold)

            y_pred  = model.predict(x_test)
            evaluation = EvaluationMatrix(y_test, y_pred)
            metrics = evaluation.get_all_metrics()

            base_metadata = {
                "Model_Type"    : expr_name,
                "Hidden_Nodes"  : hidden_size,
                "Lambda_Value"  : lambda_value,
                "Activation"    : activation_func.__name__,
                "Fold_ID"       : fold_idx,
                "Seed"          : seed
            }
            testing_results.append({**base_metadata, **metrics})

            if is_abc_opt:
                convergence_result.extend([{**base_metadata, "Iteration": i + 1, "Fitness": f} for i, f in
                                           enumerate(model.convergence_curve)])
                scout_history.extend([{**base_metadata, "Iteration": i + 1, "Scout_Triggers": s} for i, s in
                                      enumerate(model.scout_trigger_history)])


    cols_to_keep = ['Model_Type', 'Hidden_Nodes', 'Lambda_Value', 'Activation', 'Fold_ID', 'Seed', 'Accuracy',
                    'Precision', 'Recall', 'NPV', 'Specificity', 'F1-Score', 'F2-Score', 'Bal Accuracy', 'MCC']
    df_testing = pd.DataFrame(testing_results)[cols_to_keep].sort_values(by=['Fold_ID', 'Seed'])

    print()
    GlobalSetting.save_dataframe_to_record(df_testing, f"{expr_name}_Results.csv")

    if is_abc_opt:
        df_convergence = pd.DataFrame(convergence_result)
        df_scout_history = pd.DataFrame(scout_history)
        GlobalSetting.save_dataframe_to_record(df_convergence,f"{expr_name}_Convergence.csv")
        GlobalSetting.save_dataframe_to_record(df_scout_history,f"{expr_name}_Scout_History.csv")

        return df_testing , df_convergence , df_scout_history

    return df_testing


def generate_model(model_class,
                   feature_size,
                   hidden_size,
                   lambda_value,
                   is_abc_opt,
                   activation_func,
                   seed,
                   employed_bee_algo3 = False,
                   onlooker_bee_algo3 = False):
    if is_abc_opt:
        model = model_class(
            feature_size=feature_size,
            hidden_size=hidden_size,
            regularization_lambda=lambda_value,
            activation_function=activation_func,
            fitness_function=GlobalSetting.evaluation_function,
            solution_size=GlobalSetting.solution_size,
            trial_limit=GlobalSetting.trial_limit,
            max_iteration=GlobalSetting.max_iteration
        )
        if employed_bee_algo3:
            model.employed_bee_apply_algo3()
        else:
            model.employed_bee_apply_algo2()

        if onlooker_bee_algo3:
            model.onlooker_bee_apply_algo3()
        else:
            model.onlooker_bee_apply_algo2()

        model.init_random_state(seed)
    else:
        model = model_class(
            features_size=feature_size,
            hidden_size=hidden_size,
            activation_function=activation_func,
            regularization_lambda=lambda_value
        )
        model.initialize_random_weights(random_seed=seed)

    return model