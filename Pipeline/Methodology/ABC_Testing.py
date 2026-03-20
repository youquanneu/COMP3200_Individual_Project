import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from Pipeline.Algorithm.ArtificialBeeColonyElm import ArtificialBeeColonyElm
from Pipeline.Algorithm.ArtificialBeeColonyElmCV import ArtificialBeeColonyElmCV
from Pipeline.Algorithm.ArtificialBeeColonyElmCVEnsemble import ArtificialBeeColonyElmCVEnsemble
from Pipeline.Global.GallstoneDataSet import GallstoneDataSet
from Pipeline.Global.Plotting import Plotting
from Pipeline.Methodology.EvaluationMatrix import EvaluationMatrix
from Pipeline.Global.GlobalSetting import GlobalSetting

def abc_testing(abc_model,x_train, y_train , x_test , y_test):

    monte_carlo_results = []
    convergence_history = {}
    scout_history = {}

    for seed in GlobalSetting.seed_test_range:

        abc_model_tested = abc_model
        abc_model_tested.init_random_state(seed)
        abc_model_tested.fit(x_train, y_train)

        convergence_history[seed] = abc_model_tested.convergence_curve

        scout_history[seed] = abc_model_tested.scout_trigger_history

        y_pred = abc_model_tested.predict(x_test)

        eval_metrics = EvaluationMatrix(y_true= y_test, y_pred=y_pred).get_all_metrics()
        eval_metrics['Seed'] = seed

        eval_metrics['Hidden_Nodes'] = int(abc_model_tested.hidden_size)
        eval_metrics['Lambda_Value'] = float(abc_model_tested.regularization_lambda)
        eval_metrics['Activation']   = 'sigmoid'

        eval_metrics['Solution_Size']   = abc_model_tested.solution_size
        eval_metrics['Trial_Limit']     = abc_model_tested.trial_limit
        eval_metrics['Max_Iter']        = abc_model_tested.max_iteration

        monte_carlo_results.append(eval_metrics)

    return pd.DataFrame(monte_carlo_results), pd.DataFrame(convergence_history), pd.DataFrame(scout_history)


def cv_fold_testing(model_name, config, fold_id):

    gallstone_dataset = GallstoneDataSet()
    gallstone_dataset.fetch_data_path_0()
    gallstone_dataset.cross_validate_test(5)

    x_train, y_train, x_test, y_test = gallstone_dataset.fold_split[fold_id]

    if model_name == "RELM":
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)


    features_size = x_train.shape[1]

    best_lambda_config = GlobalSetting.get_config_by_type(config)
    best_lambda_hidden_size = best_lambda_config["Hidden_Nodes"] if best_lambda_config else None
    best_lambda_lambda_value = best_lambda_config["Lambda_Value"] if best_lambda_config else None

    model_registry = {
        "RELM"              : ArtificialBeeColonyElm,
        "RELM_CV"           : ArtificialBeeColonyElmCV,
        "RELM_CV_Ensemble"  : ArtificialBeeColonyElmCVEnsemble
    }

    if model_name not in model_registry:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_registry.keys())}")

    selected_model_class = model_registry[model_name]

    abc_model = selected_model_class(
        feature_size       = features_size,
        hidden_size         = best_lambda_hidden_size ,
        activation_function = GlobalSetting.sigmoid,
        regularization_lambda= best_lambda_lambda_value,
        fitness_function    = GlobalSetting.evaluation_function,
        solution_size       = GlobalSetting.solution_size,
        trial_limit         = GlobalSetting.trial_limit,
        max_iteration       = GlobalSetting.max_iteration
    )

    abc_model.employed_bee_apply_algo3()
    abc_model.init_algo3(initial_probability=0.1,final_probability=0.9)
    abc_model.onlooker_bee_apply_algo2()

    results_df, convergence_df, scout_df = abc_testing(
        abc_model,
        x_train, y_train,
        x_test, y_test
    )

    results_df['Fold_ID'] = fold_id

    cols_to_keep = [
        'Fold_ID', 'Accuracy', 'Precision', 'Recall', 'NPV', 'Specificity',
        'F1-Score', 'F2-Score', 'Bal Accuracy', 'MCC', 'Seed'
    ]

    results_df = results_df[cols_to_keep]

    GlobalSetting.save_dataframe_to_record(results_df    , f"ABC_{model_name}_{config}_Fold_{fold_id}_Results.csv")
    GlobalSetting.save_dataframe_to_record(convergence_df, f"ABC_{model_name}_{config}_Fold_{fold_id}_Convergence.csv")
    GlobalSetting.save_dataframe_to_record(scout_df      , f"ABC_{model_name}_{config}_Fold_{fold_id}_Scout_History.csv")

    Plotting.plot_abc_dashboard(convergence_df, scout_df, f"ABC_{model_name}_{config}_Fold_{fold_id}", results_df=results_df,
                                is_final_record=True)

    return results_df, convergence_df , scout_df



def cross_seed_testing(model_class,
                       model_name   : str,
                       model_types  : str   = 'Grid_Optimization',
                       cv_folds     : int   = 5,
                       is_abc_opt   : bool  = True,
                       abc_cv_fold  : int   = None,
                       data_scaling : bool  = False,
                       force_h_size : int   = None,
                       force_lambda : float = None):

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
            if is_abc_opt:
                model = model_class(
                    feature_size        = feature_size,
                    hidden_size         = hidden_size,
                    regularization_lambda= lambda_value,
                    activation_function = activation_func,
                    fitness_function    = GlobalSetting.evaluation_function,
                    solution_size       = GlobalSetting.solution_size,
                    trial_limit         = GlobalSetting.trial_limit,
                    max_iteration       = GlobalSetting.max_iteration
                )
                model.employed_bee_apply_algo3()
                model.onlooker_bee_apply_algo2()
                model.init_random_state(seed)
            else:
                model = model_class(
                    features_size           = feature_size,
                    hidden_size             = hidden_size,
                    activation_function     = activation_func,
                    regularization_lambda   = lambda_value
                )
                model.initialize_random_weights(random_seed=seed)

            if abc_cv_fold is None:
                model.fit(x_train, y_train)
            else:
                model.fit(x_train, y_train, cv_folds = abc_cv_fold)

            y_pred  = model.predict(x_test)
            evaluation = EvaluationMatrix(y_test, y_pred)
            metrics = evaluation.get_all_metrics()

            base_metadata = {
                "Model_Type"    : model_name,
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
    GlobalSetting.save_dataframe_to_record(df_testing, f"{model_name}_Results.csv")

    if is_abc_opt:
        df_convergence = pd.DataFrame(convergence_result)
        df_scout_history = pd.DataFrame(scout_history)
        GlobalSetting.save_dataframe_to_record(df_convergence,f"{model_name}_Convergence.csv")
        GlobalSetting.save_dataframe_to_record(df_scout_history,f"{model_name}_Scout_History.csv")

        return df_testing , df_convergence , df_scout_history

    return df_testing