import time

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from Pipeline.Algorithm.ArtificialBeeColonyElm import ArtificialBeeColonyElm
from Pipeline.Algorithm.ArtificialBeeColonyElmCV import ArtificialBeeColonyElmCV
from Pipeline.Algorithm.ArtificialBeeColonyElmCVEnsemble import ArtificialBeeColonyElmCVEnsemble
from Pipeline.Global.GallstoneDataSet import GallstoneDataSet
from Pipeline.Methodology.EvaluationMatrix import EvaluationMatrix
from Pipeline.Global.GlobalSetting import GlobalSetting

def abc_testing(abc_model,x_train, y_train , x_test , y_test):

    monte_carlo_results = []
    convergence_history = {}
    scout_history = {}

    for seed in GlobalSetting.elm_initial_state_range:
        print(f"\nRunning simulation for Seed: {seed}...", flush=True)

        abc_model_tested = abc_model
        abc_model_tested.init_random_state(seed)
        abc_model_tested.fit(x_train, y_train)

        convergence_history[seed] = abc_model_tested.convergence_curve
        scout_history[seed] = abc_model_tested.scout_trigger_history

        y_pred = abc_model_tested.predict(x_test)

        eval_metrics = EvaluationMatrix(y_true= y_test, y_pred=y_pred).get_all_metrics()
        eval_metrics['ABC_Seed'] = seed

        monte_carlo_results.append(eval_metrics)

    return pd.DataFrame(monte_carlo_results), pd.DataFrame(convergence_history), pd.DataFrame(scout_history)


def cv_fold_testing(model_name, config, fold_id):
    print(f"--- Initializing {model_name} on Fold {fold_id} ---")

    gallstone_dataset = GallstoneDataSet()
    gallstone_dataset.fetch_data_path_1()
    gallstone_dataset.cross_validate_test(5)

    x_train, y_train, x_test, y_test = gallstone_dataset.fold_split[fold_id]

    if model_name == "RELM":
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)


    features_size = x_train.shape[1]

    print(f"[Data] Fold {fold_id} loaded. Training samples: {len(x_train)}, Testing samples: {len(x_test)}")

    best_lambda_config = GlobalSetting.get_config_by_type(config)
    best_lambda_hidden_size = best_lambda_config["Hidden_Nodes"] if best_lambda_config else None
    best_lambda_lambda_value = best_lambda_config["Lambda_Value"] if best_lambda_config else None

    # Map the command-line argument to the actual Class
    model_registry = {
        "RELM"              : ArtificialBeeColonyElm,
        "RELM_CV"           : ArtificialBeeColonyElmCV,
        "RELM_CV_Ensemble"  : ArtificialBeeColonyElmCVEnsemble
    }

    if model_name not in model_registry:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_registry.keys())}")

    selected_model_class = model_registry[model_name]

    # Instantiate the ABC Swarm with the injected configurations
    abc_model = selected_model_class(
        features_size       = features_size,
        hidden_size         = best_lambda_hidden_size,
        activation_function = GlobalSetting.sigmoid,
        regularization_lambda=best_lambda_lambda_value,
        fitness_function    = GlobalSetting.evaluation_function,
        solution_size       = GlobalSetting.solution_size,
        trial_limit         = GlobalSetting.trial_limit,
        max_iteration       = GlobalSetting.max_iteration
    )

    abc_model.employed_bee_apply_algo3()
    abc_model.onlooker_bee_apply_algo2()

    start_time = time.time()

    results_df, convergence_df, scout_df = abc_testing(
        abc_model,
        x_train, y_train,
        x_test, y_test
    )

    duration = time.time() - start_time
    print(f"\n[Execution] Fold {fold_id} completed in {duration:.2f} seconds.")

    GlobalSetting.save_dataframe_to_record(results_df    , f"{config}_ABC_{model_name}_Fold_{fold_id}_Results.csv")
    GlobalSetting.save_dataframe_to_record(convergence_df, f"{config}_ABC_{model_name}_Fold_{fold_id}_Convergence.csv")
    GlobalSetting.save_dataframe_to_record(scout_df      , f"{config}_ABC_{model_name}_Fold_{fold_id}_Scout_History.csv")

    return results_df, convergence_df , scout_df