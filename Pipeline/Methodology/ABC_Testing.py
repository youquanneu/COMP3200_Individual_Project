import math
import time
from pathlib import Path
from typing import List

import pandas as pd

from Pipeline.Global.GallstoneDataSet import GallstoneDataSet
from Pipeline.Methodology.EvaluationMatrix import EvaluationMatrix
from Pipeline.Global.GlobalSetting import GlobalSetting


def cross_seed_testing(model_class,
                       expr_name    : str,
                       use_raw_data : bool  = False,
                       model_types  : str   = 'Grid_Optimization',
                       cv_folds     : int   = 5,
                       is_abc_opt   : bool  = False,
                       mod_cv_fold  : int   = None,
                       data_scaling : bool  = False,
                       force_h_size : int   = None,
                       force_lambda : float = None,
                       force_sn     : int   = None,
                       force_tl     : int   = None,
                       force_mi     : int   = None,
                       employed_bee_algo3   = False,
                       onlooker_bee_algo3   = False):

    prefix = "raw_" if use_raw_data else "cleaned_"
    expr_name = f"{prefix}{expr_name}"

    gallstone_dataset = GallstoneDataSet()

    if use_raw_data :
        gallstone_dataset.fetch_raw_data_path()
    else:
        gallstone_dataset.fetch_cleaned_data_path()

    gallstone_dataset.cv_test_split(cv_folds)

    feature_size = gallstone_dataset.x.shape[1]

    model_configs = GlobalSetting.get_model_configs()
    config = next((item for item in model_configs if item.get('Model_Types') == model_types), model_configs[0])

    hidden_size  = force_h_size if force_h_size is not None else config.get('Hidden_Nodes', feature_size)
    lambda_value = force_lambda if force_lambda is not None else config.get('Lambda_Value', 0.0)
    activation_func = config['Activation']

    if force_tl is None and force_sn is not None:
        force_tl = force_sn // 2

    outer_test_list = gallstone_dataset.test_scaled_fold_split if data_scaling else gallstone_dataset.test_fold_split

    convergence_result, scout_history, testing_results = [], [], []

    for fold_idx, (x_train, y_train, x_test, y_test) in enumerate(outer_test_list):

        print(f"\nTesting - Fold {fold_idx}")
        fold_start_time = time.time()

        for seed in GlobalSetting.seed_test_range:
            model = generate_model(model_class,
                                   feature_size,
                                   hidden_size,
                                   lambda_value,
                                   is_abc_opt,
                                   activation_func,
                                   seed,
                                   force_sn,
                                   force_tl,
                                   force_mi,
                                   employed_bee_algo3,
                                   onlooker_bee_algo3)

            if mod_cv_fold is None:
                model.fit(x_train, y_train)
            else:
                model.fit(x_train, y_train, cv_folds = mod_cv_fold )

            y_pred  = model.predict(x_test)
            evaluation = EvaluationMatrix(y_test, y_pred)
            metrics = evaluation.get_all_metrics()

            if is_abc_opt:
                solution_size   = model.solution_size
                trial_limit     = model.trial_limit
                max_iteration   = model.max_iteration
            else :
                solution_size   = None
                trial_limit     = None
                max_iteration   = None

            base_metadata = {
                "Model_Type"    : expr_name,
                "Is_ABC_Opt"    : is_abc_opt,
                "Data_Scaled"   : data_scaling,
                "Hidden_Nodes"  : hidden_size,
                "Lambda_Value"  : lambda_value,
                "Activation"    : activation_func.__name__,
                "Solution_Size" : solution_size,
                "Trial_Limit"   : trial_limit,
                "Max_Iteration" : max_iteration,
                "Emp_Algo3"     : employed_bee_algo3,
                "Onl_Algo3"     : onlooker_bee_algo3,
                "Fold_ID"       : fold_idx,
                "Seed"          : seed
            }

            testing_results.append({**base_metadata, **metrics})

            if is_abc_opt:
                convergence_result.extend([{**base_metadata, "Iteration": i + 1, "Fitness": f} for i, f in
                                           enumerate(model.convergence_curve)])
                scout_history.extend([{**base_metadata, "Iteration": i + 1, "Scout_Triggers": s} for i, s in
                                      enumerate(model.scout_trigger_history)])

        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        print(f"\n\nTesting - Fold {fold_idx} Completed in {fold_duration:.4f}")

    df_testing_raw = pd.DataFrame(testing_results)

    expected_cols = [
        'Model_Type', 'Is_ABC_Opt', 'Data_Scaled', 'Hidden_Nodes', 'Lambda_Value', 'Activation',
        'Solution_Size', 'Trial_Limit', 'Max_Iteration', 'Emp_Algo3', 'Onl_Algo3', 'Fold_ID', 'Seed',
        'Accuracy', 'Precision', 'Recall', 'NPV', 'Specificity', 'F1-Score', 'F2-Score', 'Bal Accuracy', 'MCC'
    ]

    actual_cols = [col for col in expected_cols if col in df_testing_raw.columns]

    df_testing = df_testing_raw[actual_cols].sort_values(by=['Fold_ID', 'Seed'])

    print()
    GlobalSetting.save_dataframe_to_record(df_testing, f"Test History/{expr_name}_Results.csv")

    if is_abc_opt:
        df_convergence = pd.DataFrame(convergence_result)
        df_scout_history = pd.DataFrame(scout_history)
        GlobalSetting.save_dataframe_to_record(df_convergence,f"Test Convergence History/{expr_name}_Convergence.csv")
        GlobalSetting.save_dataframe_to_record(df_scout_history,f"Test Scout History/{expr_name}_Scout_History.csv")

        return df_testing , df_convergence , df_scout_history

    return df_testing


def generate_model(model_class,
                   feature_size,
                   hidden_size,
                   lambda_value,
                   is_abc_opt,
                   activation_func,
                   seed,
                   force_sn,
                   force_tl,
                   force_mi,
                   employed_bee_algo3 = False,
                   onlooker_bee_algo3 = False):
    if is_abc_opt:
        model = model_class(
            feature_size            = feature_size,
            hidden_size             = hidden_size,
            regularization_lambda   = lambda_value,
            activation_function     = activation_func,
            fitness_function        = GlobalSetting.evaluation_function,
            solution_size           = force_sn if force_sn is not None else GlobalSetting.solution_size,
            trial_limit             = force_tl if force_tl is not None else GlobalSetting.trial_limit,
            max_iteration           = force_mi if force_mi is not None else GlobalSetting.max_iteration
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
            feature_size            = feature_size,
            hidden_size             = hidden_size,
            activation_function     = activation_func,
            regularization_lambda   = lambda_value
        )
        model.initialize_random_weights( random_seed = seed )

    return model


def evaluate_abc_parameters(model_class,
                            expr_name       : str,
                            solution_size   : int,
                            trial_limit     : int,
                            max_iteration   : int,
                            use_raw_data    : bool  = False,
                            cv_folds        : int   = 5,
                            employed_bee_algo3      = False,
                            onlooker_bee_algo3      = False) -> pd.DataFrame:

    prefix = "raw_" if use_raw_data else "cleaned_"
    expr_name = f"{prefix}{expr_name}"

    gallstone_dataset = GallstoneDataSet()
    if use_raw_data:
        gallstone_dataset.fetch_raw_data_path()
    else:
        gallstone_dataset.fetch_cleaned_data_path()

    gallstone_dataset.cv_test_split(cv_folds)
    inner_val_list = gallstone_dataset.val_fold_split

    history_records = []
    trace_metric = GlobalSetting.evaluation_function

    for fold_idx,(x_tr, y_tr, x_val, y_val) in enumerate(inner_val_list):

        print(f"\nTracing - Fold {fold_idx}")
        fold_start_time = time.time()

        for random_seed in GlobalSetting.elm_initial_state_range:

            abc_model = model_class(
                feature_size            = x_tr.shape[1],
                hidden_size             = GlobalSetting.abc_trace_h_size,
                activation_function     = GlobalSetting.sigmoid,
                regularization_lambda   = GlobalSetting.abc_trace_lambda,
                fitness_function        = trace_metric,
                random_state            = random_seed,
                solution_size           = solution_size,
                trial_limit             = trial_limit,
                max_iteration           = max_iteration
            )
            if employed_bee_algo3:
                abc_model.employed_bee_apply_algo3()
            else:
                abc_model.employed_bee_apply_algo2()

            if onlooker_bee_algo3:
                abc_model.onlooker_bee_apply_algo3()
            else:
                abc_model.onlooker_bee_apply_algo2()

            abc_model.apply_validation_dataset(x_val, y_val)
            abc_model.fit(x_tr, y_tr)

            train_curve = abc_model.convergence_curve
            val_curve   = abc_model.val_fitness_curve
            scout_curve = abc_model.scout_trigger_history

            for iter_idx in range(abc_model.max_iteration):
                history_records.append({
                    # Model Configs
                    "Solution_Size" : solution_size,
                    "Trial_Limit"   : trial_limit,
                    "Max_Iteration" : max_iteration,
                    "Employed_Algo3": employed_bee_algo3,
                    "Onlooker_Algo3": onlooker_bee_algo3,
                    # Trace Configs
                    "Fold_ID"       : fold_idx,
                    "Seed"          : random_seed,
                    "Iteration"     : iter_idx + 1,
                    # Trace Target
                    "Train_Fitness" : train_curve[iter_idx],
                    "Val_Fitness"   : val_curve[iter_idx] if len(val_curve) > iter_idx else None,
                    "Trace_Metric"  : trace_metric,
                    "Scout_Triggers": scout_curve[iter_idx]
                })

        fold_end_time = time.time()
        fold_duration = fold_end_time - fold_start_time
        print(f"\n\nTracing - Fold {fold_idx} Completed in {fold_duration:.4f}")

    train_val_records = pd.DataFrame(history_records)

    GlobalSetting.save_dataframe_to_record(train_val_records, f"Trace History/{expr_name}_Trace.csv")

    return train_val_records

def lcb_trace_evaluation(folder_path: str,
                         punish_coefficient: float = None) -> tuple[list[pd.DataFrame],pd.DataFrame]:

    trace_records: List[pd.DataFrame] = []
    trace_final_results : List[pd.DataFrame] = []

    trace_dir = Path(folder_path)

    if punish_coefficient is None:
        punish_coefficient = GlobalSetting.seed_punish_coe

    for file_path in trace_dir.glob("*.csv"):
        df_trace = pd.read_csv(file_path)
        if df_trace.empty:
            continue

        expr_name = file_path.stem
        metric_name = df_trace['Trace_Metric'].iloc[0] \
            if 'Trace_Metric' in df_trace.columns else GlobalSetting.evaluation_function
        sn = df_trace['Solution_Size'].iloc[0]  if 'Solution_Size'  in df_trace.columns else None
        tl = df_trace['Trial_Limit'].iloc[0]    if 'Trial_Limit'    in df_trace.columns else None
        mi = df_trace['Max_Iteration'].iloc[0]  if 'Max_Iteration'  in df_trace.columns else None

        df_seed_agg = df_trace.groupby(['Iteration', 'Seed']).agg(
            Train_Fit_Mean_by_Fold  = ('Train_Fitness', 'mean'),
            Train_Fit_Std_by_Fold   = ('Train_Fitness', 'std'),
            Val_Fit_Mean_by_Fold    = ('Val_Fitness', 'mean'),
            Val_Fit_Std_by_Fold     = ('Val_Fitness', 'std'),
            Scout_Triggers          = ('Scout_Triggers', 'mean')
        ).fillna(0)

        df_seed_agg['Train_LCB'] = df_seed_agg['Train_Fit_Mean_by_Fold'] - df_seed_agg['Train_Fit_Std_by_Fold']
        df_seed_agg['Val_LCB']   = df_seed_agg['Val_Fit_Mean_by_Fold']   - df_seed_agg['Val_Fit_Std_by_Fold']
        def aggregate_seeds(group):
            n_seeds = len(group)

            t_mean = group['Train_LCB'].mean()
            t_std = group['Train_LCB'].std()
            t_sem = t_std / math.sqrt(n_seeds) if n_seeds > 1 else 0.0

            v_mean = group['Val_LCB'].mean()
            v_std = group['Val_LCB'].std()
            v_sem = v_std / math.sqrt(n_seeds) if n_seeds > 1 else 0.0

            scout_mean = group['Scout_Triggers'].mean()
            scout_std = group['Scout_Triggers'].std()

            return pd.Series({
                f'train_{metric_name}_LCB_Mean': t_mean,
                f'train_{metric_name}_LCB_std': t_std,
                f'train_{metric_name}_LCB_sem': t_sem,
                f'val_{metric_name}_LCB_Mean': v_mean,
                f'val_{metric_name}_LCB_std': v_std,
                f'val_{metric_name}_LCB_sem': v_sem,
                'scout_avg': scout_mean,
                'scout_std': scout_std
            })

        df_iter_results = df_seed_agg.groupby('Iteration').apply(aggregate_seeds).reset_index()

        df_iter_results[f'train_{metric_name}_trace_floor'] = (
                df_iter_results[f'train_{metric_name}_LCB_Mean'] - (
                    punish_coefficient * df_iter_results[f'train_{metric_name}_LCB_sem'])
        )
        df_iter_results[f'val_{metric_name}_trace_floor'] = (
                df_iter_results[f'val_{metric_name}_LCB_Mean'] - (
                    punish_coefficient * df_iter_results[f'val_{metric_name}_LCB_sem'])
        )
        metadata_df = pd.DataFrame({
            'Solution_Size' : sn,
            'Trial_Limit'   : tl,
            'Max_Iteration' : mi,
            'expr_name'     : expr_name,
            'metric_name'   : metric_name
        }, index=df_iter_results.index)
        df_iter_results = pd.concat([metadata_df, df_iter_results], axis=1)

        last_iter_result = df_iter_results.loc[[df_iter_results['Iteration'].idxmax()]]
        last_iter_result = last_iter_result.drop(columns=['Iteration'], errors='ignore')

        trace_records.append(df_iter_results)
        trace_final_results.append(last_iter_result)

    summary_df = pd.concat(trace_final_results, ignore_index=True) if trace_final_results else pd.DataFrame()

    return trace_records, summary_df