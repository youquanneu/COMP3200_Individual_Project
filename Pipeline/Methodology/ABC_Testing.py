import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

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

    params = (model_class, feature_size, hidden_size, lambda_value, is_abc_opt,
              activation_func, mod_cv_fold, force_sn, force_tl, force_mi,
              employed_bee_algo3, onlooker_bee_algo3, expr_name, data_scaling)

    parallel_outputs = Parallel(n_jobs=min(cv_folds, 8))(
        delayed(_fold_worker)(idx, fold_data, params)
        for idx, fold_data in enumerate(outer_test_list)
    )

    testing_results, convergence_result, scout_history = [], [], []
    for f_res, f_conv, f_scout in parallel_outputs:
        testing_results.extend(f_res)
        convergence_result.extend(f_conv)
        scout_history.extend(f_scout)

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

def _fold_worker(fold_idx, fold_data, params):

    x_train, y_train, x_test, y_test = fold_data

    (model_class, feature_size, hidden_size, lambda_value, is_abc_opt,
     activation_func, mod_cv_fold, force_sn, force_tl, force_mi,
     employed_bee_algo3, onlooker_bee_algo3, expr_name, data_scaling) = params

    convergence_result, scout_history, testing_results = [], [], []

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

        if hasattr(model, 'n_jobs'):
            model.n_jobs = 1

        if mod_cv_fold is None:
            model.fit(x_train, y_train)
        else:
            model.fit(x_train, y_train, cv_folds=mod_cv_fold)

        y_pred = model.predict(x_test)
        evaluation  = EvaluationMatrix(y_test, y_pred)
        metrics     = evaluation.get_all_metrics()

        solution_size   = getattr(model, 'solution_size', None) if is_abc_opt else None
        trial_limit     = getattr(model, 'trial_limit', None)   if is_abc_opt else None
        max_iteration   = getattr(model, 'max_iteration', None) if is_abc_opt else None

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

    return testing_results, convergence_result, scout_history

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
                            force_h_size : int   = None,
                            force_lambda : float = None,
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

    trace_metric = GlobalSetting.evaluation_function
    hidden_size  = force_h_size if force_h_size is not None else GlobalSetting.abc_trace_h_size
    lambda_value = force_lambda if force_lambda is not None else GlobalSetting.abc_trace_lambda

    params = (model_class, solution_size, trial_limit, max_iteration,
              employed_bee_algo3, onlooker_bee_algo3, trace_metric,
              hidden_size, lambda_value)

    parallel_outputs = Parallel(n_jobs=min(cv_folds, 8))(
        delayed(_trace_worker)(idx, fold_data, params)
        for idx, fold_data in enumerate(inner_val_list)
    )
    history_records = []
    for fold_records in parallel_outputs:
        history_records.extend(fold_records)

    train_val_records = pd.DataFrame(history_records)

    GlobalSetting.save_dataframe_to_record(train_val_records, f"Trace History/{expr_name}_Trace.csv")

    return train_val_records

def _trace_worker(fold_idx, fold_data, params):
    x_tr, y_tr, x_val, y_val = fold_data

    (model_class, solution_size, trial_limit, max_iteration,
     employed_bee_algo3, onlooker_bee_algo3, trace_metric,
     hidden_size, lambda_value) = params

    fold_history = []

    for random_seed in GlobalSetting.elm_initial_state_range:
        abc_model = model_class(
            feature_size        = x_tr.shape[1],
            hidden_size         = hidden_size,
            activation_function = GlobalSetting.sigmoid,
            regularization_lambda= lambda_value,
            fitness_function    = trace_metric,
            random_state        = random_seed,
            solution_size       = solution_size,
            trial_limit         = trial_limit,
            max_iteration       = max_iteration
        )
        if hasattr(abc_model, 'n_jobs'):
            abc_model.n_jobs = 1

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
        val_curve = abc_model.val_fitness_curve
        scout_curve = abc_model.scout_trigger_history

        for iter_idx in range(abc_model.max_iteration):
            fold_history.append({
                # Model Configs
                "Solution_Size": solution_size,
                "Trial_Limit": trial_limit,
                "Max_Iteration": max_iteration,
                "Employed_Algo3": employed_bee_algo3,
                "Onlooker_Algo3": onlooker_bee_algo3,
                # Trace Configs
                "Fold_ID": fold_idx,
                "Seed": random_seed,
                "Iteration": iter_idx + 1,
                # Trace Target
                "Train_Fitness": train_curve[iter_idx],
                "Val_Fitness": val_curve[iter_idx] if len(val_curve) > iter_idx else None,
                "Trace_Metric": trace_metric,
                "Scout_Triggers": scout_curve[iter_idx]
            })

    return fold_history
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
        eb3 = df_trace['Employed_Algo3'].iloc[0] if 'Employed_Algo3'  in df_trace.columns else None
        ob3 = df_trace['Onlooker_Algo3'].iloc[0] if 'Onlooker_Algo3'  in df_trace.columns else None

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
            'Employed_Algo3': eb3,
            'Onlooker_Algo3': ob3,
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

def get_result_stats(model_dict,
                     target_order,
                     main_model_name   = "ABC RELM CV",
                     metric_name       = 'MCC',
                     show_macro        = True):

    if main_model_name not in target_order:
        raise ValueError(f"Model '{main_model_name}' not found in target_order.")
    k_comparisons = len(target_order) - 1

    dfs_seed, dfs_fold = [], []
    p_vals_s, p_vals_f = {}, {}
    for model_name in target_order:
        if model_name in model_dict:
            df = pd.read_csv(model_dict[model_name])
            seed_lcb = df.groupby('Seed')[metric_name].agg(['mean', 'std'])
            seed_lcb['lcb'] = seed_lcb['mean'] - (GlobalSetting.cv_punish_coe * seed_lcb['std'])
            dfs_seed.append(seed_lcb[['lcb']].reset_index()
                            .rename(columns={'lcb': metric_name})
                            .assign(Model=model_name))

            dfs_fold.append(df.groupby('Fold_ID')[metric_name].mean().reset_index().assign(Model=model_name))

    df_s_all = None

    if show_macro:
        df_s_all = pd.concat(dfs_seed, ignore_index=True)
        df_s_pivot = df_s_all.pivot_table(index='Seed', columns='Model', values=metric_name, aggfunc='mean')[
            target_order]

        for m in target_order:
            if m != main_model_name:
                diff = df_s_pivot[main_model_name] - df_s_pivot[m]
                diff_mean = diff.mean()
                if np.all(diff == 0):
                    p_vals_s[m] = (1.0, diff_mean)
                else:
                    try:
                        _, p_raw = stats.wilcoxon(diff)
                        p_vals_s[m] = (min(1.0, p_raw * k_comparisons), diff_mean)
                    except ValueError:
                        p_vals_s[m] = (1.0, diff_mean)

    df_f_all = pd.concat(dfs_fold, ignore_index=True)
    df_f_pivot = df_f_all.pivot_table(index='Fold_ID', columns='Model', values=metric_name, aggfunc='mean')[
        target_order]


    for m in target_order:
        if m != main_model_name:
            diff = df_f_pivot[main_model_name] - df_f_pivot[m]
            diff_mean = diff.mean()
            if np.all(diff == 0):
                p_vals_f[m] = (1.0, diff_mean)
            else:
                _, p_raw = stats.ttest_rel(df_f_pivot[main_model_name], df_f_pivot[m])
                if np.isnan(p_raw):
                    p_raw = 0.0 if diff_mean != 0 else 1.0

                p_vals_f[m] = (min(1.0, p_raw * k_comparisons), diff_mean)

    return (df_s_all, p_vals_s), (df_f_all, df_f_pivot, p_vals_f)

def get_test_result_summaries(model_dict,
                              p_vals_s= None, p_vals_f= None,
                              metric_name= 'MCC',
                              alpha=0.05):

    stable_list, general_list = [], []

    unused_column = [
        'Hidden_Nodes', 'Lambda_Value',
        'Solution_Size', 'Trial_Limit', 'Max_Iteration',
        'Precision', 'Recall', 'NPV', 'Specificity',
        'F2-Score', 'Bal Accuracy'
    ]

    for model_name, path in model_dict.items():
        df = pd.read_csv(path)
        df = df.drop(columns=unused_column, errors='ignore')

        numeric_df = df.select_dtypes(include=['number'])
        calc_cols = [c for c in numeric_df.columns if c not in ['Seed', 'Fold_ID']]

        grouped_s = numeric_df.groupby('Seed')
        seed_lcb = grouped_s[calc_cols].mean() - (GlobalSetting.cv_punish_coe * grouped_s[calc_cols].std())

        s_summary = seed_lcb.agg(['mean', 'std']).unstack()
        s_flat = pd.DataFrame([s_summary.values],
                              columns=[f"{c}_{s}" for c, s in s_summary.index])
        s_flat.insert(0, 'model_name', model_name)
        stable_list.append(s_flat)

        grouped_f = numeric_df.groupby('Fold_ID')
        fold_avg = grouped_f[calc_cols].mean()

        f_summary = fold_avg.agg(['mean', 'std']).unstack()
        f_flat = pd.DataFrame([f_summary.values],
                              columns=[f"{c}_{s}" for c, s in f_summary.index])
        f_flat.insert(0, 'model_name', model_name)
        general_list.append(f_flat)

    df_stable = pd.concat(stable_list, ignore_index=True) if stable_list else pd.DataFrame()
    df_general = pd.concat(general_list, ignore_index=True) if general_list else pd.DataFrame()

    def get_numeric_direction(p_val, diff):
        if pd.isna(p_val) or pd.isna(diff): return None
        if p_val > alpha: return 0
        return 1 if diff > 0 else -1
    def apply_stats_to_df(target_df, p_vals_dict, test_prefix):
        if p_vals_dict is not None and not target_df.empty:
            target_df[f'{test_prefix}_p_{metric_name}'] = target_df['model_name'].map(
                lambda m: p_vals_dict[m][0] if m in p_vals_dict else None
            )
            target_df[f'{test_prefix}_dir_{metric_name}'] = target_df['model_name'].map(
                lambda m: get_numeric_direction(p_vals_dict[m][0], p_vals_dict[m][1]) if m in p_vals_dict else None
            )

    apply_stats_to_df(df_stable, p_vals_s, "Wilcoxon")
    apply_stats_to_df(df_general, p_vals_f, "T_test")

    return df_stable, df_general

def format_summaries_for_academic_report(df: pd.DataFrame, decimal_places: int = 4):
    report_df = df.copy()
    mean_cols = [c for c in report_df.columns if c.endswith('_mean')]

    for m_col in mean_cols:
        base_name = m_col.replace('_mean', '')
        s_col = f"{base_name}_std"

        if s_col in report_df.columns:
            loc = report_df.columns.get_loc(m_col)

            formatted_series = report_df.apply(
                lambda row: f"{row[m_col]:.{decimal_places}f} ± {row[s_col]:.{decimal_places}f}"
                if pd.notna(row[m_col]) and pd.notna(row[s_col]) else "N/A",
                axis=1
            )

            report_df = report_df.drop(columns=[m_col, s_col])
            report_df.insert(loc, base_name, formatted_series)

    dir_cols = [c for c in report_df.columns if '_dir_' in c]
    sig_map = {1.0: "+1", -1.0: "-1", 0.0: "0"}

    for d_col in dir_cols:
        report_df[d_col] = report_df[d_col].map(sig_map).fillna("N/A")

    return report_df

def overall_result_summaries(model_dict, unused_column):
    final_list = []

    for model_name, path in model_dict.items():
        df = pd.read_csv(path)
        df = df.drop(columns=unused_column, errors='ignore')

        numeric_df = df.select_dtypes(include=['number'])
        calc_cols = [c for c in numeric_df.columns if c not in ['Seed', 'Fold_ID']]

        fold_level_performance = numeric_df.groupby('Fold_ID')[calc_cols].mean()

        res_mean = fold_level_performance.mean().add_suffix('_mean')
        res_std = fold_level_performance.std().add_suffix('_std')

        row_df = pd.DataFrame([pd.concat([res_mean, res_std])])
        row_df.insert(0, 'model_name', model_name)
        final_list.append(row_df)

    return pd.concat(final_list, ignore_index=True)