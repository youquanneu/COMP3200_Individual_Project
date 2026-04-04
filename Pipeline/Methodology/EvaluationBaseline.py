import time
from typing import Any

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

from Pipeline.Global.GallstoneDataSet import GallstoneDataSet
from Pipeline.Methodology.EvaluationMatrix import EvaluationMatrix
from Pipeline.Global.GlobalSetting import GlobalSetting


class EvaluationBaseline:
    def __init__(self,
                 model_registry : dict,
                 use_raw_data   : bool  = False,
                 data_scaling   : bool  = False,
                 cv_folds       : int   = 5):

        self.model_registry = model_registry
        self.use_raw_data   = use_raw_data
        self.data_scaling   = data_scaling
        self.prefix = "raw_" if use_raw_data else "cleaned_"
        self.cv_folds = cv_folds

        self.gallstone_dataset = GallstoneDataSet()
        if self.use_raw_data:
            self.gallstone_dataset.fetch_raw_data_path()
        else:
            self.gallstone_dataset.fetch_cleaned_data_path()

        self.gallstone_dataset.cv_test_split(cv_folds)

    def _scikit_learn_parameter_tuning(self) -> dict:

        val_splits = self.gallstone_dataset.val_scaled_fold_split \
            if self.data_scaling else self.gallstone_dataset.val_fold_split

        x_stacked, y_stacked, custom_cv_indices = [], [], []
        current_offset = 0
        for x_tr, y_tr, x_val, y_val in val_splits:
            x_fold_combined = np.vstack((x_tr, x_val))
            y_fold_combined = np.concatenate((y_tr, y_val))

            x_stacked.append(x_fold_combined)
            y_stacked.append(y_fold_combined)

            train_idx = np.arange(current_offset, current_offset + len(x_tr))
            val_idx = np.arange(current_offset + len(x_tr), current_offset + len(x_fold_combined))

            custom_cv_indices.append((train_idx, val_idx))
            current_offset += len(x_fold_combined)

        x_all = np.vstack(x_stacked)
        y_all = np.concatenate(y_stacked)

        best_parameters = {}

        for model_name, config in self.model_registry.items():
            base_model = config["model_class"](**config["base_kwargs"])

            searcher = RandomizedSearchCV(
                estimator       = base_model,
                param_distributions = config["tuning_grid"],
                n_iter          = 20,
                cv              = custom_cv_indices,
                scoring         = 'matthews_corrcoef',
                n_jobs          = -1,
                random_state    = 42
            )

            searcher.fit(x_all, np.ravel(y_all))
            best_parameters[model_name] = searcher.best_params_

        return best_parameters

    def stage_2_testing(self, best_parameters: dict) -> list[Any]:

        outer_test_list = self.gallstone_dataset.test_scaled_fold_split \
            if self.data_scaling else self.gallstone_dataset.test_fold_split

        model_results_buckets = {model_name: [] for model_name in self.model_registry.keys()}

        for fold_idx, (x_train, y_train, x_test, y_test) in enumerate(outer_test_list):

            print(f"\nTesting - Fold {fold_idx}")
            fold_start_time = time.time()

            for seed in GlobalSetting.seed_test_range:
                for model_name, config in self.model_registry.items():
                    try:
                        optimal_params = {**config["base_kwargs"], **best_parameters[model_name]}
                        if 'random_state' in config["model_class"]().get_params():
                            optimal_params['random_state'] = seed

                        model = config["model_class"](**optimal_params)

                        model.fit(x_train, np.ravel(y_train))
                        y_pred = model.predict(x_test)

                        evaluation = EvaluationMatrix(y_test, y_pred)
                        metrics = evaluation.get_all_metrics()

                        model_results_buckets[model_name].append({
                            "Model_Type"    : f"{self.prefix}{model_name}",
                            "Is_ABC_Opt"    : False,
                            "Data_Scaled"   : self.data_scaling,
                            "Fold_ID"       : fold_idx,
                            "Seed"          : seed,
                            **metrics
                        })

                    except Exception as e:
                        print(f"\n    [!] ERROR: Failed at Fold {fold_idx}, Model {model_name}, Seed {seed}: {e}")
                        continue

            fold_end_time = time.time()
            fold_duration = fold_end_time - fold_start_time
            print(f"\n\nTesting - Fold {fold_idx} Completed in {fold_duration:.4f}")

        expected_cols = [
            'Model_Type', 'Is_ABC_Opt', 'Data_Scaled', 'Fold_ID', 'Seed',
            'Accuracy', 'Precision', 'Recall', 'NPV', 'Specificity', 'F1-Score', 'F2-Score', 'Bal Accuracy', 'MCC'
        ]

        result_list = []
        for model_name, results in model_results_buckets.items():
            if not results:
                continue

            full_model_type = f"{self.prefix}{model_name}"

            df_model_raw = pd.DataFrame(results)

            actual_cols = [col for col in expected_cols if col in df_model_raw.columns]
            df_model = df_model_raw[actual_cols].sort_values(by=['Fold_ID', 'Seed'])

            file_path = f"Test History/{full_model_type}_Results.csv"
            GlobalSetting.save_dataframe_to_record(df_model, file_path)

            result_list.append(df_model)

        return result_list

    def baseline_pipeline_running(self):
        best_params = self._scikit_learn_parameter_tuning()
        results_list = self.stage_2_testing(best_params)
        return results_list