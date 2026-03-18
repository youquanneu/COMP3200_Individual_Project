import json
import os
import numpy as np
import pandas as pd


class GlobalSetting:

    dataset_dir = '../../Storage/'
    record_dir  = os.path.join(dataset_dir, 'Record')
    json_dir    = os.path.join(dataset_dir, 'JSON')
    figure_dir  = os.path.join(dataset_dir, 'Figure')

    config_file = os.path.join(json_dir, 'full_model_configs.json')

    elm_initial_state_range     = range(161, 191)
    hidden_size_explore_range   = range(1, 101)
    lambda_value_explore_range  = 2.0 ** np.arange(-25, 3)

    seed_punish_coefficient     = 1.0

    cv_punish_coefficient       = 1.0


    test_set_size       = 0.2
    data_split_seed     = 42

    data_cv_fold        = 5
    data_shuffle_seed   = 42

    evaluation_function = 'MCC'

    solution_size = 80
    trial_limit   = 50
    max_iteration = 100
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def get_model_configs(cls):

        activation_map = {
            "sigmoid": GlobalSetting.sigmoid,
            "tanh": np.tanh
        }

        model_configs = []
        try:
            with open(cls.config_file, 'r') as f:
                configs = json.load(f)

                for config in configs:
                    act_str = config.get("Activation")

                    if act_str in activation_map:
                        config["Activation"] = activation_map[act_str]
                    else:
                        raise ValueError(f"Unsupported activation function: '{act_str}' found in config.")

                model_configs.extend(configs)
            return model_configs

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Critical configuration missing: {cls.config_file}. Execution aborted.") from e

    @classmethod
    def upsert_model_configs(cls, payload, config_file=None):
        target_file = config_file if config_file else cls.config_file
        os.makedirs(os.path.dirname(target_file), exist_ok=True)

        if os.path.exists(target_file):
            with open(target_file, 'r') as f:
                existing_configs = json.load(f)
        else:
            existing_configs = []

        existing_map = {config.get("Model_Types"): i for i, config in enumerate(existing_configs)}

        for new_config in payload:
            model_type = new_config.get("Model_Types")

            if model_type in existing_map:
                existing_configs[existing_map[model_type]] = new_config
            else:
                existing_configs.append(new_config)
                existing_map[model_type] = len(existing_configs) - 1

        with open(target_file, 'w') as f:
            json.dump(existing_configs, f, indent=4)

    @classmethod
    def get_config_by_type(cls, model_type):
        configs = cls.get_model_configs()

        return next((config for config in configs if config.get("Model_Types") == model_type), None)

    @classmethod
    def get_record_dir(cls):
        os.makedirs(cls.record_dir, exist_ok=True)
        return cls.record_dir
    @classmethod
    def save_dataframe_to_record(cls, df, filename):
        if not filename.endswith('.csv'):
            filename += '.csv'

        target_dir = cls.get_record_dir()
        file_path = os.path.join(target_dir, filename)

        df.to_csv(file_path, index=False)

        print(f"[I/O Trace] Record exported successfully: {file_path}")

    @classmethod
    def get_dataframe_from_record(cls, filename):
        """Fetches a saved CSV record and returns it as a Pandas DataFrame."""
        if not filename.endswith('.csv'):
            filename += '.csv'

        target_dir = cls.get_record_dir()
        file_path = os.path.join(target_dir, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Critical Error: The record '{file_path}' does not exist.")

        df = pd.read_csv(file_path)
        print(f"[I/O Trace] Record imported successfully: {file_path}")

        return df

    @classmethod
    def get_figure_dir(cls):
        """Safely fetches or creates the Figure directory"""
        os.makedirs(cls.figure_dir, exist_ok=True)
        return cls.figure_dir