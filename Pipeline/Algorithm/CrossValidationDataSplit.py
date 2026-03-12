import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


class CrossValidationDataSplit:
    def __init__(self, k_fold : int = 5):
        self.k_fold         = k_fold

        self.k_fold_dataset = None
        self.main_scaler    = None

    def k_fold_data_spiting(self, x_train, y_train):

        kf = StratifiedKFold(
            n_splits        = self.k_fold,
            shuffle         = False
        )

        self.k_fold_dataset = {}

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x_train, y_train)):
            x_train_fold_raw = x_train.iloc[train_idx]
            x_val_fold_raw   = x_train.iloc[val_idx]

            fold_scaler      = MinMaxScaler()

            x_train_fold_scaled = pd.DataFrame(fold_scaler.fit_transform(x_train_fold_raw),columns=x_train.columns)
            x_val_fold_scaled   = pd.DataFrame(fold_scaler.transform(x_val_fold_raw),columns=x_train.columns)

            self.k_fold_dataset[fold_idx] = {
                'X_train_fold'  : x_train_fold_scaled,
                'X_val_fold'    : x_val_fold_scaled,
                'y_train_fold'  : y_train.iloc[train_idx].reset_index(drop=True),
                'y_val_fold'    : y_train.iloc[val_idx].reset_index(drop=True),
                'scaler'        : fold_scaler
            }

        return self.k_fold_dataset