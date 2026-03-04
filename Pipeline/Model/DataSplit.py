import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler


class DataSplit:
    def __init__(self, random_state: int = 42 , test_size = 0.2 , k_fold:int = 5):
        self.random_state = random_state
        self.test_size = test_size
        self.k_fold = k_fold

        self.x_test = None
        self.y_test = None

        self.x_train_scaled = None
        self.x_test_scaled  = None

        self.k_fold_dataset = None
        self.main_scaler    = None

    def k_fold_data_spiting(self, x, y):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size= self.test_size,
            random_state=self.random_state,
            stratify=y)

        self.x_test = x_test.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        x_train = x_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        self.main_scaler = StandardScaler()
        self.x_train_scaled = pd.DataFrame(self.main_scaler.fit_transform(x_train), columns=x.columns)
        self.x_test_scaled  = pd.DataFrame(self.main_scaler.transform(x_test), columns=x.columns)

        kf = StratifiedKFold(
            n_splits=self.k_fold,
            shuffle=True,
            random_state=self.random_state
        )

        self.k_fold_dataset = {}

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(x_train, y_train)):
            x_train_fold_raw = x_train.iloc[train_idx]
            x_val_fold_raw   = x_train.iloc[val_idx]
            fold_scaler      = StandardScaler()

            x_train_fold_scaled = pd.DataFrame(fold_scaler.fit_transform(x_train_fold_raw),columns=x.columns)
            x_val_fold_scaled   = pd.DataFrame(fold_scaler.transform(x_val_fold_raw),columns=x.columns)

            self.k_fold_dataset[fold_idx] = {
                'X_train_fold': x_train_fold_scaled,
                'X_val_fold': x_val_fold_scaled,
                'y_train_fold': y_train.iloc[train_idx].reset_index(drop=True),
                'y_val_fold': y_train.iloc[val_idx].reset_index(drop=True),
                'scaler': fold_scaler
            }

        return self.x_test,self.y_test,self.k_fold_dataset