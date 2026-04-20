import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from Pipeline.Global.GlobalSetting import GlobalSetting

class GallstoneDataSet:
    def __init__(self):

        self.x = None
        self.y = None

        self.file_path  = None
        self.target_col = None

        self.splits     = None

        self.val_fold_split         = []
        self.val_scaled_fold_split  = []

        self.test_fold_split        = []
        self.test_scaled_fold_split = []

    def apply_data(self):
        df = pd.read_csv(self.file_path)
        self.x = df.drop(self.target_col, axis=1)
        self.y = df[self.target_col]
    def fetch_cleaned_data_path(self):
        self.__init__()
        self.file_path = '../../Storage/Dataset/Cleaned_Gallstone_Dataset_FE.csv'
        self.target_col = ['Gallstone Status']
        self.apply_data()

    def fetch_raw_data_path(self):
        self.__init__()
        self.file_path = '../../Storage/Dataset/UCI_Gallstone_Dataset.csv'
        self.target_col = ['Gallstone Status']
        self.apply_data()
    @staticmethod
    def generate_k_fold_splits(base_x, base_y, n_splits):

        raw_folds       = []
        scaled_folds    = []

        skf = StratifiedKFold(n_splits  = n_splits,
                              shuffle   = True,
                              random_state = GlobalSetting.data_split_seed)

        for train_idx, val_idx in skf.split(base_x, base_y):
            x_train_raw = base_x.iloc[train_idx].reset_index(drop=True)
            x_val_raw   = base_x.iloc[val_idx].reset_index(drop=True)
            y_train_raw = base_y.iloc[train_idx].reset_index(drop=True)
            y_val_raw   = base_y.iloc[val_idx].reset_index(drop=True)

            scaler = MinMaxScaler()
            x_train_scaled  = pd.DataFrame(scaler.fit_transform(x_train_raw), columns=base_x.columns)
            x_val_scaled    = pd.DataFrame(scaler.transform(x_val_raw)      , columns=base_x.columns)

            raw_folds.append((x_train_raw, y_train_raw, x_val_raw, y_val_raw))
            scaled_folds.append((x_train_scaled, y_train_raw, x_val_scaled, y_val_raw))

        return raw_folds, scaled_folds

    @staticmethod
    def generate_inner_holdout(base_x, base_y):
        x_tr, x_val, y_tr, y_val = train_test_split(
            base_x, base_y,
            test_size       = 0.2,
            random_state    = GlobalSetting.data_split_seed,
            stratify        = base_y
        )
        x_tr    = x_tr.reset_index (drop=True)
        x_val   = x_val.reset_index(drop=True)
        y_tr    = y_tr.reset_index (drop=True)
        y_val   = y_val.reset_index(drop=True)

        scaler = MinMaxScaler()
        x_tr_scaled     = pd.DataFrame(scaler.fit_transform(x_tr) , columns = base_x.columns)
        x_val_scaled    = pd.DataFrame(scaler.transform(x_val)    , columns = base_x.columns)

        return (x_tr, y_tr, x_val, y_val), (x_tr_scaled, y_tr, x_val_scaled, y_val)
    def cv_test_split(self, n_splits = None):

        self.val_fold_split = []
        self.val_scaled_fold_split = []

        self.test_fold_split = []
        self.test_scaled_fold_split = []

        self.splits = GlobalSetting.data_test_split if n_splits is None else n_splits
        self.test_fold_split, self.test_scaled_fold_split = self.generate_k_fold_splits(
            self.x, self.y, self.splits
        )

        for fold_idx in range(self.splits):
            x_outer_tr_raw, y_outer_tr_raw, _, _ = self.test_fold_split[fold_idx]

            raw_inner, scaled_inner = self.generate_inner_holdout(x_outer_tr_raw, y_outer_tr_raw)
            self.val_fold_split.append(raw_inner)
            self.val_scaled_fold_split.append(scaled_inner)