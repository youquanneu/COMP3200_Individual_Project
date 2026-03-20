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

        self.splits = None

        self.x_train        = None
        self.x_train_scaled = None
        self.y_train        = None

        self.x_test         = None
        self.x_test_scaled  = None
        self.y_test         = None

        self.main_scaler    = None

        self.fold_split = []
        self.scaled_fold_split = []

    def apply_data(self):
        df = pd.read_csv(self.file_path)
        self.x = df.drop(self.target_col, axis=1)
        self.y = df[self.target_col]
    def fetch_data_path_0(self):
        self.__init__()
        self.file_path = '../../Storage/Dataset/Dataset.csv'
        self.target_col = ['Gallstone Status']
        self.apply_data()

    def fetch_data_path_1(self):
        self.__init__()
        self.file_path = '../../Storage/Dataset/UCI_Gallstone_Dataset.csv'
        self.target_col = ['Gallstone Status']
        self.apply_data()

    def normal_data_split(self):

        x_train, x_test, y_train, y_test = train_test_split(
            self.x,self.y,
            test_size       = GlobalSetting.test_set_size,
            random_state    = GlobalSetting.data_split_seed,
            stratify        = self.y
        )

        self.x_train = x_train.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)

        self.x_test = x_test.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)

        self.main_scaler = MinMaxScaler()
        self.x_train_scaled = pd.DataFrame(self.main_scaler.fit_transform(x_train), columns=self.x.columns)
        self.x_test_scaled  = pd.DataFrame(self.main_scaler.transform(x_test), columns=self.x.columns)

    def cross_validate_test(self, n_splits = None):

        self.splits = GlobalSetting.data_test_split if n_splits is None else n_splits
        skf = StratifiedKFold(n_splits=self.splits,
                              shuffle=True,
                              random_state = GlobalSetting.data_split_seed
                              )

        self.fold_split = []
        self.scaled_fold_split = []

        for train_idx, test_idx in skf.split(self.x, self.y):

            x_train_raw = self.x.iloc[train_idx].reset_index(drop=True)
            x_test_raw  = self.x.iloc[test_idx] .reset_index(drop=True)
            y_train = self.y.iloc[train_idx].reset_index(drop=True)
            y_test  = self.y.iloc[test_idx] .reset_index(drop=True)

            scaler = MinMaxScaler()
            x_train_scaled  = pd.DataFrame(scaler.fit_transform(x_train_raw), columns=self.x.columns)
            x_test_scaled   = pd.DataFrame(scaler.transform(x_test_raw), columns=self.x.columns)

            self.fold_split.append((x_train_raw, y_train, x_test_raw, y_test))
            self.scaled_fold_split.append((x_train_scaled, y_train, x_test_scaled , y_test))