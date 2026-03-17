import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from Pipeline.Global.GlobalSetting import GlobalSetting

class GallstoneDataSet:
    def __init__(self):


        self.test_size      = GlobalSetting.test_set_size
        self.random_state   = GlobalSetting.data_split_seed

        self.file_path  = None
        self.target_col = None

        self.x_train        = None
        self.x_train_scaled = None
        self.y_train        = None

        self.x_test         = None
        self.x_test_scaled  = None
        self.y_test         = None

        self.main_scaler    = None

        self.fold_split = []

    def fetch_data_path_1(self):
        self.file_path = '../../Storage/Dataset/Dataset.csv'
        self.target_col = ['Gallstone Status']

    def normal_data_split(self):
        df = pd.read_csv(self.file_path)

        x = df.drop(self.target_col, axis=1)
        y = df[self.target_col]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y)

        self.x_test = x_test.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        self.x_train = x_train.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)

        self.main_scaler = MinMaxScaler()
        self.x_train_scaled = pd.DataFrame(self.main_scaler.fit_transform(x_train), columns=x.columns)
        self.x_test_scaled = pd.DataFrame(self.main_scaler.transform(x_test), columns=x.columns)

    def cross_validate_test(self, n_splits=5):
        if self.file_path is None or self.target_col is None:
            raise ValueError("File path or target column not set. Call fetch_data_path_1() first.")

        df = pd.read_csv(self.file_path)
        x = df.drop(self.target_col, axis=1)
        y = df[self.target_col]

        # 2. Initialize the Stratified K-Fold generator
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        # Clear the list just in case this method is called multiple times
        self.fold_split = []

        # 3. Iterate through the splits and pack them into the tuple list
        for train_idx, test_idx in skf.split(x, y):
            # Use iloc to slice by the raw numpy indices provided by skf.split
            x_train = x.iloc[train_idx].reset_index(drop=True)
            x_test  = x.iloc[test_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_test  = y.iloc[test_idx].reset_index(drop=True)

            # Append the unscaled, raw DataFrames to the class attribute
            self.fold_split.append((x_train, y_train, x_test, y_test))