import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class GallstoneDataSet:
    def __init__(self,test_size = 0.2 ,random_state = 42):

        self.test_size = test_size
        self.random_state = random_state

        self.x_train        = None
        self.x_train_scaled = None
        self.y_train        = None
        self.x_test_scaled  = None
        self.y_test         = None
        self.main_scaler    = None

    def fetch_data_path_1(self):
        file_path = '../../Dataset/UCI_Gallstone_Dataset.csv'
        target_col = ['Gallstone Status']
        df = pd.read_csv(file_path)

        x = df.drop(target_col, axis=1)
        y = df[target_col]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y)

        x_test = x_test.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        self.x_train = x_train.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)

        self.main_scaler = MinMaxScaler()
        self.x_train_scaled = pd.DataFrame(self.main_scaler.fit_transform(x_train), columns=x.columns)
        self.x_test_scaled = pd.DataFrame(self.main_scaler.transform(x_test), columns=x.columns)
