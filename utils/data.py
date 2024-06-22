import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


from constants import DATA_PATH

def load_data():
    return pd.read_csv(DATA_PATH, index_col=False)


def encode_categorical(data: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(data, columns=['city', 'state'], dtype=float, sparse=True)


def split_and_scale(data: pd.DataFrame, test_size: float = 0.2) -> np.array:
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['price']), data['price'], test_size=test_size)

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

