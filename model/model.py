import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from model.configurations import RANDOM_FOREST_CONFIG
from utils import load_data, Sample


class Predictor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(**RANDOM_FOREST_CONFIG)
        self._fit()

    def _fit(self):
        data = load_data()[['bed', 'bath', 'house_size', 'price']].copy()
        features = self.scaler.fit_transform(data.drop('price', axis=1))
        print('Fitting model...')
        self.model.fit(features, data['price'])

        predictions = self.model.predict(features)
        mse = r2_score(y_true=data['price'], y_pred=predictions)
        print('Approximation R2:', mse)

    def predict(self, sample: Sample) -> float:
        input_sample = np.array([sample['bed'], sample['bath'], sample['house_size']]).reshape(1, -1)
        scaled_input_sample = self.scaler.transform(input_sample)
        return self.model.predict(scaled_input_sample)[0]

if __name__ == "__main__":
    Predictor()
