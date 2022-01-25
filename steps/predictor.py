import pandas as pd

from zenml.steps import step
from sklearn.base import RegressorMixin

from steps.utils import get_label_encoder


@step
def predictor(model: RegressorMixin,
              data: pd.DataFrame,
              ) -> pd.DataFrame:
    feature_cols = model.feature_names_in_

    data = data[feature_cols]

    predicted_y = model.predict(data)

    data['PREDICTION'] = predicted_y

    le_seasons = get_label_encoder('le_seasons')
    data['SEASON_ID'] = le_seasons.inverse_transform(data['SEASON_ID'])

    return data