import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor


from zenml.steps import step
from zenml.steps.base_step_config import BaseStepConfig


class RandomForestTrainerConfig(BaseStepConfig):
    """Config class for the sklearn trainer"""
    max_depth: int = 10000
    target_col: str = 'FG3M'


@step(enable_cache=False)
def random_forest_trainer(train_df_x: pd.DataFrame, train_df_y: pd.DataFrame,
                          eval_df_x: pd.DataFrame, eval_df_y: pd.DataFrame,
                          config: RandomForestTrainerConfig) -> RegressorMixin:

    # mlflow.sklearn.autolog()
    clf = RandomForestRegressor(max_depth=config.max_depth)
    clf.fit(train_df_x, np.squeeze(train_df_y.values.T))
    eval_score = clf.score(eval_df_x, np.squeeze(eval_df_y.values.T))
    print(f"Eval score is: {eval_score}")
    return clf
