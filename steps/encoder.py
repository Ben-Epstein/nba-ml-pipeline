import os
import pandas as pd
import numpy as np
from typing import Any, Type, Union
import pickle

from zenml.materializers.base_materializer import BaseMaterializer
from zenml.artifacts import ModelArtifact
from zenml.steps.step_output import Output

from zenml.io import fileio

from sklearn import preprocessing
from zenml.steps import step

from steps.utils import get_label_encoder

DEFAULT_FILENAME = 'label_encoder'


class SklearnLEMaterializer(BaseMaterializer):
    """Materializer to read data to and from sklearn."""

    ASSOCIATED_TYPES = [preprocessing.LabelEncoder]

    ASSOCIATED_ARTIFACT_TYPES = [ModelArtifact]

    def handle_input(
            self, data_type: Type[Any]
    ) -> Union[preprocessing.LabelEncoder]:
        """Reads a base sklearn label encoder from a pickle file."""
        super().handle_input(data_type)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as fid:
            clf = pickle.load(fid)
        return clf

    def handle_return(
            self,
            clf: Union[preprocessing.LabelEncoder],
    ) -> None:
        """Creates a pickle for a sklearn label encoder.

        Args:
            clf: A sklearn label encoder.
        """
        super().handle_return(clf)
        filepath = os.path.join(self.artifact.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as fid:
            pickle.dump(clf, fid)


def apply_encoder(label_encoder: preprocessing.LabelEncoder,
                  one_hot_encoder: preprocessing.OneHotEncoder,
                  dataframe: pd.DataFrame) -> pd.DataFrame:
    """Use Encoders to turn the categorical columns into LabelEncoded/ OneHotEncoded columns"""

    dataframe['SEASON_ID'] = label_encoder.fit_transform(dataframe['SEASON_ID'])

    one_hot_team = one_hot_encoder.transform(
        dataframe['TEAM_ABBREVIATION'].values.reshape(-1, 1)).todense()
    one_hot_team_df = pd.DataFrame(one_hot_team,
                                   columns=one_hot_encoder.categories_[0],
                                   index=dataframe.index)

    one_hot_opponent = one_hot_encoder.transform(
        dataframe['OPPONENT_TEAM_ABBREVIATION'].values.reshape(-1, 1)).todense()
    one_hot_opponent_df = pd.DataFrame(one_hot_opponent,
                                       columns=one_hot_encoder.categories_[0],
                                       index=dataframe.index)

    dataframe = dataframe.drop(
        ['TEAM_ABBREVIATION', 'OPPONENT_TEAM_ABBREVIATION'], axis=1)

    new_df = pd.concat([dataframe, one_hot_team_df.add_prefix('home_'),
                        one_hot_opponent_df.add_prefix('away_')], axis=1)

    return new_df


@step(enable_cache=False)
def encoder(pandas_df: pd.DataFrame) -> Output(
    encoded_data=pd.DataFrame,
    le_seasons=preprocessing.LabelEncoder,
    ohe_teams=preprocessing.LabelEncoder):
    # convert categorical to ints
    le_seasons = preprocessing.LabelEncoder()
    le_seasons.fit(pandas_df['SEASON_ID'])

    ohe_teams = preprocessing.OneHotEncoder(dtype=np.int32)
    ohe_teams.fit(pandas_df['TEAM_ABBREVIATION'].values.reshape(-1, 1))

    new_df = apply_encoder(label_encoder=le_seasons, one_hot_encoder=ohe_teams,
                           dataframe=pandas_df)

    return new_df, le_seasons, ohe_teams


@step
def encode_columns_and_clean(pandas_df: pd.DataFrame) -> pd.DataFrame:
    # convert categorical to ints
    le_seasons = get_label_encoder('le_seasons')

    ohe_teams = get_label_encoder('ohe_teams')

    # Clean data with missing date
    pandas_df = pandas_df.drop(pandas_df[pandas_df['GAME_DAY'] == 'TBD'].index)
    pandas_df['GAME_TIME'].mask(pandas_df['GAME_TIME'] == 'TBD', '00:00:00',
                                inplace=True)

    # Apply label encoders using the same function as during training
    new_df = apply_encoder(label_encoder=le_seasons, one_hot_encoder=ohe_teams,
                           dataframe=pandas_df)

    return new_df
