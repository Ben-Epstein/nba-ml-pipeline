import pandas as pd
from zenml.steps import step


@step
def data_post_processor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Go from one hot encoded teams to team columns
    """
    home = [team for team in df.columns.to_list() if 'home_' in team]
    away = [team for team in df.columns.to_list() if 'away_' in team]

    df['Home_Team'] = df[home].idxmax(1)
    df['Away_Team'] = df[away].idxmax(1)

    df = df.drop(columns=home+away+['SEASON_ID'])
    df = df.rename(columns={'PREDICTION': 'Predicted_3_Pointers_Home_Team'})
    return df

