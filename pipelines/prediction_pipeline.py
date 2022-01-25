from zenml.repository import Repository
from zenml.pipelines import pipeline

from steps.encoder import encode_columns_and_clean
from steps.importer import import_season_schedule, SeasonScheduleConfig
from steps.model_picker import model_picker
from steps.predictor import predictor
from steps.splitter import get_coming_week_data, TimeWindowConfig


@pipeline(enable_cache=False)
def inference_pipeline(
        importer,
        # Import data that contains data about which teams will be
        # playing each other in the coming week
        preprocessor,  # Label encode SEASON_ID and TEAMS
        extract_next_week,
        # Extract games from next week to infer on
        # -> SEASON_ID, TEAM_ABBREVIATION, OPPONENT_TEAM_ABBREVIATION
        model_picker,  # Go into last pipeline runs and pick best model
        predictor,  # Predict three-pointer
):
    """Links all the steps together in a pipeline"""
    season_schedule = importer()
    processed_season_schedule = preprocessor(season_schedule)
    upcoming_week = extract_next_week(processed_season_schedule)
    model, run_id = model_picker()
    predictions = predictor(model, upcoming_week)


if __name__ == "__main__":
    # Initialize the pipeline
    inference_pipeline = inference_pipeline(
        importer=import_season_schedule(
            SeasonScheduleConfig(current_season='2021-22')),
        preprocessor=encode_columns_and_clean(),
        extract_next_week=get_coming_week_data(TimeWindowConfig(time_window=7)),
        model_picker=model_picker(),
        predictor=predictor()
    )

    inference_pipeline.run()
