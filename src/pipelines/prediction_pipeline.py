from zenml.pipelines import pipeline


@pipeline(enable_cache=False)
def inference_pipeline(
        importer,
        preprocessor,
        extract_next_week,
        model_picker,
        predictor,
):
    """Links all the steps together in a pipeline"""
    season_schedule = importer()
    processed_season_schedule = preprocessor(season_schedule)
    upcoming_week = extract_next_week(processed_season_schedule)
    model, run_id = model_picker()
    predictions = predictor(model, upcoming_week)


if __name__ == "__main__":
    from src.steps.encoder import encode_columns_and_clean
    from src.steps.importer import import_season_schedule, SeasonScheduleConfig
    from src.steps.model_picker import model_picker
    from src.steps.predictor import predictor
    from src.steps.splitter import get_coming_week_data, TimeWindowConfig

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
