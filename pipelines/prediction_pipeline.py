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
