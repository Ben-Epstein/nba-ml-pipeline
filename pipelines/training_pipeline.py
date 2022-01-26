from datetime import date, timedelta, datetime

from zenml.pipelines import pipeline, Schedule


@pipeline(enable_cache=False, requirements_file='requirements.txt')
def training_pipeline(
        importer,
        feature_engineerer,
        encoder,
        ml_splitter,
        trainer,
        tester,

        drift_splitter,
        drift_detector,
        drift_alert
):
    """Links all the steps together in a pipeline"""
    # Training pipeline
    raw_data = importer()
    transformed_data = feature_engineerer(raw_data)
    encoded_data, le_seasons, ohe_teams = encoder(transformed_data)
    train_df_x, train_df_y, test_df_x, test_df_y, eval_df_x, eval_df_y = ml_splitter(encoded_data)
    model = trainer(train_df_x, train_df_y, eval_df_x, eval_df_y)
    test_results = tester(model, test_df_x, test_df_y)

    # drift
    reference_dataset, comparison_dataset = drift_splitter(raw_data)
    drift_report, _ = drift_detector(reference_dataset, comparison_dataset)
    drift_alert(drift_report)
