from datetime import date, timedelta, datetime

from zenml.pipelines import pipeline, Schedule

last_week = date.today() - timedelta(days=7)
ONE_WEEK_AGO = last_week.strftime("%Y-%m-%d")
CURRY_FROM_DOWNTOWN = '2016-02-27'


@pipeline(enable_cache=False)
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


if __name__ == "__main__":
    from ..steps.discord_bot import discord_alert
    from ..steps.encoder import data_encoder
    from ..steps.evaluator import tester
    from ..steps.feature_engineer import feature_engineer
    from ..steps.importer import game_data_importer
    from ..steps.profiler import evidently_drift_detector
    from ..steps.splitter import sklearn_splitter, SklearnSplitterConfig, \
        reference_data_splitter, TrainingSplitConfig
    from ..steps.trainer import random_forest_trainer

    # Initialize the pipeline
    training_pipeline = training_pipeline(
        importer=game_data_importer(),
        # Train Model
        feature_engineerer=feature_engineer(),
        encoder=data_encoder(),
        ml_splitter=sklearn_splitter(SklearnSplitterConfig(
            ratios={'train': 0.6, 'test': 0.2, 'validation': 0.2})),
        trainer=random_forest_trainer(),
        tester=tester(),
        # Drift detection
        drift_splitter=reference_data_splitter(
            TrainingSplitConfig(
                new_data_split_date=ONE_WEEK_AGO,
                start_reference_time_frame=CURRY_FROM_DOWNTOWN,
                end_reference_time_frame="2019-02-27",
                columns=["FG3M"])
        ),
        drift_detector=evidently_drift_detector,
        drift_alert=discord_alert(),
    )

    training_pipeline.run(
        Schedule(start_time=datetime.now(), interval_seconds=30)
    )
