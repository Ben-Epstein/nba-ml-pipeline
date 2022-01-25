from zenml.pipelines import pipeline

CURRY_FROM_DOWNTOWN = '2016-02-27'


@pipeline
def one_shot_drift_detector(
        importer,
        drift_splitter,
        drift_detecter,
        drift_analyzer
):
    """Links all the steps together in a pipeline"""
    # drift
    raw_data = importer()
    reference_dataset, comparison_dataset = drift_splitter(raw_data)
    drift_report, _ = drift_detecter(reference_dataset, comparison_dataset)
    drift_analyzer(drift_report)


if __name__ == "__main__":
    from steps.importer import game_data_importer
    from steps.splitter import date_based_splitter, SplitConfig
    from steps.analyzer import analyze_drift
    from steps.profiler import drift_detecter

    # Initialize the pipeline
    one_shot_pipeline = one_shot_drift_detector(
        importer=game_data_importer(),
        drift_splitter=date_based_splitter(
            SplitConfig(date_split=CURRY_FROM_DOWNTOWN, columns=['FG3M'])),
        drift_detecter=drift_detecter,
        drift_analyzer=analyze_drift()
    )

    one_shot_pipeline.run()
