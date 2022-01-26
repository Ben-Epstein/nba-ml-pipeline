from zenml.steps import step

@step
def analyze_drift(
    input: dict,
) -> bool:
    """Analyze the Evidently drift report and return a true/false value indicating
    whether data drift was detected."""
    drift = input['data_drift']['data']['metrics']['dataset_drift']
    print("Drift detected" if drift else "No drift detected")
    return drift
