from zenml.integrations.evidently.steps import (
    EvidentlyProfileConfig,
    EvidentlyProfileStep,
)

drift_detecter = EvidentlyProfileStep(
    EvidentlyProfileConfig(
        column_mapping=None,
        profile_sections=["datadrift"],
    )
)