from sklearn.base import RegressorMixin

from zenml.steps import step
from zenml.steps.step_output import Output


@step
def model_picker() -> Output(model=RegressorMixin,
                             associated_run_id=str):
    repo = Repository()
    training_pipeline = repo.get_pipeline(pipeline_name="training_pipeline")

    best_score = None
    best_model = None
    best_run = None

    for run in training_pipeline.runs:
        model = None
        mae = None
        try:
            mae = run.get_step(name='tester').output.read()
            print(f"Run {run.name} yielded a model with mae={mae}")
        except KeyError:
            print(f"Skipping {run.name} as it does not contain the tester step")

        try:
            model = run.get_step(name='trainer').output.read()
        except KeyError:
            print(
                f"Skipping {run.name} as it does not contain the trainer step")

        if mae and model:
            if not best_score or (best_score and mae <= best_score):
                best_model = model
                best_score = mae
                best_run = run.name

    return best_model, best_run
