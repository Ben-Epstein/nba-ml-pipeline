from sklearn import preprocessing
from zenml.repository import Repository


def get_label_encoder(name: str) -> preprocessing.LabelEncoder:
    repo = Repository()
    training_pipeline = repo.get_pipeline(pipeline_name="training_pipeline")
    return training_pipeline.runs[-1].get_step(name='encoder').outputs[
        name].read()