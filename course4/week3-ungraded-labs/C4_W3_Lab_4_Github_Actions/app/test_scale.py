from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from main import clf


def test_pipeline_and_scaler():
    isPipeline = isinstance(clf, Pipeline)
    assert isPipeline
    if isPipeline:
        firstStep = [v for v in clf.named_steps.values()][0]
    assert isinstance(firstStep, StandardScaler)
