import os
from joblib import load

import yaml
import pytest
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from model.ml.data import process_data
from model.ml.model import load_artifact, compute_slice_metrics, \
    inference, RFModel

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'artifact',
    'model',
    'random_forest.pkl')
BINARIZER_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'artifact',
    'model',
    'label_binarizer.pkl')
ENCODER_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'artifact',
    'model',
    'oh_encoder.pkl')
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'artifact',
    'data',
    'clean_census.csv')

# Loads config
config_path = os.path.join(os.path.dirname(__file__),
                           '..', 'model', 'model_config.yaml')
with open(config_path, 'r') as fp:
    config = yaml.safe_load(fp)


@pytest.fixture
def random_forest():
    return load(MODEL_PATH)


@pytest.fixture
def binarizer():
    return load(BINARIZER_PATH)


@pytest.fixture
def encoder():
    return load(ENCODER_PATH)


@pytest.fixture(scope='function')
def df():
    import pandas as pd

    dataframe = pd.read_csv(DATA_PATH)
    return dataframe


def test_rf_model_attributes(random_forest):
    model = RFModel(model=random_forest)
    assert hasattr(
        model,
        'model') and isinstance(
        model.model,
        RandomForestClassifier)
    assert hasattr(
        model,
        'binarizer') and isinstance(
        model.binarizer,
        LabelBinarizer)
    assert hasattr(
        model,
        'encoder') and isinstance(
        model.encoder,
        OneHotEncoder)


def test_load_model():
    assert isinstance(load_artifact(MODEL_PATH), RandomForestClassifier)


def test_compute_slice_metrics(df):
    SLICES = ['race', 'sex']
    for slice in SLICES:
        predictions = compute_slice_metrics(df, slice)
        for feature, metrics in predictions.items():
            assert isinstance(feature, str)
            assert metrics['precision'] > 0.5
            assert isinstance(
                metrics['precision'],
                float) and isinstance(
                metrics['recall'],
                float) and isinstance(
                metrics['fbeta'],
                float)


def test_process_data(encoder, binarizer, df):
    X, y, _, _ = process_data(df,
                              categorical_features=config['categorical_features'],
                              label='salary', training=False, encoder=encoder,
                              lb=binarizer)
    assert isinstance(X, np.ndarray)
    assert len(X) > 0
    assert isinstance(y, np.ndarray)


def test_inference(random_forest, encoder, binarizer, df):
    X, y, _, _ = process_data(df,
                              categorical_features=config['categorical_features'],
                              label='salary', training=False, encoder=encoder,
                              lb=binarizer)

    preds = inference(random_forest, X)
    assert isinstance(preds, np.ndarray)
    assert len(preds) > 0
