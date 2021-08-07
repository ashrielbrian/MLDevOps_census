# Script to train machine learning model.
import os
from joblib import dump
import yaml
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.model import train_model, compute_model_metrics, inference, save_artifact, compute_slice_metrics
from ml.data import process_data, preprocess_data

# Set up logging
logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), '..', 'logs', 'model.log'), 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='w', 
                    level=logging.INFO, )

slice_logger = logging.getLogger('slice_metrics')
slice_logger.setLevel(logging.INFO)
slice_logger.addHandler(logging.FileHandler(
    filename=os.path.join(os.path.dirname(__file__), '..', 'logs', 'slice_output.txt'),
    mode='w',
))

# Loads config
with open(os.path.join(os.path.dirname(__file__), 'model_config.yaml'), 'r') as fp:
    config = yaml.safe_load(fp)


if __name__ == '__main__':
    CAT_FEATURES = config['categorical_features']

    # Loads census data
    data_filename = 'census.csv'
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'artifact', 'data')
    data_path = os.path.join(data_dir, data_filename)

    raw_df = pd.read_csv(data_path)
    df = preprocess_data(raw_df, dest_path=data_dir)

    train, test = train_test_split(df, test_size=0.20, random_state=config['random_seed'])

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label='salary', training=False, encoder=encoder, lb=lb
    )

    logging.info(f"Model parameters: {config['random_forest']}")

    # Train and save a model.
    model = train_model(X_train, y_train, config['random_forest'])

    # test predictions
    y_test_preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_test_preds)
    logging.info(f'Overall model predictions - precision: {precision}, recall: {recall}, fbeta: {fbeta}')

    # export artifacts
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'artifact', 'model')

    # model export
    model_dest_path = os.path.join(model_dir, 'random_forest.pkl')
    save_artifact(model, model_dest_path)

    # encoder and labelbinarizer export for inference
    save_artifact(encoder, os.path.join(model_dir, 'oh_encoder.pkl'))
    save_artifact(lb, os.path.join(model_dir, 'label_binarizer.pkl'))

    # compute metrics based on slice
    clean_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'artifact', 'data', 'clean_census.csv'))

    for slice in ['race', 'sex']:
        slice_metrics = compute_slice_metrics(clean_df, slice)

        for feature_val, metrics in slice_metrics.items():
            slice_logger.info(f"`{slice}` category -> precision: {metrics['precision']:.3f}, recall: {metrics['recall']:.3f}, fbeta: {metrics['fbeta']:.3f} -- {feature_val}.")
        slice_logger.info('\n')