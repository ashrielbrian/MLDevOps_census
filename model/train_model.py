# Script to train machine learning model.
import os
from joblib import dump
import yaml
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data, preprocess_data

# Set up logging
logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), '..', 'logs', 'model.log'), 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode='a', 
                    level=logging.INFO, )

# Loads config
with open('./model_config.yaml', 'r') as fp:
    config = yaml.safe_load(fp)


# Loads census data
data_filename = 'census.csv'
data_dir = os.path.join(os.path.dirname(__file__), '..', 'artifact', 'data')
data_path = os.path.join(data_dir, data_filename)

raw_df = pd.read_csv(data_path)
df = preprocess_data(raw_df, dest_path=data_dir)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(df, test_size=0.20, random_state=config['random_seed'])

cat_features = config['categorical_features']
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label='salary', training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train, config['random_forest'])

# test predictions
y_test_preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_test_preds)
logging.info(f'Model predictions gave precision: {precision}, recall: {recall}, fbeta: {fbeta}')

# export model
model_dest_path = os.path.join(os.path.dirname(__file__), '..', 'artifact', 'model', 'random_forest.pkl')
dump(model, model_dest_path)
