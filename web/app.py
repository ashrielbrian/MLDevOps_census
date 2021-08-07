import os

""" 
    This is used by Heroku deployment to pull
    the necessary artifacts using dvc. Must be
    executed prior to importing any other modules.
"""
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

from io import StringIO
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from model.ml.model import RFModel
from model.ml.data import process_data
from .census_class import CensusClass



rf_model = RFModel()
app = FastAPI()


@app.get('/')
def welcome():
    return "Hello from Census Predictor!"


@app.post('/batch_inference')
async def batch_inference(csv_file: UploadFile = File(...)):

    try:
        # loads the uploaded file
        str_buf = StringIO(str(csv_file.file.read(), 'utf-8'))
        df = pd.read_csv(str_buf, encoding='utf-8')
        df.columns = [col.strip().replace('-', '_') for col in df.columns]

    except Exception as e:
        print(str(e))
        return {'success': False, 'results': None,
                'error': 'Failed to parse uploaded file.'}

    try:
        # process and perform inference
        X, _, _, _ = process_data(df,
                                  categorical_features=rf_model.CAT_FEATURES,
                                  training=False,
                                  encoder=rf_model.encoder,
                                  lb=rf_model.binarizer)
        y_preds = rf_model.inference(X)
    except Exception as e:
        print(str(e))
        return {'success': False, 'results': None,
                'error': 'Failed to perform inference.'}

    return {'success': True, 'results': y_preds, 'error': None}


@app.post('/inference')
async def inference(individual: CensusClass):

    try:
        indv_dict = {k: [v] for k, v in individual.dict().items()}
        df = pd.DataFrame.from_dict(indv_dict)
    except Exception as e:
        print(str(e))
        return {'success': False, 'results': None,
                'error': 'Failed to parse JSON body.'}

    try:
        # process and perform inference
        X, _, _, _ = process_data(df,
                                  categorical_features=rf_model.CAT_FEATURES,
                                  training=False,
                                  encoder=rf_model.encoder,
                                  lb=rf_model.binarizer)
        y_preds = rf_model.inference(X)
    except Exception as e:
        print(str(e))
        return {'success': False, 'results': None,
                'error': 'Failed to perform inference.'}

    return {'success': True, 'results': y_preds, 'error': None}
