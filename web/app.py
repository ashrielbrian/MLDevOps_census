import os
from io import StringIO
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from typing import Union, Optional
from pydantic import BaseModel
from model.ml.model import RFModel
from model.ml.data import process_data
from .census_class import CensusClass

class InferenceInput(BaseModel):
    pass

# model_artifact_dir = os.path.join(os.path.dirname(__file__), '..', 'artifact', 'model')
# model       = load_artifact(os.path.join(model_artifact_dir, 'random_forest.pkl'))
# binarizer   = load_artifact(os.path.join(model_artifact_dir, 'label_binarizer.pkl'))
# encoder     = load_artifact(os.path.join(model_artifact_dir, 'oh_encoder.pkl'))

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
        return {'success': False, 'results': None, 'error': 'Failed to parse uploaded file.'}

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
        return {'success': False, 'results': None, 'error': 'Failed to perform inference.'}
    
    return {'success': True, 'results': y_preds, 'error': None}

@app.post('/inference')
async def inference(individual: CensusClass):

    try:
        indv_dict = {k: [v] for k,v in individual.dict().items()}
        df = pd.DataFrame.from_dict(indv_dict)
    except Exception as e:
        print(str(e))
        return {'success': False, 'results': None, 'error': 'Failed to parse JSON body.'}
    
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
        return {'success': False, 'results': None, 'error': 'Failed to perform inference.'}
    
    return {'success': True, 'results': y_preds, 'error': None}
    