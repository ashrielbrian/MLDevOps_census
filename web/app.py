from fastapi import FastAPI
from typing import Union, Optional
from pydantic import BaseModel

class InferenceInput(BaseModel):
    pass

app = FastAPI()

@app.get('/')
def welcome():
    return "Hello from Census Predictor!"

@app.post('/inference')
def inference():
    pass