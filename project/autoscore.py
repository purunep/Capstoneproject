import json
import numpy as np
import os
import pickle
import joblib
from azureml.core.model import Model
import pandas as pd

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automlmodel.pkl')
    model = joblib.load(model_path)


def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = pd.DataFrame.from_dict(data)
        # make prediction
        mypredict = model.predict(data)
        # you can return any data type as long as it is JSON-serializable
        return mypredict.tolist()
    except Exception as ex:
        error = str(ex)
        return error
