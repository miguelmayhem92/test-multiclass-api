import json
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request, FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.templating import Jinja2Templates

from loguru import logger
from multiclass_model import __version__ as model_version
from multiclass_model.predict import make_prediction

from application import __version__, schemas
from application.config import settings

import jinja2

from multiclass_model.config.core import config
from multiclass_model.processing.data_manager import load_dataset

api_router = APIRouter()
templates = Jinja2Templates(directory="templates")


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


@api_router.get("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    """
    Make predictions with the TID model
    """

    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    # Advanced: You can improve performance of your API by rewriting the
    # `make prediction` function to be async and using await here.
    logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction(input_data=input_df.replace({np.nan: None}))
    logger.info(f"Prediction results: {results.get('predictions')}")

    return results

@api_router.get("/display_test_data")
async def predict_using_testData(request: Request, 
    From: Optional[str]= 0, To: Optional[str] = 5):

    test_data = load_dataset(file_name=config.app_config.test_data_file)
    predictions = make_prediction(input_data=test_data.replace({np.nan: None})).get('predictions')
    test_data['Prediction'] = predictions

    display_data = test_data.iloc[int(From): int(To),:]

    return templates.TemplateResponse(
        'df_representation.html',
        {'request': request, 'data': display_data.to_html()}
    )
