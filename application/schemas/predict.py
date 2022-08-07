from typing import Any, List, Optional

from pydantic import BaseModel
from multiclass_model.processing.validation import InvoiceDataSchema


class PredictionResults(BaseModel):
    version: str
    predictions: Optional[List[float]]


class MultipleDataInputs(BaseModel):
    inputs: List[InvoiceDataSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Inv_Id": 15000,
                        "Vendor_Code": "VENDOR-0000",
                        "GL_Code": "GL-6100410",
                        "Inv_Amt": 85.25,
                        "Item_Description": "This is not a description"

                    }
                ]
            }
        }
