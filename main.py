from fastapi import FastAPI
import pickle
import pandas as pd

from  pydantic import BaseModel


app = FastAPI()


@app.get("/")

def home():
    
    return {"message": "welcome!"}





class InsuranceDataSchema(BaseModel):
    Gender: str
    Age: int
    Driving_License: int
    Region_Code: float
    Previously_Insured: int
    Vehicle_Age: str
    Vehicle_Damage: str
    Annual_Premium: float
    Policy_Sales_Channel: float
    Vintage: int
    Response: int
