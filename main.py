from fastapi import FastAPI
import pickle
import pandas as pd

from  pydantic import BaseModel


app = FastAPI()


@app.get("/")

def home():
    
    return {"message": "welcome!"}



 # Load the trained model

class InsuranceDataSchema(BaseModel):
    Age: int
    Driving_License: int
    Region_Code: float
    Previously_Insured: int
    Annual_Premium: float
    Policy_Sales_Channel: float
    Vintage: int
    

@app.post("/predict/NB/")
def NB_predict(predict_values:InsuranceDataSchema):
    load_model = pickle.load(open("NB_model.pkl","rb"))

    # predict_values -> gelen verileri bir dataframe'e çeviriyoruz.
    # predict_values.dict().values() -> gelen verilerin değerlerini alıyoruz.
    df = pd.DataFrame(
        [predict_values.model_dump().values()],
        columns=predict_values.model_dump().keys()
        )


    # print(df)
    predict =load_model.predict(df)
    return {"Predict":int(predict[0])}

@app.post("/predict/qgb/")
def qgb_predict(predict_values:InsuranceDataSchema):
    load_model = pickle.load(open("qgb_model.pkl","rb"))
    df = pd.DataFrame([predict_values.model_dump().values()],columns=predict_values.model_dump().keys())


    predict =load_model.predict(df)
    return {"Predict":int(predict[0])}