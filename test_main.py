from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

def test_rf_predict():
    payload = {
        "Gender": "Male",
        "Age": 28,
        "Driving_License": 1 ,
        "Region_Code": 11,
        "Previously_Insured": 1,
        "Vehicle_Age": 14,
        "Vehicle_Damage": "No",
        "Annual_Premium": 12000,
        "Policy_Sales_Channel": "float",
        "Vintage": 435,
        "Response": 1
    }
    response = client.post("/predict/rf/",json=payload)

    assert response.status_code == 200

    # response.json() -> gelen veriyi json formatına çevirir.
    # assert "Predict" in response.json() -> gelen veride "Predict" kelimesi var mı diye kontrol eder.
    assert "Predict" in response.json()

    # assert response.json()["Predict"] in [0,1] -> gelen verinin içindeki "Predict" key'ine karşılık gelen değer 0 veya 1 mi diye kontrol eder.
    assert response.json()["Predict"] in [0,1]

def test_qgb_predict():
    payload = {
        "Gender": "Male",
        "Age": 28,
        "Driving_License": 1 ,
        "Region_Code": 11,
        "Previously_Insured": 1,
        "Vehicle_Age": 14,
        "Vehicle_Damage": "No",
        "Annual_Premium": 12000,
        "Policy_Sales_Channel": "float",
        "Vintage": 435,
        "Response": 1
        }
    response = client.post("/pridict/qgb/",json=payload)

    assert response.status_code == 200
    assert "Predict" in response.json()
    assert response.json()["Predict"] in [0,1]