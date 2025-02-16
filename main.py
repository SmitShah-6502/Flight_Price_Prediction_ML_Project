from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel

# Load the trained model
model_filename = "flight_price_model.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Initialize FastAPI app
app = FastAPI()

# Define request model
class FlightInput(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Total_Stops: int
    Duration: float
    Additional_Info: str

# Define prediction endpoint
@app.post("/predict")
def predict_price(input_data: FlightInput):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data.dict()])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        return {"predicted_price": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Root endpoint
@app.get("/")
def home():
    return {"message": "Flight Price Prediction API is running!"}
