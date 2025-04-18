from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import mode
from typing import List
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import uvicorn
import re

# Initialize FastAPI app
app = FastAPI(title="Heart Disease Prediction API")
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("API Key not found! Please set DEEPSEEK_API_KEY in .env file.")

# Initialize DeepSeek model
model_name = "deepseek-r1-distill-llama-70b"
deepseek = ChatGroq(api_key=api_key, model_name=model_name)
parser = StrOutputParser()
deepseek_chain = deepseek | parser

# Load templates
templates = Jinja2Templates(directory="templates")
# Load pre-trained models and scaler
try:
    model_lr = joblib.load('models/LogisticRegression.pkl')
    model_dt = joblib.load('models/DecisionTreeClassifier.pkl')
    model_rf = joblib.load('models/RandomForestClassifier.pkl')
    model_svm = joblib.load('models/SVC.pkl')
    model_knn = joblib.load('models/KNeighborsClassifier.pkl')
    scaler = joblib.load('models/scaler.pkl')
except Exception as e:
    raise Exception(f"Error loading models or scaler: {str(e)}")

# Initialize LabelEncoder
le = LabelEncoder()

# Define the input data model for a single patient (user-submitted data)
class PatientData(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str

# Define a batch input model for multiple patients
class PatientDataBatch(BaseModel):
    patients: List[PatientData]

# Helper function to preprocess user-submitted data
def preprocess_data(df):
    df_processed = df.copy()
    
    # Encode categorical variables
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for col in categorical_columns:
        if col in df_processed.columns:
            df_processed[col] = le.fit_transform(df_processed[col])
    
    # Ensure all expected columns are present (based on scaler's training data)
    expected_columns = [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 
            'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
        ]
    for col in expected_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    # Reorder columns to match scaler's expectation
    df_processed = df_processed[expected_columns]
    
    # Scale numerical data
    scaled_data = scaler.transform(df_processed)
    
    return scaled_data

# Helper function to perform ensemble prediction
def ensemble_predict(scaled_data):
    # Predict with each model
    pred_lr = model_lr.predict(scaled_data)
    pred_dt = model_dt.predict(scaled_data)
    pred_rf = model_rf.predict(scaled_data)
    pred_svm = model_svm.predict(scaled_data)
    pred_knn = model_knn.predict(scaled_data)
    
    # Collect predictions
    predictions = np.array([pred_lr, pred_dt, pred_rf, pred_svm, pred_knn])
    
    # Perform majority voting
    final_pred, _ = mode(predictions, axis=0)
    final_pred = final_pred.flatten()
    
    return final_pred
# Helper function to clean DeepSeek response
def clean_response(response: str) -> str:
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()


@app.post("/chat")
async def chat(message: str = Form(...)):
    try:
        if not message.strip():
            return JSONResponse(content={"response": "Vui lòng nhập câu hỏi."}, status_code=400)

        answer = deepseek_chain.invoke(message)
        clean_answer = clean_response(answer)
        return JSONResponse(content={"response": clean_answer})

    except Exception as e:
        return JSONResponse(content={"response": f"Lỗi hệ thống: {str(e)}"}, status_code=500)
# API endpoint for single patient prediction
@app.post("/predict")
async def predict(data: PatientData):
    try:
        # Convert user-submitted JSON data to DataFrame
        df = pd.DataFrame([data.dict()])
        
        # Preprocess data
        scaled_data = preprocess_data(df)
        
        # Perform ensemble prediction
        prediction = ensemble_predict(scaled_data)
        
        # Return result
        return {
            "prediction": int(prediction[0]),
            "interpretation": "Heart disease" if prediction[0] == 1 else "No heart disease"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# API endpoint for batch prediction
@app.post("/predict_batch")
async def predict_batch(data: PatientDataBatch):
    try:
        # Convert user-submitted list of patients to DataFrame
        df = pd.DataFrame([patient.dict() for patient in data.patients])
        
        # Preprocess data
        scaled_data = preprocess_data(df)
        
        # Perform ensemble prediction
        predictions = ensemble_predict(scaled_data)
        
        # Prepare results
        results = [
            {
                "patient_index": idx,
                "prediction": int(pred),
                "interpretation": "Heart disease" if pred == 1 else "No heart disease"
            }
            for idx, pred in enumerate(predictions)
        ]
        
        return {"predictions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Root endpoint for health check
@app.get("/")
async def root():
    return {"message": "Heart Disease Prediction API is running"}