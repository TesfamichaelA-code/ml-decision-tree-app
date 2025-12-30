"""
FastAPI backend for Titanic Survival Prediction using Decision Tree Classifier
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os

from schemas import (
    PassengerInput,
    PassengerBatchInput,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict Titanic passenger survival using Decision Tree Classifier",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "decision_tree_model.joblib")

model = None

def load_model():
    """Load the trained Decision Tree model"""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def categorize_age(age: float) -> str:
    """Categorize age into groups matching the training data"""
    if age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teenager'
    elif age <= 35:
        return 'YoungAdult'
    elif age <= 55:
        return 'MiddleAged'
    else:
        return 'Senior'


def prepare_features(passenger: PassengerInput) -> pd.DataFrame:
    """Prepare features for prediction matching the training pipeline"""
    # Calculate derived features
    family_size = passenger.sibsp + passenger.parch + 1
    is_alone = 1 if family_size == 1 else 0
    fare_per_person = passenger.fare / family_size if family_size > 0 else passenger.fare
    age_group = categorize_age(passenger.age)
    
    # Create DataFrame with all features
    data = {
        'Pclass': [passenger.pclass],
        'Sex': [passenger.sex.lower()],
        'Age': [passenger.age],
        'SibSp': [passenger.sibsp],
        'Parch': [passenger.parch],
        'Fare': [passenger.fare],
        'Embarked': [passenger.embarked.upper()],
        'FamilySize': [family_size],
        'IsAlone': [is_alone],
        'FarePerPerson': [fare_per_person],
        'AgeGroup': [age_group]
    }
    
    return pd.DataFrame(data)


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API health check"""
    return HealthResponse(
        status="healthy",
        message="Titanic Survival Prediction API (Decision Tree) is running",
        model_loaded=model is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="API is running",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(passenger: PassengerInput):
    """Predict survival for a single passenger"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features
        features = prepare_features(passenger)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        survival_prob = float(probabilities[1])
        survived = bool(prediction == 1)
        confidence = float(max(probabilities) * 100)
        
        return PredictionResponse(
            survived=survived,
            survival_probability=round(survival_prob, 4),
            confidence=round(confidence, 2),
            message="Survived" if survived else "Did Not Survive"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: PassengerBatchInput):
    """Predict survival for multiple passengers"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        survived_count = 0
        
        for passenger in batch.passengers:
            features = prepare_features(passenger)
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            survival_prob = float(probabilities[1])
            survived = bool(prediction == 1)
            confidence = float(max(probabilities) * 100)
            
            if survived:
                survived_count += 1
            
            predictions.append(PredictionResponse(
                survived=survived,
                survival_probability=round(survival_prob, 4),
                confidence=round(confidence, 2),
                message="Survived" if survived else "Did Not Survive"
            ))
        
        total_count = len(predictions)
        survival_rate = (survived_count / total_count * 100) if total_count > 0 else 0
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=total_count,
            survived_count=survived_count,
            survival_rate=round(survival_rate, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        classifier = model.named_steps.get('classifier', None)
        
        info = {
            "model_type": "Decision Tree Classifier",
            "pipeline_steps": list(model.named_steps.keys()),
            "model_loaded": True
        }
        
        if classifier:
            info["tree_depth"] = classifier.get_depth()
            info["n_leaves"] = classifier.get_n_leaves()
            info["n_features"] = classifier.n_features_in_
        
        return info
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
