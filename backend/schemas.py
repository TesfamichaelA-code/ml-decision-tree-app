"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PassengerInput(BaseModel):
    """Input schema for a single passenger prediction"""
    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    sex: str = Field(..., description="Sex of the passenger ('male' or 'female')")
    age: float = Field(..., ge=0, le=120, description="Age in years")
    sibsp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, description="Passenger fare")
    embarked: str = Field(..., description="Port of embarkation (C, Q, or S)")

    class Config:
        json_schema_extra = {
            "example": {
                "pclass": 1,
                "sex": "female",
                "age": 25,
                "sibsp": 1,
                "parch": 0,
                "fare": 100.0,
                "embarked": "S"
            }
        }


class PassengerBatchInput(BaseModel):
    """Input schema for batch passenger predictions"""
    passengers: List[PassengerInput]


class PredictionResponse(BaseModel):
    """Response schema for a single prediction"""
    survived: bool
    survival_probability: float
    confidence: float
    message: str


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[PredictionResponse]
    total_count: int
    survived_count: int
    survival_rate: float


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    message: str
    model_loaded: bool
