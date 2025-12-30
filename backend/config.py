"""
Configuration settings for the Decision Tree API
"""

import os

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", "../model/decision_tree_model.joblib")

# API configuration
API_TITLE = "Titanic Survival Prediction API"
API_DESCRIPTION = "Predict Titanic passenger survival using Decision Tree Classifier"
API_VERSION = "1.0.0"

# Server configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 10000))
