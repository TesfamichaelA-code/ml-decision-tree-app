"""
Configuration settings for the Decision Tree backend
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Model path
MODEL_PATH = BASE_DIR / "model" / "decision_tree_model.joblib"

# API Settings
API_TITLE = "Titanic Survival Prediction API"
API_DESCRIPTION = "A REST API for predicting Titanic passenger survival using Decision Tree Classifier"
API_VERSION = "1.0.0"

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 10000))

# CORS settings
CORS_ORIGINS = ["*"]  # Allow all origins for deployment flexibility
