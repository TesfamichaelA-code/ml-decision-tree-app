# Titanic Survival Prediction - Decision Tree

A complete end-to-end machine learning project using **Decision Tree Classifier** to predict Titanic passenger survival.

This project is an academic ML lab submission demonstrating the full ML workflow from data exploration to model deployment.

---

## Project Structure

```
ml-decision-tree-app/
├── README.md                      # Project documentation
├── .gitignore                     # Git ignore file
├── backend/                       # FastAPI backend
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── schemas.py                 # Pydantic schemas
│   ├── config.py                  # Configuration settings
│   └── requirements.txt           # Backend dependencies
├── frontend/                      # Streamlit frontend
│   ├── app.py                     # Streamlit application
│   └── requirements.txt           # Frontend dependencies
├── model/
│   └── decision_tree_model.joblib # Trained Decision Tree model
└── notebook/
    └── decision_tree_pipeline.ipynb # ML pipeline notebook
```

---

## Features

### Decision Tree Model

- **Data Loading**: Titanic dataset from public URL
- **Exploratory Data Analysis**: Comprehensive visualizations
- **Data Cleaning**: Missing value handling, feature selection
- **Feature Engineering**: FamilySize, IsAlone, AgeGroup, FarePerPerson
- **sklearn Pipeline**: Preprocessing + Model in a single pipeline
- **Model Training**: Decision Tree Classifier with balanced class weights
- **Evaluation**: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
- **Tree Visualization**: Visual representation of decision rules
- **Model Export**: Saved as `decision_tree_model.joblib`

### FastAPI Backend

- RESTful API for predictions
- Endpoints: `GET /`, `GET /health`, `POST /predict`, `POST /predict/batch`, `GET /model/info`
- CORS enabled for all origins
- Pydantic validation
- OpenAPI documentation (`/docs`)

### Streamlit Frontend

- Interactive user interface
- Real-time predictions
- Configurable API URL
- Visual probability displays

---

## Quick Start

### Prerequisites

```bash
# Python 3.9 or higher
python --version
```

---

## Running the Notebook Locally

### Step 1: Create and activate a virtual environment

```bash
cd ml-decision-tree-app
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 2: Install notebook dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### Step 3: Run the Decision Tree notebook

```bash
jupyter notebook notebook/decision_tree_pipeline.ipynb
```

Or open in VS Code with the Jupyter extension.

### Step 4: Run all cells

Execute all cells from top to bottom. The trained model will be saved to `model/decision_tree_model.joblib`.

---

## Running the Backend Locally

```bash
# Navigate to backend directory
cd ml-decision-tree-app/backend
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
# Run the server
uvicorn main:app --host 0.0.0.0 --port 10000 --reload
```

### Test the API:

- Open browser: `http://localhost:10000`
- API docs: `http://localhost:10000/docs`
- Health check: `http://localhost:10000/health`

---

## Running the Frontend Locally

```bash
# Navigate to frontend directory
cd ml-decision-tree-app/frontend
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
# Run Streamlit
streamlit run app.py
```

### Configure API URL (optional):

By default, the frontend connects to `http://localhost:10000`. To change this:

```bash
API_URL=https://your-api-url.com streamlit run app.py
```

---

## Deployment Instructions

### Deploying Backend to Render

1. Create a new Web Service on [Render](https://render.com)
2. Connect your GitHub repository
3. Configure the service:
   - Name: `ml-decision-tree-api`
   - Root Directory: `backend`
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port 10000`
4. Deploy
5. Note your API URL: `https://ml-decision-tree-api.onrender.com`

### Deploying Frontend to Streamlit Cloud

1. Go to [Streamlit Cloud](https://share.streamlit.io)
2. Create a new app
3. Configure:
   - Repository: Your GitHub repo
   - Branch: `main`
   - Main file path: `frontend/app.py`
4. Advanced Settings - Secrets:
   ```toml
   API_URL = "https://ml-decision-tree-api.onrender.com"
   ```
5. Deploy

---

## Model Performance

| Metric    | Training | Test |
| --------- | -------- | ---- |
| Accuracy  | ~80%     | ~75% |
| Precision | ~73%     | ~67% |
| Recall    | ~75%     | ~72% |
| F1-Score  | ~74%     | ~69% |
| ROC-AUC   | ~84%     | ~80% |

_Note: Actual values may vary slightly based on random state._

---

## API Endpoints

### GET `/`

Root endpoint - API health check

**Response:**

```json
{
  "status": "healthy",
  "message": "Titanic Survival Prediction API (Decision Tree) is running",
  "model_loaded": true
}
```

### GET `/health`

Health check endpoint

### POST `/predict`

Predict survival for a passenger

**Request Body:**

```json
{
  "pclass": 1,
  "sex": "female",
  "age": 25,
  "sibsp": 1,
  "parch": 0,
  "fare": 100.0,
  "embarked": "S"
}
```

**Response:**

```json
{
  "survived": true,
  "survival_probability": 0.85,
  "confidence": 85.0,
  "message": "Survived"
}
```

### POST `/predict/batch`

Predict survival for multiple passengers

### GET `/model/info`

Get information about the loaded model

---

## Using the Trained Model

```python
import joblib
import pandas as pd
# Load the Decision Tree model
model = joblib.load('model/decision_tree_model.joblib')
# Create new passenger data
new_passenger = pd.DataFrame({
    'Pclass': [1],
    'Sex': ['female'],
    'Age': [25],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [100.0],
    'Embarked': ['S'],
    'FamilySize': [2],
    'IsAlone': [0],
    'FarePerPerson': [50.0],
    'AgeGroup': ['YoungAdult']
})
# Make prediction
prediction = model.predict(new_passenger)
probability = model.predict_proba(new_passenger)
print(f"Survival Prediction: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")
print(f"Probability of Survival: {probability[0][1]*100:.2f}%")
```

---

## Why Decision Tree?

| Feature             | Advantage                               |
| ------------------- | --------------------------------------- |
| Interpretability    | Easy to visualize and explain decisions |
| No Feature Scaling  | Handles different scales naturally      |
| Non-linear Patterns | Captures complex relationships          |
| Feature Importance  | Built-in importance scores              |
| Fast Inference      | Quick predictions                       |

---

## Key Findings

- **Sex** is the most important predictor (females had higher survival rates)
- **Passenger Class** significantly affected survival (1st class had better odds)
- **Family Size** had a non-linear effect on survival
- **Age** played a role, with children having better survival rates

---

## License

This project is for academic purposes - ML Lab Submission.

---

## Author

- Academic ML Lab Project

---

## Acknowledgments

- Titanic Dataset: [Kaggle](https://www.kaggle.com/c/titanic)
- scikit-learn Documentation
- FastAPI Documentation
- Streamlit Documentation
