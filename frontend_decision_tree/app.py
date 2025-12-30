"""
Streamlit Frontend Application for Titanic Survival Prediction

This application provides a user-friendly interface to interact with the
FastAPI backend for predicting Titanic passenger survival using Decision Tree.
"""

import streamlit as st
import requests
import os
from typing import Dict, Any

# =============================================================================
# Configuration
# =============================================================================

# FastAPI backend URL - configurable via environment variable
# Default to local development URL, can be overridden for deployment
API_BASE_URL = os.getenv("API_URL", "http://localhost:10000")
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Titanic Survival Predictor - Decision Tree",
    page_icon="🌳",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Helper Functions
# =============================================================================


def check_api_health() -> bool:
    """Check if the FastAPI backend is running and healthy."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def make_prediction(passenger_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send passenger data to the FastAPI backend for prediction.
    
    Args:
        passenger_data: Dictionary containing passenger features
        
    Returns:
        Prediction response from the API
    """
    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            json=passenger_data,
            timeout=10
        )
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to the API server. Please ensure the FastAPI backend is running."}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Please try again."}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"API error: {e.response.text}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {str(e)}"}


# =============================================================================
# Sidebar - API Status and Instructions
# =============================================================================

with st.sidebar:
    st.header("🔌 API Status")
    
    # Display current API URL
    st.text(f"API URL: {API_BASE_URL}")
    
    # Check API health
    if check_api_health():
        st.success("✅ Backend API is running")
    else:
        st.error("❌ Backend API is not available")
        st.info(
            "Start the backend with:\n\n"
            "```bash\n"
            "cd backend_decision_tree\n"
            "uvicorn main:app --host 0.0.0.0 --port 10000\n"
            "```"
        )
    
    st.divider()
    
    # Instructions
    st.header("📋 Instructions")
    st.markdown("""
    1. Fill in all passenger details
    2. Click **Predict Survival**
    3. View the prediction results
    
    **Feature Descriptions:**
    - **Pclass**: Passenger class (1st, 2nd, 3rd)
    - **Sex**: Gender of passenger
    - **Age**: Age in years
    - **SibSp**: Siblings/spouses aboard
    - **Parch**: Parents/children aboard
    - **Fare**: Ticket price
    - **Embarked**: Port of embarkation
    """)
    
    st.divider()
    
    # About section
    st.header("ℹ️ About")
    st.markdown("""
    This application uses a **Decision Tree Classifier** 
    trained on the Titanic dataset to predict passenger survival.
    
    The model considers factors like passenger class, 
    age, gender, and family size to make predictions.
    
    **Model Advantages:**
    - Easy to interpret
    - Visual decision rules
    - Handles non-linear patterns
    """)


# =============================================================================
# Main Content
# =============================================================================

# Title and description
st.title("🌳 Titanic Survival Predictor")
st.markdown("### Using Decision Tree Classifier")
st.markdown("""
Predict whether a passenger would have survived the Titanic disaster 
based on their characteristics. Enter the passenger details below and 
click **Predict Survival** to see the results.
""")

st.divider()

# =============================================================================
# Input Form
# =============================================================================

st.subheader("👤 Passenger Details")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Passenger Class
    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1st Class 🥇", 2: "2nd Class 🥈", 3: "3rd Class 🥉"}[x],
        help="The class of the passenger's ticket"
    )
    
    # Sex
    sex = st.selectbox(
        "Sex",
        options=["male", "female"],
        format_func=lambda x: f"{'👨 Male' if x == 'male' else '👩 Female'}",
        help="Gender of the passenger"
    )
    
    # Age
    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=120.0,
        value=30.0,
        step=1.0,
        help="Age of the passenger in years"
    )
    
    # Fare
    fare = st.number_input(
        "Fare ($)",
        min_value=0.0,
        max_value=1000.0,
        value=32.0,
        step=1.0,
        help="Price of the ticket"
    )

with col2:
    # Siblings/Spouses
    sibsp = st.number_input(
        "Siblings/Spouses Aboard",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Number of siblings or spouses aboard"
    )
    
    # Parents/Children
    parch = st.number_input(
        "Parents/Children Aboard",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Number of parents or children aboard"
    )
    
    # Embarked
    embarked = st.selectbox(
        "Port of Embarkation",
        options=["S", "C", "Q"],
        format_func=lambda x: {
            "S": "🚢 Southampton",
            "C": "⚓ Cherbourg",
            "Q": "🏴 Queenstown"
        }[x],
        help="The port where the passenger boarded"
    )

st.divider()

# =============================================================================
# Prediction Button and Results
# =============================================================================

# Center the predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button(
        "🔮 Predict Survival",
        type="primary",
        use_container_width=True
    )

# Handle prediction
if predict_button:
    # Prepare passenger data
    passenger_data = {
        "pclass": pclass,
        "sex": sex,
        "age": age,
        "sibsp": sibsp,
        "parch": parch,
        "fare": fare,
        "embarked": embarked
    }
    
    # Show loading spinner
    with st.spinner("Making prediction..."):
        result = make_prediction(passenger_data)
    
    st.divider()
    
    if result["success"]:
        prediction = result["data"]
        
        # Display results in a styled container
        st.subheader("📊 Prediction Results")
        
        # Main prediction result with color coding
        if prediction["survived"]:
            st.success(f"🎉 **Prediction: SURVIVED**")
        else:
            st.error(f"💔 **Prediction: DID NOT SURVIVE**")
        
        # Detailed metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric(
                label="Survival Probability",
                value=f"{prediction['survival_probability']:.1%}"
            )
        
        with metrics_col2:
            st.metric(
                label="Confidence",
                value=f"{prediction['confidence']:.1f}%"
            )
        
        with metrics_col3:
            st.metric(
                label="Prediction",
                value="Survived ✅" if prediction["survived"] else "Not Survived ❌"
            )
        
        # Show probability bar
        st.progress(
            prediction["survival_probability"],
            text=f"Survival Probability: {prediction['survival_probability']:.1%}"
        )
        
        # Additional context
        with st.expander("📝 View Input Summary"):
            st.json(passenger_data)
            
    else:
        # Display error message
        st.error(result["error"])


# =============================================================================
# Footer
# =============================================================================

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        Built with Streamlit | Model: Decision Tree Classifier | Dataset: Titanic<br>
        Academic ML Lab Project
    </div>
    """,
    unsafe_allow_html=True
)
