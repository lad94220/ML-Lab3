import uvicorn 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from utils_pkl import predict_all, get_chart_arrays

app = FastAPI(title="Protein Prediction API")

# Enable CORS for client communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_length=9, max_length=9, description="Array of 9 features (f1-f9)")

# Response model
class PredictionArrays(BaseModel):
    y_truth: List[float]
    y_pred: List[float]

class PredictResponse(BaseModel):
    predictions: dict  # {'GAR': float, 'GAR-EXP': float, 'MAE': float}

@app.get("/")
def read_root():
    return {
        "message": "Protein Prediction API",
        "endpoints": {
            "/predict": "POST - Predict protein properties using 9 features",
            "/chart-data": "GET - Get chart data for visualization (300 samples)",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/chart-data")
def get_chart_data():
    """
    Get chart data for visualization (300 pre-sampled points)
    
    Response:
    {
        "arrays": {
            "GAR": {"y_truth": [...], "y_pred": [...]},
            "GAR-EXP": {"y_truth": [...], "y_pred": [...]},
            "MAE": {"y_truth": [...], "y_pred": [...]}
        }
    }
    """
    try:
        arrays = get_chart_arrays()
        return {"arrays": arrays}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data: {str(e)}")

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict protein properties using all 3 loss functions (GAR, GAR-EXP, MAE)
    
    Request body:
    {
        "features": [f1, f2, f3, f4, f5, f6, f7, f8, f9]
    }
    
    Response:
    {
        "predictions": {
            "GAR": <prediction_value>,
            "GAR-EXP": <prediction_value>,
            "MAE": <prediction_value>
        }
    }
    """
    try:
        # Validate features
        if len(request.features) != 9:
            raise HTTPException(status_code=400, detail="Exactly 9 features are required (f1-f9)")
        
        # Check if all features are positive
        if any(f <= 0 for f in request.features):
            raise HTTPException(status_code=400, detail="All features must be greater than 0")
        
        # Get predictions from all models
        results = predict_all(request.features)
        
        # Return only predictions, not arrays
        return {"predictions": results['predictions']}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

