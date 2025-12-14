import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from models import MLP  # Import MLP from models.py for pickle compatibility


# 1. Load scaler
scaler = None
try:
    with open('data/protein_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("[OK] Loaded scaler successfully")
except Exception as e:
    print(f"[ERROR] Error loading scaler: {e}")

# 2. Cấu hình thiết bị (App thường chạy CPU)
device = torch.device('cpu') 

# 3. Load all 3 models from .pkl files (GAR, GAR-EXP, MAE)
models = {}
model_configs = {
    'GAR': 'data/FINAL_DEPLOYMENT_MODEL_protein_GAR_20251213_225421.pkl',
    'GAR-EXP': 'data/FINAL_DEPLOYMENT_MODEL_protein_GAR-EXP_20251213_231705.pkl',
    'MAE': 'data/FINAL_DEPLOYMENT_MODEL_protein_MAE_20251213_230400.pkl'
}

for loss_name, model_path in model_configs.items():
    try:
        # Load the entire model object using pickle.load
        # The .pkl files were saved with pickle.dump()
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Move to device and set to eval mode
        model.to(device)
        model.eval()
        
        models[loss_name] = model
        print(f"[OK] Loaded {loss_name} model from .pkl successfully")
    except Exception as e:
        print(f"[ERROR] Error loading {loss_name} model from .pkl: {e}")
        print(f"       Make sure the model was saved with the MLP class definition available")

# 4. Load pre-sampled arrays (300 random samples)
sample_arrays = {}
try:
    # Load ground truth (shared across all models)
    y_truth = np.load('data/y_truth_sample.npy')
    
    # Load predictions for each model
    y_pred_gar = np.load('data/y_pred_gar_sample.npy')
    y_pred_mae = np.load('data/y_pred_mae_sample.npy')
    y_pred_gar_exp = np.load('data/y_pred_gar_exp_sample.npy')
    
    # Store in dictionary
    sample_arrays = {
        'GAR': {
            'y_truth': y_truth.tolist(),
            'y_pred': y_pred_gar.tolist()
        },
        'MAE': {
            'y_truth': y_truth.tolist(),
            'y_pred': y_pred_mae.tolist()
        },
        'GAR-EXP': {
            'y_truth': y_truth.tolist(),
            'y_pred': y_pred_gar_exp.tolist()
        }
    }
    
    print(f"[OK] Loaded sample arrays ({len(y_truth)} samples)")
except Exception as e:
    print(f"[ERROR] Error loading sample arrays: {e}")

# 5. Dự đoán với một model cụ thể
def predict_single(model, input_data):
    """
    Predict using a single model
    input_data: list of 9 features [f1, f2, ..., f9]
    """
    # Convert to numpy array and reshape
    x = np.array(input_data).reshape(1, -1)
    
    # Transform using scaler (BAT BUOC - REQUIRED)
    if scaler is not None:
        x = scaler.transform(x)
    else:
        print("[WARNING] Scaler not loaded, using raw features")
    
    # Convert to tensor
    tensor_data = torch.FloatTensor(x).to(device)
    
    with torch.no_grad():
        output, _ = model(tensor_data)
    return output.item()

# 6. Dự đoán với tất cả 3 models
def predict_all(features):
    """
    Predict using all 3 models (GAR, GAR-EXP, MAE)
    
    Args:
        features: list of 9 features [f1, f2, ..., f9]
    
    Returns:
        dict with predictions and arrays (300 pre-sampled points):
        {
            'predictions': {
                'GAR': float,
                'GAR-EXP': float,
                'MAE': float
            },
            'arrays': {
                'GAR': {'y_truth': [...], 'y_pred': [...]},
                'GAR-EXP': {'y_truth': [...], 'y_pred': [...]},
                'MAE': {'y_truth': [...], 'y_pred': [...]}
            }
        }
    """
    results = {
        'predictions': {},
        'arrays': {}
    }
    
    # Get predictions from all models
    for loss_name, model in models.items():
        try:
            prediction = predict_single(model, features)
            results['predictions'][loss_name] = prediction
        except Exception as e:
            print(f"Error predicting with {loss_name}: {e}")
            results['predictions'][loss_name] = None
    
    # Use pre-sampled arrays (300 samples)
    for loss_name in ['GAR', 'GAR-EXP', 'MAE']:
        if loss_name in sample_arrays:
            results['arrays'][loss_name] = sample_arrays[loss_name]
    
    return results

# 7. Get chart arrays only (for initial load)
def get_chart_arrays():
    """
    Get pre-loaded chart arrays without making a prediction
    
    Returns:
        dict with arrays for all 3 models:
        {
            'GAR': {'y_truth': [...], 'y_pred': [...]},
            'GAR-EXP': {'y_truth': [...], 'y_pred': [...]},
            'MAE': {'y_truth': [...], 'y_pred': [...]}
        }
    """
    return sample_arrays

