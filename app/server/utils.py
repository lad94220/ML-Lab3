import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os

# 1. Định nghĩa lại class MLP (hoặc import từ file models.py nếu bạn copy cả file đó sang)
class MLP(torch.nn.Module):
    r"""
        An implementation of Multilayer Perceptron (MLP).
    """
    def __init__(self, input_dim=1025, hidden_sizes=(256,), activation='elu', num_classes=64):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        if sum(self.hidden_sizes) > 0: # multi-layer model
            layers = []
            for i in range(len(hidden_sizes)):
                layers.append(torch.nn.Linear(input_dim, hidden_sizes[i])) 
                if activation=='relu':
                  layers.append(torch.nn.ReLU())
                elif activation=='elu':
                  layers.append(torch.nn.ELU())
                else:
                  pass 
                input_dim = hidden_sizes[i]
            self.layers = torch.nn.Sequential(*layers)
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        """forward pass"""
        if sum(self.hidden_sizes) > 0:
            x = self.layers(x)
        return self.fc(x), x 


# 2. Cấu hình thiết bị (App thường chạy CPU)
device = torch.device('cpu') 

# 3. Load all 3 models (GAR, GAR-EXP, MAE)
models = {}
model_configs = {
    'GAR': 'data/FINAL_DEPLOYMENT_MODEL_protein_GAR_20251213_225421.pth',
    'GAR-EXP': 'data/FINAL_DEPLOYMENT_MODEL_protein_GAR-EXP_20251213_231705.pth',
    'MAE': 'data/FINAL_DEPLOYMENT_MODEL_protein_MAE_20251213_230400.pth'
}

for loss_name, model_path in model_configs.items():
    try:
        # Initialize empty model with correct architecture
        # Based on the protein dataset: input_dim=9 (features), output=1 (prediction)
        model = MLP(input_dim=9, hidden_sizes=(64, 128, 64, 32, ), num_classes=1)
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        models[loss_name] = model
        print(f"[OK] Loaded {loss_name} model successfully")
    except Exception as e:
        print(f"[ERROR] Error loading {loss_name} model: {e}")

# 4. Load prediction logs for y_truth and y_pred arrays
prediction_logs = {}
log_configs = {
    'GAR': 'data/prediction_logs/preds_protein_GAR_fold4_para0.1.csv',
    'GAR-EXP': 'data/prediction_logs/preds_protein_GAR-EXP_fold4_para0.1.csv',
    'MAE': 'data/prediction_logs/preds_protein_MAE_fold4_para0.1.csv'
}

for loss_name, log_path in log_configs.items():
    try:
        df = pd.read_csv(log_path)
        prediction_logs[loss_name] = {
            'y_truth': df['Truth_0'].tolist(),
            'y_pred': df['Prediction_0'].tolist()
        }
        print(f"[OK] Loaded {loss_name} prediction logs ({len(df)} samples)")
    except Exception as e:
        print(f"[ERROR] Error loading {loss_name} prediction logs: {e}")

# 5. Dự đoán với một model cụ thể
def predict_single(model, input_data):
    """
    Predict using a single model
    input_data: list of 9 features [f1, f2, ..., f9]
    """
    tensor_data = torch.FloatTensor(input_data).unsqueeze(0).to(device)
    with torch.no_grad():
        output, _ = model(tensor_data)
    return output.item()

# 6. Dự đoán với tất cả 3 models
def predict_all(features, sample_size=200):
    """
    Predict using all 3 models (GAR, GAR-EXP, MAE)
    
    Args:
        features: list of 9 features [f1, f2, ..., f9]
        sample_size: number of data points to return for visualization (default: 50)
    
    Returns:
        dict with predictions and arrays:
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
    
    # Sample prediction logs for better visualization
    for loss_name, logs in prediction_logs.items():
        total_points = len(logs['y_truth'])
        
        if total_points > sample_size:
            # Create evenly spaced indices for sampling
            indices = np.linspace(0, total_points - 1, sample_size, dtype=int)
            
            results['arrays'][loss_name] = {
                'y_truth': [logs['y_truth'][i] for i in indices],
                'y_pred': [logs['y_pred'][i] for i in indices]
            }
        else:
            # If we have fewer points than sample_size, return all
            results['arrays'][loss_name] = logs
    
    return results