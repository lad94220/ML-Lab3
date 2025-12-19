import pandas as pd
import pickle
import models
import numpy as np
import torch

INPUT_CSV_FILE = 'data/protein.csv' 
MODEL_PKL_FILE = 'FINAL_DEPLOYMENT_MODEL_protein_MAE_20251213_230400.pkl'
OUTPUT_CSV_FILE = 'MAE_PROTEIN_PREDICTIONS.csv'
TARGET_COLUMN = 'RMSD' 
SCALER_PKL_FILE = 'data/protein_scaler.pkl'


try:
    # Read input data
    data = pd.read_csv(INPUT_CSV_FILE)
    
    print(f"Đã đọc {len(data)} dòng dữ liệu từ file: {INPUT_CSV_FILE}")
    print(f"Các cột dữ liệu: {data.columns.tolist()}")
    X = data.drop(columns=[TARGET_COLUMN]) 
    
    # Get ground truth
    Y_ground_truth = data[TARGET_COLUMN].copy() 

    # Change DataFrame to NumPy array for processing
    X_features = X.values 
    
    # Print check information
    print(f"\nFeature dimensions (X): {X_features.shape}")
    print(f"Ground Truth dimensions (Y): {Y_ground_truth.shape}")

except FileNotFoundError:
    print(f"ERROR: Cannot find input data file '{INPUT_CSV_FILE}'. Please check the path.")
    exit()
except KeyError:
    print(f"ERROR: Target column '{TARGET_COLUMN}' not found in the data.")
    exit()
try:
    with open(SCALER_PKL_FILE, 'rb') as f:
        scaler = pickle.load(f)
    print(f"\nSuccessfully loaded scaler object from file: {SCALER_PKL_FILE}")

except FileNotFoundError:
    print(f"\nERROR: Cannot find SCALER file '{SCALER_PKL_FILE}'. Please check the path.")
    print("You must save the scaler object from the training process to use it for prediction.")
    exit()
except Exception as e:
    print(f"ERROR loading SCALER from PKL file: {e}")
    exit()

try:
    with open(MODEL_PKL_FILE, 'rb') as file:
        model = pickle.load(file)
    
    print(f"\nSuccessfully loaded model from file: {MODEL_PKL_FILE}")
    # print(f"Model type: {type(model)}") # You can check the model type if needed

except FileNotFoundError:
    print(f"ERROR: Cannot find model file '{MODEL_PKL_FILE}'. Please check the path.")
    exit()
except Exception as e:
    print(f"ERROR loading model from PKL file: {e}")
    exit()

print("\nStarting prediction...")

try:
    # 1. Set the model to evaluation mode (Very important for PyTorch)
    model.eval() 
    
    # 2. Scale input data (The scaler variable was loaded in step 1.5)
    X_scaled_np = scaler.transform(X_features) 
    
    # 3. Convert scaled features to PyTorch Tensor
    X_tensor = torch.tensor(X_scaled_np, dtype=torch.float32)
    
    # 4. Perform prediction
    with torch.no_grad():
        # Call the PyTorch model, it returns y_pred and feat
        y_pred_tensor, _ = model(X_tensor) 
    
    # 5. Convert the result to a NumPy array to save into a Pandas DataFrame
    predictions = y_pred_tensor.cpu().numpy().flatten()
    
    print(f"Prediction completed. Result size: {predictions.shape}")

except Exception as e:
    print(f"ERROR during PyTorch prediction: {e}")
    print("Please check the input and output structure of the MLP model.")
    exit()


# Create results DataFrame
results_df = pd.DataFrame({
    'Y_Ground_Truth': Y_ground_truth,
    'Prediction': predictions
})

# Add original feature columns to the results DataFrame for easier tracking
# This helps you know which input data corresponds to the predictions and Ground Truth
results_df = pd.concat([X, results_df], axis=1)

# Save the results DataFrame to a CSV file
results_df.to_csv(OUTPUT_CSV_FILE, index=False)

print(f"Done! Prediction results and Ground Truth have been saved to the file:")
print(f"{OUTPUT_CSV_FILE}")

print(results_df.head())