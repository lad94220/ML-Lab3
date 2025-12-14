import pandas as pd
import torch
from utils import models, predict_single

# Load one of the prediction logs to see what the input data looks like
pred_log = pd.read_csv('data/prediction_logs/preds_protein_GAR_fold0_para0.1.csv')

print("Prediction log shape:", pred_log.shape)
print("\nFirst few predictions:")
print(pred_log.head(10))

# Check if we can find the original dataset to compare
# The prediction logs show Truth_0 and Prediction_0
# Let's test with a sample from the dataset

# Try to make a prediction with the GAR model
print("\n" + "="*50)
print("Testing prediction with GAR model...")

# Sample features (we need to find what the actual input was)
# For now, let's try to reverse engineer from the prediction logs

# The issue is likely that the model expects NORMALIZED data
# but we're feeding it RAW data

print("\nModel architecture:")
print(models['GAR'])
