import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Read all three prediction CSV files
df_gar = pd.read_csv('data/GAR_PROTEIN_PREDICTIONS.csv')
df_mae = pd.read_csv('data/MAE_PROTEIN_PREDICTIONS.csv')
df_gar_exp = pd.read_csv('data/GAR-EXP_PROTEIN_PREDICTIONS.csv')

# Randomly select 300 samples from the entire dataset
sample_indices = np.random.choice(len(df_gar), size=300, replace=False)
sample_indices = np.sort(sample_indices)  # Sort to maintain order from beginning to end

# Create y_truth array (same for all models)
y_truth = df_gar.iloc[sample_indices]['Y_Ground_Truth'].values

# Create y_pred arrays for each model
y_pred_gar = df_gar.iloc[sample_indices]['Prediction'].values
y_pred_mae = df_mae.iloc[sample_indices]['Prediction'].values
y_pred_gar_exp = df_gar_exp.iloc[sample_indices]['Prediction'].values

# Print the arrays
print("=" * 60)
print("SAMPLE ARRAYS CREATED")
print("=" * 60)
print(f"\ny_truth shape: {y_truth.shape}")
print(f"y_pred_gar shape: {y_pred_gar.shape}")
print(f"y_pred_mae shape: {y_pred_mae.shape}")
print(f"y_pred_gar_exp shape: {y_pred_gar_exp.shape}")

print("\n" + "=" * 60)
print("FIRST 10 VALUES")
print("=" * 60)
print("\ny_truth:")
print(y_truth[:10])
print("\ny_pred_gar:")
print(y_pred_gar[:10])
print("\ny_pred_mae:")
print(y_pred_mae[:10])
print("\ny_pred_gar_exp:")
print(y_pred_gar_exp[:10])

# Save to numpy files for later use
np.save('data/y_truth_sample.npy', y_truth)
np.save('data/y_pred_gar_sample.npy', y_pred_gar)
np.save('data/y_pred_mae_sample.npy', y_pred_mae)
np.save('data/y_pred_gar_exp_sample.npy', y_pred_gar_exp)

print("\n" + "=" * 60)
print("ARRAYS SAVED TO:")
print("=" * 60)
print("- data/y_truth_sample.npy")
print("- data/y_pred_gar_sample.npy")
print("- data/y_pred_mae_sample.npy")
print("- data/y_pred_gar_exp_sample.npy")
