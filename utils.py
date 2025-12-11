import torch 
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class pair_dataset(torch.utils.data.Dataset):
  def __init__(self, X, Y):
    self.X = torch.from_numpy(X.astype(np.float32))
    self.Y = torch.from_numpy(Y.astype(np.float32))
  def __len__(self):
    try:
      L = len(self.X)
    except:
      L = self.X.shape[0]
    return L 
  def __getitem__(self, idx):
    return self.X[idx], self.Y[idx]


def bike_sharing(path='./data/bike_sharing.npz', std_flag=True, y_std_flag=False):
  try:
    dat = np.load(path)
    trX = dat['trX']
    trY = dat['trY']
    teX = dat['teX']
    teY = dat['teY']
    print('Loaded bike_sharing from cache')
  except:
    # Fetch dataset 
    bike_sharing_data = fetch_ucirepo(id=275)
    X = bike_sharing_data.data.features 
    y = bike_sharing_data.data.targets 
    
    # The 'dteday' column contains dates and cannot be converted to float.
    if 'dteday' in X.columns:
        X = X.drop(columns=['dteday'])

    # Convert to numpy
    X = X.to_numpy()
    # Ensure data is numeric before calculations
    X = X.astype(np.float64)
    
    # Standardization
    if std_flag:
      X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Split into train and test (80/20)
    ids = np.random.permutation(X.shape[0])
    teN = int(0.2 * X.shape[0])
    te_ids = ids[:teN]
    tr_ids = ids[teN:]
    trX = X[tr_ids] # NumPy array indexing
    trY = y.iloc[tr_ids].to_numpy() # Pandas row indexing by position
    teX = X[te_ids] # NumPy array indexing
    teY = y.iloc[te_ids].to_numpy() # Pandas row indexing by position
    
    print('Processed bike_sharing dataset')
    print(f'Train samples: {trX.shape[0]}, Test samples: {teX.shape[0]}')
    print(f'Features: {trX.shape[1]}, Targets: {trY.shape[1]}')
    
    # Save for future use
    output_dir = os.path.dirname(path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savez(path, trX=trX, trY=trY, teX=teX, teY=teY)
    print(f'Saved processed bike_sharing data to {path}')
  
  if y_std_flag:
    mY, stdY = trY.mean(axis=0), trY.std(axis=0)
    trY = (trY - mY) / stdY
    teY = (teY - mY) / stdY
    
  return trX, trY, teX, teY


def protein_data(path='./data/protein.csv', std_flag=True, y_std_flag=False, target_column='RMSD'):
  """
  Load protein dataset from a cached NPZ file or process from CSV.

  Args:
    path: Path to the NPZ file.
    std_flag: Whether to standardize features
    y_std_flag: Whether to standardize targets
    target_column: Name of the target column in the CSV. Defaults to 'RMSD'.

  Returns:
    trX, trY, teX, teY: Train/test features and targets
  """
  try:
    dat = np.load(path)
    trX = dat['trX']
    trY = dat['trY']
    teX = dat['teX']
    teY = dat['teY']
    print(f'Loaded protein data from {path}')
  except:
    # Fall back to loading from CSV
    csv_path = 'data/protein.csv'
    if not os.path.isabs(csv_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.normpath(os.path.join(current_dir, csv_path))
    
    try:
      df = pd.read_csv(csv_path)
      print(f'Loaded protein data from {csv_path}')
    except FileNotFoundError as e:
      print(f"Error: {e}")
      print("Please make sure the 'protein.csv' file is located in the 'data' directory.")
      raise

    # Separate features and targets
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values.reshape(-1, 1)

    # Standardization of features
    if std_flag:
      scaler = StandardScaler()
      X = scaler.fit_transform(X)

    # Split into train and test (80/20)
    trX, teX, trY, teY = train_test_split(X, y, test_size=0.2, random_state=123)
    print(f'Train samples: {trX.shape[0]}, Test samples: {teX.shape[0]}')
    
    # Save for future use
    output_dir = os.path.dirname(path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savez(path, trX=trX, trY=trY, teX=teX, teY=teY)
    print(f'Saved processed protein data to {path}')

  if y_std_flag:
    mY, stdY = trY.mean(axis=0), trY.std(axis=0)
    trY = (trY - mY) / stdY
    teY = (teY - mY) / stdY

  return trX, trY, teX, teY


def sine_data(path='data/sine.npz', n_samples=700, scale=1.0, noise_level=0.1, test_size=0.2, random_state=123, split_data=True, force_regenerate=False):
  """
  Generates or loads a synthetic sine wave dataset with noise.

  Args:
    path (str): Path to save/load the cached .npz file.
    n_samples (int): Total number of samples to generate.
    scale (float): The amplitude (scaling factor) of the sine wave.
    noise_level (float): The standard deviation of the Gaussian noise to add to the targets.
    test_size (float): The proportion of the dataset to allocate to the test set.
    random_state (int): Seed for reproducibility of the train/test split.
    split_data (bool): If True, splits and returns train/test sets. If False, returns full X and y.
    force_regenerate (bool): If True, always generates new data, ignoring the cache.

  Returns:
    If split_data is True: trX, trY, teX, teY (Train/test features and targets)
    If split_data is False: X, y (Full features and targets)
  """
  if not force_regenerate:
    try:
      dat = np.load(path)
      print(f'Loaded sine data from cache: {path}')
      X, y = dat['X'], dat['y']
    except FileNotFoundError:
      force_regenerate = True # File not found, so we must regenerate

  if force_regenerate:
    print(f"Generating new sine dataset with scale={scale}, noise={noise_level}")
    # Generate feature values (X) - e.g., time steps
    X = np.linspace(-10 * np.pi, 10 * np.pi, n_samples).reshape(-1, 1)

    # Generate clean target values (Y) based on a scaled sine wave
    y_clean = scale * np.sin(X)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, y_clean.shape)
    y = y_clean + noise

    # Save for future use
    output_dir = os.path.dirname(path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savez(path, X=X, y=y)
    print(f'Saved processed sine data to {path}')

  if split_data:
    # Split into training and testing sets
    trX, teX, trY, teY = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return trX, trY, teX, teY
  else:
    return X, y


def add_noise_and_save(dataset_name, noise_level=0.1, output_folder='advanced-data'):
  """
  Loads, scales, adds noise to a dataset, and saves it to a specified folder.

  Args:
      dataset_name (str): The name of the dataset ('bike_sharing' or 'protein').
      noise_level (float): The standard deviation of the Gaussian noise to add to features.
      output_folder (str): The directory to save the noisy dataset.
  """
  import os
  if not os.path.exists(output_folder):
      os.makedirs(output_folder)
      print(f"Created directory: {output_folder}")

  print(f"Processing dataset: {dataset_name} with noise level: {noise_level}")

  if dataset_name == 'bike_sharing':
      # Load and scale data using existing function
      trX, trY, teX, teY = bike_sharing(std_flag=True)

      # Add Gaussian noise to features
      trX_noisy = trX + np.random.normal(0, noise_level, trX.shape)
      teX_noisy = teX + np.random.normal(0, noise_level, teX.shape)

      # Save the noisy dataset
      output_path = os.path.join(output_folder, 'bike_sharing_noisy.npz')
      np.savez(output_path, trX=trX_noisy, trY=trY, teX=teX_noisy, teY=teY)
      print(f"Saved noisy bike sharing data to {output_path}")

  elif dataset_name == 'protein':
      # Load and scale data using existing function
      # This will load from '../data/protein.csv'
      trX, trY, teX, teY = protein_data(std_flag=True)

      # Add Gaussian noise to features
      trX_noisy = trX + np.random.normal(0, noise_level, trX.shape)
      teX_noisy = teX + np.random.normal(0, noise_level, teX.shape)

      # Save the noisy dataset
      output_path = os.path.join(output_folder, 'protein_noisy.npz')
      np.savez(output_path, trX=trX_noisy, trY=trY, teX=teX_noisy, teY=teY)
      print(f"Saved noisy protein data to {output_path}")
  else:
      print(f"Dataset '{dataset_name}' is not supported.")