import os
from utils import sine_data, set_all_seeds

if __name__ == '__main__':
    """
    This script generates a synthetic sine wave dataset with specified
    scaling and noise level, and saves it to a file.
    """
    SEED = 123
    set_all_seeds(SEED)

    output_folder = 'sine-data'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_path = os.path.join(output_folder, 'sine.npz')

    print("Generating sine dataset with scale=5.0 and noise=0.3")
    # The function will generate, split, and save the data to the specified path.
    # force_regenerate=True ensures that the file is always overwritten.
    sine_data(path=file_path, n_samples=2000, scale=1.0, noise_level=0, split_data=False, force_regenerate=True)

    print(f"\nSine dataset generation complete. Data saved to '{file_path}'")