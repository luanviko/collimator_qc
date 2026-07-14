"""Standard library"""
import os
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = '/home/lkoerich/Data/collimator/photos'
PREPROCESSED_DIR = '/home/lkoerich/Data/collimator/preprocessed/'

# Working from home
DATA_DIR = '/Users/lkoerich/Downloads/collimator_photos'
PREPROCESSED_DIR = '/Users/lkoerich/Downloads/collimator_preprocessed'

# RUN_STRS = [f'{i:04d}' for i in range(14, 21)]
# RUN_STRS = [f'{i:04d}' for i in [9903, 9907, 9911]]
# RUN_STRS = [f'{i:04d}' for i in [9913, 9914, 9915]]
RUN_STRS = [f'{i:04d}' for i in range(9969, 9981)]
# RUN_STRS = ['9952', '9953', '9954']

width, height = 5544, 3684
bpp = 16
MAX_WORKERS = 1

def load_raw_image(path, width, height, bpp):
    """Fastest binary load."""
    return np.fromfile(path, dtype=np.uint16).reshape((height, width))

def process_single_run(run_number):
    try:
        directory = Path(DATA_DIR)
        save_path = Path(PREPROCESSED_DIR)
        save_path.mkdir(parents=True, exist_ok=True)
        
        prefix = f'Run-{run_number}'
        raw_files = sorted(list(directory.glob(f"{prefix}*.raw")))
        n_photos = len(raw_files)
        
        if n_photos == 0:
            return f"Run {run_number}: No files found."

        # 1. LOAD DATA
        all_photos = np.zeros((n_photos, height, width), dtype=np.float32)
        for i, raw_file in enumerate(raw_files):
            all_photos[i] = load_raw_image(raw_file, width, height, bpp)

        # 2. STATS (Full frame)
        mean_full = np.mean(all_photos, axis=0)
        sum_full  = np.sum(all_photos, axis=0)

        if n_photos > 1:
            std_full  = np.std(all_photos, axis=0, ddof=1)
            mean_err = std_full / np.sqrt(n_photos)
            sum_err  = std_full * np.sqrt(n_photos)
        else:
            std_full  = np.zeros((height, width), dtype=np.float32)
            mean_err = np.zeros((height, width), dtype=np.float32)
            sum_err  = np.zeros((height, width), dtype=np.float32)
        
        mean_err = std_full / np.sqrt(n_photos)
        sum_err  = std_full * np.sqrt(n_photos)

        # 3. SAVE
        np.savez_compressed(
            save_path / f'preprocessed_{run_number}.npz',
            means=mean_full, # We still save the full mean
            sums=sum_full,
            means_error=mean_err,
            sums_error=sum_err,
        )
        return f"Run {run_number} success."
    
    except Exception as e:
        return f"Run {run_number} failed: {str(e)}"

if __name__ == "__main__":
    print(f"Processing {len(RUN_STRS)} runs using {MAX_WORKERS} cores...")
    
    # The 'with' context manager ensures the worker pool stays alive 
    # until every internal file operation is 100% complete.
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        # Wrapping executor.map into a list forces the main script to block
        # and wait for the files to completely save before moving forward.
        results = list(tqdm(executor.map(process_single_run, RUN_STRS), total=len(RUN_STRS)))

    # Print out the success/failure strings returned from process_single_run
    print("\n--- Run Execution Summary ---")
    for result in results:
        print(result)