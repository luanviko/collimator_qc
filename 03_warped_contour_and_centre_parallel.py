"""Standard library"""
import os
import math
import glob
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

"""Third party library"""
import cv2 as cv
import numpy as np
from scipy import interpolate
from scipy.optimize import least_squares

# Force Matplotlib to use a non-interactive backend (Crucial for multi-processing!)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec

# Custom tools library
from tools import *

# --- CONFIGURATION ---
DATA_DIR = '/home/lkoerich/Data/collimator/preprocessed/'
PLOT_DIR = './plots'
FIT_DIR  = './fits'

# Working from home
DATA_DIR = '/Users/lkoerich/Downloads/collimator_preprocessed'

bg_cmap = 'bone'
fg_cmap = 'turbo'
MAX_WORKERS = 4  # Adjust based on available CPU cores

RUN_STRS = [f'{i:04d}' for i in range(9969, 9981)]

def process_single_run(run_str):
    """Worker function to process a single run independently."""
    try:
        # Paths setup
        plot_path = Path(PLOT_DIR)
        fit_path = Path(FIT_DIR)
        
        # Load preprocessed image data
        MEAN  = np.load(f'{DATA_DIR}/warped_mean_{run_str}.npz')['means']
        ERROR = np.load(f'{DATA_DIR}/warped_mean_{run_str}.npz')['error']

        # Geometrical centre and bright spot
        threshold = np.percentile(MEAN, 80)
        mask = MEAN > threshold
        y_coords, x_coords = np.where(mask)
        
        # Initialize fallback variables in case mask is empty
        center_x, center_y = 0.0, 0.0
        err_center_x, err_center_y = 0.0, 0.0
        
        bright_x, bright_y = 0.0, 0.0
        err_bright_x, err_bright_y = 0.0, 0.0
        
        if len(x_coords) > 0:
            N = len(x_coords)
            weights = MEAN[mask]
            w_err = ERROR[mask]
            sum_w = np.sum(weights)

            center_x = np.mean(x_coords)
            center_y = np.mean(y_coords)
            err_center_x = np.std(x_coords) / np.sqrt(N)
            err_center_y = np.std(y_coords) / np.sqrt(N)

            bright_x = np.average(x_coords, weights=MEAN[mask])
            bright_y = np.average(y_coords, weights=MEAN[mask])
            
            if sum_w > 0:
                err_bright_x = np.sqrt(np.sum(( (x_coords - bright_x) / sum_w )**2 * w_err**2))
                err_bright_y = np.sqrt(np.sum(( (y_coords - bright_y) / sum_w )**2 * w_err**2))

        # Save centers directly to the destination
        np.savez(
            f'{FIT_DIR}/centers_run_{run_str}.npz', 
            center_x=center_x, 
            center_y=center_y, 
            bright_x=bright_x, 
            bright_y=bright_y,
            err_center_x=err_center_x,
            err_center_y=err_center_y,
            err_bright_x=err_bright_x,
            err_bright_y=err_bright_y
            )

        # Plot 1: Spot Centers
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        im1 = ax1.imshow(MEAN, cmap='jet', origin='upper')
        ax1.plot(center_x, center_y, marker='x', color='red', markersize=10, label='Geometrical Centre')
        ax1.plot(bright_x, bright_y, marker='o', color='blue', markersize=10, label='Bright Centre')
        ax1.set_title(f'{run_str}')
        ax1.set_xlabel('[px]')
        ax1.set_ylabel('[px]')
        plt.colorbar(im1, ax=ax1)
        plt.tight_layout()
        plt.savefig(plot_path / f'warped_spot_run_{run_str}.png', dpi=300)
        plt.close(fig1)

        # Determine contour levels based on percentiles of the MEAN image
        base_percentiles = [0.1, 0.2, 0.4, 0.65]
        contour_segments, contour_levels = find_contours(MEAN, base_percentiles=base_percentiles)
        
        # Safe colormap fetching for parallel execution context
        cmap_obj = matplotlib.colormaps[fg_cmap] if hasattr(matplotlib, 'colormaps') else plt.cm.get_cmap(fg_cmap)
        colors = cmap_obj(np.linspace(0, 1, len(contour_segments)))

        # Plot 2: Contours and Ellipse Fitting
        fig2, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(MEAN, cmap=bg_cmap, origin='upper')
        contour_data_dict = {}
        fit_data_dict = {}
        
        if contour_segments:    
            for level_idx, level_segs in enumerate(contour_segments):
                level_value = contour_levels[level_idx]
                level_color = colors[level_idx]
                all_points = np.vstack(level_segs)
                x_all = all_points[:, 0]
                y_all = all_points[:, 1]
                
                contour_data_dict[f"level_{level_idx}_x_raw"] = x_all
                contour_data_dict[f"level_{level_idx}_y_raw"] = y_all
                contour_data_dict[f"level_{level_idx}_value"] = np.array([level_value])
                
                distances = np.sqrt((x_all - center_x)**2 + (y_all - center_y)**2)
                valid_mask = distances <= 950
                x_all = x_all[valid_mask]
                y_all = y_all[valid_mask]
                
                if len(x_all) >= 5:
                    try:
                        x0, y0, a, b, theta_fit = fit_stable_ellipse(x_all, y_all)
                        if not (np.isnan(x0) or np.isnan(y0) or np.isnan(a) or np.isnan(b) or np.isnan(theta_fit)):
                            t = np.linspace(0, 2 * np.pi, 100)
                            x_canonical = a * np.cos(t)
                            y_canonical = b * np.sin(t)
                            x_ellipse = x0 + x_canonical * np.cos(theta_fit) - y_canonical * np.sin(theta_fit)
                            y_ellipse = y0 + x_canonical * np.sin(theta_fit) + y_canonical * np.cos(theta_fit)
                            
                            ax.scatter(x0, y0, color='red', edgecolors='white', s=40, zorder=5, 
                                       label='Fitted Center' if level_idx == 0 else None)
                            ax.plot(x_ellipse, y_ellipse, color='red', linestyle='--', linewidth=1.5, zorder=4, 
                                    label='Fitted Ellipse' if level_idx == 0 else None)
                            
                            fit_data_dict[f"level_{level_idx}_params"] = np.array([x0, y0, a, b, theta_fit])
                            fit_data_dict[f"level_{level_idx}_x_fit"] = x_ellipse
                            fit_data_dict[f"level_{level_idx}_y_fit"] = y_ellipse
                            fit_data_dict[f"level_{level_idx}_success"] = np.array([True])
                        else:
                            fit_data_dict[f"level_{level_idx}_success"] = np.array([False])
                    except Exception:
                        fit_data_dict[f"level_{level_idx}_success"] = np.array([False])
                else:
                    fit_data_dict[f"level_{level_idx}_success"] = np.array([False])
                    
                for seg_idx, segment in enumerate(level_segs):
                    x_data = segment[:, 0]
                    y_data = segment[:, 1]
                    label = f"{level_value:.1f}" if seg_idx == 0 else None
                    ax.plot(x_data, y_data, color=level_color, linewidth=2, label=label)
                    
        # Save data direct to destinations
        np.savez(f'{FIT_DIR}/contour_data_run_{run_str}.npz', **contour_data_dict, allow_pickle=True)
        np.savez(f'{FIT_DIR}/fit_data_run_{run_str}.npz', **fit_data_dict, allow_pickle=True)

        ax.set_title(f'MEAN {run_str}')
        ax.set_xlabel('[px]')
        ax.set_ylabel('[px]')
        ax.legend(title='Contour Levels', bbox_to_anchor=(1.15, 1), loc='upper left', fontsize='small')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(plot_path / f'warped_contours_run_{run_str}.png', dpi=300)
        
        # Free up system graphics memory completely
        plt.close(fig2)
        plt.close('all')
        
        return f"Run {run_str} successfully processed."
    
    except Exception as e:
        plt.close('all')
        return f"Run {run_str} failed: [{type(e).__name__}] {str(e)}"


if __name__ == "__main__":
    # Ensure standard directories exist before spinning up workers
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(FIT_DIR, exist_ok=True)

    print(f"Processing {len(RUN_STRS)} runs in parallel across {MAX_WORKERS} cores...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Pass the strings directly, list() ensures execution blocks until finished
        results = list(tqdm(executor.map(process_single_run, RUN_STRS), total=len(RUN_STRS)))

    print("\n--- Parallel Run Execution Summary ---")
    for result in results:
        print(result)