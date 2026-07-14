"""Standard library"""
import os
import math
import glob
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse

"""Third party library"""
import cv2 as cv
import numpy as np
from scipy import interpolate
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt

""" --- CUSTOM FUNCTIONS --- """

def parse_arguments():
    """Parses command-line arguments for the image processing pipeline."""
    parser = argparse.ArgumentParser(
        description="Process a single raw image to calculate spot centers, contours, and profiles."
    )

    parser.add_argument(
        "-p",
        "--photo",
        type=str,
        required=True,
        help="Path to the raw binary image file (e.g., /path/to/image.raw)",
    )

    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        required=True,
        help="Percentile value for centre-finding threshold (e.g., 99.999).",
    )

    args = parser.parse_args()

    # Validate that the file actually exists
    input_path = Path(args.photo)
    if not input_path.is_file():
        parser.error(f"The specified file does not exist: '{input_path}'")

    return Path(input_path), args.threshold

import numpy as np

def pseudo_voigt_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, eta, offset):
    """
    A 2D Pseudo-Voigt profile: a linear combination of a 2D Gaussian and 
    a 2D Lorentzian sharing the same spatial parameters.
    
    Parameters:
    -----------
    xy : tuple of arrays (X, Y)
        The meshgrid coordinates.
    amplitude : float
        Peak amplitude of the mixture.
    xo, yo : float
        Center coordinates.
    sigma_x, sigma_y : float
        Width parameters (standard deviations for Gaussian, HWHM-like for Lorentzian).
    theta : float
        Rotation angle in radians.
    eta : float
        Mixing fraction (0.0 = pure Gaussian, 1.0 = pure Lorentzian). 
        Should be bounded between 0 and 1.
    offset : float
        Background baseline.
    """
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    
    # 1. Coordinate rotation and scaling transformations
    # To keep things clean, we transform the coordinates instead of the equations.
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    # Shift to center
    x_shifted = x - xo
    y_shifted = y - yo
    
    # Rotate coordinates
    x_rot = x_shifted * cos_t + y_shifted * sin_t
    y_rot = -x_shifted * sin_t + y_shifted * cos_t
    
    # Normalized squared distances
    d2 = (x_rot / sigma_x)**2 + (y_rot / sigma_y)**2
    
    # 2. Component 1: 2D Gaussian component
    # Equivalent to standard Gaussian when d2 expression expands
    gaussian = np.exp(-0.5 * d2)
    
    # 3. Component 2: 2D Cauchy/Lorentzian component
    # Standard 2D Lorentzian profile shape
    lorentzian = 1.0 / (1.0 + 0.5 * d2)
    
    # 4. Mix them linearly and apply amplitude/offset
    g = offset + amplitude * ((1.0 - eta) * gaussian + eta * lorentzian)
    
    return g.ravel()

""" CAMERA CONFIG """
IMG_WIDTH, IMG_HEIGHT = 5544, 3684
IMG_BPP = 16

""" MATPLOBLIB CONFIG """
bg_cmap = 'bone'
fg_cmap = 'turbo'
PLOT_DIR = './laser_plots'
plot_path = Path(PLOT_DIR)
plot_path.mkdir(parents=True, exist_ok=True)
crop_size = 20

""" ITERATE OVER IMAGES """
RUN_STRS = [f'{i:04d}' for i in range(9952, 9969)]
DISTANCES = [4.650, 4.549, 4.452, 4.352, 4.250, 4.150,
             4.048, 3.945, 3.847, 3.743, 3.643, 3.546,
             3.447, 3.348, 3.250, 3.151, 3.029]

""" RUN THROUGH EVERY RUN """
all_xo, all_yo = [], []
all_xo_err, all_yo_err = [], []
for run_str in RUN_STRS:

    """ LOAD IMAGE """
    # raw_image_path, percentile = parse_arguments()
    DATA_DIR = '/home/lkoerich/Data/collimator/preprocessed/'
    # raw_image_path = f'{DATA_DIR}/'
    # PHOTO = load_raw_image(raw_image_path, IMG_WIDTH, IMG_HEIGHT, IMG_BPP)
    PHOTO = np.load(f'{DATA_DIR}/warped_mean_{run_str}.npz')['means']
    ERROR = np.load(f'{DATA_DIR}/warped_mean_{run_str}.npz')['error']
    file_stem = f'RUN-{run_str}'
    centre_filename  = Path(PLOT_DIR) / f"centre_{file_stem}.png"
    profile_filename = Path(PLOT_DIR) / f"profile_{file_stem}.png"

    """ FIND SPOT CENTRE """
    percentile = 99.999
    threshold = np.percentile(PHOTO, percentile)
    mask = PHOTO > threshold
    y_coords, x_coords = np.where(mask)
    center_x, center_y = 0.0, 0.0
    if len(x_coords) > 0:
        center_x = np.mean(x_coords)
        center_y = np.mean(y_coords)

    """ 2D Gaussian Area of Interest """
    x_min, x_max = int(center_x - crop_size), int(center_x + crop_size)
    y_min, y_max = int(center_y - crop_size), int(center_y + crop_size)
    x_indices = np.arange(x_min, x_max)
    y_indices = np.arange(y_min, y_max)
    X, Y = np.meshgrid(x_indices, y_indices)
    cropped_photo = PHOTO[y_min:y_max, x_min:x_max]

    """ Attempt Gaussian fit"""
    initial_guess = (
        np.max(cropped_photo),  # amplitude
        center_x,              # xo
        center_y,              # yo
        crop_size / 4,         # sigma_x
        crop_size / 4,         # sigma_y
        0.0,                   # theta
        0.5,                   # eta (mix fraction)
        np.min(cropped_photo)  # offset
    )

    # Bounds to ensure stable convergence:
    # (lower_bounds, upper_bounds)
    bounds = (
        [0, center_x - crop_size, center_y - crop_size, 0.1, 0.1, -np.pi, 0.0, 0],
        [np.inf, center_x + crop_size, center_y + crop_size, crop_size, crop_size, np.pi, 1.0, np.inf]
    )

    try:
        popt, pcov = curve_fit(
            pseudo_voigt_2d, 
            (X, Y), 
            cropped_photo.ravel(), 
            p0=initial_guess, 
            bounds=bounds
        )
        fit_success = True
        fit_data = pseudo_voigt_2d((X, Y), *popt).reshape(cropped_photo.shape)
        perr = np.sqrt(np.diag(pcov))
        residuals = cropped_photo - fit_data

    except Exception as e:
        print(f"Fit failed for RUN-{run_str}: {e}")
        fit_success = False

    """ PLOT SPOT CENTRE """
    fig1, (ax1a, ax1b, ax1c) = plt.subplots(3, 1, figsize=(5, 18))
    extent = [x_min, x_max, y_max, y_min]

    im1a = ax1a.imshow(cropped_photo, cmap=bg_cmap)
    ax1a.plot(center_x-x_min, center_y-y_min, marker='x', color='red', markersize=10, label='Geometrical Centre')
    ax1a.set_xlabel('[px]')
    ax1a.set_ylabel('[px]')
    ax1a.set_xlim([center_x-x_min-crop_size, center_x-x_min+crop_size])
    ax1a.set_ylim([center_y-y_min+crop_size, center_y-y_min-crop_size])
    plt.colorbar(im1a, ax=ax1a)
    plt.tight_layout()

    if fit_success:
        im1b = ax1b.imshow(fit_data, cmap=bg_cmap)
        ax1b.set_title('2D Gaussian Fit')
    else:
        im1b = ax1b.imshow(np.zeros_like(cropped_photo), cmap=bg_cmap)
        ax1b.set_title('Fit Failed')
    plt.colorbar(im1b, ax=ax1b)

    if fit_success:
        v_max = np.max(np.abs(residuals))
        im1c = ax1c.imshow(residuals, cmap='seismic')
        ax1c.set_title('Residuals (Data - Fit)')
    else:
        im1c = ax1c.imshow(np.zeros_like(cropped_photo), cmap='seismic')
        ax1c.set_title('No Residuals')
    plt.colorbar(im1c, ax=ax1c)
    plt.savefig(centre_filename, dpi=300)

    """ PLOT PROFILE """
    if fit_success:
        
        y_profile_fit = fit_data[:, int(center_x - x_min)]  # Vertical cut (Y)
        x_profile_fit = fit_data[int(center_y - y_min), :]  # Horizontal cut (X)
        
        x_fit_pixels = np.arange(x_min, x_max)
        y_fit_pixels = np.arange(y_min, y_max)

        # Real data cuts across the full image size
        x_profile = PHOTO[int(center_y), :]
        y_profile = PHOTO[:, int(center_x)]

        # Respective errors:
        x_profile_err = ERROR[int(center_y), :]
        y_profile_err = ERROR[:, int(center_x)]

        fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(8, 6))

        x = np.arange(0, len(x_profile))
        ax3a.plot(x_profile, label='Data', color='black', alpha=0.7)
        ax3a.plot(x_fit_pixels, x_profile_fit, label='2D Gaussian Fit', color='red', linestyle='--')
        ax3a.set_xlim([center_x - crop_size, center_x + crop_size])
        ax3a.set_title('Horizontal Profile (X)')
        
        ax3b.plot(y_profile, label='Data', color='black', alpha=0.7)
        ax3b.plot(y_fit_pixels, y_profile_fit, label='2D Gaussian Fit', color='red', linestyle='--')
        
        ax3b.set_xlim([center_y + crop_size, center_y - crop_size])
        ax3b.set_title('Vertical Profile (Y)')
        
        for ax in [ax3a, ax3b]:
            ax.set_xlabel('[px]')
            ax.set_ylabel('Intensity')  
            ax.legend(loc='upper right')
            
        plt.tight_layout()
        plt.savefig(profile_filename, dpi=300)

    # plt.show()

    # EXTRACT REAL_X_CENTER AND REAL_Y_CENTER FROM fit parameters.
    # PROBABLY xo and yo
    
    # ADD THESE TO A GLOBAL ARRAY

    print(f"------ RUN {run_str} ------")
    print(f'xo = {popt[1]}, yo = {popt[2]}')
    print(f'xo = {perr[1]}, yo = {perr[2]}')
    print("----------------------")
    all_xo.append(popt[1])
    all_yo.append(popt[2])
    all_xo_err.append(perr[1])
    all_yo_err.append(perr[2])
    plt.close('all')

np.savez(
    './laser_field.npz', 
    xo=all_xo, 
    yo=all_yo, 
    z=DISTANCES, 
    xo_err = all_xo_err, 
    yo_err = all_yo_err
    )