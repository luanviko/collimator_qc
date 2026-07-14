import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import glob
import numpy as np
import math, glob, os
from scipy import interpolate
import sys, os
from PIL import Image, ImageOps
from tools import *
from scipy import odr
import pandas as pd
from scipy import ndimage

''' LOAD RUN INFORMATION '''
print("Loading run_info dataframe from hdf5...")
try:
    run_info = pd.read_hdf('./run_info_with_centres.h5', key='run_info')
    print("Run information loaded successfully.")
except Exception as e:
    print(f"Error loading run_info dataframe: {e}")

''' ORGANIZE RUNS BY CHANNEL AND DISTANCE '''
DIST = run_info['distance (m)'].drop_duplicates().sort_values().values
z_min = float(DIST[0])*1000.
z_max = float(DIST[2])*1000.
RUN_CH_DICT = {ch: [run_info['run'][run_info['channel'] == ch].values[i] for i in range(len(DIST))] for ch in range(4)}

''' LOAD CALIBRATION DATA FOR PX-MM CONVERSION '''
calib = np.load('./unit_conversion_9997/px_mm_conversion_with_error_9997.npz')
x_fit = np.poly1d(calib['x_coeff'])
y_fit = np.poly1d(calib['y_coeff'])
x_cov = calib['x_cov']
y_cov = calib['y_cov']
a_x = calib['x_coeff'][0]
a_y = calib['y_coeff'][0]

''' VARIANCES FOR PX-MM CONVERSION '''
var_a_x  = x_cov[0, 0]
var_b_x  = x_cov[1, 1]
cov_ab_x = x_cov[0, 1]
var_a_y  = y_cov[0, 0]
var_b_y  = y_cov[1, 1]
cov_ab_y = y_cov[0, 1]

''' CAMERA SETTINGS '''
width, height = 5544, 3684
bpp = 16

''' DIRECTORIES '''
PLOT_DIR = './spot_plots'
FIT_DIR = './fits'
DATA_DIR = '/home/lkoerich/Data/collimator/preprocessed/'

# Working from home
DATA_DIR = '/Users/lkoerich/Downloads/collimator_preprocessed'

def plot_spot(
        MEAN, 
        center_x_px, 
        center_y_px, 
        file_name, 
        x_fit, 
        y_fit, 
        width  = 5544, #px
        height = 3684, #px
        crop_x = 1025, #px
        crop_y = 1025, #px
        ) -> None:

    '''PLOT PARAMETERS'''
    extent_mm = [x_fit(0), x_fit(width), y_fit(height), y_fit(0)]
    x_lim_spot = [int(max(0., x_fit(center_x_px-crop_x))), int(max(0., x_fit(center_x_px+crop_x)))]
    y_lim_spot = [int(max(0., y_fit(center_y_px+crop_y))), int(max(0., y_fit(center_y_px-crop_y)))]

    '''CREATE IMAGE'''
    fig, (ax1) = plt.subplots(1, 1, figsize=(5, 7))
    ax1.imshow(MEAN, cmap='jet', extent=extent_mm)
    ax1.plot(x_fit(center_x_px), y_fit(center_y_px), 'ko')
    # ax1.imshow(MEAN, cmap='jet', origin='upper')
    # ax1.plot(center_x_px, center_y_px, 'wo')
    ax1.set_xlim(x_lim_spot)
    ax1.set_ylim(y_lim_spot)
    plt.savefig(file_name, dpi=300)
    plt.close()
    return None

def split_pixels(data, centre, error, radius, radial_division, angular_division):
    x0, y0 = centre
    h, w = data.shape
    
    X, Y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    distance = np.sqrt((X - y0)**2 + (Y - x0)**2)
    angle = np.arctan2(X - y0, Y - x0)
    angle = np.where(angle < 0, angle + 2 * np.pi, angle)

    Rs = np.arange(0, radius, radial_division)
    Ts = np.arange(0, 360 + angular_division, angular_division) * np.pi / 180.0

    num_r = len(Rs) - 1
    num_t = len(Ts) - 1

    z_matrix = np.zeros((len(Rs), len(Ts)))
    std_matrix = np.zeros((len(Rs), len(Ts)))
    n_pixel_matrix = np.zeros((len(Rs), len(Ts)))
    z_matrix_weighted = np.zeros((len(Rs), len(Ts)))
    err_matrix = np.zeros((len(Rs), len(Ts)))
    
    z_binned_image = data.copy()

    r_idx = np.digitize(distance, Rs) - 1
    t_idx = np.digitize(angle, Ts) - 1

    valid_mask = (r_idx >= 0) & (r_idx < num_r) & (t_idx >= 0) & (t_idx < num_t)
    
    r_valid = r_idx[valid_mask]
    t_valid = t_idx[valid_mask]
    data_valid = data[valid_mask]
    error_valid = error[valid_mask]
    
    flat_bin_keys = r_valid * num_t + t_valid
    total_flat_bins = num_r * num_t

    sort_order = np.argsort(flat_bin_keys)
    flat_bin_keys = flat_bin_keys[sort_order]
    data_valid = data_valid[sort_order]
    error_valid = error_valid[sort_order]

    splits = np.nonzero(np.diff(flat_bin_keys))[0] + 1
    
    grouped_data = np.split(data_valid, splits)
    grouped_error = np.split(error_valid, splits)
    unique_bins = flat_bin_keys[np.insert(splits, 0, 0)]

    for b_idx, bin_key in enumerate(unique_bins):
        i = bin_key // num_t
        j = bin_key % num_t
        
        vals = grouped_data[b_idx]
        errs = grouped_error[b_idx]
        
        n_pixels_in_bin = len(vals)
        n_pixel_matrix[i, j] = n_pixels_in_bin
        
        mean_val = np.mean(vals)
        z_matrix[i, j] = mean_val
        std_matrix[i, j] = np.std(vals, ddof=1) if n_pixels_in_bin > 1 else 0.0

        weights = 1.0 / (errs ** 2)
        sum_weights = np.sum(weights)
        
        z_matrix_weighted[i, j] = np.sum(weights * vals) / sum_weights
        err_matrix[i, j] = 1.0 / np.sqrt(sum_weights)

    z_binned_image[valid_mask] = z_matrix[r_idx[valid_mask], t_idx[valid_mask]]

    return z_binned_image, (z_matrix, std_matrix, z_matrix_weighted, err_matrix, n_pixel_matrix, Rs, Ts)

'''BOOLEANS FOR DEBUGGING OR PERFORMANCE'''
PLOT_SPOT = False

''' CALCULATE RELATIVE CENTRES AND ERRORS FOR EACH RUN '''
for ch_idx, (ch, runs) in enumerate(RUN_CH_DICT.items()):
     for run_idx, RUN_STR in enumerate(runs):
        try:
            contour_file = np.load(f'{FIT_DIR}/contour_data_run_{RUN_STR}.npz')
            fit_file = np.load(f'{FIT_DIR}/fit_data_run_{RUN_STR}.npz')
            center_file = np.load(f'{FIT_DIR}/centers_run_{RUN_STR}.npz')
            MEAN  = np.load(f'{DATA_DIR}/warped_mean_{RUN_STR}.npz')["means"]
            ERROR = np.load(f'{DATA_DIR}/warped_mean_{RUN_STR}.npz')["error"]

        except FileNotFoundError:
            print(f"Data files missing for run {RUN_STR}. Skipping.")
            continue

        current_run_info = run_info[run_info['run'] == RUN_STR].iloc[0]
        center_x_px = current_run_info['centre_x_px']
        center_y_px = current_run_info['centre_y_px']
        
        center_x = current_run_info['centre_x']
        center_y = current_run_info['centre_y']
        center_z = current_run_info['centre_z']
        
        err_center_x_px = current_run_info['err_centre_x']
        err_center_y_px = current_run_info['err_centre_y']
        err_center_x = current_run_info['centre_x']
        err_center_y = current_run_info['centre_y']
        
        upright_width  = 3684
        upright_height = 5544

        r_bins = 20
        t_bins = 20
        binned_image, binned_info = split_pixels(MEAN, [center_x_px, center_y_px], ERROR, radius=3000, radial_division=r_bins, angular_division=t_bins)
        z_matrixes, std_matrixes, wz_matrices, werror_matrices, n_pixels, Rs, Ts = binned_info
        # binned_image = bin_image(MEAN, center_x, center_y, num_r_bins=20, num_theta_bins=20)
        
        current_height, current_width = MEAN.shape[:2]
        extent_mm = [
            y_fit(0), y_fit(current_width),   # New X bounds
            x_fit(current_height), x_fit(0)    # New Y bounds (keeping top-left origin)
        ]
        plt.imshow(binned_image, cmap='jet', extent=extent_mm, aspect='equal')
        plt.show()
        sys.exit()

        if PLOT_SPOT == True:
            try:
                file_name = f"./{PLOT_DIR}/spot_{RUN_STR}.png"
                plot_spot(
                    MEAN, 
                    center_x_px, 
                    center_y_px, 
                    file_name,
                    x_fit  = x_fit, 
                    y_fit  = y_fit,
                    width  = upright_width,
                    height = upright_height 
                )
            except Exception as e_plot:
                print(f"An error occurred while plotting the spot for run {RUN_STR}: {e_plot}")

        


sys.exit()

# Create figure and add gridspec
fig = plt.figure(figsize=(10, 8))
gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 3])
colors = ['crimson', 'darkorange', 'teal', 'mediumpurple', 'grey']

# Add subplots to grid
ax_top_left  = fig.add_subplot(gs[0, 0])
ax_top_right = fig.add_subplot(gs[0, 1], projection='3d')
ax_bot_left  = fig.add_subplot(gs[1, 0])                 
ax_bot_right = fig.add_subplot(gs[1, 1])
axes = [ax_top_left, ax_top_right, ax_bot_left, ax_bot_right]

# Set the projection plot aspect ratio
ax_top_right.set_box_aspect((1, 1, 3))

directions = []

for ch_idx, points in relative_centres.items():
    print(f"\n--- Channel {ch_idx} ---")
    points = np.array(points)              # Shape: (N, 3) -> [x, y, z]
    errors = np.array(err_centres[ch_idx]) # Shape: (N, 3) -> [err_x, err_y, err_z]

    for idx, (x, y, z) in enumerate(points):
        print(f"Original coordinates {idx + 1}: ({x:.2f}, {y:.2f}, {z:.2f}) mm")
        ax_top_right.plot(x, y, z)

    x_data, y_data, z_data = points[:, 0], points[:, 1], points[:, 2]
    x_err,  y_err,  z_err  = errors[:, 0], errors[:, 1], errors[:, 2]
    N = len(z_data)

    z_input = np.tile(z_data, 2)
    sz_input = np.tile(z_err, 2)

    xy_data = np.concatenate([x_data, y_data])
    sy_data = np.concatenate([x_err, y_err])

    def flat_multi_line_model(beta, z):
        half = len(z) // 2
        
        x_pred = beta[0] * z[:half] + beta[2]
        y_pred = beta[1] * z[half:] + beta[3]
        
        return np.concatenate([x_pred, y_pred])

    model_3d = odr.Model(flat_multi_line_model)
    data_3d = odr.RealData(x=z_input, y=xy_data, sx=sz_input, sy=sy_data)

    # Initial guess
    guess_mx = (x_data[-1] - x_data[0]) / (z_data[-1] - z_data[0] + 1e-6)
    guess_my = (y_data[-1] - y_data[0]) / (z_data[-1] - z_data[0] + 1e-6)
    guess = [guess_mx, guess_my, np.mean(x_data), np.mean(y_data)]

    # Run the fit
    odr_3d = odr.ODR(data_3d, model_3d, beta0=guess)
    output_3d = odr_3d.run()

    # Extract clean parameters
    m_x, m_y, c_x, c_y = output_3d.beta
    err_mx = output_3d.sd_beta[0]
    err_my = output_3d.sd_beta[1]
    var_mx = err_mx ** 2
    var_my = err_my ** 2

    # Plotting
    direction = np.array([m_x, m_y, 1.0])
    direction /= np.linalg.norm(direction)
    directions.append(direction)

    t_span = np.linspace(z_min, z_max, 100)
    line_points = np.zeros((100, 3))
    line_points[:, 0] = m_x * t_span + c_x  
    line_points[:, 1] = m_y * t_span + c_y  
    line_points[:, 2] = t_span              

    l_starting_point = (line_points[0, 0], line_points[0, 1], line_points[0, 2])
    l_ending_point = (line_points[-1, 0], line_points[-1, 1], line_points[-1, 2])
    line_vector = np.array(l_ending_point) - np.array(l_starting_point)

    n_ending_point = (line_points[0, 0], line_points[0, 1], line_points[-1, 2])
    normal_vector  = n_ending_point - np.array(l_starting_point)

    # Calculate angle between the line vector and the normal vector
    v1 = line_vector
    v2 = normal_vector
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(normal_vector)
    cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    channel_angles_deg[ch_idx] = angle_deg

    # Propagate the error in the angle using the variances of m_x and m_y
    slope_magnitude_sq = m_x**2 + m_y**2
    if slope_magnitude_sq > 1e-12:
        denom = (slope_magnitude_sq + 1) * np.sqrt(slope_magnitude_sq)
        d_theta_d_mx = m_x / denom
        d_theta_d_my = m_y / denom

        # Propagating using the true variances derived from sd_beta
        var_angle_rad = (d_theta_d_mx**2 * var_mx) + (d_theta_d_my**2 * var_my)
        angle_err_deg = np.degrees(np.sqrt(max(0.0, var_angle_rad)))
    else:
        angle_err_deg = 0.0

    print(f"Angle relative to normal: {angle_deg:.2f}° ± {angle_err_deg:.2f}°")
    
    # Plot Styles
    style_kw = {
        'marker':channel_markers[ch_idx],
        'markersize':7,
        'linestyle':'none', 
        'mfc':colors[ch_idx], 
        'mec':'black', 
        'ecolor':'black', 
        'capsize':3, 
        'label':f'Channel {ch_idx}'
        }

    ## Projection plot
    errors = np.zeros_like(points[:,0])
    ax_top_right.errorbar(points[:, 1], points[:, 0], points[:, 2], xerr=y_err, yerr=x_err, zerr=z_err, **style_kw)
    ax_top_right.plot(line_points[:, 1], line_points[:, 0], line_points[:, 2], color=colors[ch_idx], linestyle='--')
    ax_top_right.set_xlabel('X [mm]')
    ax_top_right.set_ylabel('Y [mm]')
    ax_top_right.set_zlabel('Z [mm]')
    ax_top_right.view_init(elev=-33, azim=165, roll=-80)
    ax_top_right.grid(True, linestyle='--', alpha=0.6, color='gray')

    ## XY Plot
    ax_bot_left.errorbar(points[:, 1], points[:, 0], xerr=y_err, yerr=x_err, **style_kw)
    print(f'XY Plot: ({points[:, 1]}, {points[:, 0]}) mm')
    ax_bot_left.plot(line_points[:, 1], line_points[:, 0], color=colors[ch_idx], linestyle='--')
    ax_bot_left.set_xlabel('X [mm]')
    ax_bot_left.set_ylabel('Y [mm]')

    xlim = ax_bot_left.get_xlim()
    ylim = ax_bot_left.get_ylim()

    ## XZ Plot
    ax_top_left.errorbar(points[:, 1], points[:, 2], xerr=y_err, yerr=z_err, **style_kw)
    ax_top_left.plot(line_points[:, 1], line_points[:, 2], color=colors[ch_idx], linestyle='--')
    ax_top_left.set_ylabel('Z [mm]') 
    ax_top_left.xaxis.set_tick_params(length=0)
    ax_top_left.xaxis.set_ticklabels([])

    ## ZY Plot
    ax_bot_right.errorbar(points[:, 2], points[:, 0], xerr=z_err, yerr=x_err, **style_kw)
    ax_bot_right.plot(line_points[:, 2], line_points[:, 0], color=colors[ch_idx], linestyle='--')
    ax_bot_right.set_xlabel('Z [mm]')
    ax_bot_right.yaxis.set_tick_params(length=0)
    ax_bot_right.yaxis.set_ticklabels([])

for ax in axes:
    ax.grid(True, linestyle='--', alpha=0.6, color='gray')

for ch_idx, angle in channel_angles_deg.items():
    print(f"--- Channel {ch_idx} ---")
    print(f"Radians: {angle*np.pi/180.:.8f}")
    print(f"Degrees: {angle:.8f}°")

all_d = np.vstack(directions)
median_direction = np.median(all_d, axis=0)
median_direction /= np.linalg.norm(median_direction)
relative_angles = [np.degrees(np.arccos(np.clip(np.dot(d, median_direction), -1.0, 1.0))) for d in directions]
for ch, ang in enumerate(relative_angles):
    print(f"Channel {ch} relative deviation: {ang:.8f}°")

plt.tight_layout()
plt.savefig('./plots/exp_display_laser-field.png')
plt.show()