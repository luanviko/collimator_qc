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

## LOAD UP UNIT-CONVERSION DATA FOR ABSOLUTE POSITION CALCULATIONS
# calib = np.load('./unit_conversion/px_mm_conversion.npz')
calib = np.load('./unit_conversion_9997/px_mm_conversion_with_error_9997.npz')
x_fit = np.poly1d(calib['x_coeff'])
y_fit = np.poly1d(calib['y_coeff'])
x_cov = calib['x_cov']
y_cov = calib['y_cov']

## UNIT CONVERSION SLOPES FOR RELATIVE SIZE CALCULATIONS (e.g., ellipse axes)
a_x = calib['x_coeff'][0]
a_y = calib['y_coeff'][0]

# Load calibration variances and covariances
var_a_x  = x_cov[0, 0]
var_b_x  = x_cov[1, 1]
cov_ab_x = x_cov[0, 1]
var_a_y  = y_cov[0, 0]
var_b_y  = y_cov[1, 1]
cov_ab_y = y_cov[0, 1]

## CAMERA SETTINGS
width, height = 5544, 3684
bpp = 16

## ELLIPSE FITTING SETTINGS
FIT_DIR = './fits'

## RUN PARAMETERS
DIST = ['3.580', '4.369', '5.162']
# RUN_STRS_DICT = {
#     DIST[0]:['0005', '0006', '0007', '0008'],
#     DIST[1]:['0009', '0010', '0011', '0012'],
#     DIST[2]:['0001', '0002', '0003', '0004']
# }

RUN_STRS_DICT = {
    DIST[0]:['9927', '9928', '9929', '9930'],
    DIST[1]:['9922', '9923', '9924', '9925'],
    DIST[2]:['9917', '9918', '9919', '9920']
}

## Min and max distances
z_min = float(DIST[0])*1000.
z_max = float(DIST[2])*1000.

# Closest, middle, farthest
RUN_CH_DICT = {
    0:['9927', '9922', '9917'],
    1:['9928', '9923', '9918'],
    2:['9929', '9924', '9919'],
    3:['9930', '9925', '9920']
}

## COLORS FOR CHANNELS
channel_cmaps = {
    0: 'Blues',
    1: 'Greens',
    2: 'Reds',
    3: 'Purples'
}

# Centroids will be added to this dictionary
centres = {    
    0: [],  # Channel 0
    1: [],  # Channel 1
    2: [],  # Channel 2
    3: []   # Channel 3 
}

err_centres = {
    0: [],
    1: [],
    2: [],
    3: []
}

channel_markers = {
    0: 'o',  # Circle
    1: '^',  # Triangle
    2: '*',  # Star
    3: 's'   # Square
}

channel_angles_deg = {
    0: 0,
    1: 0,
    2: 0,
    3: 0
}

num_distances = len(RUN_STRS_DICT)


for ch_idx, (ch, runs) in enumerate(RUN_CH_DICT.items()):
     for run_idx, RUN_STR in enumerate(runs):
        try:
            contour_file = np.load(f'{FIT_DIR}/contour_data_run_{RUN_STR}.npz')
            fit_file = np.load(f'{FIT_DIR}/fit_data_run_{RUN_STR}.npz')
            center_file = np.load(f'{FIT_DIR}/centers_run_{RUN_STR}.npz')
        except FileNotFoundError:
            print(f"Data files missing for run {RUN_STR}. Skipping.")
            continue

        # Convert centers to mm
        center_x_px = center_file['center_x']
        center_y_px = center_file['center_y']
        center_x = x_fit(center_file['center_x'])
        center_y = y_fit(center_file['center_y'])
        center_z = 1000.*np.float64(DIST[run_idx])

        err_center_x_px = center_file['err_center_x']
        err_center_y_px = center_file['err_center_y']

        err_center_x = np.sqrt(
            (a_x * err_center_x_px)**2 + (center_x_px**2 * var_a_x) + var_b_x + (2 * center_x_px * cov_ab_x)
        )

        err_center_y = np.sqrt(
            (a_y * err_center_y_px)**2 + (center_y_px**2 * var_a_y) + var_b_y + (2 * center_y_px * cov_ab_y)
        )

        centres[ch].append((center_x, center_y, center_z))
        err_centres[ch].append((err_center_x, err_center_y, 1.))

        print(f"\n--- Run {RUN_STR} Centre Coordinates ---")
        print(f"  • Center X: {center_x:.2f} ± {err_center_x:.2f} mm")
        print(f"  • Center Y: {center_y:.2f} ± {err_center_y:.2f} mm")
        print(f"  • Center Z: {int(center_z):d} ± 1 mm")
        print("-" * 48)

        # Print 
        print(f"\n--- Run {RUN_STR} (Channel: {ch}) --- Centroid Coordinates: ({center_x:.2f}, {center_y:.2f}, {center_z}) mm")


# Create figure and add gridspec
fig = plt.figure(figsize=(10, 8))
gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 3])
colors = ['crimson', 'darkorange', 'teal', 'mediumpurple']

# Add subplots to grid
ax_top_left  = fig.add_subplot(gs[0, 0])
ax_top_right = fig.add_subplot(gs[0, 1], projection='3d')
ax_bot_left  = fig.add_subplot(gs[1, 0])                 
ax_bot_right = fig.add_subplot(gs[1, 1])
axes = [ax_top_left, ax_top_right, ax_bot_left, ax_bot_right]

# Set the projection plot aspect ratio
ax_top_right.set_box_aspect((1, 1, 3))

directions = []

for ch_idx, points in centres.items():
    print(f"\n--- Channel {ch_idx} ---")
    points = np.array(points)          # Shape: (N, 3) -> [x, y, z]
    errors = np.array(err_centres[ch_idx]) # Shape: (N, 3) -> [err_x, err_y, err_z]

    for idx, (x, y, z) in enumerate(points):
        print(f"Original coordinates {idx + 1}: ({x:.2f}, {y:.2f}, {z:.2f}) mm")
        ax_top_right.plot(x, y, z)

    # Extract clean vectors
    x_data, y_data, z_data = points[:, 0], points[:, 1], points[:, 2]
    x_err,  y_err,  z_err  = errors[:, 0], errors[:, 1], errors[:, 2]
    N = len(z_data)

    # ====================================================
    # FIXED: FLAT MULTIDIMENSIONAL ODR STRUCTURE
    # ====================================================
    # 1. Tile the independent variable Z to match the lengths of X and Y combined
    # Z becomes a single flat 1D array of length 2*N
    z_input = np.tile(z_data, 2)
    sz_input = np.tile(z_err, 2)

    # 2. Concatenate X and Y into a single flat 1D dependent array
    xy_data = np.concatenate([x_data, y_data])
    sy_data = np.concatenate([x_err, y_err])

    # 3. Adjust the model to process a flat 1D array based on an index split
    # beta = [slope_x, slope_y, intercept_x, intercept_y]
    def flat_multi_line_model(beta, z):
        # Find the split point in the flat array
        half = len(z) // 2
        
        # Calculate X for the first half, Y for the second half
        x_pred = beta[0] * z[:half] + beta[2]
        y_pred = beta[1] * z[half:] + beta[3]
        
        # Return a completely flat 1D array
        return np.concatenate([x_pred, y_pred])

    model_3d = odr.Model(flat_multi_line_model)
    data_3d = odr.RealData(x=z_input, y=xy_data, sx=sz_input, sy=sy_data)

    # Robust initial guesses based on endpoints
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
    
    # GENERATE PLOTTING & DIRECTION VARIABLES
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

    # ====================================================
    # CALCULATE NOMINAL ANGLE & ANALYTICAL PROPAGATION
    # ====================================================
    v1 = line_vector
    v2 = normal_vector
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(normal_vector)
    cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    channel_angles_deg[ch_idx] = angle_deg

    # ====================================================
    # CORRECTED DELTA METHOD DERIVATIVES
    # ====================================================
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
    style_kw = {'marker':channel_markers[ch_idx], 'linestyle':'none', 'mfc':colors[ch_idx], 'mec':'black'}

    ## Projection plot
    errors = np.zeros_like(points[:,0])
    ax_top_right.errorbar(points[:, 0], points[:, 1], points[:, 2], xerr=errors, yerr=errors, zerr=errors, **style_kw)
    ax_top_right.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], color=colors[ch_idx], linestyle='--')
    ax_top_right.set_xlabel('X [mm]')
    ax_top_right.set_ylabel('Y [mm]')
    ax_top_right.set_zlabel('Z [mm]')
    ax_top_right.view_init(elev=-33, azim=165, roll=-80)
    ax_top_right.grid(True, linestyle='--', alpha=0.6, color='gray')

    ## XY Plot
    ax_bot_left.errorbar(points[:, 0], points[:, 1], xerr=errors, yerr=errors, **style_kw)
    print(f'XY Plot: ({points[:, 0]}, {points[:, 1]}) mm')
    ax_bot_left.plot(line_points[:, 0], line_points[:, 1], color=colors[ch_idx], linestyle='--')
    ax_bot_left.set_xlabel('X [mm]')
    ax_bot_left.set_ylabel('Y [mm]')

    xlim = ax_bot_left.get_xlim()
    ylim = ax_bot_left.get_ylim()

    ## XZ Plot
    ax_top_left.errorbar(points[:, 0], points[:, 2], xerr=errors, yerr=errors, **style_kw)
    ax_top_left.plot(line_points[:, 0], line_points[:, 2], color=colors[ch_idx], linestyle='--')
    ax_top_left.set_ylabel('Z [mm]')
    # ax_top_left.set_xlim(xlim)    
    ax_top_left.xaxis.set_tick_params(length=0)
    ax_top_left.xaxis.set_ticklabels([])

    ## ZY Plot
    ax_bot_right.errorbar(points[:, 2], points[:, 1], xerr=errors, yerr=errors, **style_kw)
    ax_bot_right.plot(line_points[:, 2], line_points[:, 1], color=colors[ch_idx], linestyle='--')
    ax_bot_right.set_xlabel('Z [mm]')
    # ax_bot_right.set_ylim(ylim)
    ax_bot_right.yaxis.set_tick_params(length=0)
    ax_bot_right.yaxis.set_ticklabels([])

for ax in axes:
    ax.grid(True, linestyle='--', alpha=0.6, color='gray')

for ch_idx, angle in channel_angles_deg.items():
    print(f"--- Channel {ch_idx} ---")
    print(f"Radians: {angle*np.pi/180.:.4f}")
    print(f"Degrees: {angle:.2f}°")

all_d = np.vstack(directions)
median_direction = np.median(all_d, axis=0)
median_direction /= np.linalg.norm(median_direction)
relative_angles = [np.degrees(np.arccos(np.clip(np.dot(d, median_direction), -1.0, 1.0))) for d in directions]
for ch, ang in enumerate(relative_angles):
    print(f"Channel {ch+1} relative deviation: {ang:.3f}°")

plt.tight_layout()
plt.savefig('./plots/exp_display.png')
# plt.show()



