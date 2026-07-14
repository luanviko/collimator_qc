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

''' LOAD RUN INFORMATION '''
column_types = {
    'run':int,
    'channel':int,
    'distance (m)': float
}
run_info = pd.read_csv('./experiment01.csv', dtype=column_types)

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

''' LASER CALIBRATION '''
fits_data = np.load('./laser_field_fits.npz')
y_coeff_1 = fits_data['y_coeff_1']
y_coeff_n = fits_data['y_coeff_n']
x_coeff_1 = fits_data['x_coeff_1']
x_coeff_n = fits_data['x_coeff_n']
x_cov_1   = fits_data['x_cov_1']
x_cov_n   = fits_data['x_cov_n']
y_cov_1   = fits_data['y_cov_1']
y_cov_n   = fits_data['y_cov_n']
x_fit_n   = np.poly1d(x_coeff_n)
x_fit_1   = np.poly1d(x_coeff_1)
y_fit_1   = np.poly1d(y_coeff_1)
y_fit_n   = np.poly1d(y_coeff_n)
a_x_1 = x_coeff_1[0]
a_x_n = x_coeff_n[0]
a_y_1 = y_coeff_1[0]
a_y_n = y_coeff_n[0]
var_a_x_1  = x_cov_1[0, 0]
var_b_x_1  = x_cov_1[1, 1]
cov_ab_x_1 = x_cov_1[0, 1]
var_a_x_n  = x_cov_n[0, 0]
var_b_x_n  = x_cov_n[1, 1]
cov_ab_x_n = x_cov_n[0, 1]
var_a_y_1  = y_cov_1[0, 0]
var_b_y_1  = y_cov_1[1, 1]
cov_ab_y_1 = y_cov_1[0, 1]
var_a_y_n  = y_cov_n[0, 0]
var_b_y_n  = y_cov_n[1, 1]
cov_ab_y_n = y_cov_n[0, 1]

## CAMERA SETTINGS
width, height = 5544, 3684
bpp = 16

## ELLIPSE FITTING SETTINGS
FIT_DIR = './fits'


## COLORS FOR CHANNELS
channel_cmaps = {
    0: 'Blues',
    1: 'Greens',
    2: 'Reds',
    3: 'Purples',
    4: 'Greys'
}

# Centroids will be added to this dictionary
centres = {    
    0: [],  # Channel 0
    1: [],  # Channel 1
    2: [],  # Channel 2
    3: [],  # Channel 3 
}

relative_centres = {    
    0: [],  # Channel 0
    1: [],  # Channel 1
    2: [],  # Channel 2
    3: [],  # Channel 3 
}

err_centres = {
    0: [],
    1: [],
    2: [],
    3: [],
}

err_relative_centres = {
    0: [],
    1: [],
    2: [],
    3: [],
}

channel_markers = {
    0: 'o',  # Circle
    1: '^',  # Triangle
    2: '*',  # Star
    3: 's',  # Square
    4: 'D',  # Diamond
}

distance_markers = {
    str(float(DIST[0])): 'o',  # Circle
    str(float(DIST[1])): '^',  # Triangle
    str(float(DIST[2])): '*',  # Star
}

channel_angles_deg = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0
}

num_distances = len(DIST)

lsr_centres = {DIST:() for DIST in DIST}
err_lsr_centres = {DIST:() for DIST in DIST}

lsr_y_offset = 0.

""" CALCULATE LASER FIELD CENTERS AND ERRORS """
USE_LASER_CORRECTION = True
N = 5
for distance in DIST:
    dist_val = np.float64(distance)

    if N == 1:
        laser_x = x_fit_1(dist_val)
        laser_y = y_fit_1(dist_val)
        laser_z = dist_val

        laser_x_err = np.sqrt((a_x_1 * 0.5)**2 + (dist_val**2 * var_a_x_1) + var_b_x_1 + (2 * dist_val * cov_ab_x_1))
        laser_y_err = np.sqrt((a_y_1 * 0.5)**2 + (dist_val**2 * var_a_y_1) + var_b_y_1 + (2 * dist_val * cov_ab_y_1))
        laser_z_err = 1.0 

    elif N == 5:
        laser_x = x_fit_n(dist_val)
        laser_y = y_fit_n(dist_val)
        laser_z = dist_val

        powers = np.arange(N, -1, -1)
        jacobian = dist_val ** powers

        poly_var_x = np.dot(jacobian, np.dot(x_cov_n, jacobian))
        laser_x_err = np.sqrt(poly_var_x)

        poly_var_y = np.dot(jacobian, np.dot(y_cov_n, jacobian))
        laser_y_err = np.sqrt(poly_var_y)

        laser_z_err = 1.0

    else: 
        raise ValueError("N must be either 1 or 5.")

    if USE_LASER_CORRECTION:
        lsr_centres[distance] = (laser_x, laser_y, laser_z)
        err_lsr_centres[distance] = (laser_x_err, laser_y_err, laser_z_err)
    else:
        lsr_centres[distance] = (0.0, 0.0, 0.0)
        err_lsr_centres[distance] = (0.0, 0.0, 0.0)

print("\n--- Laser Field Centres ---")
for distance, (x, y, z) in lsr_centres.items():
    print(f"  • Centre at {distance} m: ({x:.2f}, {y:.2f}, {z:.2f}) mm")

""" CALCULATE RELATIVE CENTRES AND ERRORS FOR EACH RUN """
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

        laser_x_err = err_lsr_centres[DIST[run_idx]][0]
        laser_y_err = err_lsr_centres[DIST[run_idx]][1]

        err_center_x = np.sqrt(
            (a_x * err_center_x_px)**2 + (center_x_px**2 * var_a_x) + var_b_x + (2 * center_x_px * cov_ab_x)
            + laser_x_err**2
        )

        err_center_y = np.sqrt(
            (a_y * err_center_y_px)**2 + (center_y_px**2 * var_a_y) + var_b_y + (2 * center_y_px * cov_ab_y)
            + laser_y_err**2
        )

        centres[ch].append((center_x, center_y, center_z))
        relative_centres[ch].append(
            (
                center_x - lsr_centres[DIST[run_idx]][0],
                center_y - lsr_centres[DIST[run_idx]][1],
                center_z
            )
        )
        err_centres[ch].append((err_center_x, err_center_y, 1.))

        row_iloc = run_info.index[run_info['run'] == int(RUN_STR)][0]

        run_info.at[row_iloc, 'centre_x'] = center_x
        run_info.at[row_iloc, 'centre_y'] = center_y
        run_info.at[row_iloc, 'centre_z'] = center_z
        run_info.at[row_iloc, 'err_centre_x'] = err_center_x
        run_info.at[row_iloc, 'err_centre_y'] = err_center_y
        run_info.at[row_iloc, 'err_centre_z'] = 1.

        run_info.at[row_iloc, 'rel_centre_x'] = center_x - lsr_centres[DIST[run_idx]][0]
        run_info.at[row_iloc, 'rel_centre_y'] = center_y - lsr_centres[DIST[run_idx]][1]
        run_info.at[row_iloc, 'rel_centre_z'] = center_z
        run_info.at[row_iloc, 'err_rel_centre_x'] = err_center_x
        run_info.at[row_iloc, 'err_rel_centre_y'] = err_center_y
        run_info.at[row_iloc, 'err_rel_centre_z'] = 1.

        run_info.at[row_iloc, 'laser_centre_x'] = lsr_centres[DIST[run_idx]][0]
        run_info.at[row_iloc, 'laser_centre_y'] = lsr_centres[DIST[run_idx]][1]
        run_info.at[row_iloc, 'laser_centre_z'] = lsr_centres[DIST[run_idx]][2]
        run_info.at[row_iloc, 'err_laser_centre_x'] = err_lsr_centres[DIST[run_idx]][0]
        run_info.at[row_iloc, 'err_laser_centre_y'] = err_lsr_centres[DIST[run_idx]][1]
        run_info.at[row_iloc, 'err_laser_centre_z'] = err_lsr_centres[DIST[run_idx]][2] 

        run_info.at[row_iloc, 'distance'] = DIST[run_idx]
        run_info.at[row_iloc, 'centre_x_px'] = center_x_px
        run_info.at[row_iloc, 'centre_y_px'] = center_y_px

        # print(f"\n--- Run {RUN_STR} Centre Coordinates ---")
        # print(f"  • Center X: {center_x:.2f} ± {err_center_x:.2f} mm")
        # print(f"  • Center Y: {center_y:.2f} ± {err_center_y:.2f} mm")
        # print(f"  • Center Z: {int(center_z):d} ± 1 mm")
        # print("-" * 48)

        # Print 
        # print(f"\n--- Run {RUN_STR} (Channel: {ch}) --- Centroid Coordinates: ({center_x:.2f}, {center_y:.2f}, {center_z}) mm")

print("Saving run_info dataframe to hdf5...")
try:
    run_info.to_hdf('./run_info_with_centres.h5', key='run_info', mode='w')
    run_info.to_csv('./run_info_with_centres.csv', index=False)
    print("Run information saved successfully.")
except Exception as e:
    print(f"Error saving run_info dataframe: {e}")
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
mxs = []
mys = []
err_mxs = []
err_mys = []

for ch_idx, points in relative_centres.items():
    print(f"\n--- Channel {ch_idx} ---")
    points = np.array(points)              # Shape: (N, 3) -> [x, y, z]
    errors = np.array(err_centres[ch_idx]) # Shape: (N, 3) -> [err_x, err_y, err_z]

    # for idx, (x, y, z) in enumerate(points):
    #     print(f"Original coordinates {idx + 1}: ({x:.2f}, {y:.2f}, {z:.2f}) mm")
    #     ax_top_right.plot(x, y, z)

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

    mxs.append(m_x)
    mys.append(m_y)
    err_mxs.append(err_mx)
    err_mys.append(err_my)

    # Plotting
    direction = np.array([m_x, m_y, 1.0])
    direction /= np.linalg.norm(direction)
    directions.append(direction)

    t_span = np.linspace(z_min, z_max, 100)
    line_points = np.zeros((100, 3))
    line_points[:, 0] = m_x * t_span + c_x  
    line_points[:, 1] = m_y * t_span + c_y  
    line_points[:, 2] = t_span    

    theta = np.sqrt(m_x**2 + m_y**2)
    D60 = 60*theta
    sigma_theta = np.sqrt( (m_x/theta * err_mx)**2 + (m_y/theta * err_my)**2 )    
    sigma_D60 = 60.*sigma_theta
    D_QC = D60 + 2.*sigma_D60

    print(f"θ: {theta*180./np.pi}°")
    print(f"σ_θ: {sigma_theta*180./np.pi}")
    print(f"D₆₀: {D60}")
    print(f"σ_D₆₀: {sigma_D60}")
    print(f"D_QC: {D_QC}")

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


    # print(f"{theta*180./np.pi = }, {D60 = }, {sigma_theta*180./np.pi = }, {sigma_D60 = }, {D_QC = }")


theta_i = [ (mxs[i], mys[i]) for i in range(0, len(mxs)) ]
average_theta = np.mean(theta_i, axis=0)
delta = np.array([np.abs(d - average_theta) for d in theta_i])

N = len(mxs)
err_theta_i = np.array([ (err_mxs[i], err_mys[i]) for i in range(0, len(mxs)) ])

err_avg_mx = np.sqrt(np.sum(np.array(err_mxs)**2)) / N
err_avg_my = np.sqrt(np.sum(np.array(err_mys)**2)) / N
err_avg = np.array([err_avg_mx, err_avg_my])

err_delta_i_x = [np.sqrt(err_theta_i[i][0]**2 + err_avg_mx**2) for i in range(0, len(mxs))]
err_delta_i_y = [np.sqrt(err_theta_i[i][1]**2 + err_avg_my**2) for i in range(0, len(mxs))]

sigma_delta = np.array([ [err_delta_i_x[i], err_delta_i_y[i]] for i in range(0, len(mxs)) ])

D_QC_rel = 60.*delta + 2*sigma_delta

for ch_idx, D_QC_rel_i in enumerate(D_QC_rel):
    print(f"Channel {ch_idx}. D_QC_rel_i: {D_QC_rel_i}")

sys.exit()

# all_d = np.vstack(directions)
# median_direction = np.median(all_d, axis=0)
# median_direction /= np.linalg.norm(median_direction)
# relative_angles = [np.degrees(np.arccos(np.clip(np.dot(d, median_direction), -1.0, 1.0))) for d in directions]

# delta = [np.abs(d - median_direction) for d in directions] 
# D_rel = 60.*delta
# D_QC_rel = D_rel + 2.*

# # for ch, ang in enumerate(relative_angles):
# #     print(f"Channel {ch} relative deviation: {ang:.8f}°")

# plt.tight_layout()
# plt.savefig('./plots/exp_display_laser-field.png')
# plt.show()