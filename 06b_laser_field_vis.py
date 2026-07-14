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

""" PX-MM CONVERSION """
calib = np.load('./unit_conversion_9997/px_mm_conversion_with_error_9997.npz')
mm_x_fit = np.poly1d(calib['x_coeff'])
mm_y_fit = np.poly1d(calib['y_coeff'])
a_x, b_x = mm_x_fit.coefficients
a_y, b_y = mm_y_fit.coefficients
mm_x_cov = calib['x_cov']
mm_y_cov = calib['y_cov']

""" VARIANCES FOR PX-MM CONVERSION """
mm_var_a_x  = mm_x_cov[0, 0]
mm_var_b_x  = mm_x_cov[1, 1]
mm_cov_ab_x = mm_x_cov[0, 1]
mm_var_a_y  = mm_y_cov[0, 0]
mm_var_b_y  = mm_y_cov[1, 1]
mm_cov_ab_y = mm_y_cov[0, 1]

""" LOAD CENTRES """
data = np.load('./laser_field.npz')
xo_px = data['xo']
yo_px = data['yo']
xo = mm_x_fit(xo_px)
yo = mm_y_fit(yo_px)
xo_err_px = data['xo_err']
yo_err_px = data['yo_err']

""" ERROR PROPAGATION (px -> mm) """
xo_err_mm = np.sqrt(
    (a_x**2 * xo_err_px**2) + 
    (xo_px**2 * mm_var_a_x) + 
    mm_var_b_x + 
    (2 * xo_px * mm_cov_ab_x)
)

yo_err_mm = np.sqrt(
    (a_y**2 * yo_err_px**2) + 
    (yo_px**2 * mm_var_a_y) + 
    mm_var_b_y + 
    (2 * yo_px * mm_cov_ab_y)
)

""" DISTANCES """
z      = data['z']
z_err  = np.array([0.001]*len(z))
z_fine = np.linspace(z.min(), z.max(), 500)

""" FIT Y """
m = 5
y_coeff_1, y_cov_1 = np.polyfit(z, yo, 1, w=1.0/yo_err_mm, cov='scaled')
y_coeff_n, y_cov_n = np.polyfit(z, yo, m, w=1.0/yo_err_mm, cov='scaled')
y_fit_1   = np.poly1d(y_coeff_1)
y_fit_n   = np.poly1d(y_coeff_n)

""" FIT X """
n = 5
x_coeff_n, x_cov_n = np.polyfit(z, xo, n, w=1.0/xo_err_mm, cov='scaled')
x_coeff_1, x_cov_1 = np.polyfit(z, xo, 1, w=1.0/xo_err_mm, cov='scaled')
x_fit_n   = np.poly1d(x_coeff_n)
x_fit_1   = np.poly1d(x_coeff_1)

""" SAVE ALL COEFFS TO FILE """
np.savez(
    './laser_field_fits.npz',
    y_coeff_1 = y_coeff_1,
    y_coeff_n = y_coeff_n,
    x_coeff_1 = x_coeff_1,
    x_coeff_n = x_coeff_n,
    x_cov_1 = x_cov_1,
    x_cov_n = x_cov_n,
    y_cov_1 = y_cov_1,
    y_cov_n = y_cov_n,
)

c_marker = 'oldlace'   
c_fit1   = 'darkorange'
c_fit2   = 'sandybrown'

fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1a.errorbar(z, yo, yerr=yo_err_mm, xerr=z_err, fmt='o', mfc=c_marker, mec='black', ecolor='black', capsize=3, label='Data')
ax1a.plot(z_fine, y_fit_1(z_fine), color=c_fit1, linestyle=':', linewidth=2, label='Linear Fit')
ax1a.plot(z_fine, y_fit_n(z_fine), color=c_fit2, linestyle='--', linewidth=2, label=f'n-th Order Poly (n = {m})')
ax1a.set_ylabel('Y [mm]')
ax1a.set_xlabel('Z [m]')
ax1a.legend()

c_marker = 'honeydew'
c_fit1 = 'forestgreen'
c_fit2 = 'darkseagreen'

ax1b.errorbar(z, xo, yerr=xo_err_mm, xerr=z_err, fmt='o', mfc=c_marker, mec='black', ecolor='black', capsize=3, label='Data')
ax1b.plot(z_fine, x_fit_1(z_fine), color=c_fit1, linestyle=':', linewidth=2, label='Linear')
ax1b.plot(z_fine, x_fit_n(z_fine), color=c_fit2, linestyle='--', linewidth=2, label=f'n-th Order Poly (n = {n})')
ax1b.set_ylabel('X [mm]')
ax1b.set_xlabel('Z [m]')
ax1b.legend()

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/laser_field.png", dpi=300)
plt.show()