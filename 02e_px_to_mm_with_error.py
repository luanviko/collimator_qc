import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np, sys
from scipy.signal import find_peaks

haxis = 100
yaxis = 2851

x_mm, _ = np.loadtxt('./unit_conversion/mmpx_short.csv', delimiter=',', skiprows=1, unpack=True)
y_mm, _ = np.loadtxt('./unit_conversion/mmpx_long.csv', delimiter=',', skiprows=1, unpack=True)
print(x_mm, x_mm.shape)
print(y_mm, y_mm.shape)
# sys.exit(0)

img = Image.open('./perspective_correction_9998/calibration_fixed.png')
img = img.convert("L")
img_inverted = ImageOps.invert(img)
horizontal_line = np.array(img_inverted)[haxis, :]
vertical_line = np.array(img_inverted)[:, yaxis]

horizontal_samples = np.arange(horizontal_line.shape[0])
vertical_samples = np.arange(vertical_line.shape[0])
print(horizontal_line.shape, vertical_line.shape)


fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
ax1.imshow(img, cmap='gray')
ax1.axhline(y=haxis, color="red", linestyle="--", linewidth=1.5, label=f"Horizontal Axis (y={haxis})")
ax1.set_title("Horizontal Axis", fontsize=14, pad=15)
ax1.set_ylabel("[Pixels]", fontsize=11)
ax1.set_xlim(0, img.width)
ax1.set_ylim(220, 0)
x_min = 300
x_max = 2600
peaks, _ = find_peaks(horizontal_line[x_min:x_max], distance=20, prominence=30)
# peaks, _ = find_peaks(horizontal_line[x_min:x_max], distance = 20, height=20 )
ax2.plot(horizontal_samples, horizontal_line, 'k-')
ax2.plot(horizontal_samples[peaks+x_min], horizontal_line[peaks+x_min], "x", color='red', markersize=8)
ax2.grid('on', linestyle='--', linewidth=0.5)
ax2.set_xlabel("[Pixels]", fontsize=11)
ax2.axis('on')
x_px = peaks + x_min
plt.savefig('./unit_conversion_9997/horizontal.png', dpi=300, bbox_inches='tight')
plt.close()

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 12), sharey=True)
ax1.imshow(img, cmap='gray')
ax1.axvline(x=yaxis, color="green", linestyle="--", linewidth=1.5, label=f"Vertical Axis (x={yaxis})")
ax1.set_title("Vertical Axis", fontsize=14, pad=15)
ax1.set_xlabel("[Pixels]", fontsize=11)
ax1.set_ylabel("[Pixels]", fontsize=11)
ax1.set_xlim(2680, 2900)

x_min = 260
x_max = 4300
peaks, _ = find_peaks(vertical_line[x_min:x_max], height = 25, distance=20, prominence=10)
ax2.plot(vertical_line, vertical_samples, 'k-')
ax2.plot(vertical_line[peaks+x_min], vertical_samples[peaks+x_min], "x", color='green', markersize=8)
ax2.grid('on', linestyle='--', linewidth=0.5)
ax2.axis('on')
y_px = peaks + x_min
print(len(y_px), len(x_px))

plt.savefig('./unit_conversion_9997/vertical.png', dpi=300, bbox_inches='tight')

plt.close()

mm_error = 0.5
weights_x = np.ones_like(x_mm) / mm_error
weights_y = np.ones_like(y_mm) / mm_error

fig2, (ax3) = plt.subplots(1, 1, figsize=(10, 6), sharey=True)
x_coeff, x_cov = np.polyfit(x_px, x_mm, 1, w=weights_x, cov='unscaled')
x_fit = np.poly1d(x_coeff)
x_fit_sample = np.linspace(x_px.min(), x_px.max(), 100)
ax3.plot(x_fit_sample, x_fit(x_fit_sample), 'k--')
ax3.plot(x_px, x_mm, 'o', color='red', label='Horizontal Axis')

y_coeff, y_cov = np.polyfit(y_px, y_mm, 1, w=weights_y, cov='unscaled')
y_fit = np.poly1d(y_coeff)
y_fit_sample = np.linspace(y_px.min(), y_px.max(), 100)
ax3.plot(y_fit_sample, y_fit(y_fit_sample), 'k--')
ax3.plot(y_px, y_mm, 'o', color='green', label='Vertical Axis')

np.savez(
    'px_mm_conversion_with_error_9997.npz', 
    x_coeff=x_coeff, 
    y_coeff=y_coeff,
    x_cov=x_cov,
    y_cov=y_cov)
print("Fitting coefficients saved to 'px_mm_conversion.npz'")
plt.close()

img_width_mm = x_fit(img.width) - x_fit(0)
img_height_mm = y_fit(img.height) - y_fit(0)

# Calculate the precise boundaries in mm
x_start_mm = x_fit(0)
x_end_mm = x_fit(img.width)
y_start_mm = y_fit(0)
y_end_mm = y_fit(img.height)

# Plot the image using the physical millimeter extent
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img, cmap='gray', extent=[x_start_mm, x_end_mm, y_end_mm, y_start_mm])

ax.set_title("Calibrated Image (Millimeters)", fontsize=14)
ax.set_xlabel("Width [mm]", fontsize=11)
ax.set_ylabel("Height [mm]", fontsize=11)

plt.savefig('./unit_conversion_9997/calibrated_image_mm.png', dpi=300, bbox_inches='tight')
# plt.show()

# ==========================================
# 4. Clean and Formatted Printout
# ==========================================
print("\n" + "="*50)
print("          CALIBRATION RESULTS (1D Polyfit)          ")
print("="*50)

# Extract linear parameters (a = slope, b = intercept)
a_x = x_coeff[0]
b_x = x_coeff[1]

# Extract variances and covariance from the matrix elements
var_a_x  = x_cov[0, 0]
var_b_x  = x_cov[1, 1]
cov_ab_x = x_cov[0, 1]

# Compute standard errors (square root of variance)
err_a_x = np.sqrt(var_a_x)
err_b_x = np.sqrt(var_b_x)
print(f"▶ HORIZONTAL AXIS (X):")
print(f"  • Scale Factor (a):  {a_x:11.6f} ± {err_a_x:.6f} mm/px")
print(f"  • Intercept    (b):  {b_x:11.4f} ± {err_b_x:.4f} mm")
print(f"  • Covariance (a,b):  {x_cov[0,1]:11.8f}")
print("-"*50)

a_y = y_coeff[0]
b_y = y_coeff[1]

# Extract variances and covariance from the matrix elements
var_a_y  = y_cov[0, 0]
var_b_y  = y_cov[1, 1]
cov_ab_y = y_cov[0, 1]

# Compute standard errors (square root of variance)
err_a_y = np.sqrt(var_a_y)
err_b_y = np.sqrt(var_b_y)
print(f"▶ VERTICAL AXIS (Y):")
print(f"  • Scale Factor (a):  {a_y:11.6f} ± {err_a_y:.6f} mm/px")
print(f"  • Intercept    (b):  {b_y:11.4f} ± {err_b_y:.4f} mm")
print(f"  • Covariance (a,b):  {y_cov[0,1]:11.8f}")
print("="*50 + "\n")