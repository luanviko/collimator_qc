import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import least_squares
from tqdm import tqdm   

# def load_raw_image(file_path, width, height, bpp):
#     if bpp == 8:
#         dtype = np.uint8
#     elif bpp == 16:
#         dtype = np.uint16
#     else:
#         raise ValueError("Unsupported bit depth. Use 8 or 16.")

#     img = np.fromfile(file_path, dtype=dtype)

#     try:
#         img = img.reshape((height, width))
#     except ValueError as e:
#         print(f"Error: Could not reshape. Check if dimensions {width}x{height} are correct.")
#         raise e

#     return img


def load_raw_image(path, width, height, bpp):
    # This is near-instantaneous compared to other methods
    return np.fromfile(path, dtype=np.uint16).reshape((height, width))


def preview_processed(img):
    plt.figure(figsize=(10, 8))
    plt.imshow(img, cmap='gray')
    plt.colorbar(label='Intensity')
    plt.show()


# def find_contours(z, n=100, steps=10):

#     z=z/z.sum()

#     t = np.linspace(0, z.max(), n)
#     integral = ((z >= t[:, None, None]) * z).sum(axis=(1,2))
#     f = interpolate.interp1d(integral, t)
    
#     f_min = f(float(integral.max()))
#     f_max = f(float(integral.min()))
#     t_contours = np.linspace(f_min, f_max, steps)
#     contours = plt.contour(z, t_contours)
#     plt.close()
#     return contours

# def find_contours(z, n=100, steps=10):

#     z_sum = z.sum()
#     z=z/z_sum
#     t = np.linspace(0, z.max(), n)
#     integral = ((z >= t[:, None, None]) * z).sum(axis=(1,2))
#     integral, unique_indices = np.unique(integral, return_index=True)
#     f = interpolate.interp1d(integral, t[unique_indices])
    
#     f_min = f(float(integral.max()))
#     f_max = f(float(integral.min()))
#     t_contours = np.linspace(f_min, f_max, steps)
#     contours = plt.contour(z, t_contours)
#     plt.close()
#     return contours.allsegs, t_contours*z_sum


def find_contours(z, n=100, base_percentiles=[0.25, 0.50, 0.80, 0.95]):
    z_sum = z.sum()
    z = z / z_sum
    t = np.linspace(0, z.max(), n)
    integral = ((z >= t[:, None, None]) * z).sum(axis=(1,2))
    integral, unique_indices = np.unique(integral, return_index=True)
    f = interpolate.interp1d(integral, t[unique_indices], bounds_error=False, fill_value="extrapolate")
    base_percentiles = np.array(base_percentiles)
    energy_min = float(integral.min())
    energy_max = float(integral.max())
    target_percentiles = energy_min + base_percentiles * (energy_max - energy_min)
    t_contours = np.sort(f(target_percentiles))
    contours = plt.contour(z, t_contours)
    plt.close()
    return contours.allsegs, t_contours * z_sum


def circle_residuals(params, x, y, x_error=None, y_error=None):

    x0, y0, R = params
    dx = x - x0
    dy = y - y0
    r = np.sqrt(dx**2 + dy**2)
    residuals = r - R

    if x_error is not None and y_error is not None:
        sigma_r = np.sqrt(((dx / r) * x_error)**2 + ((dy / r) * y_error)**2)
        residuals = residuals / sigma_r

    return residuals


# def ellipse_residuals(params, x, y, x_error=None, y_error=None):
#     x0, y0, a, b, theta = params

#     x_rot = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
#     y_rot = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
        
#     if x_error is not None and y_error is not None:
#         res_x = x_rot/(a*x_error)
#         res_y = y_rot/(b*y_error)
#     else:
#         res_x = x_rot/a
#         res_y = x_rot/b

#     return res_x**2 + res_y**2 - 1


def ellipse_residuals(params, x, y, x_error=None, y_error=None):
    x0, y0, a, b, theta = params
    x_rot = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
    y_rot = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
    if x_error is not None and y_error is not None:
        res_x = x_rot / (a * x_error)
        res_y = y_rot / (b * y_error)
    else:
        res_x = x_rot / a
        res_y = y_rot / b
    return (res_x**2 + res_y**2) - 1

def downsample(data, cluster_size):
    """
    Takes your calculated MEAN or SUM and shrinks it by averaging clusters.
    """
    h, w = data.shape
    # Shrink dimensions to be divisible by cluster_size
    new_h, new_w = h // cluster_size, w // cluster_size
    
    # Reshape and average across the clusters
    return data[:new_h*cluster_size, :new_w*cluster_size].reshape(
        new_h, cluster_size, new_w, cluster_size
    ).mean(axis=(1, 3)), new_h, new_w


def fit_stable_ellipse(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones_like(x)]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=float)
    M = np.linalg.inv(C1) @ M
    evals, evecs = np.linalg.eig(M)
    cond = 4 * evecs[0, :] * evecs[2, :] - evecs[1, :]**2
    a1 = evecs[:, cond > 0]
    a = np.vstack([a1, T @ a1]).flatten()
    b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b**2 - a*c
    x0 = (c*d - b*f) / num
    y0 = (a*f - b*d) / num
    up = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    down1 = (b**2 - a*c) * (np.sqrt((a - c)**2 + 4*b**2) - (a + c))
    down2 = (b**2 - a*c) * (-np.sqrt((a - c)**2 + 4*b**2) - (a + c))
    res_a = np.sqrt(up / down1)
    res_b = np.sqrt(up / down2)
    if b == 0:
        theta = 0.0 if a < c else np.pi/2
    else:
        theta = 0.5 * np.arctan2(2*b, (a - c))
    return x0, y0, res_a, res_b, theta


def gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude * np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2))) + offset
    return g.ravel() 


def tophat_super_gaussian_2d(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, p, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    
    # Rotation and scaling transformation
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    x_prime = cos_t * (x - xo) + sin_t * (y - yo)
    y_prime = -sin_t * (x - xo) + cos_t * (y - yo)
    
    # Super-gaussian core calculation (p controls edge sharpness/flatness)
    term = (np.abs(x_prime) / sigma_x)**(2 * p) + (np.abs(y_prime) / sigma_y)**(2 * p)
    g = amplitude * np.exp(-0.5 * term) + offset
    return g.ravel()