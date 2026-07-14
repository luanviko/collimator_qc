import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- CONFIGURATION ---
CALIB_FILE = "./perspective_correction_9998/calibration_9997.npz"
DATA_DIR = "/home/lkoerich/Data/collimator/preprocessed"

# Working from home
DATA_DIR = '/Users/lkoerich/Downloads/collimator_preprocessed'

# RUN_STRS = [f'{i:04d}' for i in range(9916, 9931)]
# RUN_STRS = ['9947']
# RUN_STRS = ['9950', '9951']
# RUN_STRS = ['9952', '9953', '9954']
# RUN_STRS = [f'{i:04d}' for i in range(9952, 9969)]
RUN_STRS = [f'{i:04d}' for i in range(9969, 9981)]
# RUN_STRS = ['9969']

# 1. Load the calibration parameters
calib_data = np.load(CALIB_FILE)
matrix = calib_data["matrix"]
out_w = int(calib_data["out_w"])
out_h = int(calib_data["out_h"])

for RUN_STR in RUN_STRS:

    try:
        MEAN = np.load(f"{DATA_DIR}/preprocessed_{RUN_STR}.npz")["means"]
        ERROR = np.load(f"{DATA_DIR}/preprocessed_{RUN_STR}.npz")["means_error"]

        precision_mean = MEAN.astype(np.float64)
        precision_error = ERROR.astype(np.float64)

        warped_mean = cv2.warpPerspective(
            precision_mean, 
            matrix, 
            (out_w, out_h)
        )

        warped_error = cv2.warpPerspective(
            precision_error, 
            matrix, 
            (out_w, out_h)
        )

        # rotated_mean  = cv2.rotate(precision_mean, cv2.ROTATE_90_CLOCKWISE)
        # rotated_error = cv2.rotate(precision_error, cv2.ROTATE_90_CLOCKWISE)

        OUTPUT_FILE = f"{DATA_DIR}/warped_mean_{RUN_STR}.npz"
        np.savez(OUTPUT_FILE, means=warped_mean, error=warped_error)
        print(f"Successfully applied calibration matrix to MEAN!")
        print(f"Saved warped array to: {OUTPUT_FILE}")
        print(f"Output shape: {warped_mean.shape} (Height x Width)")

        # plt.imshow(warped_mean, cmap='jet', vmin=0, vmax=100)
        # plt.show()


    except Exception as e:
        print(f"⚠️ Error processing {RUN_STR}: {e}.")
