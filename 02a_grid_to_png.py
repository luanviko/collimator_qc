import cv2
import numpy as np


def load_raw_image(path, width=5544, height=3684):
    # 1. Read the raw data
    img = np.fromfile(path, dtype=np.uint16)

    # 2. UNCOMMENT THE LINE BELOW if the image looks like corrupted television static:
    # img = img.byteswap()

    return img.reshape((height, width))


def convert_raw_to_16bit_png(raw_path, output_png_path):
    img_16bit = load_raw_image(raw_path)

    # 3. Scale the 12-bit camera data (0-4095) to full 16-bit range (0-65535)
    # This prevents the image from looking pitch black in image viewers.
    img_min = img_16bit.min()
    img_max = img_16bit.max()

    if img_max - img_min > 0:
        img_scaled = (
            ((img_16bit - img_min) / (img_max - img_min)) * 65535
        ).astype(np.uint16)
    else:
        img_scaled = np.zeros(img_16bit.shape, dtype=np.uint16)

    success = cv2.imwrite(output_png_path, img_scaled)

    if success:
        print(f"Successfully saved visible 16-bit PNG: {output_png_path}")
        print(f"Resolution: {img_16bit.shape[1]}x{img_16bit.shape[0]}")
    else:
        print("Failed to save the PNG file.")


if __name__ == "__main__":
    # raw_input = "./calibration/calibration_original.raw"
    # png_output = "./calibration/calibration_original.png"

    raw_input = "./perspective_correction_9998/calibration_9997.raw"
    png_output = "./perspective_correction_9998/calibration_9997.png"

    convert_raw_to_16bit_png(raw_input, png_output)