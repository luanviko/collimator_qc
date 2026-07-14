import cv2
import cv2.aruco as aruco
import numpy as np


def save_true_aspect_grid(
    config_filename, input_path, output_filename="natural_grid.jpg"
):
    img = cv2.imread(input_path)
    if img is None:
        return print("Error: Image not found.")

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 3:
        return print(
            f"Error: Need at least 3 markers. Found {len(ids) if ids is not None else 0}."
        )

    pm = {
        int(id_val): np.mean(c[0], axis=0)
        for id_val, c in zip(ids.flatten(), corners)
    }

    # Reconstruct ID 1 using the updated 90-degree CW layout:
    # True Layout: ID 2 (TL), ID 0 (TR), ID 1 (BR), ID 3 (BL)
    if 1 not in pm and all(k in pm for k in [0, 2, 3]):
        print(
            "Marker ID 1 is missing. Mathematically reconstructing it for 90° CW rotation..."
        )
        # Parallelogram rule for this configuration: BR = TR + BL - TL
        pm[1] = pm[0] + pm[3] - pm[2]

    try:
        # Reordered array to match the updated orientation: [TL, TR, BR, BL]
        src_pts = np.array([pm[2], pm[0], pm[1], pm[3]], dtype="float32")
    except KeyError as e:
        return print(
            f"Error: Cannot reconstruct grid. Missing required ID {e}"
        )

    # Width and height calculations swap due to the rotation
    width = int(
        max(np.linalg.norm(pm[0] - pm[2]), np.linalg.norm(pm[1] - pm[3]))
    )
    height = int(
        max(np.linalg.norm(pm[3] - pm[2]), np.linalg.norm(pm[1] - pm[0]))
    )

    margin = 150
    dst_pts = np.array(
        [
            [margin, margin],  # TL
            [width + margin, margin],  # TR
            [width + margin, height + margin],  # BR
            [margin, height + margin],  # BL
        ],
        dtype="float32",
    )

    out_w = width + 2 * margin
    out_h = height + 2 * margin

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    np.savez(config_filename, matrix=matrix, out_w=out_w, out_h=out_h)
    print(f"Calibration successful! Saved parameters to '{config_filename}'")

    warped = cv2.warpPerspective(img, matrix, (out_w, out_h))
    cv2.imwrite(output_filename, warped)
    print(f"Success! Saved as {output_filename}")
    print(f"Final Image Resolution: {out_w}x{out_h} px")


if __name__ == "__main__":
    config_file = "./perspective_correction_9998/calibration_9997.npz"
    calibration_image = "./perspective_correction_9998/calibration_9997.png"
    output_image = "./perspective_correction_9998/calibration_fixed.png"
    save_true_aspect_grid(config_file, calibration_image, output_filename=output_image)