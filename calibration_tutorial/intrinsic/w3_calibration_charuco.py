"""Week 3, option B: calibrate a camera with a ChArUco board.

Same steps as w3_calibration.py, but the target is a ChArUco board: a chessboard
with an ArUco marker in every white square. Every inner corner has an ID, so a
photo still counts when part of the board is outside the frame or covered.
That lets you push the board right into the corners of the image, where lens
distortion is largest. For stereo calibration later, it also lets two cameras
use the corners they both see.

Capture photos with "python w3_calibration_cap.py charuco", then run:
    python w3_calibration_charuco.py                 # reads ./calibration_images next to this file
    python w3_calibration_charuco.py path/to/photos  # or any folder you name
Add "nogui" to skip the pop-up windows (automatic when there is no display, e.g. over SSH).
Optional: add "fisheye" to fit OpenCV's fisheye lens model instead (Kannala-Brandt, four coefficients), for
lenses wider than about 120 degrees. Ordinary webcams don't need it; to try it without a fisheye lens:
    python w3_calibration_charuco.py sample_fisheye_images fisheye

Needs OpenCV 4.7 or newer, including OpenCV 5 (pip install opencv-contrib-python).
"""
import glob
import os
import sys

import cv2
import numpy as np

if not hasattr(cv2, 'aruco') or not hasattr(cv2.aruco, 'CharucoDetector'):
    sys.exit(f"Option B needs OpenCV 4.7 or newer with its ArUco module, but this is OpenCV {cv2.__version__}.\n"
             "Update it inside your environment: pip install -U opencv-contrib-python")

HERE = os.path.dirname(os.path.abspath(__file__))

# Step 1: Describe the board. These must match the board you printed. The defaults match
# docs/charuco_board_A4.pdf next to this script (also linked from the tutorial page)
# and the board in sample_charuco_images/.
SQUARES_X = 7            # squares across
SQUARES_Y = 5            # squares down
SQUARE_LENGTH = 30.0     # side of a chessboard square, in mm (measure your print)
MARKER_LENGTH = 15.0     # side of an ArUco marker, in mm
DICTIONARY = cv2.aruco.DICT_6X6_250
MIN_CORNERS = 6          # skip photos with fewer corners than this
# Windows need a display: none over SSH, or when "nogui" is given
SHOW = 'nogui' not in [a.lower() for a in sys.argv[1:]] and (not sys.platform.startswith('linux')
                                                             or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))
FISHEYE = 'fisheye' in [a.lower() for a in sys.argv[1:]]  # optional: the fisheye lens model, see calibrate_fisheye()

dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
detector = cv2.aruco.CharucoDetector(board)


def detect_board(gray):
    """Find the ChArUco corners in one grayscale image.

    Returns (object_points, image_points, ids, marker_corners, marker_ids).
    The first three are None if too few corners were found, or if they all lie
    on one line (a single row of corners cannot constrain the camera).
    """
    corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS:
        return None, None, None, marker_corners, marker_ids
    obj, img = board.matchImagePoints(corners, ids)  # each corner's 3D position on the board, found by its ID
    xy = obj.reshape(-1, 3)[:, :2]
    if np.linalg.matrix_rank(xy - xy.mean(axis=0), tol=1e-6) < 2:
        return None, None, None, marker_corners, marker_ids
    return obj, img, ids, marker_corners, marker_ids


def calibrate_fisheye(objpoints, imgpoints, image_size):
    """Optional: calibrate with OpenCV's fisheye model (cv2.fisheye) instead of the standard one.

    A fisheye lens maps a ray at angle theta from the optical axis to the radius r = f * theta_d, with
    theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8) (Kannala and Brandt, 2006).
    The standard model works with tan(theta) instead, which grows without limit as theta nears 90 degrees.
    Returns the same five values as cv2.calibrateCamera; D then holds k1..k4.
    """
    obj = [o.reshape(-1, 1, 3).astype(np.float64) for o in objpoints]  # cv2.fisheye wants N x 1 x 3 (refuses N x 3)
    img = [i.reshape(-1, 1, 2).astype(np.float64) for i in imgpoints]
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7)
    try:
        return cv2.fisheye.calibrate(obj, img, image_size, np.zeros((3, 3)), np.zeros((4, 1)),
                                     flags=flags, criteria=criteria)
    except cv2.error as error:
        sys.exit(f"The fisheye calibration failed: {str(error).strip().splitlines()[-1]}\n"
                 "Use more photos, with the board in every part of the image, and leave out very tilted views.")


def project(obj, rvec, tvec, camera_matrix, dist_coeffs):
    """Where the calibrated camera puts the board corners, with the model that was fitted."""
    if FISHEYE:
        return cv2.fisheye.projectPoints(obj.reshape(-1, 1, 3).astype(np.float64), rvec, tvec,
                                         camera_matrix, dist_coeffs)[0]
    return cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)[0]


def main():
    # Step 2: Load the photos (any .jpg/.jpeg/.png, any case) from the folder.
    folders = [a for a in sys.argv[1:] if a.lower() not in ('nogui', 'fisheye')]
    image_folder = folders[0] if folders else os.path.join(HERE, 'calibration_images')
    files = sorted(f for f in glob.glob(os.path.join(image_folder, '*'))
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '_corners_detected' not in os.path.basename(f))
    if not files:
        sys.exit(f"No images found in {image_folder}")

    # Detection overlays go to their own folder, so a second run never reads them back. A run on the bundled
    # samples (sample_...) keeps its overlays and result apart, so it never replaces your own calibration.
    tag = os.path.basename(os.path.normpath(image_folder))
    sample = tag.startswith('sample_')
    output_folder = os.path.join(HERE, 'charuco_detections', tag) if sample else os.path.join(HERE, 'charuco_detections')
    os.makedirs(output_folder, exist_ok=True)

    objpoints, imgpoints, names = [], [], []
    image_size = None
    for fname in files:
        name = os.path.basename(fname)
        img = cv2.imread(fname)
        if img is None:
            print(f"{name}: cannot read, skipped")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]  # (width, height)
        if image_size is not None and size != image_size:
            print(f"{name}: size {size} differs from {image_size}, skipped")
            continue

        # Step 3: Find the ChArUco corners and match them to their 3D positions on the board.
        obj, imgp, ids, marker_corners, marker_ids = detect_board(gray)
        vis = img.copy()
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
        if obj is None:
            print(f"{name}: not enough corners, skipped")
            continue
        image_size = image_size or size  # the first photo with a board sets the image size
        cv2.aruco.drawDetectedCornersCharuco(vis, imgp.reshape(-1, 1, 2), ids, (0, 0, 255))
        cv2.imwrite(os.path.join(output_folder, name), vis)
        if SHOW:
            cv2.imshow('ChArUco', vis)
            cv2.waitKey(300)
        objpoints.append(obj)
        imgpoints.append(imgp)
        names.append(name)
        print(f"{name}: {len(ids)} of {(SQUARES_X - 1) * (SQUARES_Y - 1)} corners")
    if SHOW:
        cv2.destroyAllWindows()

    if len(objpoints) < 3:
        sys.exit(f"Only {len(objpoints)} usable photos; take at least 10.")

    # Step 4: Calibrate. ret is the RMS reprojection error of slide 10, in pixels.
    if FISHEYE:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_fisheye(objpoints, imgpoints, image_size)
    else:
        # up to 500 iterations: the default 30 can stop before the answer settles (e.g. on wide-angle lenses)
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, image_size, None, None,
            criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 500, 1e-9))

    # Step 5: Display results.
    print(f"\nUsed {len(objpoints)} of {len(files)} photos, image size {image_size}")
    print("Camera Matrix:\n", camera_matrix)
    print("\nDistortion Coefficients" + (" (fisheye k1, k2, k3, k4)" if FISHEYE else "") + ":\n", dist_coeffs.ravel())
    print("\nRMS reprojection error (px):", ret)
    print("\nPer-photo RMS (px):")
    for i, name in enumerate(names):
        proj = project(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = imgpoints[i].reshape(-1, 2) - proj.reshape(-1, 2)
        print(f"  {name}: {np.sqrt((err ** 2).sum(axis=1).mean()):.3f}")

    # Step 6: Save the results next to this script (a fisheye result in its own file: it needs the cv2.fisheye functions).
    result_file = ('calibration_data_fisheye' if FISHEYE else 'calibration_data_charuco') + (f'_{tag}' if sample else '') + '.npz'
    np.savez(os.path.join(HERE, result_file), camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
             image_size=np.array(image_size), rms=ret, model='fisheye' if FISHEYE else 'standard')
    print(f"Saved {result_file}")

    # Step 7: Undistort the first usable photo and show it next to the original.
    img = cv2.imread(os.path.join(image_folder, names[0]))
    w, h = image_size
    if FISHEYE:  # balance=0 zooms in until the whole output is image; 1 zooms out until each edge's middle is in view
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            camera_matrix, dist_coeffs, (w, h), np.eye(3), balance=0.0)
        if not np.isfinite(new_camera_matrix).all() or new_camera_matrix[0, 0] < 0.2 * camera_matrix[0, 0]:
            new_camera_matrix = camera_matrix  # the photos don't reach the image edges, so the edges can't be undistorted
        undistorted_img = cv2.fisheye.undistortImage(img, camera_matrix, dist_coeffs, Knew=new_camera_matrix)
    else:
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    cv2.imwrite(os.path.join(output_folder, 'undistorted_' + names[0]), undistorted_img)
    print(f"Undistorted {names[0]} saved in {os.path.relpath(output_folder, HERE)}/")
    if SHOW:
        print("Showing the original and undistorted images: press any key in one of the windows to finish "
              "(add nogui to skip the windows).")
        cv2.imshow('Original Image', img)
        cv2.imshow('Undistorted Image', undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
