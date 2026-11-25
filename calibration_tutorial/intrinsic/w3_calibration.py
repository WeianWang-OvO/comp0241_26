import cv2
import numpy as np
import glob
import os
import sys
from collections import Counter

# Step 1: Prepare object points
# Define the chessboard size (number of internal corners)
chessboard_size = (4, 7)  # INNER corners = (squares - 1) per direction; the calib.io 5x8-square board is (4, 7)
square_size = 34.0  # Edge of one printed square in mm. It only scales tvecs: K, distortion and errors do not depend on it
fix_k3 = True  # Few views that do not reach the image corners cannot pin down k3, so keep it at 0

# Paths are relative to this script, so it works from any working directory (e.g. the VS Code Run button)
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(script_dir, 'calibration_images')
# Windows need a display. Without one (e.g. over SSH), or with "nogui", pictures are only saved to files
show = 'nogui' not in [a.lower() for a in sys.argv[1:]] and (not sys.platform.startswith('linux')
                                                             or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))

# Prepare the object points for a standard chessboard
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all the images
objpoints = []  # 3D points in the real world
imgpoints = []  # 2D points in the image plane
used_files = []  # images whose chessboard was found
detections = []  # (file name, image size, corners)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # for cornerSubPix

# Step 2: Load calibration images (the captures only, not the *_corners_detected overlays written below)
images = sorted(f for f in glob.glob(os.path.join(image_folder, '*'))
                if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '_corners_detected' not in os.path.basename(f))
if not images:
    sys.exit(f"No images found in {image_folder}\n"
             "Run w3_calibration_cap.py first, or copy the files from sample_calibration_images/ into calibration_images/.")

for fname in images:
    name = os.path.basename(fname)
    img = cv2.imread(fname)
    if img is None:
        print(f"{name}: cannot be read - skipped")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 3: Find the chessboard corners, then refine them to sub-pixel accuracy
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if not ret:
        print(f"{name}: chessboard NOT found - skipped")
        continue
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    detections.append((fname, gray.shape[::-1], corners))
    print(f"{name}: chessboard found")

    # Optional: Draw and display the corners for visualization
    cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
    if show:
        cv2.imshow('Chessboard', img)
        cv2.waitKey(500)
    root, ext = os.path.splitext(fname)
    cv2.imwrite(root + '_corners_detected' + ext, img)

if show:
    cv2.destroyAllWindows()

if not detections:
    sys.exit(f"No {chessboard_size} chessboard found in {len(images)} images. chessboard_size counts INNER corners, "
             "i.e. (squares - 1) in each direction: the calib.io 5x8-square board is (4, 7).")

# All views must come from one camera at one resolution: keep the most common image size
image_size = Counter(size for _, size, _ in detections).most_common(1)[0][0]
for fname, size, corners in detections:
    if size != image_size:
        print(f"{os.path.basename(fname)}: size {size} differs from {image_size} - skipped")
        continue
    objpoints.append(objp)  # Add object points
    imgpoints.append(corners)  # Add image points
    used_files.append(fname)

print(f"\nChessboard found in {len(objpoints)} of {len(images)} images ({image_size[0]}x{image_size[1]})")
if len(objpoints) < 3:
    sys.exit("At least 3 views are needed. Capture more images with the whole board visible.")
if len(objpoints) < 10:
    print("WARNING: fewer than 10 views - the result will be unreliable (slide 9: capture 10-20)")

# Step 4: Perform camera calibration (ret = RMS re-projection error in pixels, the formula on slide 10)
flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None,
                                                                    flags=flags)

# Step 5: Display results
print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)

# Step 6: Save the calibration results
np.savez(os.path.join(script_dir, 'calibration_data.npz'), camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
         rvecs=rvecs, tvecs=tvecs, image_size=image_size, rms=ret)

# Re-projection error (printed before the undistortion demo, so it is never lost)
mean_error = 0
print("\nRMS re-projection error per image (px):")
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i].reshape(-1, 2), imgpoints2.reshape(-1, 2), cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
    print(f"  {os.path.basename(used_files[i])}: {error * np.sqrt(len(imgpoints2)):.3f}")

print("total error: {}".format(mean_error / len(objpoints)))  # OpenCV-tutorial metric: about RMS / sqrt(corners per image)
print(f"RMS re-projection error: {ret:.4f} px")

# Step 7: Use the calibration results to undistort an image
img = cv2.imread(used_files[0])  # Replace with an image to undistort (same camera, same resolution)
h, w = img.shape[:2]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# Undistort the image
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)


# Crop the image (optional, based on ROI), unless the valid region is tiny (over-fitted distortion model)
x, y, rw, rh = roi
if rw * rh >= 0.3 * w * h:
    undistorted_img = undistorted_img[y:y+rh, x:x+rw]
else:
    print("WARNING: the valid region after undistortion is tiny - capture more views that reach the image corners")

# Save the undistorted image, and display it next to the original
cv2.imwrite(os.path.join(script_dir, 'undistorted_example.jpg'), undistorted_img)
print(f"Undistorted image saved as undistorted_example.jpg ({undistorted_img.shape[1]}x{undistorted_img.shape[0]})")
if show:
    print("Showing the original and undistorted images: press any key in one of the windows to finish "
          "(add nogui to skip the windows).")
    cv2.imshow('Original Image', img)
    cv2.imshow('Undistorted Image', undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
