import cv2
import os
import re
import sys

# Parameters
script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_dir, "calibration_images")  # Folder next to this script, from any working directory
chessboard_size = (4, 7)  # Inner corners in the chessboard (must match w3_calibration.py)
capture_key = 'c'  # Key to capture the image
exit_key = 'q'  # Key to exit the program
target = 17  # the 17-photo shot list in the tutorial (slide 9: at least 10-20 images)

# Command-line options, in any order:
#   a number  camera index, e.g. "python w3_calibration_cap.py 1" for a second camera
#   charuco   capture a ChArUco board for w3_calibration_charuco.py (option B) instead of the chessboard
args = [a.lower() for a in sys.argv[1:]]
camera_index = next((int(a) for a in args if a.isdigit()), 0)
use_charuco = 'charuco' in args

flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK  # fast enough for live video
if use_charuco:
    sys.path.insert(0, script_dir)
    from w3_calibration_charuco import detect_board  # same board settings as the ChArUco calibration


def find_board(gray):
    """Return (found, corners) for the chosen board. A ChArUco board may be partly outside the frame."""
    if use_charuco:
        obj, corners, ids, _, _ = detect_board(gray)
        return obj is not None, corners
    return cv2.findChessboardCorners(gray, chessboard_size, flags=flags)


# Open the webcam (use the appropriate index if multiple cameras are connected)
cap = cv2.VideoCapture(camera_index)

# A camera that is warming up can drop a few frames, so allow ten tries
if not (cap.isOpened() and any(cap.read()[0] for _ in range(10))):
    sys.exit(f"Error: Unable to access camera {camera_index}. Try another index: python w3_calibration_cap.py 1\n"
             "macOS: allow your terminal/IDE in System Settings > Privacy & Security > Camera, then restart it. "
             "Close Zoom/Teams/OBS if they are using the camera.")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Continue numbering after earlier captures instead of overwriting them
existing = sorted(f for f in os.listdir(output_folder) if re.fullmatch(r"calibration_image_\d+\.jpg", f))
image_count = max(int(f[18:-4]) for f in existing) + 1 if existing else 0  # number for the next file name
saved_count = len(existing)  # images in the folder
coverage = []  # corner positions of saved views, drawn on the preview so uncovered image areas stand out
for f in existing:
    old = cv2.imread(os.path.join(output_folder, f), cv2.IMREAD_GRAYSCALE)
    found, corners = find_board(old) if old is not None else (False, None)
    if found:
        coverage.extend(corners.reshape(-1, 2))
if existing:
    print(f"Note: {len(existing)} earlier image(s) in calibration_images/ will be calibrated with the new ones. "
          "Empty the folder first if you changed camera or resolution.")

print(f"Board: {'ChArUco (option B)' if use_charuco else f'chessboard {chessboard_size}'}")
print("Click the preview window first, then:")
print(f"Press '{capture_key}' to capture an image and save it (only when the corners are drawn).")
print(f"Press '{exit_key}' to quit.")

failed_reads = 0
first_frame = True

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        failed_reads += 1
        if failed_reads < 10:  # a camera that is warming up can drop a few frames
            continue
        print("Error: Unable to read from the camera.")
        break
    failed_reads = 0
    if first_frame:
        print(f"Capturing at {frame.shape[1]}x{frame.shape[0]}. Use the calibration only at this resolution.")
        first_frame = False

    # Convert the frame to grayscale for board detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = find_board(gray)

    # Show the frame with the detected corners, the saved corners and a counter (the saved image stays clean)
    preview = frame.copy()
    for x, y in coverage:
        cv2.circle(preview, (int(x), int(y)), 3, (0, 255, 0), -1)
    if found and use_charuco:
        for x, y in corners.reshape(-1, 2):
            cv2.circle(preview, (int(x), int(y)), 5, (0, 0, 255), -1)
    elif found:
        cv2.drawChessboardCorners(preview, chessboard_size, corners, found)
    cv2.putText(preview, f"saved {saved_count}/{target}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if found else (0, 0, 255), 2)
    cv2.imshow("Calibration Capture", preview)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    if key in (ord(capture_key), ord(capture_key.upper())):  # Capture and save the image
        if found:
            image_path = os.path.join(output_folder, f"calibration_image_{image_count:02d}.jpg")
            if cv2.imwrite(image_path, frame):
                image_count += 1
                saved_count += 1
                coverage.extend(corners.reshape(-1, 2))
                print(f"Image saved: {os.path.relpath(image_path, script_dir)}  ({saved_count}/{target})")
            else:
                print(f"Error: could not write {image_path} - check that the folder is writable.")
        else:
            print(f"No {'ChArUco board' if use_charuco else 'chessboard'} detected. Please adjust the position and try again.")

    elif key in (ord(exit_key), ord(exit_key.upper()), 27):  # Exit the program: q, Q or Esc
        print("Exiting...")
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
