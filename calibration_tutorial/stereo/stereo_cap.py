"""Capture pairs of photos of the ChArUco board with two USB webcams, for stereo_calibration.py.

    python stereo_cap.py list     # save one photo from each camera index 0-5 in camera_list/, to find your webcams
    python stereo_cap.py 1 2      # left camera index, then right camera index (left and right as the cameras see)
    python stereo_cap.py 1 2 chessboard   # the Week 3 chessboard instead: it must be whole in both photos
    python stereo_cap.py 1 2 nostill      # save without waiting for a still board (hold it steady yourself)
Click the preview window first, then press c to save a pair, q or Esc to quit.

Two USB webcams are not synchronised: the two photos of a pair can be taken several frame times apart.
So hold the board still: the script saves a pair only after the board has been still for half a second
(unless you add nostill).
"""
import os
import re
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stereo_calibration import detect_board, common_corners  # noqa: E402  (same board and checks as the calibration)
from camera_names import (camera_label, camera_names, compare_cameras, record_cameras,  # noqa: E402
                          recorded_board, recorded_cameras)  # what the operating system calls each camera, which was left

output_folder = os.path.join(HERE, 'stereo_pairs')  # pairs go to stereo_pairs/left and stereo_pairs/right
frame_size = (640, 480)  # asked of both cameras; both must deliver the same size
target = 20  # pairs to aim for
max_motion = 1.5  # px: the board counts as still if its corners moved less than this (median) per frame ...
still_time = 0.5  # s: ... for this long (one camera's frames can lag the other's by a few frames)
max_drift = 3.0  # px: ... and drifted less than this (median) from where they were when the board stopped
capture_key = 'c'
exit_key = 'q'


def list_cameras(count=6):
    """Save one photo from each camera index, so you can see which index is which webcam."""
    folder = os.path.join(HERE, 'camera_list')
    os.makedirs(folder, exist_ok=True)
    names = camera_names()  # {index: name} on macOS and Linux
    for index in sorted(set(range(count)) | set(names)):
        name = f" {names[index]}" if index in names else ""
        cap = cv2.VideoCapture(index)
        ok, frame = False, None
        for _ in range(10 if cap.isOpened() else 0):  # a camera that is warming up can drop a few frames
            ok, frame = cap.read()
            if ok:
                break
        cap.release()
        if ok:
            cv2.imwrite(os.path.join(folder, f"camera_{index}.jpg"), frame)
            print(f"camera {index}{name}: {frame.shape[1]}x{frame.shape[0]}, photo saved as camera_list/camera_{index}.jpg")
        else:
            print(f"camera {index}{name}: no image" if name else f"camera {index}: no camera")


def open_camera(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # compressed video, so two webcams fit on one USB
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep few old frames (both requests work on Linux; others may ignore them)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
    if cap.isOpened() and any(cap.read()[0] for _ in range(10)):  # a camera that is warming up can drop a few frames
        return cap
    cap.release()
    sys.exit(f"Error: unable to read camera {index}. Run 'python stereo_cap.py list' to see which indices work.\n"
             "macOS: allow your terminal/IDE in System Settings > Privacy & Security > Camera, then restart it. "
             "Two webcams on one USB hub may not both start: use two USB ports.")


def corners_by_id(detection):
    """{corner ID: (x, y)} of one detect_board() result."""
    if detection[0] is None:
        return {}
    return dict(zip(detection[2].ravel().tolist(), detection[1].reshape(-1, 2)))


def corner_motion(now, before):
    """How far the corners seen in both frames moved (px, the median, so one jittery corner doesn't count),
    or None if they share no corner."""
    common = now.keys() & before.keys()
    return float(np.median([np.linalg.norm(now[i] - before[i]) for i in common])) if common else None


args = [a.lower() for a in sys.argv[1:]]
if 'list' in args:
    list_cameras()
    sys.exit()
indices = [int(a) for a in args if a.isdigit()]
board = 'chessboard' if 'chessboard' in args else 'charuco'
board_name = 'chessboard' if board == 'chessboard' else 'ChArUco board'
need_still = 'nostill' not in args  # wait for a still board before saving (recommended: the cameras are not synchronised)
if len(indices) != 2 or indices[0] == indices[1]:
    sys.exit("Give the left and the right camera index, e.g.: python stereo_cap.py 1 2\n"
             "To find the indices of your webcams: python stereo_cap.py list")

cameras = [open_camera(i) for i in indices]
names = camera_names()
labels = [camera_label(i, names) for i in indices]

# Create stereo_pairs/left and stereo_pairs/right, and continue numbering after earlier pairs
folders = [os.path.join(output_folder, side) for side in ('left', 'right')]
for folder in folders:
    os.makedirs(folder, exist_ok=True)
existing = sorted(f for f in os.listdir(folders[0])
                  if re.fullmatch(r"pair_\d+\.jpg", f) and os.path.exists(os.path.join(folders[1], f)))
pair_count = max(int(f[5:-4]) for f in existing) + 1 if existing else 0  # number for the next file name
saved_count = len(existing)  # pairs in the folders
recorded = recorded_cameras(output_folder)  # which cameras took the earlier pairs
if existing and recorded_board(output_folder) not in (None, board):
    sys.exit(f"The pairs in stereo_pairs/ were taken with the {recorded_board(output_folder)}: one calibration needs one "
             f"board. Use the same board{' (add chessboard)' if board == 'charuco' else ' (leave out chessboard)'}, "
             "or empty stereo_pairs/ to start again.")
if existing and recorded and compare_cameras(recorded, labels) == 'swapped':
    sys.exit(f"The pairs in stereo_pairs/ were taken with {recorded[0]} as the LEFT camera, but now it is the right one.\n"
             f"To add to them: python stereo_cap.py {indices[1]} {indices[0]}  (or empty stereo_pairs/ to start again)")
if existing and recorded and compare_cameras(recorded, labels) == 'different':
    print(f"WARNING: the earlier pairs were taken with {recorded[0]} (left) and {recorded[1]} (right). "
          "If these are other cameras or you moved one, empty stereo_pairs/ first.")
if existing:
    print(f"Note: {len(existing)} earlier pair(s) in stereo_pairs/ will be calibrated with the new ones. "
          "Empty the folders first if you moved a camera.")
coverage = [[], []]  # corners of saved pairs in each camera, drawn on the preview
for name in existing:
    for saved, folder in zip(coverage, folders):
        old = cv2.imread(os.path.join(folder, name), cv2.IMREAD_GRAYSCALE)
        if old is not None:
            saved.extend(corners_by_id(detect_board(old, board)).values())

print(f"Left camera {labels[0]}, right camera {labels[1]}, {board_name}. Click the preview window first, then:")
print(f"Press '{capture_key}' to save a pair (only when both cameras see the board"
      + (" and it is still)." if need_still else "; no still check, so hold it steady)."))
print(f"Press '{exit_key}' to quit.")

failed_reads = 0
first_frame = True
previous = [{}, {}]  # corners in the previous frame of each camera, for the stillness check
still_since = None  # when the board stopped moving
anchor = [{}, {}]  # the corners at that moment, to catch a slow drift

while True:
    # Grab both frames first and decode them afterwards: this keeps the two photos as close in time as possible
    grabbed = [cam.grab() for cam in cameras]
    frames = [cam.retrieve()[1] if ok else None for cam, ok in zip(cameras, grabbed)]
    if any(frame is None for frame in frames):
        failed_reads += 1
        if failed_reads < 10:  # a camera that is warming up can drop a few frames
            continue
        print("Error: unable to read from the cameras.")
        break
    failed_reads = 0
    if first_frame:
        sizes = [frame.shape[1::-1] for frame in frames]
        if sizes[0] != sizes[1]:
            print(f"Error: the cameras give different image sizes, left {sizes[0]} and right {sizes[1]}. "
                  "Use two webcams of the same model, or change frame_size at the top of this script.")
            break
        if existing:  # earlier pairs must have the same size, or the calibration has to drop some
            old = cv2.imread(os.path.join(folders[0], existing[0]))
            if old is not None and old.shape[1::-1] != sizes[0]:
                print(f"Error: the pairs in stereo_pairs/ are {old.shape[1]}x{old.shape[0]}, but the cameras now give "
                      f"{sizes[0][0]}x{sizes[0][1]}. Capture at the old size (frame_size), or empty stereo_pairs/ first.")
                break
        print(f"Capturing at {sizes[0][0]}x{sizes[0][1]}. Use the calibration only at this resolution.")
        first_frame = False

    # Find the board in both photos, the corners both cameras see, and how much the board moved
    detections = [detect_board(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), board, fast=True) for frame in frames]
    both = common_corners(detections[0], detections[1], board)
    now = [corners_by_id(d) for d in detections]
    motions = [corner_motion(n, p) for n, p in zip(now, previous)]
    previous = now
    drifts = [corner_motion(n, a) for n, a in zip(now, anchor)]
    if not all(m is not None and m <= max_motion for m in motions):
        still_since = None
    elif still_since is None or not all(d is not None and d <= max_drift for d in drifts):
        still_since, anchor = time.time(), now  # (re)start the still period here
    still = not need_still or (still_since is not None and time.time() - still_since >= still_time)

    # Preview: both photos side by side, corners in red, the corners of saved pairs as green dots
    views = [frame.copy() for frame in frames]
    for view, corners, saved in zip(views, now, coverage):
        for x, y in saved:
            cv2.circle(view, (int(x), int(y)), 3, (0, 255, 0), -1)
        for x, y in corners.values():
            cv2.circle(view, (int(x), int(y)), 5, (0, 0, 255), -1)
    if both is None:
        status, colour = "board not seen by both cameras", (0, 0, 255)
    elif not still:
        status, colour = f"{len(both[0])} shared corners - hold still", (0, 165, 255)
    else:
        status, colour = f"{len(both[0])} shared corners - ready", (0, 255, 0)
    preview = np.hstack(views)
    cv2.putText(preview, f"saved {saved_count}/{target}   {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.imshow("Stereo capture (left | right)", preview)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord(capture_key), ord(capture_key.upper())):  # Save the pair
        if both is None:
            print("Both cameras must see the " + ("whole chessboard" if board == 'chessboard' else
                                                   "board (at least 6 of the same corners)") + ". Move it and try again.")
        elif not still:
            print("The board is moving: hold it still for half a second (or rest it on something) and press c again.")
        else:
            name = f"pair_{pair_count:02d}.jpg"
            if all(cv2.imwrite(os.path.join(folder, name), frame) for folder, frame in zip(folders, frames)):
                record_cameras(output_folder, *labels, board)  # which cameras took the left and right photos, which board
                pair_count += 1
                saved_count += 1
                for saved, corners in zip(coverage, now):
                    saved.extend(corners.values())
                print(f"Pair saved: stereo_pairs/left/{name} and right/{name}  "
                      f"({saved_count}/{target}, {len(both[0])} shared corners)")
            else:
                print(f"Error: could not write {name} - check that {output_folder} is writable.")

    elif key in (ord(exit_key), ord(exit_key.upper()), 27):  # Exit the program: q, Q or Esc
        print("Exiting...")
        break

# Release the cameras and close all windows
for cam in cameras:
    cam.release()
cv2.destroyAllWindows()
