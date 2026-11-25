"""See depth with your calibrated webcam pair: the rectified images and a disparity map.

    python stereo_depth.py 1 2                     # live: the same left and right camera as when capturing
    python stereo_depth.py sample_pairs/left/pair_00.jpg sample_pairs/right/pair_00.jpg     # or one pair of photos
It uses the calibration saved next to the pairs: stereo_pairs/stereo_calibration.npz (live), or the one in the folder
the photos come from. To use another one, add its path: python stereo_depth.py 1 2 my_pairs/stereo_calibration.npz
Add min=250 to measure points closer than the default 300 mm (the matcher then searches further along each row);
points closer than that get a wrong, far distance rather than none.
Click on the left image or on the disparity map to measure the distance to that point, and click a second point to
also measure the distance between the two. A ruler lying in view checks the calibration's scale: put two marks on it
200 mm apart (a cross on each of two sticky notes; its evenly spaced ticks can fool the matcher), click their centres
and compare. Keys: c clear the points, s save a snapshot, q or Esc quit.
With two photos, add "nogui" to only save the results (depth_rectified.jpg and depth_disparity.jpg).
"""
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from camera_names import camera_label, camera_names, compare_cameras  # noqa: E402

min_distance = 300.0  # mm: the closest distance to measure; sets how far the matcher searches along a row
block_size = 5  # side of the patches that are matched between the two images (odd, 3-11)
max_screen = (1280, 720)  # the window is scaled down to fit this

args = [a for a in sys.argv[1:] if a.lower() != 'nogui']
show = 'nogui' not in [a.lower() for a in sys.argv[1:]] and (not sys.platform.startswith('linux')
                                                             or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))
for a in args:
    if re.fullmatch(r'min=\d+(\.\d+)?', a.lower()):
        min_distance = float(a[4:])
files = [a for a in args if a.lower().endswith(('.jpg', '.jpeg', '.png'))]
indices = [int(a) for a in args if a.isdigit()]

# Load the calibration: a path you give, else the one next to the photos, else stereo_pairs/
given = [a for a in args if a.lower().endswith('.npz')]
if given:
    calibration_file = given[0]
elif len(files) == 2:
    calibration_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(files[0]))), 'stereo_calibration.npz')
else:
    calibration_file = os.path.join(HERE, 'stereo_pairs', 'stereo_calibration.npz')
if not os.path.exists(calibration_file):
    sys.exit(f"No calibration at {calibration_file}: run stereo_calibration.py on your pairs first.")
data = np.load(calibration_file)
size = tuple(int(v) for v in data['image_size'])
P2 = data['P2']
if abs(P2[1, 3]) > abs(P2[0, 3]):
    sys.exit("This calibration is for cameras above each other; this script needs them side by side.")
if P2[0, 3] > 0:
    sys.exit("In this calibration the 'left' camera is on the right (stereo_calibration.py warned about it). Swap the "
             "names of the left/ and right/ folders, delete cameras.txt, calibrate again, and use the camera indices "
             "the other way round.")
maps = [cv2.initUndistortRectifyMap(data[f'K{i}'], data[f'D{i}'], data[f'R{i}'], data[f'P{i}'], size, cv2.CV_16SC2)
        for i in (1, 2)]
Q = data['Q']
f = data['P1'][0, 0]  # focal length of the rectified images, in pixels
baseline = np.linalg.norm(data['T'])  # mm

# Disparity d = x_left - x_right of the same point; its distance is Z = f * baseline / d. The closest point
# we want to measure has the largest disparity, so the matcher searches up to that (a multiple of 16).
num_disparities = int(np.clip(16 * np.ceil(f * baseline / min_distance / 16), 16, 1024))
matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disparities, blockSize=block_size,
                                P1=8 * block_size ** 2, P2=32 * block_size ** 2, uniquenessRatio=10,
                                speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
print(f"Calibration {os.path.relpath(calibration_file)}: baseline {baseline:.1f} mm, f {f:.1f} px. Searching "
      f"disparities 0-{num_disparities} px: distances from about {f * baseline / num_disparities:.0f} mm "
      f"(add min=... to change; closer points get wrong, far distances). The left {num_disparities} px of the map "
      "cannot be matched.")


def looks_swapped(rectified_pair):
    """True if the scene in the right image is shifted to the RIGHT of the left image: the cameras are swapped."""
    grays = [np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in rectified_pair]
    (dx, _), response = cv2.phaseCorrelate(grays[0], grays[1], cv2.createHanningWindow(size, cv2.CV_32F))
    return dx > 5 and response > 0.05  # correct order: dx < 0 (near points sit further left in the right image)


SWAP_HINT = ("The images look swapped: the scene is shifted the wrong way between left and right. "
             "Are the two cameras (or photos) the wrong way round?")


def depth(rectified_pair):
    """Disparity (px, float; <= 0 where no match was found) of a rectified (left, right) pair."""
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in rectified_pair]
    return matcher.compute(grays[0], grays[1]).astype(np.float32) / 16  # SGBM returns 16 * disparity


def disparity_near(disparity, x, y, r=3):
    """Disparity at pixel (x, y), steadied by its neighbours in the (2r+1) x (2r+1) square around it: the median of
    those on the same surface (within 2 px of its own disparity). If (x, y) itself has no match: the median of the
    square's matched pixels, or None if fewer than half of them matched."""
    x, y = int(round(x)), int(round(y))
    square = disparity[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1]
    matched = square[square > 0]
    own = disparity[y, x]
    if own > 0:  # near an object's edge, part of the square sees what is behind it: leave that out
        return float(np.median(matched[np.abs(matched - own) <= 2]))
    return float(np.median(matched)) if 2 * matched.size >= square.size else None


def point_at(disparity, x, y):
    """3D point (mm, rectified left camera) at pixel (x, y), or None without a match."""
    d = disparity_near(disparity, x, y)
    if d is None:
        return None
    X, Y, Z, W = Q @ np.array([x, y, d, 1.0])  # Z / W is the same as f * baseline / d
    return np.array([X, Y, Z]) / W


# The window: rectified pair on top, disparity map below, scaled to fit the screen
scale = min(1.0, max_screen[0] / (2 * size[0]), max_screen[1] / (2 * size[1]))
points = []  # up to two clicked points, (x, y) in pixels of the rectified left image (the same on the map)
clicked = False  # a new click: print the measurement to the terminal once


def measurement(disparity):
    """Lines of text: the distance to each clicked point and, with two points, the distance between them."""
    found = [point_at(disparity, *p) for p in points]
    lines = [f"{i}: no match here" if p is None else
             f"{i}: Z = {p[2]:.0f} mm, straight-line distance {np.linalg.norm(p):.0f} mm"
             for i, p in enumerate(found, 1)]
    if len(found) == 2:
        lines.append("1 to 2: needs a match at both points" if any(p is None for p in found) else
                     f"1 to 2: {np.linalg.norm(found[0] - found[1]):.0f} mm apart")
    return lines


def report(disparity):
    """Print the measurement after a click."""
    global clicked
    if clicked:
        clicked = False
        print("   ".join(measurement(disparity)))


def mark_points(image, s):
    """The clicked points: numbered crosses (white with a black edge) and the line between them."""
    h, w = image.shape[:2]
    pixels = [(int(round(x)), int(round(y))) for x, y in points]
    labels = []  # the number goes above right of the cross, or on the other side near an edge
    for i, (x, y) in enumerate(pixels, 1):
        (tw, th), _ = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, max(1, round(4 * s)))
        gap = round(8 * s)
        labels.append((x + gap if x + gap + tw < w else x - gap - tw, y - gap if y - gap - th >= 0 else y + gap + th))
    for colour, width in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        if len(pixels) == 2:
            cv2.line(image, pixels[0], pixels[1], colour, max(1, round(width * s)), cv2.LINE_AA)
        for i, (p, label) in enumerate(zip(pixels, labels), 1):
            cv2.drawMarker(image, p, colour, cv2.MARKER_CROSS, round(20 * s), max(1, round((width + 1) * s)))
            cv2.putText(image, str(i), label, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, colour, max(1, round((width + 1) * s)),
                        cv2.LINE_AA)


def draw(rectified_pair, disparity, fit=True):
    w, h = size
    s = min(w / 640, h / 400)  # text and markers grow with the image
    pair = np.hstack(rectified_pair)
    for y in range(20, h, 40):
        cv2.line(pair, (0, y), (2 * w - 1, y), (0, 255, 0), 1)
    colour = cv2.applyColorMap(np.uint8(np.clip(disparity, 0, num_disparities) * 255 / num_disparities),
                               cv2.COLORMAP_JET)
    colour[disparity <= 0] = 0  # no match: black
    for image in (pair[:, :w], colour):  # the left image and the map have the same pixels
        mark_points(image, s)
    info = np.zeros_like(colour)
    lines = ["Disparity: red = near, blue = far, black = no match.",
             f"The left {num_disparities} px are never searched:",
             "there, the match would be outside the right image.",
             "Click the left image or the map: a point.",
             "Click again: a second point and the distance.",
             "c clears, s saves a snapshot, q or Esc quits."]
    if points:
        lines += [""] + measurement(disparity)
    for i, text in enumerate(lines):
        cv2.putText(info, text, (round(16 * s), round((36 + 32 * i) * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s,
                    (255, 255, 255), max(1, round(s)), cv2.LINE_AA)
    view = np.vstack([pair, np.hstack([colour, info])])
    return cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if fit and scale < 1 else view


def on_mouse(event, x, y, flags, param):
    global clicked
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    x, y = (x + 0.5) / scale - 0.5, (y + 0.5) / scale - 0.5  # the centre of the window pixel, in image pixels
    w, h = size
    if not (0 <= round(x) < w and 0 <= round(y) < 2 * h):  # only the left image (top) and the map (below it)
        return
    if round(y) >= h:
        y -= h  # on the map: the same pixel of the left image
    points[:] = points + [(x, y)] if len(points) == 1 else [(x, y)]  # the 3rd click starts again
    clicked = True


def save_snapshot(view):
    taken = [int(m.group(1)) for m in (re.fullmatch(r'depth_snapshot_(\d+)\.jpg', n) for n in os.listdir(HERE)) if m]
    name = f"depth_snapshot_{max(taken) + 1 if taken else 0:02d}.jpg"
    cv2.imwrite(os.path.join(HERE, name), view)
    print(f"Saved {name}")


if len(files) == 2:  # one pair of photos
    images = [cv2.imread(name) for name in files]
    if any(img is None for img in images) or any(img.shape[1::-1] != size for img in images):
        sys.exit(f"Could not read both photos at the calibrated size {size}.")
    rectified = [cv2.remap(img, m[0], m[1], cv2.INTER_LINEAR) for img, m in zip(images, maps)]
    disparity = depth(rectified)
    if looks_swapped(rectified):
        print("WARNING:", SWAP_HINT)
    centre = (size[0] // 2, size[1] // 2)
    p = point_at(disparity, *centre)
    print("At the image centre: " + ("no match" if p is None else
                                     f"Z = {p[2]:.0f} mm, straight-line distance {np.linalg.norm(p):.0f} mm"))
    matched = disparity > 0
    if matched.any():
        z = cv2.reprojectImageTo3D(disparity, Q)[..., 2][matched]
        print(f"Matched {100 * matched.mean():.0f} % of the pixels; their median Z is {np.median(z):.0f} mm "
              f"(from {np.percentile(z, 5):.0f} to {np.percentile(z, 95):.0f} mm for 90 % of them)")
    cv2.imwrite(os.path.join(HERE, 'depth_rectified.jpg'), np.hstack(rectified))
    points[:] = [centre]
    cv2.imwrite(os.path.join(HERE, 'depth_disparity.jpg'), draw(rectified, disparity, fit=False)[size[1]:, :size[0]])
    points.clear()  # in the window, the first click is point 1
    print("Saved depth_rectified.jpg and depth_disparity.jpg")
    if show:
        cv2.namedWindow("Stereo depth")
        cv2.setMouseCallback("Stereo depth", on_mouse)
        while True:
            view = draw(rectified, disparity)
            report(disparity)
            cv2.imshow("Stereo depth", view)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('s'), ord('S')):
                save_snapshot(view)
            elif key in (ord('c'), ord('C')):
                points.clear()
            elif key in (ord('q'), ord('Q'), 27):
                break
        cv2.destroyAllWindows()
    sys.exit()

if len(indices) != 2:
    sys.exit("Give the left and right camera index (python stereo_depth.py 1 2) or two photos.")
# Same cameras as during calibration? The capture scripts noted their names (with the USB location).
names = camera_names()
recorded = (str(data['camera_left']), str(data['camera_right'])) if 'camera_left' in data.files else ('', '')
if all(recorded) and names:
    now = [camera_label(i, names) for i in indices]
    verdict = compare_cameras(recorded, now)
    if verdict == 'swapped':
        print(f"WARNING: these cameras were calibrated the other way round ({recorded[0]} was left): "
              f"use python stereo_depth.py {indices[1]} {indices[0]}")
    elif verdict == 'different':
        print(f"WARNING: the calibration was made with {recorded[0]} (left) and {recorded[1]} (right), "
              f"not with {now[0]} and {now[1]}.")
cameras = []
for index in indices:
    cam = cv2.VideoCapture(index)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    cameras.append(cam)
cv2.namedWindow("Stereo depth")
cv2.setMouseCallback("Stereo depth", on_mouse)
failed_reads, empty_frames, hinted, swap_votes = 0, 0, False, []
while True:
    grabbed = [cam.grab() for cam in cameras]  # grab both first: the two photos as close in time as possible
    frames = [cam.retrieve()[1] if ok else None for cam, ok in zip(cameras, grabbed)]
    if any(frame is None or frame.shape[1::-1] != size for frame in frames):
        failed_reads += 1
        if failed_reads < 10:
            continue
        print(f"Error: no frames of the calibrated size {size} from cameras {indices}.")
        break
    failed_reads = 0
    rectified = [cv2.remap(frame, m[0], m[1], cv2.INTER_LINEAR) for frame, m in zip(frames, maps)]
    disparity = depth(rectified)
    if len(swap_votes) < 15:  # the first frames: are the cameras the right way round?
        swap_votes.append(looks_swapped(rectified))
        if len(swap_votes) == 15 and sum(swap_votes) > 7:
            print(f"WARNING: {SWAP_HINT} Try: python stereo_depth.py {indices[1]} {indices[0]}")
    empty_frames = empty_frames + 1 if (disparity > 0).mean() < 0.05 else 0
    if empty_frames > 60 and not hinted:
        print("Almost nothing matches. Are the two indices the wrong way round, is everything closer than "
              f"{min_distance:.0f} mm (add min=...), or is the scene plain? Point the cameras at something textured.")
        hinted = True
    view = draw(rectified, disparity)
    report(disparity)
    cv2.imshow("Stereo depth", view)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('s'), ord('S')):
        save_snapshot(view)
    elif key in (ord('c'), ord('C')):
        points.clear()
    elif key in (ord('q'), ord('Q'), 27):
        break
for cam in cameras:
    cam.release()
cv2.destroyAllWindows()
