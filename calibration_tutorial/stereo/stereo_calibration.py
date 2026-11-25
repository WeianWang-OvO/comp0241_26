"""Stereo calibration: where is the right webcam relative to the left one?

Uses pairs of photos of the ChArUco board (or the chessboard), taken with stereo_cap.py or stereo_capture_gui.py:
    python stereo_calibration.py                 # reads stereo_pairs/left and stereo_pairs/right next to this file
    python stereo_calibration.py sample_pairs    # or another folder with left/ and right/ inside
The result goes into that folder: stereo_calibration.npz and rectified_example.jpg.
Add "nogui" to skip the pop-up window (automatic when there is no display, e.g. over SSH).
Which board: the one noted in cameras.txt by the capture scripts, else the one found in the photos. To choose
it yourself, add "charuco" or "chessboard". The ChArUco board works best: a pair counts when both cameras see
part of it. The chessboard (the Week 3 one, 4 x 7 inner corners, 34 mm squares) must be whole in both photos.

Steps
  1. Find the board corners in every photo (the detectors of the Week 3 tutorial).
  2. Calibrate each camera on its own: K and distortion, as in Week 3.
  3. Stereo calibration: keep K and distortion fixed and find R and T, the pose of the right camera
     relative to the left one, from the corners that both cameras saw.
  4. Rectification: turn both images so that matching points lie on the same image row.
  5. Check the result and save it.
"""
import os
import sys
from collections import Counter

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'intrinsic'))
from w3_calibration_charuco import detect_board as detect_charuco  # noqa: E402  (board: top of w3_calibration_charuco.py)
sys.path.insert(0, HERE)
from camera_names import recorded_board, recorded_cameras  # noqa: E402  (noted when capturing: cameras and board)

BOARDS = ('charuco', 'chessboard')
CHESSBOARD_SIZE = (4, 7)  # the Week 3 chessboard (calib.io, 5 x 8 squares): inner corners, as in w3_calibration.py
CHESSBOARD_SQUARE = 34.0  # mm: side of one printed square, as in w3_calibration.py (measure your print)
MIN_COMMON = 6  # skip pairs in which the two cameras share fewer corners than this
FIX_K3 = True   # as in the Week 3 chessboard script: a few views cannot pin down k3, so keep it at 0
MAX_STEREO_RMS = 1.0  # px: above this, some pairs are bad (the board moved between the two photos, blur)
MAX_ROW_DIFFERENCE = 1.5  # px: after rectification, a corner must be on (almost) the same row in both images
#                          (the worst of ~20 corners per pair; good real webcam pairs reach about 1.3 px)
MAX_DISAGREEMENT = 0.02  # flag pairs whose two board distances differ by more than 2 %
# Windows need a display: none over SSH, or when "nogui" is given
SHOW = 'nogui' not in [a.lower() for a in sys.argv[1:]] and (not sys.platform.startswith('linux')
                                                             or bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')))


CHESS_OBJECT = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 1, 3), np.float32)  # the corners on the board, mm
CHESS_OBJECT[:, 0, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * CHESSBOARD_SQUARE
SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # as in w3_calibration.py


def detect_board(gray, board='charuco', fast=False):
    """Find the board in one grayscale image: (object_points, image_points, ids, marker_corners, marker_ids).

    The first three are None if the board was not found. ChArUco: the Week 3 option-B detector, which numbers
    every corner. Chessboard: the whole board must be in view; its 28 corners are numbered 0-27 in the order
    OpenCV finds them, with sub-pixel refinement as in w3_calibration.py. fast=True is the quick search for live
    video (no refinement).
    """
    if board == 'charuco':
        return detect_charuco(gray)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + (cv2.CALIB_CB_FAST_CHECK if fast else 0)
    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags=flags)
    if not found:
        return None, None, None, None, None
    if not fast:
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX)
    return CHESS_OBJECT.copy(), corners, np.arange(len(corners)).reshape(-1, 1), None, None


def same_order(left, right):
    """The right camera's chessboard corners in the same order as the left camera's.

    OpenCV starts numbering at a black corner square. The Week 3 board (5 x 8 squares) swaps its black and white
    corners when turned by 180 degrees, so both photos are numbered alike and this is not used for it. A board
    with an even total of squares along its two sides (e.g. 6 x 8 or 7 x 9 squares) looks the same turned round,
    and OpenCV may then number it from opposite ends. In the right order, every corner shifts by about the same
    amount from the left photo to the right one (this assumes both cameras are mounted the same way up).
    """
    a, b = left[1].reshape(-1, 2), right[1].reshape(-1, 2)
    if (a - b).var(axis=0).sum() <= (a - b[::-1]).var(axis=0).sum():
        return right
    return right[0], right[1][::-1].copy(), right[2], right[3], right[4]


def common_corners(left, right, board='charuco'):
    """Board points and pixels of the corners BOTH cameras saw, in the same order, or None if too few.

    left and right are detect_board() results: (object_points, image_points, ids, ...).
    """
    if left[0] is None or right[0] is None:
        return None
    if board == 'chessboard' and sum(CHESSBOARD_SIZE) % 2 == 0:  # a board that looks the same turned round
        right = same_order(left, right)
    ids_left, ids_right = left[2].ravel(), right[2].ravel()
    common = np.intersect1d(ids_left, ids_right)
    if len(common) < MIN_COMMON:
        return None
    pick_left = [np.flatnonzero(ids_left == i)[0] for i in common]
    pick_right = [np.flatnonzero(ids_right == i)[0] for i in common]
    obj = left[0][pick_left]
    xy = obj.reshape(-1, 3)[:, :2]
    if np.linalg.matrix_rank(xy - xy.mean(axis=0), tol=1e-6) < 2:  # all on one line: no use
        return None
    return obj, left[1][pick_left], right[1][pick_right]


def find_folder(name):
    """A folder named on the command line: relative to where you are, or else to this script."""
    for base in (os.getcwd(), HERE):
        if os.path.isdir(os.path.join(base, name)):
            return os.path.join(base, name)
    return name


def photos_in(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png')))


def choose_board(folder, names, args):
    """('charuco' or 'chessboard', why): given on the command line, noted in cameras.txt, or found in the photos."""
    for board in BOARDS:
        if board in args:
            return board, "given on the command line"
    board = recorded_board(folder)
    if board:
        return board, "noted in cameras.txt when capturing"
    found = Counter()
    for name in names[:6]:
        gray = cv2.imread(os.path.join(folder, 'left', name), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            found.update(b for b in BOARDS if detect_board(gray, b, fast=True)[0] is not None)
    if not found:
        return 'charuco', "no board found in the first photos, so ChArUco is assumed; add chessboard to use the other"
    board = 'chessboard' if found['chessboard'] > found['charuco'] else 'charuco'
    return board, f"found in the photos; add {'charuco' if board == 'chessboard' else 'chessboard'} to use the other"


def cameras_other_way(obj_both, img_left, img_right, names, K1, D1, K2, D2):
    """Names of the pairs taken with the cameras the other way round from the rest (the smaller group).

    Each pair on its own tells where the right camera sits relative to the left one: the board's pose seen by
    each camera, from solvePnP. Pairs taken with the cameras swapped put it on the opposite side.
    """
    centres = []
    for obj, il, ir in zip(obj_both, img_left, img_right):
        _, rvec_l, tvec_l = cv2.solvePnP(obj, il, K1, D1)
        _, rvec_r, tvec_r = cv2.solvePnP(obj, ir, K2, D2)
        Ri = cv2.Rodrigues(rvec_r)[0] @ cv2.Rodrigues(rvec_l)[0].T
        centres.append((-Ri.T @ (tvec_r - Ri @ tvec_l)).ravel())  # right camera centre, left-camera coordinates
    centres = np.array(centres)
    if len(centres) < 2:
        return []
    side = centres @ np.linalg.svd(centres, full_matrices=False)[2][0]  # along the line the cameras sit on
    group = side > 0
    if group.sum() < (~group).sum() or (group.sum() == (~group).sum() and not group[0]):
        group = ~group  # keep the larger group (on a tie, the group of the first pair)
    return [n for n, g in zip(names, group) if not g]


def uncovered(points, size):
    """Which ninths of the image (a 3 x 3 grid) no corner reached."""
    w, h = size
    seen = {(min(int(3 * x / w), 2), min(int(3 * y / h), 2)) for x, y in points}
    names = ['top left', 'top', 'top right', 'left', 'centre', 'right', 'bottom left', 'bottom', 'bottom right']
    return [names[3 * row + col] for row in range(3) for col in range(3) if (col, row) not in seen]


def main():
    # Step 1: Load the pairs (the same file name in left/ and right/) and find the corners.
    words = [a.lower() for a in sys.argv[1:]]
    args = [a for a in sys.argv[1:] if a.lower() not in ('nogui',) + BOARDS]
    if not args:  # a pairs folder may itself be called chessboard or charuco
        args = [a for a in sys.argv[1:] if a.lower() in BOARDS and os.path.isdir(os.path.join(find_folder(a), 'left'))][:1]
        if args:
            words.remove(args[0].lower())
    folder = find_folder(args[0]) if args else os.path.join(HERE, 'stereo_pairs')
    names = [n for n in photos_in(os.path.join(folder, 'left')) if os.path.exists(os.path.join(folder, 'right', n))]
    if not names:
        sys.exit(f"No pairs found in {folder}/left and {folder}/right.\n"
                 "Capture some with stereo_capture_gui.py (or stereo_cap.py), or try: python stereo_calibration.py sample_pairs")
    board, why = choose_board(folder, names, words)
    print(f"Board: {'ChArUco' if board == 'charuco' else f'chessboard, {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners of {CHESSBOARD_SQUARE:g} mm'} ({why})")
    pairs = {}
    for name in names:
        images = [cv2.imread(os.path.join(folder, side, name)) for side in ('left', 'right')]
        if images[0] is None or images[1] is None:
            print(f"{name}: cannot read, skipped")
            continue
        pairs[name] = images
    # All photos must have one size: use the most common one, so one stray photo cannot spoil the rest
    image_size = Counter(img.shape[1::-1] for images in pairs.values() for img in images).most_common(1)[0][0]

    obj_left, img_left, obj_right, img_right = [], [], [], []  # every view of each camera, for step 2
    names_left, names_right = [], []  # the pair each of those views comes from
    obj_both, img_both_left, img_both_right, pair_names = [], [], [], []  # corners seen by both, for step 3
    blank = 0  # pairs in which neither photo shows the board
    for name, images in pairs.items():
        sizes = [img.shape[1::-1] for img in images]  # (width, height)
        if sizes[0] != image_size or sizes[1] != image_size:
            print(f"{name}: sizes {sizes[0]} and {sizes[1]} differ from {image_size}, skipped")
            continue
        left, right = [detect_board(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), board) for img in images]
        if left[0] is not None:
            obj_left.append(left[0])
            img_left.append(left[1])
            names_left.append(name)
        if right[0] is not None:
            obj_right.append(right[0])
            img_right.append(right[1])
            names_right.append(name)
        both = common_corners(left, right, board)
        if both is not None:
            obj_both.append(both[0])
            img_both_left.append(both[1])
            img_both_right.append(both[2])
            pair_names.append(name)
        blank += left[0] is None and right[0] is None
        count = lambda d: 0 if d[0] is None else len(d[0])
        print(f"{name}: left {count(left)} corners, right {count(right)}, both {0 if both is None else len(both[0])}")

    other = 'charuco' if board == 'chessboard' else 'chessboard'
    blank_hint = (f" {blank} of {len(pairs)} pairs show no {'chessboard' if board == 'chessboard' else 'ChArUco board'} "
                  f"at all: were they taken with the other board? Add {other}, or delete them." if 2 * blank >= len(pairs) else "")
    if min(len(obj_left), len(obj_right), len(obj_both)) < 3:
        sys.exit(f"Too few usable photos (left {len(obj_left)}, right {len(obj_right)}, pairs {len(obj_both)}); "
                 "take at least 10 pairs in which both cameras see the board"
                 + (" (the whole chessboard)." if board == 'chessboard' else ".") + blank_hint)

    # Step 2: Calibrate each camera on its own (ret = RMS reprojection error in pixels, as in Week 3).
    flags = cv2.CALIB_FIX_K3 if FIX_K3 else 0
    rms_left, K1, D1, _, _ = cv2.calibrateCamera(obj_left, img_left, image_size, None, None, flags=flags)
    rms_right, K2, D2, _, _ = cv2.calibrateCamera(obj_right, img_right, image_size, None, None, flags=flags)

    # Were all pairs taken with the cameras the same way round? Mixing pairs taken both ways gives a meaningless
    # result (and puts photos of both cameras into each camera's step 2), so find them first and leave them out.
    other_way = cameras_other_way(obj_both, img_both_left, img_both_right, pair_names, K1, D1, K2, D2)
    if other_way:
        print(f"\nWARNING: {', '.join(other_way)} {'were' if len(other_way) > 1 else 'was'} taken with the cameras the "
              f"other way round from the other {len(pair_names) - len(other_way)} pair(s), so {'they are' if len(other_way) > 1 else 'it is'} "
              "left out. To use them, swap their left/ and right/ photos (or delete them).")
        keep = [i for i, n in enumerate(pair_names) if n not in other_way]
        obj_both, img_both_left, img_both_right, pair_names = (
            [x[i] for i in keep] for x in (obj_both, img_both_left, img_both_right, pair_names))
        keep_l = [i for i, n in enumerate(names_left) if n not in other_way]
        keep_r = [i for i, n in enumerate(names_right) if n not in other_way]
        obj_left, img_left = [obj_left[i] for i in keep_l], [img_left[i] for i in keep_l]
        obj_right, img_right = [obj_right[i] for i in keep_r], [img_right[i] for i in keep_r]
        if min(len(obj_left), len(obj_right), len(obj_both)) < 3:
            sys.exit("Too few pairs are left: swap or delete the pairs named above and capture more.")
        rms_left, K1, D1, _, _ = cv2.calibrateCamera(obj_left, img_left, image_size, None, None, flags=flags)
        rms_right, K2, D2, _, _ = cv2.calibrateCamera(obj_right, img_right, image_size, None, None, flags=flags)

    # Step 3: Stereo calibration. K and distortion stay fixed; it finds R and T, which take a point from
    # left-camera coordinates to right-camera coordinates: X_right = R @ X_left + T (in mm, like the board).
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-6)
    rms, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_both, img_both_left, img_both_right, K1, D1, K2, D2, image_size,
        criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)

    # Step 4: Rectification. R1 and R2 turn each camera so that both look the same way, P1 and P2 are the
    # new camera matrices, and Q turns (x, y, disparity) into a 3D point. alpha=0 keeps only valid pixels.
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, alpha=0)

    # Step 5: Results and checks.
    baseline = np.linalg.norm(T)
    angle = np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))
    print(f"\nUsed {len(obj_left)} left and {len(obj_right)} right views, {len(obj_both)} pairs, image size {image_size}")
    print(f"Left camera:  RMS {rms_left:.3f} px\n K1 =\n{K1}\n D1 = {D1.ravel()}")
    print(f"Right camera: RMS {rms_right:.3f} px\n K2 =\n{K2}\n D2 = {D2.ravel()}")
    print(f"\nStereo RMS reprojection error: {rms:.3f} px")
    print(f"R =\n{R}\nT = {T.ravel()} mm")
    print(f"Baseline |T|: {baseline:.1f} mm  (check it with a ruler between the centres of the two lenses)")
    print(f"Angle between the two cameras: {angle:.2f} degrees")

    warnings = []
    if len(obj_both) < 10:
        warnings.append(f"only {len(obj_both)} pairs show the board in both cameras: take at least 10.")
    for side, points in (('left', img_left), ('right', img_right)):
        gaps = uncovered(np.concatenate([p.reshape(-1, 2) for p in points]), image_size)
        if gaps:
            warnings.append(f"no corner reached the {', '.join(gaps)} of the {side} image: "
                            "add pairs with the board there, or the edges of the image are only guessed.")
    if rms > MAX_STEREO_RMS:
        warnings.append(f"the stereo RMS error is above {MAX_STEREO_RMS} px: some pairs are bad, usually because the "
                        "board moved between the two photos. Delete the pair named above as furthest over a limit, "
                        "and run again.")
    # Where is the right camera, seen from the left one? Its centre in left-camera coordinates is -R^T T.
    cx, cy, _ = (-R.T @ T).ravel()
    if angle > 45:
        warnings.append(f"one camera is turned by {angle:.0f} degrees relative to the other (upside down or on its "
                        "side?): mount both the same way up and capture again.")
    elif abs(cy) > abs(cx):
        warnings.append("the cameras are above each other, not side by side; stereo_depth.py needs them side by side.")
    elif cx < 0:
        warnings.append("the 'left' camera is on the right (the right camera's centre has x < 0). Swap the names of "
                        "the left/ and right/ folders, delete cameras.txt (it names the old left camera) and run again, "
                        "and use the camera indices the other way round from now on.")

    # After rectification, a corner must lie on the same row in both images. Its disparity d (the shift
    # along the row) gives its depth Z along the viewing direction: Z = f * B / d, with f = P1[0, 0] and B the baseline.
    f = P1[0, 0]
    print("\nPer pair, after rectification:      row difference (px)    board distance (mm)")
    print("                                      mean    worst        from disparity / from pose")
    worst_rows, flagged, swapped = [], 0, []
    worst = ('', 0.0)  # the pair furthest over a limit: one bad pair pulls R and T off and can get good pairs marked too
    side_by_side = angle <= 45 and abs(cx) > abs(cy)
    rig_ok = side_by_side and cx > 0  # else the rows and distances below can't be checked until the rig is fixed
    for name, obj, il, ir in zip(pair_names, obj_both, img_both_left, img_both_right):
        rl = cv2.undistortPoints(il, K1, D1, R=R1, P=P1).reshape(-1, 2)
        rr = cv2.undistortPoints(ir, K2, D2, R=R2, P=P2).reshape(-1, 2)
        dy = np.abs(rl[:, 1] - rr[:, 1])
        z_disparity = np.mean(f * baseline / (rl[:, 0] - rr[:, 0]))
        _, rvec, tvec = cv2.solvePnP(obj, il, K1, D1)  # the board's pose seen from the left camera
        points = (cv2.Rodrigues(rvec)[0] @ obj.reshape(-1, 3).T + tvec).T
        z_pose = np.mean((R1 @ points.T)[2])  # distance along the rectified viewing direction
        # This pair on its own: where is the right camera? (catches pairs taken with the cameras the other way round)
        _, rvec_r, tvec_r = cv2.solvePnP(obj, ir, K2, D2)
        Rl, Rr = cv2.Rodrigues(rvec)[0], cv2.Rodrigues(rvec_r)[0]
        Ri = Rr @ Rl.T
        pair_cx = (-Ri.T @ (tvec_r - Ri @ tvec)).ravel()[0]
        if side_by_side and pair_cx * cx < 0 and abs(pair_cx) > 0.3 * baseline:
            swapped.append(name)
        worst_rows.append(dy.max())
        bad = dy.max() > MAX_ROW_DIFFERENCE or abs(z_disparity - z_pose) > MAX_DISAGREEMENT * abs(z_pose)
        flagged += bad
        score = max(dy.max() / MAX_ROW_DIFFERENCE, abs(z_disparity - z_pose) / (MAX_DISAGREEMENT * abs(z_pose)))
        if score > worst[1]:
            worst = (name, score)
        print(f"  {name:32s} {dy.mean():6.2f}  {dy.max():6.2f}       {z_disparity:7.1f} / {z_pose:7.1f}"
              + ("   <-" if bad else ""))
    print(f"Worst row difference: {max(worst_rows):.2f} px. Matching searches along rows, so keep it below about "
          f"{MAX_ROW_DIFFERENCE:g} px.")
    if flagged and not rig_ok:
        print(f"{flagged} pair(s) marked <-: the rows and distances can't be checked until the rig problem in the "
              "WARNING below is fixed, so deleting pairs won't help.")
    elif flagged:
        print(f"{flagged} pair(s) marked <-: rows more than {MAX_ROW_DIFFERENCE:g} px apart, or distances more than "
              f"{100 * MAX_DISAGREEMENT:.0f} % apart. One bad pair can get good pairs marked too, so delete only the "
              f"pair furthest over a limit, {worst[0]} ({worst[1]:.1f} times), from left/ and right/, and run again.")
    print("Both distance columns use the same board, so they cannot catch a bent board or a wrong square size: "
          "the ruler check of the baseline can.")
    if swapped:
        warnings.append(f"in {', '.join(swapped)} the cameras look the other way round from the rest. Were they "
                        "re-plugged? Delete these pairs (from left/ and right/) and run again.")
    if blank_hint:
        warnings.append(blank_hint.strip())
    for w in warnings:
        print("WARNING:", w)

    recorded = recorded_cameras(folder) or ('', '')  # noted by the capture scripts
    out = os.path.join(folder, 'stereo_calibration.npz')
    np.savez(out, K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             image_size=np.array(image_size), rms=rms, roi_left=np.array(roi_left), roi_right=np.array(roi_right),
             camera_left=recorded[0], camera_right=recorded[1], board=board)
    print(f"\nSaved {os.path.relpath(out)}")

    # Rectify the first pair and draw rows: matching points must lie on the same green line.
    w, h = image_size
    rectified = []
    for img, K, D, Rr, P in ((pairs[pair_names[0]][0], K1, D1, R1, P1), (pairs[pair_names[0]][1], K2, D2, R2, P2)):
        map_x, map_y = cv2.initUndistortRectifyMap(K, D, Rr, P, (w, h), cv2.CV_32FC1)
        rectified.append(cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR))
    pair = np.hstack(rectified)
    for y in range(20, h, 40):
        cv2.line(pair, (0, y), (2 * w - 1, y), (0, 255, 0), 1)
    example = os.path.join(folder, 'rectified_example.jpg')
    cv2.imwrite(example, pair)
    print(f"Rectified pair {pair_names[0]} saved as {os.path.relpath(example)}")
    if SHOW:
        print("Showing the rectified pair: press any key in the window to finish (add nogui to skip the window).")
        cv2.imshow('Rectified pair', pair)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
