"""Capture calibration photos from one webcam in a window with buttons (the GUI version of w3_calibration_cap.py).

    python w3_calibration_gui.py

1. Pick the camera, the image size and the board you printed, then press Start.
2. A photo is saved only when the board is found and has been still for a moment, so it is sharp, and only
   once per position. With "Auto save" on, it saves by itself each time you hold the board still at a new
   position, so you can hold the board with both hands. Untick "Wait until still" to save without the still
   check (Auto save then saves each new position, at most once a second); hold the board steady yourself, or
   the photos may be blurred.
3. The window shows where the board has been. Green dots on the live image mark the corners of the photos
   so far. The "Board positions" panel fills in every saved board in green and shows in red the parts of the
   image that no photo covers yet (usually the corners), so you know where to hold the board next. From 4
   photos on it also calibrates in the background and shows the focal length, the principal point and the
   RMS error so far. Tick "Undistorted" to see the live image corrected with that estimate.
Photos go to calibration_images/calibration_image_NN.jpg next to this file, where w3_calibration.py and
w3_calibration_charuco.py look for them.
Keys: Space or c saves a photo, u undoes the last photo saved in this folder, q quits.

Credit: the green board positions, the corner dots, saving a still board by itself, the running estimate and
the undistorted live view follow OpenCV's interactive camera calibration application (apps/interactive-calibration
in the OpenCV source; https://docs.opencv.org/4.x/d7/d21/tutorial_interactive_calibration.html). Showing the
uncovered parts in red is this program's addition.
"""
import os
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FOLDER = os.path.join(HERE, 'calibration_images')  # the folder w3_calibration.py reads
sys.path.insert(0, HERE)
from camera_names import BUILT_IN, camera_names  # noqa: E402  (what the operating system calls each camera)

CHARUCO_ERROR = None
try:  # option B: the ChArUco board (needs OpenCV 4.7 or newer; the chessboard works without it)
    import w3_calibration_charuco as charuco
except SystemExit as error:
    charuco, CHARUCO_ERROR = None, str(error)
    print(CHARUCO_ERROR, file=sys.stderr)

CHESSBOARD_SIZE = (4, 7)  # inner corners, as in w3_calibration.py
SQUARE_SIZE = 34.0  # mm, as in w3_calibration.py
SIZES = ('640x480', '640x400', '800x600', '1280x720', '1280x800', '1920x1080')  # x400 and x800: 16:10 sensors
TARGET = 17  # photos: the shot list of the tutorial
MAX_MOTION = 1.5  # px: the board counts as still if its corners moved less than this (median) since the previous frame
READY_TIME = 0.5  # s: a photo can be saved once the board has been still this long
MAX_DRIFT = 3.0  # px: ... and drifted less than this (median) from where they were when the board stopped
STILL_TIME = 0.8  # s: Auto saves once the board has been still this long ...
NEW_POSITION = 1 / 20  # ... and has moved (on average) at least this fraction of the image diagonal since the last photo
AUTO_INTERVAL = 1.0  # s: without the still check, Auto save waits at least this long between photos
KEY_REPEAT_GAP = 0.6  # s: a save key pressed again within this time is a held key repeating, not a new press
MIN_ESTIMATE = 4  # photos before the running calibration estimate starts
PREVIEW_WIDTH = 720  # px: width of the live image on screen (smaller on short screens)
SIDE_WIDTH = 360  # px: width of the board-positions panel and its explanation
GREEN, ORANGE, RED = '#1a7f37', '#c25e00', '#e30613'
FAST = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
SUBPIX = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # as in w3_calibration.py
# The default 30 iterations of calibrateCamera can stop before the answer settles (wide-angle lenses, few photos)
SOLVER = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 500, 1e-9)

OBJP = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)  # the chessboard corners in mm
OBJP[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE


def find_board(gray, board, saved=False):
    """{'ids', 'image', 'object'} of the board corners found in a grayscale image, or None.

    image: N x 2 pixel positions, object: N x 3 positions on the board (mm), ids: one number per corner.
    saved=True searches a photo the way w3_calibration.py does (thorough, then sub-pixel refinement);
    saved=False is the quick search for live video.
    """
    if board == 'charuco':
        obj, img, ids, _, _ = charuco.detect_board(gray)
        if obj is None:
            return None
        return {'ids': ids.ravel().tolist(), 'image': img.reshape(-1, 2), 'object': obj.reshape(-1, 3)}
    if saved:  # OpenCV's default flags, as w3_calibration.py
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE)
    else:  # flags= by name: the third positional argument is the output array for the corners, not the flags
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags=FAST)
    if not found:
        return None
    if saved:
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX)
    return {'ids': list(range(len(OBJP))), 'image': corners.reshape(-1, 2), 'object': OBJP}


def corner_motion(now, before):
    """How far the corners seen in both detections moved (px, the median, so one jittery corner doesn't count),
    or None if they share no corner."""
    if now is None or before is None:
        return None
    old = dict(zip(before['ids'], before['image']))
    moves = [np.linalg.norm(p - old[i]) for i, p in zip(now['ids'], now['image']) if i in old]
    return float(np.median(moves)) if moves else None


def calibrate(views, size, board, fisheye):
    """(K, D, rms) from the saved views, with the model the calibration script for this board uses:
    k3 fixed for the chessboard (as in w3_calibration.py), free for ChArUco (as in w3_calibration_charuco.py),
    or the optional fisheye model."""
    obj = [v['object'].astype(np.float32) for v in views]
    img = [v['image'].reshape(-1, 1, 2).astype(np.float32) for v in views]
    if fisheye:
        rms, K, D, _, _ = cv2.fisheye.calibrate(
            [o.reshape(-1, 1, 3).astype(np.float64) for o in obj], [i.astype(np.float64) for i in img], size,
            np.zeros((3, 3)), np.zeros((4, 1)), flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW,
            criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7))
    else:
        flags = cv2.CALIB_FIX_K3 if board == 'chessboard' else 0
        rms, K, D, _, _ = cv2.calibrateCamera(obj, img, size, None, None, flags=flags, criteria=SOLVER)
    return K, D, rms


def undistortion_maps(K, D, size, fisheye):
    """Maps for cv2.remap that undistort a live image with an estimate (all of the image kept, black borders)."""
    if fisheye:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, np.eye(3), balance=0.0)
        if not np.isfinite(new_K).all() or new_K[0, 0] < 0.2 * K[0, 0]:  # no photo reaches the edges yet
            new_K = K.copy()
        return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, size, cv2.CV_16SC2)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, size, 1, size)
    return cv2.initUndistortRectifyMap(K, D, None, new_K, size, cv2.CV_16SC2)


def open_camera(index, size):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # larger sizes at full frame rate on USB webcams
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep few old frames (works on Linux; others may ignore it)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    if cap.isOpened() and any(cap.read()[0] for _ in range(10)):  # a camera that is warming up can drop a few frames
        return cap
    cap.release()
    return None


def path_hint(path):
    """Extra advice for a path OpenCV cannot use: on Windows, cv2.imread and cv2.imwrite need plain-English paths."""
    if sys.platform.startswith('win') and not path.isascii():
        return (" OpenCV on Windows cannot read or write files whose path has letters outside English "
                "(e.g. in your user name): move the lab folder to a plain path such as C:\\labs.")
    return ""


class Camera(threading.Thread):
    """Reads the camera in the background and keeps the newest frame, with the board found in it."""

    def __init__(self, index, size, board):
        super().__init__(daemon=True)
        self.index, self.size, self.board = index, size, board
        self.lock = threading.Lock()
        self.latest = None  # the newest frame, the board found in it, still or not, frame number
        self.state = 'opening'  # then 'running', or a message saying what went wrong
        self.running = True

    def run(self):
        cam = open_camera(self.index, self.size)
        if cam is None:
            self.state = f"Camera {self.index} gives no images: wrong index, or another app is using it."
            if sys.platform == 'darwin':
                self.state += (" On macOS, also allow camera access for the app you started this from "
                               "(System Settings > Privacy & Security > Camera), then restart it.")
            return
        previous, number, failures = None, 0, 0
        while self.running:
            ok, frame = cam.read()
            if not ok:
                failures += 1
                if failures > 30:
                    self.state = "The camera stopped sending images."
                    break
                time.sleep(0.01)
                continue
            failures = 0
            self.state = 'running'
            board = self.board  # read once: the board type can change while this frame is processed
            found = find_board(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), board)
            motion = corner_motion(found, previous)
            previous = found
            with self.lock:
                self.latest = {'frame': frame, 'number': number, 'found': found, 'board': board,
                               'motion': motion, 'still': motion is not None and motion <= MAX_MOTION}
            number += 1
        cam.release()

    def newest(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.running = False
        self.join(timeout=3)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        if self.tk.call('tk', 'windowingsystem') == 'aqua':
            try:  # keep the light look in macOS Dark Mode, where the grey messages would vanish on the dark background
                self.tk.call('::tk::unsupported::MacWindowStyle', 'appearance', self, 'aqua')
            except tk.TclError:  # Tk older than 8.6.10 has no Dark Mode anyway
                pass
        self.title("Calibration capture")
        self.camera = None  # the running Camera
        self.shown = -1  # number of the frame on screen; -1: check the next frame against the folder
        self.views = []  # every photo in the folder with the board in it: {'name', 'image', 'object', 'ids', 'size'}
        self.active_folder = None  # the folder in use (the Folder field only counts once Return or Browse confirms it)
        self.loaded = None  # (folder, board) that self.views belongs to
        self.found_cache = {}  # (path, modification time, board) -> board found, so a reload doesn't search again
        self.session = []  # full paths of the photos saved in this folder since the program started, for Undo
        self.still_since = None
        self.anchor = None  # board corners when the board stopped moving, to catch a slow drift
        self.saved_still = None  # still_since of the last save: one photo per still period, even if a key is held
        self.saved_number = None  # camera frame number of the last save: a frame is never saved twice
        self.last_auto = 0.0  # time of the last save, for Auto save without the still check
        self.last_save_key = 0.0  # time of the last Space or c key event, to ignore a held key's repeats
        self.folder_note = ''  # a note about the folder in use, kept on screen when the camera starts
        self.last_saved = None  # board of the last saved photo, so Auto doesn't save the same view twice
        self.frame_times = []
        self.estimate = None  # (K, D, rms, number of photos, fisheye, size) of the running calibration
        self.estimate_job = 0  # increases with every change, so an outdated background result is dropped
        self.estimate_ready = None  # (job, result) left by the background calibration for the main loop
        self.estimate_count, self.estimate_size = 0, None
        self.undistort_maps = None
        self.scanning = False
        self.scan_result = None
        self.build()
        self.load_folder()
        self.protocol('WM_DELETE_WINDOW', self.quit_app)
        for key in ('<space>', 'c', 'C'):
            self.bind(key, lambda e: self.keyboard(e, self.save))
        for key in ('u', 'U'):
            self.bind(key, lambda e: self.keyboard(e, self.undo))
        for key in ('q', 'Q', '<Escape>'):
            self.bind(key, lambda e: self.keyboard(e, self.quit_app))
        self.after(30, self.tick)

    # ------------------------------------------------------------------ layout
    def build(self):
        pad = {'padx': 6, 'pady': 4}
        top = ttk.Frame(self, padding=8)
        top.pack(side='top', fill='x')
        ttk.Label(top, text="Camera").grid(row=0, column=0, sticky='w', **pad)
        self.camera_box = ttk.Combobox(top, width=44)
        self.camera_box.grid(row=0, column=1, columnspan=3, sticky='w', **pad)
        self.scan_button = ttk.Button(top, text="Scan", command=self.scan, takefocus=0)
        self.scan_button.grid(row=0, column=4, **pad)
        ttk.Label(top, text="Size").grid(row=0, column=5, sticky='e', **pad)
        self.size = ttk.Combobox(top, values=SIZES, width=10, state='readonly')
        self.size.set(SIZES[0])
        self.size.grid(row=0, column=6, **pad)
        for box in (self.camera_box, self.size):
            box.bind('<<ComboboxSelected>>', lambda e: self.focus_set())
        self.start_button = ttk.Button(top, text="Start", command=self.start_stop, takefocus=0)
        self.start_button.grid(row=0, column=7, **pad)

        self.board = tk.StringVar(value='chessboard')
        ttk.Label(top, text="Board").grid(row=1, column=0, sticky='w', **pad)
        ttk.Radiobutton(top, text=f"Chessboard, {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners",
                        value='chessboard', variable=self.board, command=self.board_changed,
                        takefocus=0).grid(row=1, column=1, sticky='w', **pad)
        self.charuco_button = ttk.Radiobutton(top, text="ChArUco (option B)", value='charuco', variable=self.board,
                                              command=self.board_changed, takefocus=0)
        self.charuco_button.grid(row=1, column=2, sticky='w', **pad)
        if charuco is None:
            self.charuco_button.configure(state='disabled', text="ChArUco (needs OpenCV 4.7+)")
        self.need_still = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Wait until still", variable=self.need_still, command=self.auto_changed,
                        takefocus=0).grid(row=1, column=3, sticky='w', **pad)
        self.auto = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Auto save", variable=self.auto, command=self.auto_changed,
                        takefocus=0).grid(row=1, column=4, columnspan=2, sticky='w', **pad)
        self.fisheye = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Fisheye model (optional)", variable=self.fisheye, command=self.start_estimate,
                        takefocus=0).grid(row=1, column=6, columnspan=2, sticky='w', **pad)

        ttk.Label(top, text="Folder").grid(row=2, column=0, sticky='w', **pad)
        self.folder = tk.StringVar(value=DEFAULT_FOLDER)
        entry = ttk.Entry(top, textvariable=self.folder, width=60)
        entry.grid(row=2, column=1, columnspan=4, sticky='we', **pad)
        entry.bind('<Return>', lambda e: self.load_folder())
        entry.bind('<FocusOut>', lambda e: self.load_folder())
        ttk.Button(top, text="Browse…", command=self.browse, takefocus=0).grid(row=2, column=5, **pad)
        ttk.Button(top, text="Open folder", command=self.open_folder, takefocus=0).grid(row=2, column=6, **pad)
        self.show_undistorted = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Undistorted", variable=self.show_undistorted, takefocus=0).grid(
            row=2, column=7, sticky='w', **pad)

        # The rows under the image are packed first, from the bottom up: a short screen cuts the image, not them
        wrap = PREVIEW_WIDTH + SIDE_WIDTH
        self.message = tk.Label(self, anchor='w', justify='left', padx=10, pady=4, fg='#555', font=('Helvetica', 12),
                                wraplength=wrap)
        self.message.pack(side='bottom', fill='x')
        self.result = tk.Label(self, anchor='w', justify='left', padx=10, font=('Helvetica', 13), wraplength=wrap)
        self.result.pack(side='bottom', fill='x')
        self.status = tk.Label(self, font=('Helvetica', 14), anchor='w', padx=10, width=1)  # width=1: never widens
        self.status.pack(side='bottom', fill='x')
        bottom = ttk.Frame(self, padding=8)
        bottom.pack(side='bottom', fill='x')
        self.save_button = tk.Button(bottom, text="Save photo  (Space)", command=self.save,
                                     font=('Helvetica', 16, 'bold'), padx=18, pady=6, takefocus=0)
        self.save_button.pack(side='left')
        ttk.Button(bottom, text="Undo last (U)", command=self.undo, takefocus=0).pack(side='left', padx=10)
        self.counter = tk.Label(bottom, font=('Helvetica', 16, 'bold'))
        self.counter.pack(side='left', padx=16)

        self.photo = tk.PhotoImage(width=PREVIEW_WIDTH + SIDE_WIDTH, height=PREVIEW_WIDTH * 3 // 4)  # blank until Start
        self.preview = tk.Label(self, image=self.photo, bg='black', highlightthickness=5, highlightbackground='black')
        self.preview.pack(side='top', padx=8)
        self.preview.bind('<Button-1>', lambda e: self.focus_set())  # back to the keys after typing in a field

        # Fit a 4:3 image on short screens (e.g. 1366x768): the live image shrinks, the panel text stays readable
        self.update_idletasks()
        others = self.winfo_reqheight() - self.preview.winfo_reqheight()
        self.max_live_h = max(240, self.winfo_screenheight() - others - 120)  # 120: title bar, menu bar or taskbar

        self.names = {}
        self.show_names()
        if CHARUCO_ERROR:
            self.say(CHARUCO_ERROR.replace('\n', ' '), ORANGE)
        else:
            self.say("Pick the camera (Scan checks which ones give images) and the board, then press Start.")

    def say(self, text, colour='#555'):
        self.message.configure(text=text, fg=colour)

    def keyboard(self, event, action):
        if isinstance(event.widget, (ttk.Entry, ttk.Combobox, ttk.Spinbox, tk.Entry)):  # typing in a field
            return
        if action == self.save:  # a held key repeats itself: one save per press
            now = time.time()
            held = now - self.last_save_key < KEY_REPEAT_GAP
            self.last_save_key = now
            if held:
                return
        action()

    # ------------------------------------------------------------------ camera
    def show_names(self, indices=None):
        """Fill the camera list with the names the operating system gives (just numbers where it gives none)."""
        self.names = camera_names()
        if indices is None:
            indices = sorted(self.names) if self.names else list(range(6))
        self.camera_box.configure(values=[self.label(i) for i in indices])
        external = [i for i in indices if self.names.get(i) and not BUILT_IN.search(self.names[i])]
        if indices:
            self.camera_box.set(self.label((external or indices)[0]))  # an external webcam if there is one

    def label(self, index):
        return f"{index}: {self.names[index]}" if index in self.names else str(index)

    @staticmethod
    def index(label):
        """The camera index at the start of a list entry such as '1: Arducam OV9782 USB Camera'."""
        return int(label.split(':')[0].strip())

    def scan(self):
        if self.camera is not None:
            self.say("Stop the camera before scanning.", RED)
            return
        self.scanning = True
        self.scan_button.configure(state='disabled')
        self.start_button.configure(state='disabled')  # a scan opens every camera, the running one included
        self.names = camera_names()
        self.say("Scanning the cameras ...")
        self.scan_result = None
        to_try = sorted(set(range(6)) | set(self.names))

        def work():
            found = []
            for index in to_try:
                cap = cv2.VideoCapture(index)
                ok, frame = False, None
                for _ in range(10 if cap.isOpened() else 0):
                    ok, frame = cap.read()
                    if ok:
                        break
                cap.release()
                if ok:
                    found.append((index, frame.shape[1], frame.shape[0]))
            self.scan_result = found

        threading.Thread(target=work, daemon=True).start()
        self.after(200, self.scan_done)

    def scan_done(self):
        if self.scan_result is None:
            self.after(200, self.scan_done)
            return
        self.scanning = False
        self.scan_button.configure(state='normal')
        self.start_button.configure(state='normal')
        found = self.scan_result
        if not found:
            self.say("No camera found. Check the USB cable and the camera permission of your terminal or IDE.", RED)
            return
        self.show_names([i for i, _, _ in found])
        self.say("Found: " + "; ".join(f"{self.label(i)}, {w}x{h}" for i, w, h in found), GREEN)

    def start_stop(self):
        if self.camera is None:
            self.start()
        else:
            self.stop()

    def start(self):
        if self.scanning:
            self.say("Wait for the scan to finish.", RED)
            return
        try:
            index = self.index(self.camera_box.get())
        except ValueError:
            self.say("Pick a camera from the list (or type its number).", RED)
            return
        size = tuple(int(v) for v in self.size.get().split('x'))
        self.load_folder()
        self.camera = Camera(index, size, self.board.get())
        self.camera.start()
        self.saved_number = None  # the new camera numbers its frames from 0 again
        self.shown = -1
        self.start_button.configure(text="Stop")
        self.camera_box.configure(state='disabled')  # both are read at Start: press Stop to change them
        self.size.configure(state='disabled')
        self.focus_set()
        self.say(f"Opening {self.label(index)} ...")

    def stop(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera = None
        self.start_button.configure(text="Start")
        self.camera_box.configure(state='normal')
        self.size.configure(state='readonly')

    def board_changed(self):
        if self.camera is not None:
            self.camera.board = self.board.get()
        self.still_since = self.last_saved = self.saved_still = None
        self.load_folder()  # the photos in the folder are searched for the other board

    def auto_changed(self):
        self.still_since = None

    # ------------------------------------------------------------------ folder
    def photo_files(self, folder=None):
        """The photos in the folder, as w3_calibration.py reads them (not its *_corners_detected overlays)."""
        folder = folder or self.active_folder
        if not folder or not os.path.isdir(folder):
            return []
        return sorted(f for f in os.listdir(folder)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '_corners_detected' not in f)

    def folder_changed(self):
        """True (with a message) if the Folder field shows another folder than the one in use."""
        if os.path.abspath(self.folder.get()) != self.active_folder:
            self.say("The Folder field was changed: press Return in it (or use Browse…) to switch folder first.", ORANGE)
            return True
        return False

    def load_folder(self):
        """Find the board in the photos already in the folder, for the dots, the positions panel and the estimate."""
        folder = os.path.abspath(self.folder.get())
        board = self.board.get()
        if folder != self.active_folder:  # Undo history belongs to one folder
            self.session = []
        self.active_folder = folder
        self.session = [p for p in self.session if os.path.exists(p)]  # photos deleted by hand can't be undone
        views, unreadable = [], 0
        for name in self.photo_files():
            path = os.path.join(folder, name)
            key = (path, os.path.getmtime(path), board)
            if key not in self.found_cache:
                gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                found = find_board(gray, board, saved=True) if gray is not None else None
                self.found_cache[key] = (found, None if gray is None else gray.shape[::-1])
            found, size = self.found_cache[key]
            unreadable += size is None
            if found is not None:
                views.append(dict(found, name=name, size=size))
        self.views = views
        if self.loaded != (folder, board):
            self.loaded = (folder, board)
            self.last_saved = self.saved_still = None
            if self.camera is not None:
                self.shown = -1  # check the running camera against the photos of this folder and board
        notes = []
        if folder != os.path.abspath(DEFAULT_FOLDER):
            notes.append("Note: w3_calibration.py reads only calibration_images/. For photos in this folder, run "
                         f"python w3_calibration_charuco.py \"{folder}\" (ChArUco board).")
        if unreadable:
            notes.append(f"{unreadable} photo(s) in this folder could not be read.{path_hint(folder)}")
        self.folder_note = " ".join(notes)  # shown again when the camera starts, so it isn't lost
        if self.folder_note:
            self.say(self.folder_note, ORANGE)
        self.update_counter()
        self.start_estimate()

    def browse(self):
        folder = filedialog.askdirectory(initialdir=self.active_folder if os.path.isdir(self.active_folder) else HERE,
                                         title="Folder for the calibration photos")
        if folder:
            self.folder.set(folder)
            self.load_folder()

    def open_folder(self):
        if self.folder_changed():
            return
        folder = self.active_folder
        try:
            os.makedirs(folder, exist_ok=True)
            if sys.platform == 'darwin':
                subprocess.Popen(['open', folder])
            elif sys.platform.startswith('win'):
                os.startfile(folder)
            else:
                subprocess.Popen(['xdg-open', folder])
        except OSError as error:
            self.say(f"Could not open {folder}: {error.strerror}.", RED)

    def update_counter(self):
        files = len(self.photo_files())
        extra = f"  ({files - len(self.views)} without this board)" if files > len(self.views) else ""
        self.counter.configure(text=f"{len(self.views)} / {TARGET} photos{extra}")

    # ------------------------------------------------------------------ running estimate
    def start_estimate(self):
        """Calibrate in the background with the photos so far (the main window stays responsive)."""
        self.estimate_job += 1
        job, fisheye, board = self.estimate_job, self.fisheye.get(), self.board.get()
        sizes = [v['size'] for v in self.views]
        size = max(set(sizes), key=sizes.count) if sizes else None  # one camera at one size, as w3_calibration.py
        views = [v for v in self.views if v['size'] == size]
        self.estimate_count, self.estimate_size = len(views), size
        if len(views) < MIN_ESTIMATE:
            self.estimate, self.undistort_maps = None, None
            self.show_estimate()
            return

        def work():
            try:
                K, D, rms = calibrate(views, size, board, fisheye)
                result = (K, D, rms, len(views), fisheye, size)
            except cv2.error:
                result = None
            self.estimate_ready = (job, result)  # picked up by tick(): Tk may only be used from the main thread

        threading.Thread(target=work, daemon=True).start()

    def estimate_done(self, job, result):
        if job != self.estimate_job:  # a newer estimate is on its way
            return
        self.estimate = result
        self.undistort_maps = None
        if result is not None:
            K, D, _, _, fisheye, size = result
            self.undistort_maps = undistortion_maps(K, D, size, fisheye)
        self.show_estimate()

    def show_estimate(self):
        other = len(self.views) - self.estimate_count
        note = (f"  {other} photo{'s' if other != 1 else ''} at another size {'are' if other != 1 else 'is'} left out "
                f"(a calibration needs one size: {self.estimate_size[0]}x{self.estimate_size[1]})." if other else "")
        if self.estimate is None:
            more = max(0, MIN_ESTIMATE - self.estimate_count)
            text = (f"Estimate: after {more} more photo{'s' if more != 1 else ''}." if more else
                    "Estimate: the calibration failed with these photos; add photos with the board in other places.")
            self.result.configure(text=text + note, fg='#555')
            return
        K, D, rms, n, fisheye, _ = self.estimate
        # the verdict table of the tutorial (written for 640x480)
        colour, verdict = ((GREEN, "good") if rms < 0.5 else
                           (ORANGE, "usable: retake blurred or badly detected photos") if rms < 1.0 else
                           (RED, "too high: a curled board, motion blur or a second camera?"))
        self.result.configure(
            text=f"Estimate from {n} photos ({'fisheye' if fisheye else 'standard'} model): fx {K[0, 0]:.0f}, "
                 f"fy {K[1, 1]:.0f}, cx {K[0, 2]:.0f}, cy {K[1, 2]:.0f} px; RMS error {rms:.2f} px ({verdict}).{note}",
            fg=colour)

    # ------------------------------------------------------------------ saving
    def next_name(self):
        """calibration_image_NN.jpg, numbered after the photos already there (any case, any image type)."""
        files = self.photo_files()
        numbers = [int(m.group(1)) for m in (re.fullmatch(r'calibration_image_(\d+)\.(?:jpe?g|png)', f, re.IGNORECASE)
                                             for f in files) if m]
        n = max(numbers) + 1 if numbers else 0
        taken = {f.lower() for f in os.listdir(self.active_folder)} if os.path.isdir(self.active_folder) else set()
        while f"calibration_image_{n:02d}.jpg" in taken:  # also safe on disks that ignore upper/lower case
            n += 1
        return f"calibration_image_{n:02d}.jpg"

    def save(self, auto=False, item=None):
        if item is None:
            item = self.camera.newest() if self.camera is not None else None
        if item is None:
            self.say("Start the camera first.", RED)
            return False
        if self.folder_changed():
            return False
        if item['found'] is None or item['board'] != self.board.get():
            self.say("No board found: the whole chessboard must be in view (part of a ChArUco board is enough).", RED)
            return False
        if self.need_still.get():
            if not item['still'] or self.still_since is None or time.time() - self.still_since < READY_TIME:
                self.say("The board is moving: hold it still for half a second (or rest it on something) and save "
                         "again, or untick Wait until still.", ORANGE)
                return False
            if self.saved_still == self.still_since:
                self.say("Already saved: move or tilt the board to a new position, then save again.", ORANGE)
                return False
        elif item['number'] == self.saved_number:
            self.say("Already saved: this camera frame is in the folder already.", ORANGE)
            return False
        gray = cv2.cvtColor(item['frame'], cv2.COLOR_BGR2GRAY)
        found = find_board(gray, item['board'], saved=True)  # the search w3_calibration.py will make in the photo
        if found is None:
            self.say("Not saved: the live view found the board, but the full search used by w3_calibration.py did "
                     "not, so it would skip this photo. Hold the board still in good light and save again.", ORANGE)
            return False
        name = self.next_name()
        path = os.path.join(self.active_folder, name)
        try:
            os.makedirs(self.active_folder, exist_ok=True)
            written = cv2.imwrite(path, item['frame'])
        except (OSError, cv2.error):
            written = False
        if not written:
            if auto:
                self.auto.set(False)  # so it doesn't try again on every frame
            self.say(f"Could not save {name} in {self.active_folder}: check that it is a folder you can write to"
                     f"{'; Auto save is now off' if auto else ''}.{path_hint(path)}", RED)
            return False
        self.views.append(dict(found, name=name, size=gray.shape[::-1]))
        self.session.append(path)
        self.last_saved, self.saved_still = item['found'], self.still_since
        self.saved_number, self.last_auto = item['number'], time.time()
        self.say(f"{'Auto-saved' if auto else 'Saved'} {name}, {len(found['ids'])} corners.", GREEN)
        self.update_counter()
        self.start_estimate()
        self.flash()
        return True

    def undo(self):
        if self.folder_changed():
            return
        if not self.session:
            self.say("Nothing to undo: Undo only removes photos saved in this folder since the program started.",
                     ORANGE)
            return
        path = self.session.pop()
        name = os.path.basename(path)
        if os.path.exists(path):
            os.remove(path)
        removed = [v for v in self.views if v['name'] == name]
        self.views = [v for v in self.views if v['name'] != name]
        if self.last_saved is None:  # otherwise keep the newest view, so Auto save doesn't save a removed one straight back
            self.last_saved = removed[0] if removed else None
        self.saved_still = self.saved_number = None  # by hand, the view can be saved again
        self.say(f"Removed {name}.", ORANGE)
        self.update_counter()
        self.start_estimate()

    def flash(self):
        self.preview.configure(highlightbackground='white')
        self.after(150, lambda: self.preview.configure(highlightbackground='black'))

    def new_position(self, found, size):
        """True if the board moved away from where it was in the last saved photo."""
        if self.last_saved is None:
            return True
        old = dict(zip(self.last_saved['ids'], self.last_saved['image']))
        moves = [np.linalg.norm(p - old[i]) for i, p in zip(found['ids'], found['image']) if i in old]
        if len(moves) < 0.5 * min(len(found['ids']), len(old)):
            return True
        return np.mean(moves) > NEW_POSITION * np.hypot(*size)

    # ------------------------------------------------------------------ drawing
    def positions_panel(self, size, current):
        """Every saved board filled in green, brighter where boards overlap (as in OpenCV's interactive
        calibration app), the parts no saved board has covered yet in red, and the board in view now as a
        white outline."""
        w, h = size
        scale = 0.5
        panel = np.zeros((int(h * scale), int(w * scale), 3), np.uint8)
        covered = np.zeros(panel.shape[:2], np.uint8)
        for view in self.views:
            if view['size'] != size:
                continue
            fill = np.zeros_like(panel)
            hull = cv2.convexHull((view['image'] * scale).astype(np.int32))
            cv2.fillConvexPoly(fill, hull, (0, 255, 0), cv2.LINE_AA)
            cv2.fillConvexPoly(covered, hull, 255)
            panel = cv2.addWeighted(fill, 0.2, panel, 1.0, 0)
        panel[covered == 0] = (0, 0, 120)  # dark red: no board there yet, so the calibration knows nothing about it
        if current is not None:
            hull = cv2.convexHull((current['image'] * scale).astype(np.int32))
            cv2.polylines(panel, [hull], True, (255, 255, 255), 2, cv2.LINE_AA)
        # the image split in thirds: a calibration needs the board in each of the nine parts
        for k in (1, 2):
            cv2.line(panel, (int(k * panel.shape[1] / 3), 0), (int(k * panel.shape[1] / 3), panel.shape[0]), (90, 90, 90), 1)
            cv2.line(panel, (0, int(k * panel.shape[0] / 3)), (panel.shape[1], int(k * panel.shape[0] / 3)), (90, 90, 90), 1)
        return panel

    def empty_ninths(self, size):
        """Number of the nine parts of the image (3 x 3) that no saved board corner has reached yet."""
        w, h = size
        seen = set()
        for view in self.views:
            if view['size'] == size:
                for x, y in view['image']:
                    seen.add((min(2, int(3 * x / w)), min(2, int(3 * y / h))))
        return 9 - len(seen)

    def tick(self):
        self.after(30, self.tick)
        if self.estimate_ready is not None:
            job, result = self.estimate_ready
            self.estimate_ready = None
            self.estimate_done(job, result)
        if self.camera is None:
            return
        if self.camera.state not in ('opening', 'running'):
            self.say(self.camera.state, RED)
            self.stop()
            return
        item = self.camera.newest()
        if item is None or item['number'] == self.shown:
            return
        h, w = item['frame'].shape[:2]
        if self.shown == -1:  # first frame, or another folder or board: the photos there must have the same size
            sizes = {v['size'] for v in self.views}
            if sizes and sizes != {(w, h)}:
                if (w, h) in sizes:
                    listed = ", ".join(f"{a}x{b}" for a, b in sorted(sizes))
                    self.say(f"The photos in this folder have different sizes ({listed}): a calibration needs one "
                             "camera at one size. Move the photos of the other size out of the folder.", RED)
                else:
                    other = sorted(sizes)[0]
                    self.say(f"The photos in this folder are {other[0]}x{other[1]}, but the camera gives {w}x{h}: pick "
                             f"{other[0]}x{other[1]} in Size, or move the old photos out of the folder (a calibration "
                             "needs one camera at one size).", RED)
                self.stop()
                return
            self.say(f"Capturing at {w}x{h}: use the calibration only at this size."
                     + (f" {self.folder_note}" if self.folder_note else ""), ORANGE if self.folder_note else '#555')
        self.shown = item['number']
        now = time.time()
        self.frame_times = [t for t in self.frame_times if now - t < 1.0] + [now]

        # Status and Auto save
        found = item['found'] if item['board'] == self.board.get() else None
        if found is None:
            text, colour = "No board", RED
            self.still_since = None
        elif not self.need_still.get():  # no still check: hold the board steady yourself
            self.still_since = None
            text, colour = f"{len(found['ids'])} corners - ready (no still check)", GREEN
            if self.auto.get() and now - self.last_auto >= AUTO_INTERVAL:
                if os.path.abspath(self.folder.get()) != self.active_folder:
                    text = "Auto save waits: press Return in the Folder field to use the new folder"
                elif self.new_position(found, (w, h)):
                    self.save(auto=True, item=item)
                else:
                    text = f"{len(found['ids'])} corners - move the board to a new position"
        elif not item['still']:
            moving = f" (moving {item['motion']:.1f} px per frame, needs under {MAX_MOTION:g})" if item['motion'] else ""
            text, colour = f"{len(found['ids'])} corners - hold still{moving}", ORANGE
            self.still_since = None
        else:
            drift = corner_motion(found, self.anchor)
            if self.still_since is None or drift is None or drift > MAX_DRIFT:
                self.still_since, self.anchor = now, found  # (re)start the still period here
            if now - self.still_since < READY_TIME:
                text, colour = f"{len(found['ids'])} corners - hold still", ORANGE
            elif self.saved_still == self.still_since:
                text, colour = f"{len(found['ids'])} corners - saved: move the board to a new position", GREEN
            else:
                text, colour = f"{len(found['ids'])} corners - ready", GREEN
            if self.auto.get() and now - self.still_since >= STILL_TIME and self.saved_still != self.still_since:
                if os.path.abspath(self.folder.get()) != self.active_folder:
                    text = "Auto save waits: press Return in the Folder field to use the new folder"
                elif self.new_position(found, (w, h)):
                    self.save(auto=True, item=item)
                else:
                    text = f"{len(found['ids'])} corners - move the board to a new position"
        empty = self.empty_ninths((w, h))
        coverage = f"   ·   {empty} of 9 image parts still empty" if empty else "   ·   every part of the image reached"
        self.status.configure(text=f"{text}{coverage}   ·   {len(self.frame_times)} fps", fg=colour)

        # Live image: saved corners as green dots, the board in view with OpenCV's coloured corners (red dots: ChArUco)
        view = item['frame'].copy()
        r = max(2, w // 220)
        for saved in self.views:
            if saved['size'] == (w, h):
                for x, y in saved['image']:
                    cv2.circle(view, (int(x), int(y)), r, (0, 255, 0), 1, cv2.LINE_AA)
        if found is not None and item['board'] == 'chessboard':
            cv2.drawChessboardCorners(view, CHESSBOARD_SIZE, found['image'].reshape(-1, 1, 2).astype(np.float32), True)
        elif found is not None:
            for x, y in found['image']:
                cv2.circle(view, (int(x), int(y)), r + 2, (0, 0, 255), -1)
        if self.show_undistorted.get() and self.undistort_maps is not None and self.estimate[5] == (w, h):
            view = cv2.remap(view, *self.undistort_maps, cv2.INTER_LINEAR)
            label = "Undistorted with the estimate"
        else:
            label = f"Camera {self.label(self.camera.index)}, {w}x{h}"
        scale = w / 1280  # the same text size on screen at any resolution
        cv2.rectangle(view, (0, 0), (w, int(44 * scale) + 4), (0, 0, 0), -1)
        cv2.putText(view, label, (int(12 * scale), int(32 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale,
                    (255, 255, 255), max(1, round(2 * scale)), cv2.LINE_AA)

        live_w = PREVIEW_WIDTH
        live_h = int(round(h * live_w / w))
        if live_h > self.max_live_h:  # short screen: a smaller live image
            live_h = self.max_live_h
            live_w = int(round(w * live_h / h))
        live = cv2.resize(view, (live_w, live_h), interpolation=cv2.INTER_AREA)
        panel = cv2.resize(self.positions_panel((w, h), found), (SIDE_WIDTH, int(round(h * SIDE_WIDTH / w))),
                           interpolation=cv2.INTER_AREA)
        lines = ("Red: no saved photo covers it yet.", "Green: the saved photos (brighter",
                 "where they overlap). White: the board", "now. Move the board into the red,",
                 "especially the corners.")
        height = max(live_h, panel.shape[0] + 52 + 22 * len(lines))
        both = np.zeros((height, live_w + SIDE_WIDTH, 3), np.uint8)
        both[:live_h, :live_w] = live
        both[:panel.shape[0], live_w:] = panel
        cv2.putText(both, "Board positions", (live_w + 10, panel.shape[0] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        for i, line in enumerate(lines):
            cv2.putText(both, line, (live_w + 10, panel.shape[0] + 52 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (190, 190, 190), 1, cv2.LINE_AA)
        ppm = b'P6 %d %d 255 ' % (both.shape[1], both.shape[0]) + cv2.cvtColor(both, cv2.COLOR_BGR2RGB).tobytes()
        self.photo = tk.PhotoImage(data=ppm, format='PPM')  # keep a reference, or Tk drops the image
        self.preview.configure(image=self.photo, width=both.shape[1], height=both.shape[0])

    def quit_app(self):
        self.stop()
        self.destroy()


if __name__ == '__main__':
    App().mainloop()
