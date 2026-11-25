"""Capture image pairs from two USB webcams in a window with buttons (the GUI version of stereo_cap.py).

    python stereo_capture_gui.py

1. Pick the left and the right camera (Scan finds the cameras that work), the image size, then press Start.
   Then check which camera is which: stand behind the cameras and hold your hand in front of the LEFT one. If the
   LEFT image goes dark, press "Confirm left/right"; if the RIGHT image does, press "Swap left/right". Nothing is
   saved until left and right are confirmed (a folder whose pairs were taken with the same cameras counts as
   confirmed). With the board in view, the window also warns if the cameras look swapped.
2. Calibration mode (the ChArUco board, or the Week 3 chessboard, which must be whole in both photos): a pair
   is saved only when both cameras see the board and it is still. Untick "Wait until still" to save without the
   still check (Auto save then saves each new position, at most once a second). The two cameras are not
   synchronised, so a moving board lands in different places in the two photos: hold it steady yourself.
   With "Auto save" on, it saves by itself whenever the board has been still for a moment at a new position,
   so you can hold the board with both hands.
3. Scene mode: saves any pair, for your own stereo dataset. With "Auto save" on, it saves a pair every few seconds.
Pairs go to <folder>/left/pair_NN.jpg and <folder>/right/pair_NN.jpg, the layout stereo_calibration.py reads.
Keys: Space or c saves a pair, u undoes the last pair saved in this session, q quits.
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
sys.path.insert(0, HERE)
from stereo_calibration import common_corners, detect_board  # noqa: E402  (same board and checks as the calibration)
from camera_names import (camera_names, compare_cameras, likely_pair, record_cameras,  # noqa: E402
                          recorded_board, recorded_cameras)  # what the system calls each camera, which was left

SIZES = ('640x480', '640x400', '800x600', '1280x720', '1280x800', '1920x1080')  # x400 and x800: 16:10 sensors
TARGET = 20  # calibration pairs to aim for
MAX_MOTION = 1.5  # px: the board counts as still if its corners moved less than this (median) since the previous frame
READY_TIME = 0.5  # s: a pair can be saved once the board has been still this long (one camera can lag the other)
MAX_DRIFT = 3.0  # px: ... and drifted less than this (median) from where they were when the board stopped
STILL_TIME = 0.7  # s: Auto saves once the board has been still this long ...
NEW_POSITION = 25.0  # px: ... and has moved at least this far (on average) since the last saved pair
AUTO_INTERVAL = 1.0  # s: without the still check, Auto save waits at least this long between pairs
KEY_REPEAT_GAP = 0.6  # s: a save key pressed again within this time is a held key repeating, not a new press
PREVIEW_WIDTH = 1100  # px: width of the side-by-side preview on screen
GREEN, ORANGE, RED = '#1a7f37', '#c25e00', '#e30613'


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


def open_camera(index, size):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # compressed video, so two webcams fit on one USB
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep few old frames (both requests work on Linux; others may ignore them)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    if cap.isOpened() and any(cap.read()[0] for _ in range(10)):  # a camera that is warming up can drop a few frames
        return cap
    cap.release()
    return None


class CameraPair(threading.Thread):
    """Reads both cameras in the background and keeps the newest pair, with the board found in it."""

    def __init__(self, indices, size):
        super().__init__(daemon=True)
        self.indices, self.size = indices, size
        self.lock = threading.Lock()
        self.latest = None  # the newest pair: frames, corners, shared corners, still or not, frame number
        self.state = 'opening'  # then 'running', or a message saying what went wrong
        self.board = 'charuco'  # the board to look for: 'charuco' or 'chessboard'; None in scene mode
        self.running = True

    def run(self):
        cameras = []
        for index in self.indices:
            cam = open_camera(index, self.size)
            if cam is None:
                self.state = f"Camera {index} gives no images: wrong index, or another app is using it."
                if sys.platform == 'darwin':
                    self.state += (" On macOS, also allow camera access for the app you started this from "
                                   "(System Settings > Privacy & Security > Camera), then restart it.")
                break
            cameras.append(cam)
        previous, number, failures = [{}, {}], 0, 0
        while self.running and len(cameras) == 2:
            # Grab both frames first and decode them afterwards: the two photos as close in time as possible
            grabbed = [cam.grab() for cam in cameras]
            frames = [cam.retrieve()[1] if ok else None for cam, ok in zip(cameras, grabbed)]
            if any(frame is None for frame in frames):
                failures += 1
                if failures > 30:
                    self.state = "The cameras stopped sending images."
                    break
                time.sleep(0.01)
                continue
            failures = 0
            if frames[0].shape != frames[1].shape:
                self.state = (f"The cameras give different image sizes ({frames[0].shape[1]}x{frames[0].shape[0]} and "
                              f"{frames[1].shape[1]}x{frames[1].shape[0]}): pick another size, or use two identical webcams.")
                break
            self.state = 'running'
            item = {'frames': frames, 'number': number, 'corners': [{}, {}], 'shared': None, 'still': False}
            board = self.board  # read once: the mode can change while this frame is processed
            item['board'] = board
            if board:
                detections = [detect_board(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), board, fast=True) for frame in frames]
                now = [corners_by_id(d) for d in detections]
                motions = [corner_motion(n, p) for n, p in zip(now, previous)]
                previous = now
                item.update(corners=now, shared=common_corners(*detections, board), motion=motions,
                            still=all(m is not None and m <= MAX_MOTION for m in motions))
            else:
                previous = [{}, {}]
            with self.lock:
                self.latest = item
            number += 1
        for cam in cameras:
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
        self.title("Stereo capture")
        self.pair = None  # the running CameraPair
        self.shown = -1  # number of the frame on screen
        self.session = []  # (name, corners left, corners right) of the pairs saved since the folder was opened, for Undo
        self.loaded_folder = self.opened_folder = None  # the folder Undo belongs to; the one the cameras started on
        self.dots_board = None  # the board the green dots were found with
        self.old_coverage = [[], []]  # corners of the pairs that were in the folder already
        self.still_since = None
        self.anchor = [{}, {}]  # corners when the board stopped moving, to catch a slow drift
        self.last_auto = 0.0
        self.last_saved = {}  # left-camera corners of the last saved pair, so Auto doesn't save the same view twice
        self.saved_still = None  # still_since of the last save: one pair per still period, even if a key is held
        self.saved_number = None  # frame number of the last saved pair: a pair of frames is never saved twice
        self.confirmed = False  # the user checked that the LEFT image comes from the left camera
        self.last_save_key = 0.0  # time of the last Space or c key event, to ignore a held key's repeats
        self.swap_votes = []  # recent frames in which the board sat further right in the right image (cameras swapped?)
        self.frame_times = []
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
        top.pack(fill='x')
        ttk.Label(top, text="Left camera").grid(row=0, column=0, sticky='w', **pad)
        self.left = ttk.Combobox(top, width=42)
        self.left.grid(row=0, column=1, **pad)
        ttk.Label(top, text="Right camera").grid(row=0, column=2, sticky='w', **pad)
        self.right = ttk.Combobox(top, width=42)
        self.right.grid(row=0, column=3, **pad)
        self.scan_button = ttk.Button(top, text="Scan", command=self.scan, takefocus=0)
        self.scan_button.grid(row=0, column=4, **pad)
        ttk.Button(top, text="Swap left/right", command=self.swap, takefocus=0).grid(row=0, column=5, **pad)
        ttk.Label(top, text="Size").grid(row=0, column=6, sticky='w', **pad)
        self.size = ttk.Combobox(top, values=SIZES, width=10, state='readonly')
        self.size.set(SIZES[0])
        self.size.grid(row=0, column=7, **pad)
        for box in (self.left, self.right, self.size):
            box.bind('<<ComboboxSelected>>', lambda e: self.focus_set())
        self.start_button = ttk.Button(top, text="Start", command=self.start_stop, takefocus=0)
        self.start_button.grid(row=0, column=8, **pad)

        self.mode = tk.StringVar(value='charuco')  # 'charuco' or 'chessboard': calibration; 'scene': any pair
        ttk.Radiobutton(top, text="Calibration: ChArUco board", value='charuco', variable=self.mode,
                        command=self.mode_changed, takefocus=0).grid(row=1, column=0, columnspan=2, sticky='w', **pad)
        ttk.Radiobutton(top, text="Calibration: chessboard", value='chessboard', variable=self.mode,
                        command=self.mode_changed, takefocus=0).grid(row=1, column=2, sticky='w', **pad)
        ttk.Radiobutton(top, text="Scene (any pair)", value='scene', variable=self.mode,
                        command=self.mode_changed, takefocus=0).grid(row=1, column=3, columnspan=2, sticky='w', **pad)
        self.auto = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Auto save", variable=self.auto, command=self.auto_changed, takefocus=0).grid(
            row=1, column=5, sticky='w', **pad)
        ttk.Label(top, text="Scene: every").grid(row=1, column=6, sticky='e', **pad)
        self.interval = ttk.Spinbox(top, from_=0.5, to=60, increment=0.5, width=6)
        self.interval.set(2.0)
        self.interval.grid(row=1, column=7, **pad)
        ttk.Label(top, text="s").grid(row=1, column=8, sticky='w', **pad)

        ttk.Label(top, text="Folder").grid(row=2, column=0, sticky='w', **pad)
        self.folder = tk.StringVar(value=os.path.join(HERE, 'stereo_pairs'))
        entry = ttk.Entry(top, textvariable=self.folder, width=60)
        entry.grid(row=2, column=1, columnspan=5, sticky='we', **pad)
        entry.bind('<Return>', lambda e: self.load_folder())
        entry.bind('<FocusOut>', lambda e: self.load_folder())
        ttk.Button(top, text="Browse…", command=self.browse, takefocus=0).grid(row=2, column=6, **pad)
        ttk.Button(top, text="Open folder", command=self.open_folder, takefocus=0).grid(row=2, column=7, columnspan=2, **pad)
        self.need_still = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Wait until the board is still (recommended: the two cameras are not synchronised)",
                        variable=self.need_still, command=self.auto_changed, takefocus=0).grid(
            row=3, column=0, columnspan=6, sticky='w', **pad)

        self.photo = tk.PhotoImage(width=PREVIEW_WIDTH, height=PREVIEW_WIDTH * 3 // 8)  # blank until the cameras run
        self.preview = tk.Label(self, image=self.photo, bg='black', highlightthickness=5, highlightbackground='black')
        self.preview.pack(padx=8)
        self.preview.bind('<Button-1>', lambda e: self.focus_set())  # back to the keys after typing in a field

        bottom = ttk.Frame(self, padding=8)
        bottom.pack(fill='x')
        self.save_button = tk.Button(bottom, text="Save pair  (Space)", command=self.save, font=('Helvetica', 16, 'bold'),
                                     padx=18, pady=6, takefocus=0)
        self.save_button.pack(side='left')
        ttk.Button(bottom, text="Undo last (U)", command=self.undo, takefocus=0).pack(side='left', padx=10)
        self.confirm_button = tk.Button(bottom, text="Confirm left/right", command=self.confirm, takefocus=0,
                                        font=('Helvetica', 13, 'bold'), padx=10, pady=4)
        self.confirm_button.pack(side='left', padx=6)
        self.counter = tk.Label(bottom, font=('Helvetica', 16, 'bold'))
        self.counter.pack(side='left', padx=16)
        self.status = tk.Label(bottom, font=('Helvetica', 14), anchor='w')
        self.status.pack(side='left', fill='x', expand=True)
        self.message = tk.Label(self, anchor='w', padx=10, pady=4, fg='#555', font=('Helvetica', 12))
        self.message.pack(fill='x')
        self.names = {}
        self.show_names()
        self.say("Pick the two cameras (Scan checks which ones give images), then press Start.")

    def board(self):
        """'charuco' or 'chessboard' in calibration mode, None in scene mode."""
        mode = self.mode.get()
        return mode if mode in ('charuco', 'chessboard') else None

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

    # ------------------------------------------------------------------ cameras
    def show_names(self):
        """Fill the camera lists with the names the operating system gives (just numbers where it gives none)."""
        self.names = camera_names()
        indices = sorted(self.names) if self.names else list(range(6))
        labels = [self.label(i) for i in indices]
        self.left.configure(values=labels)
        self.right.configure(values=labels)
        pair = likely_pair(self.names) or ((indices[0], indices[1]) if len(indices) >= 2 else (0, 1))
        self.left.set(self.label(pair[0]))
        self.right.set(self.label(pair[1]))

    def label(self, index):
        return f"{index}: {self.names[index]}" if index in self.names else str(index)

    @staticmethod
    def index(label):
        """The camera index at the start of a list entry such as '1: Arducam OV9782 USB Camera'."""
        return int(label.split(':')[0].strip())

    def scan(self):
        if self.pair is not None:
            self.say("Stop the cameras before scanning.", RED)
            return
        self.scan_button.configure(state='disabled')
        self.show_names()
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
        self.scan_button.configure(state='normal')
        found = self.scan_result
        if not found:
            self.say("No camera found. Check the USB cables and the camera permission of your terminal or IDE.", RED)
            return
        indices = [i for i, _, _ in found]
        labels = [self.label(i) for i in indices]
        self.left.configure(values=labels)
        self.right.configure(values=labels)
        pair = likely_pair({i: self.names.get(i, '') for i in indices}) if self.names else None
        if pair is None and len(indices) >= 2:
            pair = (indices[-2], indices[-1])
        if pair is not None:
            self.left.set(self.label(pair[0]))
            self.right.set(self.label(pair[1]))
        self.say("Found: " + "; ".join(f"{self.label(i)}, {w}x{h}" for i, w, h in found)
                 + ". Check which is left and which is right in the preview.", GREEN)

    def swap(self):
        a, b = self.left.get(), self.right.get()
        self.left.set(b)
        self.right.set(a)
        if self.pair is not None:
            self.stop()
            self.start()

    def confirm(self):
        """The user checked that the LEFT image comes from the camera on the left."""
        if self.pair is None:
            self.say("Start the cameras first, then check left and right.", RED)
            return
        self.confirmed = True
        self.show_confirmed()
        self.say("Left and right confirmed. Hold the board where both cameras see it.", GREEN)

    def show_confirmed(self):
        self.confirm_button.configure(text="Left/right confirmed" if self.confirmed else "Confirm left/right",
                                      state='disabled' if self.confirmed else 'normal')

    def start_stop(self):
        if self.pair is None:
            self.start()
        else:
            self.stop()

    def start(self):
        try:
            indices = (self.index(self.left.get()), self.index(self.right.get()))
        except ValueError:
            self.say("Pick a camera from each list (or type its number).", RED)
            return
        if indices[0] == indices[1]:
            self.say("Pick two different cameras.", RED)
            return
        size = tuple(int(v) for v in self.size.get().split('x'))
        if self.other_board():
            return
        self.load_folder()
        labels = [self.label(i) for i in indices]
        recorded = recorded_cameras(self.folder.get())  # which cameras took the pairs already in the folder
        verdict = compare_cameras(recorded, labels) if recorded and self.existing_pairs() else 'same'
        if verdict == 'swapped':
            self.say(f"The pairs in this folder were taken with {recorded[0]} as LEFT. Swap left/right back to add to "
                     "them, or pick an empty folder to start again this way round.", RED)
            return
        self.labels = labels  # noted in cameras.txt with every saved pair
        self.opened_folder = os.path.abspath(self.folder.get())
        # Left and right still need checking, unless this folder's pairs were taken with these cameras this way round
        self.confirmed = bool(recorded and self.existing_pairs()) and verdict == 'same'
        self.swap_votes = []
        self.show_confirmed()
        self.pair = CameraPair(indices, size)
        self.pair.board = self.board()
        self.pair.start()
        self.saved_number = None  # a new camera pair numbers its frames from 0 again
        self.shown = -1
        self.start_button.configure(text="Stop")
        self.focus_set()
        self.say(f"Opening {self.label(indices[0])} as LEFT and {self.label(indices[1])} as RIGHT ...")
        if not self.confirmed:
            self.say("Check left and right: stand behind the cameras and hold your hand in front of the LEFT one. If the "
                     "LEFT image goes dark, press Confirm left/right; if the RIGHT image does, press Swap left/right.",
                     ORANGE)
        if verdict == 'different':
            self.say(f"Opening ... NOTE: the pairs already in this folder were taken with {recorded[0]} (left) and "
                     f"{recorded[1]} (right). If these are other cameras or you moved one, use another folder.", ORANGE)

    def stop(self):
        if self.pair is not None:
            self.pair.stop()
            self.pair = None
        self.start_button.configure(text="Start")

    def other_board(self):
        """True (with a message) if the pairs in the folder were taken with the other board than this mode's."""
        old_board = recorded_board(self.folder.get()) if self.existing_pairs() else None
        if self.board() and old_board and old_board != self.board():
            self.say(f"The pairs in this folder were taken with the {old_board}: one calibration needs one board. "
                     f"Pick the {old_board} mode, or another folder.", RED)
            return True
        return False

    def mode_changed(self):
        if self.other_board():
            self.stop()  # this folder belongs to the other board
        if self.pair is not None:
            self.pair.board = self.board()
        self.still_since = None
        if self.board() != self.dots_board:  # another board: its corners are numbered differently
            self.last_saved = {}
            self.find_dots()  # the green dots of the pairs already there, found with this board
        self.update_counter()

    def auto_changed(self):
        self.last_auto = time.time()
        self.still_since = None

    # ------------------------------------------------------------------ folder
    def folders(self):
        return [os.path.join(self.folder.get(), side) for side in ('left', 'right')]

    def existing_pairs(self):
        left, right = self.folders()
        if not os.path.isdir(left):
            return []
        return sorted(f for f in os.listdir(left)
                      if re.fullmatch(r"pair_\d+\.jpg", f) and os.path.exists(os.path.join(right, f)))

    def load_folder(self):
        """Count the pairs already in the folder and collect their corners for the green coverage dots."""
        if self.pair is not None and os.path.abspath(self.folder.get()) != self.opened_folder:
            self.stop()  # Start checks the cameras, the size and the board against the pairs of the new folder
            self.say("New folder: press Start again, so it can check the cameras, the size and the board against "
                     "the pairs there.", ORANGE)
        if os.path.abspath(self.folder.get()) != self.loaded_folder:  # Undo belongs to one folder
            self.session, self.loaded_folder = [], os.path.abspath(self.folder.get())
        self.find_dots()

    def find_dots(self):
        """The corners of the pairs in the folder, found with this mode's board, for the green dots."""
        self.old_coverage = [[], []]
        self.dots_board = board = self.board()
        if board and recorded_board(self.folder.get()) not in (None, board):
            board = None  # the folder belongs to the other board and is refused anyway: nothing to search
        for name in self.existing_pairs() if board else []:  # no dots in scene mode
            for saved, folder in zip(self.old_coverage, self.folders()):
                old = cv2.imread(os.path.join(folder, name), cv2.IMREAD_GRAYSCALE)
                if old is not None:
                    saved.extend(corners_by_id(detect_board(old, board)).values())
        self.update_counter()

    def browse(self):
        folder = filedialog.askdirectory(initialdir=self.folder.get() if os.path.isdir(self.folder.get()) else HERE,
                                         title="Folder for the pairs (left/ and right/ are made inside)")
        if folder:
            self.folder.set(folder)
            self.load_folder()

    def open_folder(self):
        folder = self.folder.get()
        os.makedirs(folder, exist_ok=True)
        if sys.platform == 'darwin':
            subprocess.Popen(['open', folder])
        elif sys.platform.startswith('win'):
            os.startfile(folder)
        else:
            subprocess.Popen(['xdg-open', folder])

    def update_counter(self):
        count = len(self.existing_pairs())
        target = f" / {TARGET}" if self.board() else ""
        self.counter.configure(text=f"{count}{target} pairs")

    # ------------------------------------------------------------------ saving
    def save(self, auto=False, item=None):
        if item is None:
            item = self.pair.newest() if self.pair is not None else None
        if item is None:
            self.say("Start the cameras first.", RED)
            return False
        if not self.confirmed:
            self.say("Check left and right first: press Confirm left/right (or Swap left/right).", ORANGE)
            return False
        if self.board():
            if item['shared'] is None or item.get('board') != self.board():
                self.say("Both cameras must see the " + ("whole chessboard" if self.board() == 'chessboard' else
                                                         "board (at least 6 of the same corners)")
                         + ". Move it and try again.", RED)
                return False
            if self.need_still.get():
                if not item['still'] or self.still_since is None or time.time() - self.still_since < READY_TIME:
                    self.say("The board is moving: hold it still for half a second (or rest it on something) and save "
                             "again, or untick Wait until still.", ORANGE)
                    return False
                if self.saved_still == self.still_since:
                    self.say("Already saved: move or tilt the board to a new position, then save again.", ORANGE)
                    return False
        if item['number'] == self.saved_number:
            self.say("Already saved: this pair of camera frames is in the folder already.", ORANGE)
            return False
        names = self.existing_pairs()
        number = max(int(f[5:-4]) for f in names) + 1 if names else 0
        name = f"pair_{number:02d}.jpg"
        for folder in self.folders():
            os.makedirs(folder, exist_ok=True)
        paths = [os.path.join(folder, name) for folder in self.folders()]
        if not all(cv2.imwrite(path, frame) for path, frame in zip(paths, item['frames'])):
            self.say(f"Could not write {name}: check that {self.folder.get()} is writable.", RED)
            return False
        # which camera took the left and which the right photos, and with which board (scene pairs keep the note)
        record_cameras(self.folder.get(), *self.labels, self.board() or recorded_board(self.folder.get()))
        self.session.append((name, list(item['corners'][0].values()), list(item['corners'][1].values())))
        self.last_saved = item['corners'][0]
        self.saved_still, self.saved_number, self.last_auto = self.still_since, item['number'], time.time()
        shared = f", {len(item['shared'][0])} shared corners" if item['shared'] is not None else ""
        self.say(f"{'Auto-saved' if auto else 'Saved'} {name} in left/ and right/{shared}.", GREEN)
        self.update_counter()
        self.flash()
        return True

    def undo(self):
        if not self.session:
            self.say("Nothing to undo: Undo only removes pairs saved since the folder was opened.", ORANGE)
            return
        name, _, _ = self.session.pop()
        for folder in self.folders():
            path = os.path.join(folder, name)
            if os.path.exists(path):
                os.remove(path)
        self.saved_still = self.saved_number = None  # by hand, the view can be saved again
        self.say(f"Removed {name} from left/ and right/.", ORANGE)
        self.update_counter()

    def flash(self):
        self.preview.configure(highlightbackground='white')
        self.after(150, lambda: self.preview.configure(highlightbackground='black'))

    def new_position(self, corners):
        """True if the board moved away from where it was in the last saved pair."""
        common = corners.keys() & self.last_saved.keys()
        if len(common) < 0.5 * max(1, min(len(corners), len(self.last_saved))):
            return True
        return np.mean([np.linalg.norm(corners[i] - self.last_saved[i]) for i in common]) > NEW_POSITION

    # ------------------------------------------------------------------ the loop
    def tick(self):
        self.after(30, self.tick)
        if self.pair is None:
            return
        if self.pair.state not in ('opening', 'running'):
            self.say(self.pair.state, RED)
            self.stop()
            return
        item = self.pair.newest()
        if item is None or item['number'] == self.shown:
            return
        if self.shown == -1:  # first frame: earlier pairs in the folder must have the same size
            earlier = self.existing_pairs()
            old = cv2.imread(os.path.join(self.folders()[0], earlier[0])) if earlier else None
            h, w = item['frames'][0].shape[:2]
            if old is not None and old.shape[:2] != (h, w):
                self.say(f"The pairs in this folder are {old.shape[1]}x{old.shape[0]}, but the cameras give {w}x{h}: "
                         "pick that size, or another folder.", RED)
                self.stop()
                return
        self.shown = item['number']
        now = time.time()
        self.frame_times = [t for t in self.frame_times if now - t < 1.0] + [now]
        calibration = self.board() is not None

        # Are the cameras the right way round? A point sits further LEFT in the right camera's image than in the
        # left one's (positive disparity), so the board shifting the other way means the cameras are swapped.
        if calibration and item['shared'] is not None:
            # x of each shared corner in the left and the right image (OpenCV 4 gives N x 1 x 2 arrays, OpenCV 5 N x 2)
            shift = np.median(item['shared'][1].reshape(-1, 2)[:, 0] - item['shared'][2].reshape(-1, 2)[:, 0])
            self.swap_votes = (self.swap_votes + [shift < -5])[-15:]
        looks_swapped = len(self.swap_votes) >= 10 and sum(self.swap_votes) >= 10
        auto_ok = self.auto.get() and self.confirmed  # nothing is saved before left and right are checked

        # Status and Auto save
        if calibration:
            if item['shared'] is None:
                text = "Chessboard not whole in both cameras" if self.board() == 'chessboard' else "Board not seen by both cameras"
                colour = RED
                self.still_since = None
            elif not self.need_still.get():  # no still check: hold the board steady yourself
                self.still_since = None
                text, colour = f"{len(item['shared'][0])} shared corners - ready (no still check)", GREEN
                if auto_ok and now - self.last_auto >= AUTO_INTERVAL:
                    if self.new_position(item['corners'][0]):
                        self.save(auto=True, item=item)
                    else:
                        text = f"{len(item['shared'][0])} shared corners - move the board to a new position"
            elif not item['still']:
                motion = item.get('motion') or [None, None]
                lost = [side for side, m in zip(('left', 'right'), motion) if m is None]
                if len(lost) == 1:  # one camera found the board only in this frame
                    why = f" (the {lost[0]} camera just lost the board: keep it fully in view)"
                elif not lost:
                    why = f" (moving {max(motion):.1f} px per frame, needs under {MAX_MOTION:g})"
                else:
                    why = ""
                text, colour = f"{len(item['shared'][0])} shared corners - hold still{why}", ORANGE
                self.still_since = None
            else:
                drift = [corner_motion(n, a) for n, a in zip(item['corners'], self.anchor)]
                if self.still_since is None or not all(d is not None and d <= MAX_DRIFT for d in drift):
                    self.still_since, self.anchor = now, item['corners']  # (re)start the still period here
                if now - self.still_since < READY_TIME:
                    text, colour = f"{len(item['shared'][0])} shared corners - hold still", ORANGE
                elif self.saved_still == self.still_since:
                    text, colour = f"{len(item['shared'][0])} shared corners - saved: move the board to a new position", GREEN
                else:
                    text, colour = f"{len(item['shared'][0])} shared corners - ready", GREEN
                if auto_ok and now - self.still_since >= STILL_TIME and self.saved_still != self.still_since:
                    if self.new_position(item['corners'][0]):
                        self.save(auto=True, item=item)
                    else:
                        text = f"{len(item['shared'][0])} shared corners - move the board to a new position"
        else:
            text, colour = "Ready", GREEN
            try:
                interval = max(0.5, float(self.interval.get()))
            except ValueError:
                interval = 2.0
            if auto_ok and now - self.last_auto >= interval:
                self.last_auto = now
                self.save(auto=True, item=item)
        if looks_swapped:
            text, colour = "The cameras look swapped (the board shifts the wrong way): press Swap left/right", RED
        elif not self.confirmed:
            text, colour = "Check left and right, then press Confirm left/right (or Swap left/right)", ORANGE
        self.status.configure(text=f"{text}   ·   {len(self.frame_times)} fps", fg=colour)

        # Preview: both photos side by side; corners in red, corners of saved pairs as green dots
        views = [frame.copy() for frame in item['frames']]
        if calibration:
            session = [[p for pair in self.session for p in pair[1]], [p for pair in self.session for p in pair[2]]]
            for view, corners, old, new in zip(views, item['corners'], self.old_coverage, session):
                r = max(2, view.shape[1] // 200)
                for x, y in old + new:
                    cv2.circle(view, (int(x), int(y)), r, (0, 255, 0), -1)
                for x, y in corners.values():
                    cv2.circle(view, (int(x), int(y)), r + 2, (0, 0, 255), -1)
        mark = "" if self.confirmed else "?"  # LEFT? and RIGHT? until the user has checked them
        labels = [f"{side}{mark}  {self.label(i)}" for side, i in zip(('LEFT', 'RIGHT'), self.pair.indices)]
        for view, label in zip(views, labels):
            scale = view.shape[1] / 1280  # the same size on screen at any resolution
            cv2.rectangle(view, (0, 0), (view.shape[1], int(48 * scale)), (0, 0, 0), -1)
            cv2.putText(view, label, (int(12 * scale), int(34 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale,
                        (255, 255, 255), max(1, round(2 * scale)), cv2.LINE_AA)
        both = np.hstack(views)
        h = int(round(both.shape[0] * PREVIEW_WIDTH / both.shape[1]))
        small = cv2.resize(both, (PREVIEW_WIDTH, h), interpolation=cv2.INTER_AREA)
        ppm = b'P6 %d %d 255 ' % (PREVIEW_WIDTH, h) + cv2.cvtColor(small, cv2.COLOR_BGR2RGB).tobytes()
        self.photo = tk.PhotoImage(data=ppm, format='PPM')  # keep a reference, or Tk drops the image
        self.preview.configure(image=self.photo, width=PREVIEW_WIDTH, height=h)

    def quit_app(self):
        self.stop()
        self.destroy()


if __name__ == '__main__':
    App().mainloop()
