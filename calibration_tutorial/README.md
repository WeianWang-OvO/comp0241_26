# Calibration tutorial

Two labs on camera calibration with OpenCV, each with a step-by-step tutorial page.

| Folder | Lab | Tutorial page |
|---|---|---|
| `intrinsic/` | Calibrate one webcam: its focal length, optical centre and lens distortion, with a chessboard or a ChArUco board | [intrinsic/docs/index.html](intrinsic/docs/index.html) |
| `stereo/` | Calibrate two USB webcams as a stereo pair, rectify them and measure depth | [stereo/docs/index.html](stereo/docs/index.html) |

GitHub shows the pages as HTML source. To read them, open `index.html` from your clone in a web browser, or go to
https://weianwang-ovo.github.io/comp0241_26/calibration_tutorial/ once GitHub Pages is turned on for the course
repository.

The scripts need OpenCV 4.7 or newer and NumPy. In the course environment:

```bash
conda activate comp0241
pip install -U opencv-contrib-python numpy
cd calibration_tutorial/intrinsic
python w3_calibration_charuco.py sample_charuco_images
```

The last line calibrates from the ten sample ChArUco photos, so you can check your setup before you print a board.
Each tutorial page then takes you through capturing and calibrating with your own camera.
