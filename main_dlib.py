# main_dlib_safe.py
import cv2
import math
import numpy as np
import dlib
import imutils
from imutils import face_utils
import vlc
import time
import os
import sys

# ---------- Helpers ----------
def euclideanDist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def ear(eye):
    # Eye Aspect Ratio
    return (euclideanDist(eye[1], eye[5]) + euclideanDist(eye[2], eye[4])) / (2.0 * euclideanDist(eye[0], eye[3]))

def writeEyes(left_pts, right_pts, rgb_img, out_prefix='eye'):
    """
    Save eye crops safely (clipped to image bounds).
    left_pts/right_pts are Nx2 arrays of landmark coordinates.
    """
    h, w = rgb_img.shape[:2]

    def crop_and_save(pts, name):
        xs = pts[:, 0]
        ys = pts[:, 1]
        x1 = int(np.clip(xs.min(), 0, w - 1))
        x2 = int(np.clip(xs.max(), 0, w - 1))
        y1 = int(np.clip(ys.min(), 0, h - 1))
        y2 = int(np.clip(ys.max(), 0, h - 1))
        if x2 > x1 and y2 > y1:
            crop = rgb_img[y1:y2, x1:x2]
            cv2.imwrite(f"{name}.jpg", crop)

    crop_and_save(left_pts, f"{out_prefix}-left")
    crop_and_save(right_pts, f"{out_prefix}-right")

# ---------- Config ----------
FRAME_THRESH = 15
CLOSE_THRESH = 0.3  # threshold for EAR to consider eye closed
ALERT_SOUND = "alert-sound.mp3"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
CAMERA_INDEX = 0  # try 0, then 1 if you need

# ---------- Init ----------
# VLC player for alert
alert = vlc.MediaPlayer(ALERT_SOUND) if os.path.exists(ALERT_SOUND) else None

# Video capture with DirectShow backend (more stable on Windows)
capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not capture.isOpened():
    print(f"ERROR: Camera index {CAMERA_INDEX} could not be opened. Try another index (0,1...) or check camera permissions.")
    sys.exit(1)

# dlib detector + predictor (with guard)
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
except Exception as e:
    print(f"ERROR loading shape predictor '{SHAPE_PREDICTOR_PATH}': {e}")
    print("Make sure the .dat file is the correct, uncorrupted file (~99MB).")
    capture.release()
    sys.exit(1)

(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

flag = 0
avgEAR = 0.0

print(f"close_thresh={CLOSE_THRESH}, frame_thresh={FRAME_THRESH}")

# ---------- Main loop ----------
try:
    while True:
        ret, frame = capture.read()

        if not ret or frame is None:
            # failed frame read — try again or break after a short wait
            print("Warning: failed to grab frame. Retrying...")
            time.sleep(0.1)
            continue

        # ensure we have a uint8 image
        if frame.dtype != np.uint8:
            print("Warning: frame dtype is not uint8:", frame.dtype)
            continue

        # optionally resize for faster processing
        # frame = imutils.resize(frame, width=640)

        # convert to gray (if already gray, skip conversion)
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            print("Unsupported frame shape:", frame.shape)
            continue

        # detector needs an 8-bit grayscale image
        if gray.dtype != np.uint8:
            print("Gray frame dtype unsupported:", gray.dtype)
            continue

        rects = detector(gray, 0)

        # create a color display image for drawing (so contours show in color)
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if len(rects) > 0:
            rect = rects[0]  # use first face
            shape = face_utils.shape_to_np(predictor(gray, rect))
            leftEye = shape[leStart:leEnd]
            rightEye = shape[reStart:reEnd]
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            leftEAR = ear(leftEye)
            rightEAR = ear(rightEye)
            avgEAR = (leftEAR + rightEAR) / 2.0

            # draw eye contours (green)
            cv2.drawContours(display, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(display, [rightEyeHull], -1, (0, 255, 0), 1)

            # optionally save eye crops
            writeEyes(leftEye, rightEye, frame, out_prefix='eye')

            # drowsiness logic
            if avgEAR < CLOSE_THRESH:
                flag += 1
                # print(frames closed for debugging)
                # print("closed frames:", flag)
                if flag >= FRAME_THRESH:
                    if alert is not None and not alert.is_playing():
                        alert.play()
                    cv2.putText(display, "DROWSY! ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                if flag != 0:
                    print("Flag reset to 0")
                flag = 0
                if alert is not None and alert.is_playing():
                    alert.stop()
        else:
            # no faces detected — optionally reset flag or take other action
            # flag = 0
            pass

        # show EAR value
        cv2.putText(display, f"EAR: {avgEAR:.2f}", (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Driver", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

finally:
    if alert is not None and alert.is_playing():
        alert.stop()
    capture.release()
    cv2.destroyAllWindows()
