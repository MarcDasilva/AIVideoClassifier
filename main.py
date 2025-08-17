import cv2
import numpy as np
import argparse
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
import joblib
from pathlib import Path

# Video Processing Functions
def load_frames(video_path, fps=2):
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(video_fps // fps), 1)

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        idx += 1
    cap.release()
    return frames

def compute_background(frames):
    stack = np.stack(frames, axis=0)
    background = np.median(stack, axis=0).astype(np.uint8)
    return background

# Metrics
def compute_ssim(frames, background):
    ssim_values = []
    for frame in frames:
        score, _ = ssim(frame, background, full=True)
        ssim_values.append(score)
    return ssim_values


if __name__ == "__main__":
    main()
