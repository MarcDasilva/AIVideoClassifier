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

def residual_entropy(frames, background):
    entropies = []
    for frame in frames:
        diff = cv2.absdiff(frame, background)
        hist, _ = np.histogram(diff.flatten(), bins=256, range=(0,255), density=True)
        hist += 1e-10
        ent = entropy(hist)
        entropies.append(ent)
    return entropies

def flicker_rate(frames, background, threshold=10):
    rates = []
    for frame in frames:
        diff = cv2.absdiff(frame, background)
        flicker = np.mean(diff > threshold)
        rates.append(flicker)
    return rates

def noise_stress_ssim(frames, sigma=5):
    """Compute SSIM drop when adding Gaussian noise to frames"""
    deltas = []
    for frame in frames:
        noise = np.random.normal(0, sigma, frame.shape)
        noisy_frame = frame + noise
        noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
        score, _ = ssim(frame, noisy_frame, full=True)
        deltas.append(score)
    return deltas

# Feature Aggregation
def aggregate_metrics(ssim_vals, res_ent, flickers, noise_ssim_vals):
    features = {
        "ssim_mean": np.mean(ssim_vals),
        "ssim_std": np.std(ssim_vals),
        "entropy_mean": np.mean(res_ent),
        "entropy_std": np.std(res_ent),
        "flicker_mean": np.mean(flickers),
        "flicker_std": np.std(flickers),
        "noise_ssim_mean": np.mean(noise_ssim_vals),
        "noise_ssim_std": np.std(noise_ssim_vals),
    }
    return features

# Heuristic Scoring
def heuristic_score(features):
    # higher flicker & entropy → AI, lower SSIM → AI, lower noise-SSIM → AI
    score = ((1 - features["ssim_mean"]) * 0.4 +
             features["entropy_mean"] * 0.25 +
             features["flicker_mean"] * 0.2 +
             (1 - features["noise_ssim_mean"]) * 0.15)
    verdict = "likely_real" if score < 0.5 else "possibly_ai"
    return score, verdict


# ML Functions 
def train_model(csv_files, labels, out_path):
    dfs = [pd.read_csv(f) for f in csv_files]
    X = pd.concat(dfs, ignore_index=True)
    y = pd.Series(labels)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, out_path)
    print(f"Model saved to {out_path}")

def apply_model(features, model_path):
    clf = joblib.load(model_path)
    X = pd.DataFrame([features])
    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X)[0]
    return int(pred), prob.tolist()


# Main Pipeline
def analyze_video(video_path, fps=2, save_csv=None, model_path=None):
    frames = load_frames(video_path, fps)
    background = compute_background(frames)
    ssim_vals = compute_ssim(frames, background)
    res_ent = residual_entropy(frames, background)
    flickers = flicker_rate(frames, background)
    noise_ssim_vals = noise_stress_ssim(frames, sigma=5)

    features = aggregate_metrics(ssim_vals, res_ent, flickers, noise_ssim_vals)
    score, verdict = heuristic_score(features)

    print(f"Video: {video_path}")
    print(f"Features: {features}")
    print(f"Heuristic Score: {score:.3f}, Verdict: {verdict}")

    if save_csv:
        df = pd.DataFrame([features])
        df.to_csv(save_csv, index=False)
        print(f"Saved features to {save_csv}")

    if model_path:
        pred, prob = apply_model(features, model_path)
        print(f"ML Prediction: {pred}, Probabilities: {prob}")

    return features, score, verdict

# CLI
def main():
    parser = argparse.ArgumentParser(description="Static-camera AI Video Detector")
    subparsers = parser.add_subparsers(dest="command")

    # Analyze command
    parser_analyze = subparsers.add_parser("analyze")
    parser_analyze.add_argument("video", type=str)
    parser_analyze.add_argument("--fps", type=int, default=2)
    parser_analyze.add_argument("--save-csv", type=str)
    parser_analyze.add_argument("--model", type=str)

    args = parser.parse_args()
    if args.command == "analyze":
        analyze_video(args.video, fps=args.fps, save_csv=args.save_csv, model_path=args.model)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
