# ai_camera.py
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# =========================
# CONFIG
# =========================
SOURCE = "assets/input.mp4"
OUT = "outputs/annotated.mp4"

# YOLO model priority:
# 1) TensorRT engine if present (fast on RTX)
# 2) yolov8s-worldv2.pt if present (open-vocab)
# 3) yolov8n.pt fallback (stable)
YOLO_ENGINE = "yolov8n.engine"
YOLO_WORLD = "yolov8s-worldv2.pt"
YOLO_FALLBACK = "yolov8n.pt"

# What you want YOLO to care about
YOLO_CLASSES = ["treadmill", "person"]  # you can add: "bench", "dumbbell", etc.

# Face detection: Haar cascade (FAST, no downloads, very stable)
# (We expand box + track between detections for better stickiness.)
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Tracking
DETECT_FACES_EVERY_N_FRAMES = 2         # detect periodically (tracking fills gaps)
MAX_MISSED_FRAMES = 20                  # how long a track can survive without detection
IOU_MATCH_THRESHOLD = 0.10              # lower = easier matching (helps fast motion)

# Blur strength (privacy-max)
EXPAND_FACE_BOX = 0.70                  # expand around face (hair/angles)
PIXEL_BLOCK = 75                        # lower = more pixelation; try 8 or 6 for harsher
GAUSSIAN_KERNEL = 121                    # must be odd; 99/121 = very strong blur
GAUSSIAN_SIGMA = 0                      # 0 lets OpenCV choose; fine for heavy blur
ADD_NOISE = True                        # anti-reconstruction noise
NOISE_STD = 124                          # higher = more distortion (try 12-28)
MULTI_PASS = 3                          # 2-3 passes increases irreversibility

# Output
FORCE_FPS = None                        # None = use input fps, or set like 30
DRAW_LABELS = True


# =========================
# HELPERS
# =========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-9)

def expand_box(x1, y1, x2, y2, w, h, expand=0.3):
    bw = x2 - x1
    bh = y2 - y1
    ex = int(bw * expand)
    ey = int(bh * expand)
    nx1 = clamp(x1 - ex, 0, w - 1)
    ny1 = clamp(y1 - ey, 0, h - 1)
    nx2 = clamp(x2 + ex, 0, w - 1)
    ny2 = clamp(y2 + ey, 0, h - 1)
    return nx1, ny1, nx2, ny2

def pixelate(img: np.ndarray, block: int) -> np.ndarray:
    if block <= 1:
        return img
    h, w = img.shape[:2]
    # downsample then upsample (nearest) = chunky pixelation
    small_w = max(1, w // block)
    small_h = max(1, h // block)
    tmp = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(tmp, (w, h), interpolation=cv2.INTER_NEAREST)
    return out

def hard_privacy_blur(frame: np.ndarray, box: Tuple[int, int, int, int]) -> None:
    """In-place: pixelation + multi-pass heavy gaussian + optional noise."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    # Pixelation first
    roi2 = pixelate(roi, PIXEL_BLOCK)

    # Multi-pass Gaussian blur
    for _ in range(max(1, MULTI_PASS)):
        k = GAUSSIAN_KERNEL
        if k % 2 == 0:
            k += 1
        roi2 = cv2.GaussianBlur(roi2, (k, k), GAUSSIAN_SIGMA)

    # Noise injection (anti-reconstruction)
    if ADD_NOISE:
        noise = np.random.normal(0, NOISE_STD, roi2.shape).astype(np.float32)
        roi2f = roi2.astype(np.float32) + noise
        roi2 = np.clip(roi2f, 0, 255).astype(np.uint8)

        # one more blur pass after noise to avoid “sparkle”
        k = GAUSSIAN_KERNEL
        if k % 2 == 0:
            k += 1
        roi2 = cv2.GaussianBlur(roi2, (k, k), 0)

    frame[y1:y2, x1:x2] = roi2


@dataclass
class FaceTrack:
    track_id: int
    box: Tuple[int, int, int, int]
    missed: int = 0


class FaceTracker:
    def __init__(self):
        self.tracks: List[FaceTrack] = []
        self.next_id = 1

    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[FaceTrack]:
        """IoU-based association + persistence."""
        # mark all as missed by default
        for t in self.tracks:
            t.missed += 1

        used_det = set()

        # try to match existing tracks to new detections
        for t in self.tracks:
            best_iou = 0.0
            best_j = -1
            for j, d in enumerate(detections):
                if j in used_det:
                    continue
                score = iou_xyxy(t.box, d)
                if score > best_iou:
                    best_iou = score
                    best_j = j
            if best_j >= 0 and best_iou >= IOU_MATCH_THRESHOLD:
                t.box = detections[best_j]
                t.missed = 0
                used_det.add(best_j)

        # any unused detections become new tracks
        for j, d in enumerate(detections):
            if j in used_det:
                continue
            self.tracks.append(FaceTrack(track_id=self.next_id, box=d, missed=0))
            self.next_id += 1

        # prune old tracks
        self.tracks = [t for t in self.tracks if t.missed <= MAX_MISSED_FRAMES]

        return self.tracks


def detect_faces_haar(gray: np.ndarray, face_cascade: cv2.CascadeClassifier) -> List[Tuple[int, int, int, int]]:
    """Returns list of boxes in xyxy."""
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    out = []
    for (x, y, w, h) in faces:
        out.append((x, y, x + w, y + h))
    return out


def load_yolo_model(device: str) -> YOLO:
    if os.path.exists(YOLO_ENGINE):
        print(f"Loading model: {YOLO_ENGINE}")
        return YOLO(YOLO_ENGINE)
    if os.path.exists(YOLO_WORLD):
        print(f"Loading model: {YOLO_WORLD}")
        return YOLO(YOLO_WORLD)
    print(f"Loading model: {YOLO_FALLBACK}")
    return YOLO(YOLO_FALLBACK)


def main():
    ensure_dir(os.path.dirname(OUT))

    print("Starting inference...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # YOLO
    model = load_yolo_model(device)

    # If it's YOLO-World, set classes by name
    # (won't crash on non-world models; we guard it)
    try:
        # YOLO-World supports set_classes
        if hasattr(model, "set_classes") and callable(getattr(model, "set_classes")):
            model.set_classes(YOLO_CLASSES)
            print(f"Model loaded (YOLO-World). Classes set to: {', '.join(YOLO_CLASSES)}")
        else:
            print("Model loaded. (Standard YOLO)")
    except Exception as e:
        print("Warning: set_classes failed, continuing:", e)

    # Face cascade
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar face cascade. OpenCV install might be broken.")

    # Video IO
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {SOURCE}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(FORCE_FPS) if FORCE_FPS else float(fps_in)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open. Try a different codec or path.")

    tracker = FaceTracker()

    frame_count = 0
    t0 = time.time()
    last_log = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1

        # ---- YOLO detection (optional annotation) ----
        # Use half precision when running on cuda; ultralytics handles it when possible
        res = None
        try:
            # keep it light: only do one forward per frame
            res_list = model.predict(frame, verbose=False, device=0 if device == "cuda" else "cpu", imgsz=640)
            res = res_list[0] if res_list else None
        except Exception as e:
            # Don't kill the run if YOLO hiccups; keep face privacy working
            res = None

        annotated = frame.copy()

        # ---- Face detection every N frames ----
        if frame_count % DETECT_FACES_EVERY_N_FRAMES == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = detect_faces_haar(gray, face_cascade)

            # Expand boxes for privacy coverage
            expanded = []
            for (x1, y1, x2, y2) in dets:
                ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, w, h, EXPAND_FACE_BOX)
                expanded.append((ex1, ey1, ex2, ey2))

            tracker.update(expanded)
        else:
            # still decay tracks even without detection
            tracker.update([])

        # ---- Apply privacy blur for all active tracks ----
        for t in tracker.tracks:
            hard_privacy_blur(annotated, t.box)

        # ---- Draw YOLO boxes/labels (optional) ----
        if DRAW_LABELS and res is not None and getattr(res, "boxes", None) is not None:
            try:
                boxes = res.boxes
                names = res.names if hasattr(res, "names") else {}
                for b in boxes:
                    xyxy = b.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = map(int, xyxy)
                    cls = int(b.cls[0].item()) if hasattr(b, "cls") else -1
                    conf = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
                    label = names.get(cls, str(cls))

                    # If you only want treadmill/person drawn:
                    if YOLO_CLASSES and label not in YOLO_CLASSES:
                        continue

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    txt = f"{label} {conf:.2f}"
                    cv2.putText(
                        annotated, txt,
                        (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA
                    )
            except Exception:
                pass

        writer.write(annotated)

        # ---- Performance log ----
        now = time.time()
        if now - last_log >= 2.0:
            elapsed = now - t0
            fps_avg = frame_count / max(1e-9, elapsed)
            print(f"Frames: {frame_count} | Face tracks: {len(tracker.tracks)} | Avg FPS: {fps_avg:.2f}")
            last_log = now

    cap.release()
    writer.release()

    size = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    print(f"Done. Frames processed: {frame_count}")
    print(f"Output saved to: {OUT} ({size} bytes)")


if __name__ == "__main__":
    main()