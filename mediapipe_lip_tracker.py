from pathlib import Path

from collections import Counter, deque
from time import monotonic

import cv2
import numpy as np

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

from vowel_classifier import (
    MouthFeatures,
    classify_vowel_from_landmarks,
    compute_profile,
    extract_mouth_features,
    get_calibrated_profiles,
    reset_calibrated_profiles,
    set_calibrated_profiles,
)

# MediaPipe Tasks の顔ランドマークモデルのパス。ダウンロード先:
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
MODEL_PATH = "face_landmarker.task"

WINDOW_NAME = "MediaPipe Lip Landmarks"

CALIBRATION_SEQUENCE = [
    ("a", "あ"),
    ("i", "い"),
    ("u", "う"),
    ("e", "え"),
    ("o", "お"),
]
CALIBRATION_DURATION_SECONDS = 3.0
CALIBRATION_MIN_SAMPLES = 25
FRAME_INTERVAL_MS = 33
SILENCE_VERTICAL_THRESHOLD = 0.05
SILENCE_FRAMES_REQUIRED = 5
VOWEL_SMOOTHING_WINDOW = 4

# MediaPipe Face Mesh ランドマーク番号（468点トポロジ）
OUTER_LIP_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308
]
INNER_LIP_INDICES = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191
]
EYE_CORNER_INDICES = {
    "left_outer": 33,
    "right_outer": 263
}


def landmarks_to_pixels(landmarks, image_width, image_height, indices):
    """指定したランドマーク番号をピクセル座標のリストに変換する。"""
    return [
        (
            int(landmarks[index].x * image_width),
            int(landmarks[index].y * image_height)
        )
        for index in indices
    ]


def run_calibration(
    cap: cv2.VideoCapture,
    landmarker: FaceLandmarker,
    start_timestamp_ms: int,
) -> int:
    """起動時に母音サンプルを収集し、個人向け重心（プロファイル）を作成する。"""

    timestamp_ms = start_timestamp_ms
    calibrated = get_calibrated_profiles()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("\n=== Calibration Start ===")
    print("Please look at the camera. For each prompt, pronounce the vowel steadily until the progress bar completes.")
    for vowel, label in CALIBRATION_SEQUENCE:
        samples: list[MouthFeatures] = []
        start_time = monotonic()
        end_time = start_time + CALIBRATION_DURATION_SECONDS

        print(f"Calibrating vowel '{label}' ({vowel.upper()}) ...")

        while monotonic() < end_time:
            success, frame = cap.read()
            if not success:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += FRAME_INTERVAL_MS

            overlay = frame.copy()
            elapsed = monotonic() - start_time
            remaining = max(end_time - monotonic(), 0.0)
            progress = min((elapsed / CALIBRATION_DURATION_SECONDS) * 100.0, 100.0)
            progress_text = f"Progress: {progress:5.1f}%"
            cv2.putText(
                overlay,
                f"Calibration: '{label}' ({vowel.upper()})",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                "Speak steadily. Press 's' to skip, 'q' to abort.",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                progress_text,
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            detection_text = ""
            if result and result.face_landmarks:
                landmarks = result.face_landmarks[0]
                try:
                    features = extract_mouth_features(landmarks)
                    samples.append(features)
                    detection_text = "Face detected"
                except ValueError:
                    detection_text = "Face too close/far"
            else:
                detection_text = "No face detected"

            cv2.putText(
                overlay,
                detection_text,
                (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Calibration aborted by user.")
                set_calibrated_profiles(calibrated)
                return timestamp_ms
            if key == ord("s"):
                print(f"Skipped vowel '{label}'. Using previous centroid.")
                break

        if len(samples) >= CALIBRATION_MIN_SAMPLES:
            profile = compute_profile(samples)
            calibrated[vowel] = profile
            print(
                f"  -> collected {len(samples)} samples (vert={profile.vertical:.3f}, horiz={profile.horizontal:.3f})"
            )
        else:
            print(
                f"  -> insufficient samples ({len(samples)}) for '{label}'. Retaining previous profile."
            )

    set_calibrated_profiles(calibrated)
    print("=== Calibration Complete ===\n")
    return timestamp_ms


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open default camera (index 0).")

    model_path = Path(MODEL_PATH)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. Download it from "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task "
            "and place it alongside this script, or update MODEL_PATH."
        )

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    reset_calibrated_profiles()
    timestamp_ms = run_calibration(cap, landmarker, start_timestamp_ms=0)

    recent_vowels = deque(maxlen=VOWEL_SMOOTHING_WINDOW)
    vowel_buffer: list[str] = []
    confirmed_text: list[str] = []
    silence_counter = 0

    print("Press 'c' to recalibrate, 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        frame_overlay = frame.copy()
        image_height, image_width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += FRAME_INTERVAL_MS

        vowel_text = ""
        top2 = []
        current_vertical: float | None = None
        current_vowel_smoothed: str | None = None

        if result and result.face_landmarks:
            face_landmarks = result.face_landmarks[0]

            outer_lip_points = landmarks_to_pixels(face_landmarks, image_width, image_height, OUTER_LIP_INDICES)
            inner_lip_points = landmarks_to_pixels(face_landmarks, image_width, image_height, INNER_LIP_INDICES)
            eye_corner_points = {
                name: landmarks_to_pixels(face_landmarks, image_width, image_height, [index])[0]
                for name, index in EYE_CORNER_INDICES.items()
            }

            try:
                vowel, features, scores = classify_vowel_from_landmarks(face_landmarks)
                recent_vowels.append(vowel)
                vowel_smoothed = Counter(recent_vowels).most_common(1)[0][0]
                top2 = sorted(scores.items(), key=lambda kv: kv[1])[:2]
                current_vertical = features.vertical
                current_vowel_smoothed = vowel_smoothed

                vowel_text = (
                    f"Vowel: {vowel_smoothed.upper()} (raw {vowel.upper()})  "
                    f"ratio={features.ratio:.2f}  vert={features.vertical:.2f}  horiz={features.horizontal:.2f}"
                )
            except ValueError as exc:
                vowel_text = f"Vowel: N/A ({exc})"

            cv2.polylines(
                frame_overlay,
                [np.array(outer_lip_points, dtype=np.int32)],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )
            cv2.polylines(
                frame_overlay,
                [np.array(inner_lip_points, dtype=np.int32)],
                isClosed=True,
                color=(0, 165, 255),
                thickness=2,
            )

            for point in eye_corner_points.values():
                cv2.circle(frame_overlay, point, radius=3, color=(255, 0, 0), thickness=-1)

            top2_text = ""
            if len(top2) == 2:
                top2_text = f"Top2: {top2[0][0].upper()}={top2[0][1]:.2f} {top2[1][0].upper()}={top2[1][1]:.2f}"

            debug_lines = [
                vowel_text or "Vowel: --",
                top2_text,
                f"Outer lip points: {outer_lip_points[:4]} ... | Eye corners: {eye_corner_points}",
            ]
            for idx, line in enumerate(debug_lines):
                cv2.putText(
                    frame_overlay,
                    line,
                    (10, image_height - 10 - idx * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if current_vertical is not None and current_vertical > SILENCE_VERTICAL_THRESHOLD:
            if current_vowel_smoothed is not None:
                vowel_buffer.append(current_vowel_smoothed)
            silence_counter = 0
        else:
            silence_counter += 1
            if silence_counter >= SILENCE_FRAMES_REQUIRED and vowel_buffer:
                majority = Counter(vowel_buffer).most_common(1)[0][0]
                confirmed_text.append(majority)
                vowel_buffer.clear()
                silence_counter = 0

        history_text = "".join(confirmed_text[-20:])
        buffer_text = "".join(vowel_buffer[-10:])

        cv2.putText(
            frame_overlay,
            f"History: {history_text}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_overlay,
            f"Buffer: {buffer_text}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_NAME, frame_overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            print("\nRe-running calibration...")
            recent_vowels.clear()
            timestamp_ms = run_calibration(cap, landmarker, start_timestamp_ms=timestamp_ms)
            print("Resuming detection.")
        if key == ord("r"):
            confirmed_text.clear()
            vowel_buffer.clear()
            silence_counter = 0
            print("History cleared.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
