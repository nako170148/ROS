from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import math

# 特徴量抽出に使用する Face Mesh のランドマーク番号
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# 口の面積近似に用いる外唇点の一部（時計回り）
OUTER_LIP_LOOP = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 306,
]


@dataclass(frozen=True)
class MouthFeatures:
# 正規化幾何特徴量

    vertical: float
    horizontal: float
    ratio: float
    area: float


DEFAULT_PROFILES: Dict[str, MouthFeatures] = {
    "a": MouthFeatures(vertical=0.45, horizontal=0.48, ratio=0.95, area=0.24),
    "i": MouthFeatures(vertical=0.14, horizontal=0.55, ratio=0.26, area=0.12),
    "u": MouthFeatures(vertical=0.04, horizontal=0.40, ratio=0.10, area=0.10),
    "e": MouthFeatures(vertical=0.16, horizontal=0.57, ratio=0.28, area=0.16),
    "o": MouthFeatures(vertical=0.08, horizontal=0.34, ratio=0.24, area=0.14),
}

CALIBRATED_PROFILES: Dict[str, MouthFeatures] = dict(DEFAULT_PROFILES)


def reset_calibrated_profiles() -> None:
    CALIBRATED_PROFILES.clear()
    CALIBRATED_PROFILES.update(DEFAULT_PROFILES)


def set_calibrated_profiles(profiles: Mapping[str, MouthFeatures]) -> None:
    CALIBRATED_PROFILES.clear()
    for vowel, profile in profiles.items():
        CALIBRATED_PROFILES[vowel] = MouthFeatures(
            vertical=float(profile.vertical),
            horizontal=float(profile.horizontal),
            ratio=float(profile.ratio),
            area=float(profile.area),
        )


def get_calibrated_profiles() -> Dict[str, MouthFeatures]:
    return dict(CALIBRATED_PROFILES)


def _euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _polygon_area(points: Sequence[Tuple[float, float]]) -> float:
    # シューレース公式により、多角形の符号なし面積を返す。
    if len(points) < 3:
        return 0.0
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _landmark_xy(landmarks: Sequence[object], index: int) -> Tuple[float, float]:
    landmark = landmarks[index]
    return float(landmark.x), float(landmark.y)


def extract_mouth_features(landmarks: Sequence[object]) -> MouthFeatures:
    """MediaPipe の顔ランドマークから、正規化した口形状特徴量を計算する。

    引数:
        landmarks: MediaPipe NormalizedLandmark のシーケンス（長さ 468）。

    戻り値:
        目尻間距離で正規化した距離を持つ MouthFeatures。

    例外:
        ValueError: 目尻間距離が 0（退化した入力）で正規化できない場合。
    """

    upper = _landmark_xy(landmarks, UPPER_LIP_CENTER)
    lower = _landmark_xy(landmarks, LOWER_LIP_CENTER)
    left = _landmark_xy(landmarks, LEFT_MOUTH_CORNER)
    right = _landmark_xy(landmarks, RIGHT_MOUTH_CORNER)
    eye_left = _landmark_xy(landmarks, LEFT_EYE_OUTER)
    eye_right = _landmark_xy(landmarks, RIGHT_EYE_OUTER)

    eye_distance = _euclidean_distance(eye_left, eye_right)
    if eye_distance <= 0.0:
        raise ValueError("Eye distance is zero; cannot normalize features.")

    vertical = _euclidean_distance(upper, lower)
    horizontal = _euclidean_distance(left, right)

    outer_loop_points = [_landmark_xy(landmarks, idx) for idx in OUTER_LIP_LOOP]
    area = _polygon_area(outer_loop_points)

    vertical_n = vertical / eye_distance
    horizontal_n = horizontal / eye_distance
    ratio = vertical_n / horizontal_n if horizontal_n > 0 else 0.0
    area_n = area / (eye_distance ** 2)

    return MouthFeatures(
        vertical=vertical_n,
        horizontal=horizontal_n,
        ratio=ratio,
        area=area_n,
    )



def compute_profile(samples: Sequence[MouthFeatures]) -> MouthFeatures:
    if not samples:
        raise ValueError("No samples provided for centroid computation.")
    vertical = sum(sample.vertical for sample in samples) / len(samples)
    horizontal = sum(sample.horizontal for sample in samples) / len(samples)
    ratio = sum(sample.ratio for sample in samples) / len(samples)
    area = sum(sample.area for sample in samples) / len(samples)
    return MouthFeatures(vertical=vertical, horizontal=horizontal, ratio=ratio, area=area)


def classify_vowel_by_profiles(
    features: MouthFeatures,
    profiles: Mapping[str, MouthFeatures] | None = None,
) -> Tuple[str, Dict[str, float]]:
    profile_map = profiles or CALIBRATED_PROFILES
    distances: Dict[str, float] = {}
    for vowel, profile in profile_map.items():
        diff_vertical = features.vertical - profile.vertical
        diff_horizontal = features.horizontal - profile.horizontal
        diff_ratio = features.ratio - profile.ratio
        diff_area = features.area - profile.area
        distances[vowel] = math.sqrt(
            diff_vertical ** 2
            + diff_horizontal ** 2
            + diff_ratio ** 2
            + diff_area ** 2
        )

    best_vowel = min(distances, key=distances.get)
    return best_vowel, distances


def classify_vowel_from_landmarks(
    landmarks: Sequence[object],
    profiles: Mapping[str, MouthFeatures] | None = None,
) -> Tuple[str, MouthFeatures, Dict[str, float]]:

    features = extract_mouth_features(landmarks)
    vowel, distances = classify_vowel_by_profiles(features, profiles=profiles)
    return vowel, features, distances
