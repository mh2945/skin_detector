"""MediaPipe 어댑터 + 정규 프레임 워프 + manual ROI 폴백.

core/ 밖이다. mediapipe · cv2 를 여기서만 쓴다.

**원본 -> 정규 프레임 변환은 이 파일 한 곳에서 한 번만 일어난다** (계약 3번).
core/ 안의 모든 좌표는 정규 프레임 기준이고, 원본 좌표는 render/report 만 쓴다.

계획 §2.2 의 순서를 그대로 구현한다:
    원본 uint8 -> 1536px 축소 -> 랜드마크 -> 원본 좌표로 역스케일
    -> similarity 행렬 -> warpAffine(원본 uint8 -> 768x768)
**행렬은 싸게, 픽셀은 원본에서.**
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .core.mask import (
    FACE_OVAL,
    IRIS_L_CENTER,
    IRIS_R_CENTER,
    N_LANDMARKS,
    NOSE_TIP,
    polygon_mask,
)
from .core.types import CANONICAL_IPD, CANONICAL_SIZE

# 정규 프레임에서 눈높이를 어디에 둘지. 위로 이마(+0.6 IPD), 아래로 턱이 들어가야 한다.
CANONICAL_EYE_Y = 292.0
CANONICAL_CENTER_X = CANONICAL_SIZE / 2.0

_LANDMARKER = None
_LANDMARKER_PATH: Optional[str] = None


@dataclass
class FaceObservation:
    """정규 프레임과 그것을 만든 근거 일체."""

    canonical_rgb: np.ndarray        # uint8 (768, 768, 3)
    frame_valid: np.ndarray          # bool — 원본에서 실제로 온 화소인가
    landmarks: np.ndarray            # float32 (478, 2) 정규 프레임 좌표
    face_poly: np.ndarray            # bool 얼굴 폴리곤
    affine: np.ndarray               # (2, 3) 원본 -> 정규
    native_ipd_px: float             # 원본 해상도에서의 IPD
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    confidence: float
    n_faces: int


# ── MediaPipe ────────────────────────────────────────────────────────

def _get_landmarker(model_path: Path):
    """FaceLandmarker 를 지연 생성하고 재사용한다. 생성이 비싸다."""
    global _LANDMARKER, _LANDMARKER_PATH
    key = str(Path(model_path).resolve())
    if _LANDMARKER is not None and _LANDMARKER_PATH == key:
        return _LANDMARKER

    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=key),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=2,                                   # 2명 이상을 '검출'해야 거부할 수 있다
        output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.3,
    )
    _LANDMARKER = mp_vision.FaceLandmarker.create_from_options(opts)
    _LANDMARKER_PATH = key
    return _LANDMARKER


def _euler_from_matrix(mat: np.ndarray) -> Tuple[float, float, float]:
    """4x4 facial transformation matrix -> (yaw, pitch, roll) 도 단위."""
    r = np.asarray(mat, dtype=np.float64)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(-r[2, 0], sy)
        yaw = math.atan2(r[1, 0], r[0, 0])
        roll = math.atan2(r[2, 1], r[2, 2])
    else:
        pitch = math.atan2(-r[2, 0], sy)
        yaw = 0.0
        roll = math.atan2(-r[1, 2], r[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _pose_from_landmarks(lm: np.ndarray) -> Tuple[float, float, float]:
    """변환 행렬이 없을 때의 기하 폴백.

    yaw 는 코끝이 양안 중점에서 얼마나 치우쳤는지로, pitch 는 코끝이 눈높이에서
    얼마나 내려왔는지로 근사한다. 정밀하지 않지만 게이트용으로는 충분하다.
    """
    r_eye = lm[IRIS_R_CENTER, :2]
    l_eye = lm[IRIS_L_CENTER, :2]
    nose = lm[NOSE_TIP, :2]
    v = l_eye - r_eye
    ipd = float(np.linalg.norm(v)) or 1.0
    roll = math.degrees(math.atan2(float(v[1]), float(v[0])))
    mid = 0.5 * (r_eye + l_eye)
    yaw = math.degrees(math.atan2(float(nose[0] - mid[0]), ipd)) * 2.0
    drop = float(nose[1] - mid[1]) / ipd
    pitch = math.degrees(math.atan2(drop - 0.62, 1.0)) * 2.0
    return yaw, pitch, roll


def similarity_to_canonical(right_eye: np.ndarray, left_eye: np.ndarray) -> np.ndarray:
    """두 홍채 중심 -> 정규 프레임 similarity 행렬 (2x3).

    피사체의 오른쪽 눈이 이미지 왼쪽(작은 x)으로 간다 — core/mask.py 의 좌우 규약과
    같은 사실이다.
    """
    rx, ry = float(right_eye[0]), float(right_eye[1])
    lx, ly = float(left_eye[0]), float(left_eye[1])
    vx, vy = lx - rx, ly - ry
    dist = math.hypot(vx, vy)
    if dist < 1e-6:
        raise ValueError("두 홍채 중심이 겹친다 — 랜드마크가 유효하지 않다")

    scale = CANONICAL_IPD / dist
    theta = -math.atan2(vy, vx)
    a = scale * math.cos(theta)
    b = scale * math.sin(theta)

    dst_rx = CANONICAL_CENTER_X - CANONICAL_IPD / 2.0
    dst_ry = CANONICAL_EYE_Y
    tx = dst_rx - (a * rx - b * ry)
    ty = dst_ry - (b * rx + a * ry)
    return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)


def apply_affine(points: np.ndarray, m: np.ndarray) -> np.ndarray:
    """(N,2) 점들에 2x3 아핀을 적용."""
    p = np.asarray(points, dtype=np.float64)
    return (p @ m[:, :2].T) + m[:, 2][None, :]


def invert_affine(m: np.ndarray) -> np.ndarray:
    """정규 -> 원본 역변환. 병변 좌표를 원본으로 되돌릴 때 쓴다."""
    import cv2
    return cv2.invertAffineTransform(np.asarray(m, dtype=np.float64))


def detect(rgb_original: np.ndarray, model_path: Path,
           detect_long_edge: int = 1536) -> Optional[FaceObservation]:
    """원본 uint8 RGB -> FaceObservation. 얼굴이 없으면 None.

    랜드마크는 축소본에서 뽑고 **워프는 원본 화소에서 직접** 뜬다.
    워프까지 축소본에서 하면 병변 디테일이 실제로 손실된다.
    """
    import cv2
    import mediapipe as mp

    h, w = rgb_original.shape[:2]
    scale = float(detect_long_edge) / float(max(h, w))
    if scale < 1.0:
        small = cv2.resize(rgb_original,
                           (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    else:
        small, scale = rgb_original, 1.0

    landmarker = _get_landmarker(model_path)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(small))
    result = landmarker.detect(mp_image)

    n_faces = len(result.face_landmarks) if result.face_landmarks else 0
    if n_faces == 0:
        return None

    sh, sw = small.shape[:2]
    pts = result.face_landmarks[0]
    lm_small = np.array([[p.x * sw, p.y * sh] for p in pts], dtype=np.float64)
    if lm_small.shape[0] < N_LANDMARKS:
        # 홍채 랜드마크(468~477)가 없으면 이 파이프라인은 성립하지 않는다.
        return None

    # 원본 좌표로 역스케일
    lm_orig = lm_small / scale

    native_ipd = float(np.linalg.norm(
        lm_orig[IRIS_L_CENTER] - lm_orig[IRIS_R_CENTER]))

    affine = similarity_to_canonical(lm_orig[IRIS_R_CENTER], lm_orig[IRIS_L_CENTER])

    size = (CANONICAL_SIZE, CANONICAL_SIZE)
    canonical = cv2.warpAffine(
        rgb_original, affine, size,
        flags=cv2.INTER_AREA if native_ipd > CANONICAL_IPD else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )

    # BORDER_REPLICATE 는 가짜 피부를 만든다 — 금지 (계획 §2.5).
    # 프레임 밖은 '없는 화소'로 표시하고 유효성 마스크에서 제외한다.
    ones = np.full((h, w), 255, dtype=np.uint8)
    valid = cv2.warpAffine(ones, affine, size, flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127

    lm_canon = apply_affine(lm_orig, affine).astype(np.float32)

    if result.facial_transformation_matrixes:
        yaw, pitch, roll = _euler_from_matrix(result.facial_transformation_matrixes[0])
    else:
        yaw, pitch, roll = _pose_from_landmarks(lm_orig)

    face_poly = polygon_mask(lm_canon[list(FACE_OVAL), :2],
                             CANONICAL_SIZE, CANONICAL_SIZE)

    return FaceObservation(
        canonical_rgb=canonical,
        frame_valid=valid,
        landmarks=lm_canon,
        face_poly=face_poly,
        affine=affine,
        native_ipd_px=native_ipd,
        yaw_deg=float(yaw),
        pitch_deg=float(pitch),
        roll_deg=float(roll),
        confidence=1.0,          # Tasks API 는 얼굴별 점수를 노출하지 않는다
        n_faces=int(n_faces),
    )


# ── manual ROI 폴백 (계획 §3) ────────────────────────────────────────

def observation_from_iris_points(
    rgb_original: np.ndarray,
    right_eye_xy: Sequence[float],
    left_eye_xy: Sequence[float],
) -> FaceObservation:
    """랜드마커 없이 홍채 두 점만으로 정규 프레임을 만든다.

    Phase 1(자기참조 정규화가 조명을 견디는가)은 **자동 랜드마크가 전혀 필요 없다.**
    랜드마킹이 막히면 이 경로로 우회해 색 과학 검증을 먼저 끝낸다.

    주의 — 이 경로로 ICC 를 재면 **손으로 찍은 점의 위치 변동이 조명 변동과 뒤섞인다.**
    같은 이미지에 3회 다시 찍은 'ROI 재현성 ICC' 를 먼저 재서 천장으로 삼아야
    해석이 성립한다 (계획 §3 폴백 주석). 그 절차 없이 나온 낮은 ICC 는
    정규화 실패의 증거가 아니다.
    """
    import cv2

    h, w = rgb_original.shape[:2]
    r = np.asarray(right_eye_xy, dtype=np.float64)
    l = np.asarray(left_eye_xy, dtype=np.float64)
    affine = similarity_to_canonical(r, l)
    size = (CANONICAL_SIZE, CANONICAL_SIZE)

    canonical = cv2.warpAffine(rgb_original, affine, size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    ones = np.full((h, w), 255, dtype=np.uint8)
    valid = cv2.warpAffine(ones, affine, size, flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 127

    lm = np.zeros((N_LANDMARKS, 2), dtype=np.float32)
    lm[IRIS_R_CENTER] = (CANONICAL_CENTER_X - CANONICAL_IPD / 2.0, CANONICAL_EYE_Y)
    lm[IRIS_L_CENTER] = (CANONICAL_CENTER_X + CANONICAL_IPD / 2.0, CANONICAL_EYE_Y)

    return FaceObservation(
        canonical_rgb=canonical,
        frame_valid=valid,
        landmarks=lm,
        face_poly=np.zeros((CANONICAL_SIZE, CANONICAL_SIZE), dtype=bool),
        affine=affine,
        native_ipd_px=float(np.linalg.norm(l - r)),
        yaw_deg=0.0, pitch_deg=0.0,
        roll_deg=math.degrees(math.atan2(float(l[1] - r[1]), float(l[0] - r[0]))),
        confidence=1.0,
        n_faces=1,
    )
