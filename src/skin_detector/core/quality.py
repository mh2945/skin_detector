"""품질 게이트 체인. 계획 §4.1.

**먼저 걸리는 게이트가 이긴다.** 뒤 조건은 평가조차 하지 않는다.
sample2 `EyeStateGate` 의 패턴을 그대로 계승한 것이고, 그 순서 자체가 설계다 —
근본 원인(얼굴이 없다)이 파생 증상(흐리다)보다 먼저 보고돼야 사용자에게 줄
안내 문장이 맞아떨어진다 (§5.3).

계약: numpy + stdlib 만. 실패는 명명된 ReasonCode 를 낸다.
"""

from typing import Dict, Optional

import numpy as np

from .config import QualityConfig
from .types import QualityReport, ReasonCode


def luminance(linear: np.ndarray) -> np.ndarray:
    """선형 RGB -> 상대 휘도 Y (Rec.709)."""
    return (0.2126 * linear[..., 0]
            + 0.7152 * linear[..., 1]
            + 0.0722 * linear[..., 2]).astype(np.float32)


def variance_of_laplacian(gray: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """초점 선명도. **반드시 정규 프레임에서 잰다.**

    원본 해상도에서 재면 해상도가 다른 이미지끼리 비교가 안 된다 — 흔한 버그다.
    정규 프레임은 IPD 가 320 으로 고정돼 있어 값이 곧바로 비교 가능하다.
    """
    g = gray.astype(np.float64)
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1] = (
        4.0 * g[1:-1, 1:-1]
        - g[:-2, 1:-1] - g[2:, 1:-1]
        - g[1:-1, :-2] - g[1:-1, 2:]
    )
    inner = np.zeros(g.shape, dtype=bool)
    inner[1:-1, 1:-1] = True
    sel = inner if mask is None else (inner & mask)
    if not sel.any():
        return 0.0
    # 값 범위를 8bit 상당으로 맞춰 임계값이 직관적인 스케일이 되게 한다.
    return float(np.var(lap[sel] * 255.0))


def evaluate(
    linear: np.ndarray,
    face_poly: np.ndarray,
    frame_valid: np.ndarray,
    native_ipd_px: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    landmark_confidence: float,
    n_faces: int,
    cfg: QualityConfig,
) -> QualityReport:
    """게이트 체인을 순서대로 통과시킨다. 먼저 걸린 사유 하나만 돌려준다.

    통과 여부와 무관하게 측정한 값은 전부 `metrics` 에 남긴다 —
    "왜 그렇게 나왔는지"를 결과 화면에 노출하려면 (§5.3) 통과한 값도 필요하다.
    """
    metrics: Dict[str, float] = {
        "native_ipd_px": float(native_ipd_px),
        "yaw_deg": float(yaw_deg),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(roll_deg),
        "landmark_confidence": float(landmark_confidence),
        "n_faces": float(n_faces),
    }

    def fail(reason: ReasonCode) -> QualityReport:
        return QualityReport(passed=False, reason=reason, metrics=metrics)

    # ── 1. 얼굴 자체 ──
    if n_faces <= 0:
        return fail(ReasonCode.NO_FACE)
    if n_faces > 1:
        return fail(ReasonCode.MULTIPLE_FACES)
    if landmark_confidence < cfg.min_landmark_confidence:
        return fail(ReasonCode.LOW_LANDMARK_CONFIDENCE)

    # ── 2. 기하 ──
    if native_ipd_px < cfg.min_native_ipd_px:
        return fail(ReasonCode.LOW_RESOLUTION)
    if (abs(yaw_deg) > cfg.max_yaw_deg
            or abs(pitch_deg) > cfg.max_pitch_deg
            or abs(roll_deg) > cfg.max_roll_deg):
        return fail(ReasonCode.POSE_OUT_OF_RANGE)

    # 9:16 세로 사진은 얼굴이 위쪽에 붙어 크롭이 프레임 밖으로 나가는 일이 흔하다.
    # BORDER_CONSTANT 로 채운 영역은 가짜 피부가 아니라 '없는 화소'다.
    face_area = float(int(face_poly.sum()))
    in_frame = (float(int((face_poly & frame_valid).sum())) / face_area
                if face_area > 0 else 0.0)
    metrics["face_in_frame"] = in_frame
    if in_frame < cfg.min_face_in_frame:
        return fail(ReasonCode.FACE_OUT_OF_FRAME)

    face = face_poly & frame_valid
    y = luminance(linear)

    # ── 3. 광학 ──
    vol = variance_of_laplacian(y, face)
    metrics["blur_vol"] = vol
    if vol < cfg.min_blur_vol:
        return fail(ReasonCode.BLURRY)

    max_ch = np.max(linear, axis=-1)
    clipped = float((max_ch[face] >= cfg.clip_level).mean()) if face.any() else 1.0
    dark = float((max_ch[face] < cfg.dark_level).mean()) if face.any() else 1.0
    metrics["clipped_fraction"] = clipped
    metrics["dark_fraction"] = dark
    if clipped > cfg.max_clipped_fraction:
        return fail(ReasonCode.OVEREXPOSED)
    if dark > cfg.max_dark_fraction:
        return fail(ReasonCode.UNDEREXPOSED)

    # 조명 균일성 — 세로 프레임의 천장 조명 구배를 잡는다.
    yv = y[face]
    if yv.size >= 64:
        p10 = float(np.percentile(yv, 10))
        p90 = float(np.percentile(yv, 90))
        ratio = p90 / max(p10, 1e-6)
    else:
        ratio = float("inf")
    metrics["illum_ratio"] = ratio
    if ratio > cfg.max_illum_ratio:
        return fail(ReasonCode.NON_UNIFORM_ILLUMINATION)

    return QualityReport(passed=True, reason=ReasonCode.OK, metrics=metrics)
