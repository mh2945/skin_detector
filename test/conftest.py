"""테스트 픽스처: 합성 얼굴.

실제 셀피 없이 마스킹·측정·점수화 전 구간을 검증하기 위한 것이다.
랜드마크를 정규 프레임 규격(768x768, IPD=320)에 맞게 기하학적으로 생성하므로
MediaPipe 없이도 core/ 전체가 돌아간다.

합성 홍반은 **로그 밀도 공간에서 헤모글로빈 방향으로** 주입한다.
붉은 원 알파 블렌딩은 비물리적이라 검출기를 실제보다 좋아 보이게 만든다 (계획 §6.2).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skin_detector.core import color, mask as mask_mod  # noqa: E402
from skin_detector.core.types import CANONICAL_IPD, CANONICAL_SIZE  # noqa: E402

EYE_Y = 292.0
CX = CANONICAL_SIZE / 2.0
IPD = CANONICAL_IPD


def _ellipse(n: int, cx: float, cy: float, a: float, b: float,
             start_deg: float) -> np.ndarray:
    ang = np.radians(start_deg + np.arange(n) * (360.0 / n))
    return np.stack([cx + a * np.cos(ang), cy + b * np.sin(ang)], axis=1)


def synthetic_landmarks() -> np.ndarray:
    """478점 합성 랜드마크. core/mask.py 가 실제로 읽는 인덱스만 의미를 갖는다."""
    lm = np.zeros((mask_mod.N_LANDMARKS, 2), dtype=np.float64)
    lm[:] = (CX, EYE_Y + 0.4 * IPD)          # 안 쓰는 인덱스도 NaN 이 되지 않게

    # 얼굴 외곽: FACE_OVAL 은 10(정수리)에서 시작해 152(턱)가 18번째다.
    oval = _ellipse(len(mask_mod.FACE_OVAL), CX, EYE_Y + 0.20 * IPD,
                    0.72 * IPD, 0.95 * IPD, -90.0)
    lm[list(mask_mod.FACE_OVAL)] = oval

    r_eye = np.array([CX - IPD / 2.0, EYE_Y])
    l_eye = np.array([CX + IPD / 2.0, EYE_Y])
    lm[mask_mod.IRIS_R_CENTER] = r_eye
    lm[mask_mod.IRIS_L_CENTER] = l_eye

    # 홍채 지름 11.7mm. IPD 320px <-> 63mm 이므로 반지름 = 11.7/2 * (320/63)
    iris_r = (11.7 / 2.0) * (IPD / 63.0)
    for center, ring in ((r_eye, mask_mod.IRIS_R_RING), (l_eye, mask_mod.IRIS_L_RING)):
        lm[list(ring)] = _ellipse(len(ring), center[0], center[1],
                                  iris_r, iris_r, 0.0)

    for center, idx in ((r_eye, mask_mod.EYE_R), (l_eye, mask_mod.EYE_L)):
        lm[list(idx)] = _ellipse(len(idx), center[0], center[1],
                                 0.11 * IPD, 0.048 * IPD, 180.0)

    for center, idx in ((r_eye, mask_mod.BROW_R), (l_eye, mask_mod.BROW_L)):
        half = len(idx) // 2
        xs = np.linspace(center[0] - 0.13 * IPD, center[0] + 0.13 * IPD, half)
        lower = np.stack([xs, np.full(half, center[1] - 0.16 * IPD)], axis=1)
        upper = np.stack([xs[::-1], np.full(half, center[1] - 0.24 * IPD)], axis=1)
        lm[list(idx)] = np.concatenate([lower, upper], axis=0)

    mouth_y = EYE_Y + 0.72 * IPD
    lm[list(mask_mod.LIPS_OUTER)] = _ellipse(
        len(mask_mod.LIPS_OUTER), CX, mouth_y, 0.18 * IPD, 0.07 * IPD, 180.0)

    lm[mask_mod.NOSE_TIP] = (CX, EYE_Y + 0.55 * IPD)
    lm[mask_mod.NOSE_BASE] = (CX, EYE_Y + 0.62 * IPD)
    lm[mask_mod.NOSTRIL_R] = (CX - 0.055 * IPD, EYE_Y + 0.625 * IPD)
    lm[mask_mod.NOSTRIL_L] = (CX + 0.055 * IPD, EYE_Y + 0.625 * IPD)
    return lm


def hemoglobin_direction() -> np.ndarray:
    from skin_detector.imageio import load_extinction_csv
    wl, hb, _ = load_extinction_csv(ROOT / "data" / "spectra"
                                    / "hb_melanin_extinction.csv")
    return color.absorber_rgb_direction(wl, hb, depth_weighted=True)


def melanin_direction() -> np.ndarray:
    from skin_detector.imageio import load_extinction_csv
    wl, _, ml = load_extinction_csv(ROOT / "data" / "spectra"
                                    / "hb_melanin_extinction.csv")
    return color.absorber_rgb_direction(wl, ml, depth_weighted=False)


def inject(linear: np.ndarray, amount: np.ndarray, direction: np.ndarray
           ) -> np.ndarray:
    """로그 밀도 공간에서 색소 방향으로 amount 만큼 더한다.

    D' = D + amount * v  ->  R' = R * 10^(-amount * v)
    """
    factor = np.power(10.0, -amount[..., None] * direction[None, None, :])
    return np.clip(linear * factor, 1e-4, 1.0).astype(np.float32)


def synthetic_face_linear(
    seed: int = 0,
    base_rgb=(0.44, 0.30, 0.25),
    shading: bool = True,
    noise: float = 0.004,
) -> np.ndarray:
    """합성 피부 선형 반사율. 음영 구배와 노이즈를 포함한다.

    음영은 로그 비율이 **정확히 소거**해야 하는 교란이므로 기본으로 넣는다.
    """
    rng = np.random.default_rng(seed)
    h = w = CANONICAL_SIZE
    lin = np.empty((h, w, 3), dtype=np.float32)
    lin[:] = np.asarray(base_rgb, dtype=np.float32)

    if shading:
        yy = np.linspace(0.75, 1.25, h)[:, None]
        xx = np.linspace(0.85, 1.15, w)[None, :]
        lin *= (yy * xx)[..., None].astype(np.float32)

    lin += rng.normal(0.0, noise, lin.shape).astype(np.float32)
    return np.clip(lin, 1e-3, 1.0)


@pytest.fixture(scope="session")
def landmarks() -> np.ndarray:
    return synthetic_landmarks()


@pytest.fixture(scope="session")
def frame_valid() -> np.ndarray:
    return np.ones((CANONICAL_SIZE, CANONICAL_SIZE), dtype=bool)


@pytest.fixture
def clean_face() -> np.ndarray:
    return synthetic_face_linear()


@pytest.fixture(scope="session")
def v_hemoglobin() -> np.ndarray:
    return hemoglobin_direction()


@pytest.fixture(scope="session")
def v_melanin() -> np.ndarray:
    return melanin_direction()
