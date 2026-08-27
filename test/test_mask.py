"""폴리곤 · 형태학 · 색도/정반사 게이트 · 부위 분할."""

import numpy as np
import pytest

from conftest import synthetic_face_linear
from skin_detector.core import color, mask as M
from skin_detector.core.config import MaskConfig, QualityConfig
from skin_detector.core.types import ALL_REGIONS, CANONICAL_SIZE, RegionId


# ── 폴리곤 ────────────────────────────────────────────────────────────

def test_polygon_area_exact():
    sq = M.polygon_mask(np.array([[10, 10], [90, 10], [90, 90], [10, 90]]), 100, 100)
    assert int(sq.sum()) == 6400


def test_polygon_handles_concave():
    """오목 다각형에서도 교차수 판정이 맞아야 한다 (얼굴 외곽은 볼록이 아니다)."""
    poly = np.array([[10, 10], [90, 10], [90, 90], [50, 50], [10, 90]])
    m = M.polygon_mask(poly, 100, 100)
    assert m[20, 50] and not m[80, 50]


def test_polygon_winding_independent():
    poly = np.array([[10, 10], [90, 10], [90, 90], [10, 90]])
    assert int(M.polygon_mask(poly, 100, 100).sum()) == \
           int(M.polygon_mask(poly[::-1], 100, 100).sum())


# ── 형태학 ────────────────────────────────────────────────────────────

def test_erode_dilate_are_inverse_in_bulk():
    m = np.zeros((60, 60), dtype=bool)
    m[20:40, 20:40] = True
    assert int(M.binary_erode(m, 3).sum()) < int(m.sum())
    assert int(M.binary_dilate(m, 3).sum()) > int(m.sum())
    assert int(M.binary_dilate(M.binary_erode(m, 2), 2).sum()) == int(m.sum())


def test_erode_does_not_eat_image_border():
    """경계 밖을 True 로 채운다 — 프레임 가장자리 때문에 마스크가 깎이면 안 된다.

    진짜 프레임 밖은 frame_valid 가 따로 처리한다.
    """
    m = np.ones((40, 40), dtype=bool)
    assert int(M.binary_erode(m, 3).sum()) == 1600


def test_zero_iterations_is_identity():
    m = np.zeros((20, 20), dtype=bool)
    m[5:15, 5:15] = True
    assert np.array_equal(M.binary_erode(m, 0), m)
    assert np.array_equal(M.binary_dilate(m, 0), m)


# ── 색도 게이트 ───────────────────────────────────────────────────────

def test_chromaticity_gate_rejects_off_distribution():
    """얼굴 자체에서 학습한 분포 밖(머리카락·배경)을 걸러야 한다."""
    rng = np.random.default_rng(5)
    e = rng.normal(0.10, 0.01, (64, 64))
    m = rng.normal(0.20, 0.01, (64, 64))
    e[:8, :] = 0.60          # 분포에서 크게 벗어난 영역
    m[:8, :] = -0.30

    seed = np.zeros((64, 64), dtype=bool)
    seed[20:60, :] = True
    ok = M.chromaticity_gate(e, m, seed, max_maha=3.0)
    assert ok[30, 30]
    assert not ok[:8, :].any()


def test_chromaticity_gate_degrades_with_tiny_seed():
    e = np.zeros((16, 16)); m = np.zeros((16, 16))
    seed = np.zeros((16, 16), dtype=bool)
    seed[0, 0] = True
    assert M.chromaticity_gate(e, m, seed, 3.0).shape == (16, 16)


# ── 정반사 ────────────────────────────────────────────────────────────

def test_specular_gate_flags_elevated_min_channel():
    """정반사는 분광 중립이라 min(R,G,B) 를 함께 끌어올린다 (계획 §4.3-b)."""
    rng = np.random.default_rng(2)
    min_ch = rng.normal(0.10, 0.005, (64, 64))
    min_ch[10:14, 10:14] = 0.35
    face = np.ones((64, 64), dtype=bool)
    ok = M.specular_gate(min_ch, face, k=3.0)
    assert not ok[10:14, 10:14].any()
    assert ok.mean() > 0.98


# ── 홍채 ──────────────────────────────────────────────────────────────

def test_iris_diameter_matches_canonical_scale(landmarks):
    """정규 IPD 320 <-> 63mm 이면 홍채 11.7mm 는 약 59.4px 여야 한다."""
    d = M.iris_diameter_px(landmarks)
    assert d == pytest.approx(11.7 * (320.0 / 63.0), rel=0.02)


# ── 부위 분할 통합 ────────────────────────────────────────────────────

@pytest.fixture
def built(landmarks, frame_valid):
    lin = synthetic_face_linear()
    e, m = color.chromophore_axes(color.optical_density(lin))
    q = QualityConfig()
    return M.build_masks(
        landmarks=landmarks, linear=lin, e=e, m=m, frame_valid=frame_valid,
        cfg=MaskConfig(), clip_level=q.clip_level, dark_level=q.dark_level,
    )


def test_all_five_regions_are_populated(built):
    for r in ALL_REGIONS:
        assert int(built.regions[r].sum()) > 500, "{} 가 비었다".format(r.value)


def test_regions_do_not_overlap(built):
    total = sum(int(built.regions[r].sum()) for r in ALL_REGIONS)
    union = np.zeros_like(built.skin)
    for r in ALL_REGIONS:
        union |= built.regions[r]
    assert total == int(union.sum()), "부위가 겹치면 화소가 두 번 세어진다"


def test_left_right_convention(built, landmarks):
    """**피사체의 오른쪽 볼은 이미지 왼쪽(작은 x)** 이다.

    이 규약은 core/mask.py 한 곳에서만 정한다. 뒤집히면 좌우 비대칭 진단(§6.4)이
    통째로 거짓말이 된다.
    """
    ys, xs_r = np.nonzero(built.regions[RegionId.CHEEK_R])
    ys, xs_l = np.nonzero(built.regions[RegionId.CHEEK_L])
    assert xs_r.mean() < CANONICAL_SIZE / 2 < xs_l.mean()


def test_eyes_and_lips_are_excluded(built, landmarks):
    for idx in (M.EYE_L, M.EYE_R, M.LIPS_OUTER):
        pts = landmarks[list(idx)].astype(int)
        cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        assert not built.skin[cy, cx], "제외 영역이 피부로 남았다"


def test_coverage_denominator_is_pre_gate_area(built):
    """coverage 분모는 게이트 **전** 기하 면적이어야 한다.

    분모를 게이트 후로 잡으면 다 걸러진 부위도 1.0 이 되어
    '측정 못 함'이 '정상'으로 둔갑한다 (계약 4번).
    """
    for r in ALL_REGIONS:
        cov = built.coverage(r)
        assert 0.0 <= cov <= 1.0
        assert built.geo_area[r] >= int(built.regions[r].sum())


def test_out_of_frame_pixels_never_become_skin(landmarks):
    """9:16 크롭이 프레임 밖으로 나간 영역은 가짜 피부가 되면 안 된다.

    BORDER_REPLICATE 를 쓰면 정확히 이 사고가 난다 — 그래서 금지다 (계획 §2.5).
    """
    lin = synthetic_face_linear()
    e, m = color.chromophore_axes(color.optical_density(lin))
    valid = np.ones((CANONICAL_SIZE, CANONICAL_SIZE), dtype=bool)
    valid[:300, :] = False                      # 상단이 프레임 밖이라고 가정
    q = QualityConfig()
    ms = M.build_masks(landmarks=landmarks, linear=lin, e=e, m=m,
                       frame_valid=valid, cfg=MaskConfig(),
                       clip_level=q.clip_level, dark_level=q.dark_level)
    assert not ms.skin[:300, :].any()
    assert ms.coverage(RegionId.FOREHEAD) < 0.5, "이마는 측정 불가로 떨어져야 한다"
