"""색 과학의 수학적 성질. 이미지가 전혀 필요 없다 (계획 §6.6).

여기 있는 불변성이 **자기참조 정규화가 성립하는 유일한 근거**다.
하나라도 깨지면 조명 하의 안정성 실험(§6.1)을 할 이유가 없다.
"""

import numpy as np
import pytest

from conftest import hemoglobin_direction, melanin_direction
from skin_detector.core import color


@pytest.fixture
def patch():
    # 상한을 0.30 으로 둔 것은 의도적이다. k=3.0 을 곱해도 포화하지 않아야
    # '음영 불변성'과 '포화로 인한 파괴'를 분리해서 검사할 수 있다.
    rng = np.random.default_rng(7)
    return rng.uniform(0.03, 0.30, size=(48, 48, 3)).astype(np.float32)


def axes(linear):
    return color.chromophore_axes(color.optical_density(linear))


# ── 불변성 ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [0.25, 0.5, 1.7, 3.0])
def test_brightness_invariance(patch, k):
    """음영·노출·거리·플래시 감쇠 = 픽셀별 스칼라 배. (e,m) 이 **정확히** 불변이어야 한다.

    이것이 log-ratio 를 고른 이유 전체다. 깨지면 YCbCr Cr 과 다를 게 없다.
    """
    e0, m0 = axes(patch)
    e1, m1 = axes(np.clip(patch * k, 1e-4, 1.0))
    assert np.abs(e1 - e0).max() < 1e-5
    assert np.abs(m1 - m0).max() < 1e-5


def test_saturation_destroys_invariance(patch):
    """포화하면 불변성이 **깨진다.** 이건 버그가 아니라 물리다.

    센서가 1.0 에서 잘리면 채널 비율 정보가 사라진다. 그래서 품질 게이트가
    `max(R,G,B) >= clip_level` 화소를 버린다 (계획 §4.3-a). 이 테스트는 그 게이트가
    왜 있어야 하는지를 고정한다 — 게이트를 지우면 여기가 알려준다.
    """
    e0, _ = axes(patch)
    saturated = np.clip(patch * 6.0, 1e-4, 1.0)
    e1, _ = axes(saturated)
    assert np.abs(e1 - e0).max() > 0.05, "포화가 비율을 파괴하지 않았다면 픽스처가 잘못됐다"


def test_per_pixel_shading_invariance(patch):
    """전역 스칼라가 아니라 **픽셀마다 다른** 음영에도 불변이어야 한다."""
    rng = np.random.default_rng(3)
    shade = rng.uniform(0.4, 1.4, patch.shape[:2]).astype(np.float32)[..., None]
    e0, m0 = axes(patch)
    e1, m1 = axes(np.clip(patch * shade, 1e-4, 1.0))
    assert np.abs(e1 - e0).max() < 1e-5
    assert np.abs(m1 - m0).max() < 1e-5


def test_illuminant_color_is_pure_constant_offset(patch):
    """화이트밸런스(채널별 게인)는 **상수 오프셋으로만** 남아야 한다.

    상수라면 자기참조 기준선 차감이 정확히 없앤다. 상수가 아니면 못 없앤다.
    """
    gain = np.array([1.18, 0.91, 1.33], dtype=np.float32)
    e0, m0 = axes(patch)
    e1, m1 = axes(np.clip(patch * gain, 1e-4, 1.0))
    assert (e1 - e0).std() < 1e-5, "조명색이 상수 오프셋이 아니다"
    assert (m1 - m0).std() < 1e-5
    assert abs((e1 - e0).mean()) > 1e-3, "오프셋이 0 이면 테스트가 무의미하다"


def test_baseline_subtraction_removes_illuminant(patch):
    """기준선을 뺀 뒤에는 조명색 차이가 남지 않아야 한다 — 파이프라인의 핵심 주장."""
    gain = np.array([1.18, 0.91, 1.33], dtype=np.float32)
    e0, m0 = axes(patch)
    e1, m1 = axes(np.clip(patch * gain, 1e-4, 1.0))
    d0 = e0 - np.median(e0)
    d1 = e1 - np.median(e1)
    assert np.abs(d1 - d0).max() < 1e-5


# ── 감마 ──────────────────────────────────────────────────────────────

def test_srgb_linearization_piecewise():
    """검정 근처 piecewise 구간을 지켜야 한다 (계획 §1.3-1)."""
    enc = np.array([[[0.0, 0.04045, 1.0]]], dtype=np.float32)
    lin = color.srgb_to_linear(enc)
    assert lin[0, 0, 0] == pytest.approx(0.0, abs=1e-7)
    assert lin[0, 0, 1] == pytest.approx(0.04045 / 12.92, rel=1e-5)
    assert lin[0, 0, 2] == pytest.approx(1.0, rel=1e-5)


def test_srgb_uint8_and_float_agree():
    u8 = np.arange(256, dtype=np.uint8).reshape(-1, 1, 1).repeat(3, axis=2)
    a = color.srgb_to_linear(u8)
    b = color.srgb_to_linear(u8.astype(np.float32) / 255.0)
    assert np.abs(a - b).max() < 1e-6


# ── CIELAB 기지값 ─────────────────────────────────────────────────────

def test_lab_known_values():
    """문헌 기지값. Delta-a* 교차검증이 문헌과 비교 가능하려면 여기가 맞아야 한다."""
    white = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
    lab = color.linear_to_lab(white)[0, 0]
    assert lab[0] == pytest.approx(100.0, abs=0.05)
    assert abs(lab[1]) < 0.05 and abs(lab[2]) < 0.05

    black = np.zeros((1, 1, 3), dtype=np.float32)
    assert color.linear_to_lab(black)[0, 0, 0] == pytest.approx(0.0, abs=0.05)

    # sRGB 순수 빨강 -> L*=53.24, a*=80.09, b*=67.20
    red_lin = color.srgb_to_linear(np.array([[[255, 0, 0]]], dtype=np.uint8))
    lab_r = color.linear_to_lab(red_lin)[0, 0]
    assert lab_r[0] == pytest.approx(53.24, abs=0.3)
    assert lab_r[1] == pytest.approx(80.09, abs=0.5)
    assert lab_r[2] == pytest.approx(67.20, abs=0.5)


def test_mid_gray_lightness():
    mid = np.full((1, 1, 3), 0.5, dtype=np.float32)
    assert color.linear_to_lab(mid)[0, 0, 0] == pytest.approx(76.07, abs=0.05)


# ── robust 회귀 ───────────────────────────────────────────────────────

def test_robust_fit_ignores_outliers():
    """병변·정반사 같은 소수 이상치가 멜라닌 기울기를 끌고 가면 안 된다."""
    rng = np.random.default_rng(11)
    x = rng.normal(0, 1, 4000)
    y = 0.5 * x + 0.2 + rng.normal(0, 0.01, 4000)
    y[:600] += 4.0                       # 15% 오염
    alpha, beta = color.robust_line_fit(x, y)
    assert beta == pytest.approx(0.5, abs=0.02)
    assert alpha == pytest.approx(0.2, abs=0.02)


def test_robust_fit_degrades_gracefully_on_tiny_sample():
    alpha, beta = color.robust_line_fit(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert beta == 0.0 and np.isfinite(alpha)


# ── 색소 방향 벡터 — 설계 전체의 물리적 근거 ─────────────────────────

def test_hemoglobin_peaks_in_green():
    """oxy-Hb 는 542/577nm Q-band 때문에 **녹색을 가장 강하게** 먹어야 한다."""
    v = hemoglobin_direction()
    assert v[1] > v[0] and v[1] > v[2], "G 가 최대가 아니면 홍반 축이 성립하지 않는다"


def test_melanin_is_monotonic_toward_blue():
    """멜라닌은 피크 없이 파랑으로 갈수록 단조 증가해야 한다."""
    v = melanin_direction()
    assert v[0] < v[1] < v[2], "단조 증가가 아니면 멜라닌 축이 성립하지 않는다"


def test_two_chromophores_separate_in_em_plane():
    """둘 다 e 를 올리지만 m 의 **부호가 갈려야** 2차원 분리가 성립한다.

    이 성질이 없으면 `Delta-m >> Delta-e` 모반 기각(§4.5)의 근거가 사라지고,
    a* 하나만 보는 것과 다를 게 없어진다.
    """
    vh = hemoglobin_direction()
    vm = melanin_direction()
    de_h, dm_h = vh[1] - vh[0], vh[2] - vh[1]
    de_m, dm_m = vm[1] - vm[0], vm[2] - vm[1]

    assert de_h > 0 and de_m > 0, "둘 다 홍반 축을 올려야 한다 (교란의 정체)"
    assert dm_h < 0 < dm_m, "멜라닌 축에서 부호가 갈려야 한다"

    # 각 분리가 충분한가 (라디안)
    ang_h = np.arctan2(dm_h, de_h)
    ang_m = np.arctan2(dm_m, de_m)
    assert abs(ang_h - ang_m) > np.radians(60.0)


def test_dermal_pathlength_suppresses_blue():
    """진피 광로장 가중이 없으면 Soret 밴드 때문에 v_h 가 뒤집힌다."""
    from skin_detector.imageio import load_extinction_csv
    from conftest import ROOT
    wl, hb, _ = load_extinction_csv(ROOT / "data/spectra/hb_melanin_extinction.csv")

    weighted = color.absorber_rgb_direction(wl, hb, depth_weighted=True)
    raw = color.absorber_rgb_direction(wl, hb, depth_weighted=False)
    assert weighted[1] > weighted[2], "가중 후에는 G 가 B 보다 커야 한다"
    assert raw[2] >= raw[1], "가중 전에는 415nm Soret 때문에 B 가 이긴다 (그래서 가중이 필요하다)"
