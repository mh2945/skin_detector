"""품질 게이트 체인. **먼저 걸리는 게이트가 이긴다.**

순서 자체가 설계다 (계획 §4.1). 근본 원인(얼굴이 없다)이 파생 증상(흐리다)보다
먼저 보고돼야 §5.3 의 안내 문장이 맞아떨어진다 — "고개를 돌리지 마세요"라고
안내해 놓고 사실은 얼굴이 두 명이었다면 사용자는 영영 통과하지 못한다.
"""

import numpy as np
import pytest

from skin_detector.core import quality
from skin_detector.core.config import QualityConfig
from skin_detector.core.types import FATAL_REASONS, ReasonCode

N = 256


def base_kwargs(**over):
    """전부 통과하는 기본 입력. 테스트마다 한 가지씩만 망가뜨린다."""
    face = np.zeros((N, N), dtype=bool)
    face[40:210, 40:210] = True

    rng = np.random.default_rng(0)
    linear = np.full((N, N, 3), 0.28, dtype=np.float32)
    # 균일한 회색은 Laplacian 분산이 0 이라 blur 게이트에 걸린다.
    # 실제 피부의 미세 질감에 해당하는 고주파를 넣어 준다.
    linear += rng.normal(0.0, 0.02, linear.shape).astype(np.float32)
    linear = np.clip(linear, 0.02, 0.9)

    kw = dict(
        linear=linear,
        face_poly=face,
        frame_valid=np.ones((N, N), dtype=bool),
        native_ipd_px=420.0,
        yaw_deg=2.0, pitch_deg=-3.0, roll_deg=1.0,
        landmark_confidence=0.95,
        n_faces=1,
        cfg=QualityConfig(),
    )
    kw.update(over)
    return kw


def test_clean_input_passes():
    r = quality.evaluate(**base_kwargs())
    assert r.passed and r.reason == ReasonCode.OK


@pytest.mark.parametrize("over,expected", [
    (dict(n_faces=0), ReasonCode.NO_FACE),
    (dict(n_faces=2), ReasonCode.MULTIPLE_FACES),
    (dict(landmark_confidence=0.1), ReasonCode.LOW_LANDMARK_CONFIDENCE),
    (dict(native_ipd_px=40.0), ReasonCode.LOW_RESOLUTION),
    (dict(yaw_deg=45.0), ReasonCode.POSE_OUT_OF_RANGE),
    (dict(pitch_deg=-40.0), ReasonCode.POSE_OUT_OF_RANGE),
    (dict(roll_deg=50.0), ReasonCode.POSE_OUT_OF_RANGE),
])
def test_each_gate_reports_its_own_reason(over, expected):
    r = quality.evaluate(**base_kwargs(**over))
    assert not r.passed and r.reason == expected


def test_gate_priority_earlier_wins():
    """여러 조건이 동시에 나빠도 **앞 게이트의 사유**가 나가야 한다.

    뒤 조건은 평가조차 하지 않는다.
    """
    r = quality.evaluate(**base_kwargs(
        n_faces=0, landmark_confidence=0.0, native_ipd_px=1.0, yaw_deg=90.0))
    assert r.reason == ReasonCode.NO_FACE

    r = quality.evaluate(**base_kwargs(
        landmark_confidence=0.0, native_ipd_px=1.0, yaw_deg=90.0))
    assert r.reason == ReasonCode.LOW_LANDMARK_CONFIDENCE

    r = quality.evaluate(**base_kwargs(native_ipd_px=1.0, yaw_deg=90.0))
    assert r.reason == ReasonCode.LOW_RESOLUTION


def test_out_of_frame_face_is_rejected():
    """9:16 세로 사진에서 얼굴이 위쪽에 붙어 크롭이 프레임을 벗어나는 흔한 경우."""
    valid = np.ones((N, N), dtype=bool)
    valid[:150, :] = False
    r = quality.evaluate(**base_kwargs(frame_valid=valid))
    assert not r.passed and r.reason == ReasonCode.FACE_OUT_OF_FRAME


def test_blur_gate():
    flat = np.full((N, N, 3), 0.28, dtype=np.float32)   # 질감 없음 = 완전 흐림
    r = quality.evaluate(**base_kwargs(linear=flat))
    assert not r.passed and r.reason == ReasonCode.BLURRY


def _textured(mean_map, sigma=0.02, seed=0):
    """평균 밝기 지도 + 고주파 질감.

    **질감을 반드시 남겨야 한다.** 게이트 순서상 BLURRY 가 노출·조명 게이트보다
    먼저 걸리므로, 평평한 이미지로는 뒤쪽 게이트에 도달조차 하지 못한다.
    이름은 노출 게이트인데 실제로는 흐림 게이트를 검사하는 테스트가 되기 쉽다.
    """
    rng = np.random.default_rng(seed)
    lin = np.asarray(mean_map, dtype=np.float32).copy()
    lin += rng.normal(0.0, sigma, lin.shape).astype(np.float32)
    return np.clip(lin, 0.002, 0.999)


def test_overexposed_gate():
    bright = np.full((N, N, 3), 0.28, dtype=np.float32)
    bright[40:210, 40:210] = 0.999
    r = quality.evaluate(**base_kwargs(linear=_textured(bright)))
    assert not r.passed
    assert r.reason == ReasonCode.OVEREXPOSED


def test_underexposed_gate_is_actually_reachable():
    """어두운 사진은 BLURRY 가 아니라 UNDEREXPOSED 로 보고돼야 한다.

    안내 문장이 달라진다 — "삼각대를 쓰세요"와 "더 밝은 곳에서 찍으세요"는
    사용자가 할 일이 완전히 다르다. 질감이 살아 있는 어두운 사진으로 검사한다.
    """
    dark = np.full((N, N, 3), 0.28, dtype=np.float32)
    dark[40:210, 40:210] = 0.03
    r = quality.evaluate(**base_kwargs(linear=_textured(dark, sigma=0.02)))
    assert not r.passed
    assert r.reason == ReasonCode.UNDEREXPOSED, r.metrics


def test_non_uniform_illumination_gate():
    """세로 프레임의 천장 조명 구배. 계획 §2.5 에서 '그만큼 더 중요하다'고 표시한 게이트."""
    # 측면광: 얼굴 **안에서** 밝기가 크게 벌어져야 P90/P10 이 올라간다.
    # 프레임 전체에 램프를 걸면 얼굴이 램프의 가운데만 덮어 비율이 희석된다.
    lin = np.full((N, N, 3), 0.5, dtype=np.float32)
    grad = np.linspace(0.12, 0.95, 210 - 40).astype(np.float32)
    lin[40:210, 40:210] = grad[None, :, None]
    r = quality.evaluate(**base_kwargs(linear=_textured(lin)))
    assert not r.passed
    assert r.reason == ReasonCode.NON_UNIFORM_ILLUMINATION, r.metrics
    assert r.metrics["illum_ratio"] > 4.0


def test_metrics_are_recorded_even_on_pass():
    """통과해도 측정값은 남아야 한다 — 결과 화면의 '근거' 노출(§5.3)에 쓰인다."""
    r = quality.evaluate(**base_kwargs())
    for key in ("native_ipd_px", "blur_vol", "clipped_fraction",
                "illum_ratio", "face_in_frame"):
        assert key in r.metrics, key


def test_metrics_survive_early_failure():
    """앞 게이트에서 떨어져도 그때까지 아는 값은 남아야 디버깅이 된다."""
    r = quality.evaluate(**base_kwargs(n_faces=0))
    assert "native_ipd_px" in r.metrics
    assert "blur_vol" not in r.metrics       # 평가 자체를 안 했으므로 없는 게 정직하다


def test_every_failure_reason_is_declared_fatal():
    """게이트가 내는 사유는 전부 FATAL_REASONS 에 있어야 한다.

    빠지면 파이프라인이 '치명적이지 않은 실패'로 오해해 계속 진행한다.
    """
    seen = set()
    for over in (dict(n_faces=0), dict(n_faces=3), dict(landmark_confidence=0.0),
                 dict(native_ipd_px=1.0), dict(yaw_deg=90.0)):
        seen.add(quality.evaluate(**base_kwargs(**over)).reason)
    assert seen <= FATAL_REASONS


def test_variance_of_laplacian_scales_with_detail():
    rng = np.random.default_rng(1)
    sharp = rng.normal(0.5, 0.05, (128, 128))
    smooth, _ = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 128))
    assert quality.variance_of_laplacian(sharp) > \
           quality.variance_of_laplacian(smooth) * 100


def test_luminance_matches_rec709():
    lin = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    assert quality.luminance(lin)[0, 0] == pytest.approx(0.2126, abs=1e-5)
    lin = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
    assert quality.luminance(lin)[0, 0] == pytest.approx(0.7152, abs=1e-5)
