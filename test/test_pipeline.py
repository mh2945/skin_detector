"""오케스트레이터 전 구간 검증 — `pipeline.analyze()` 입력부터 `AnalysisResult` 까지.

**이 파일이 생기기 전까지 `pipeline.py` 는 한 번도 실행되지 않았다.** 86개 테스트는
`core/` 만 검증했고, 사진 한 장을 결과로 바꾸는 계층은 통째로 미검증이었다.

MediaPipe 는 쓰지 않는다. 합성 픽스처가 이미 정규 프레임 규격이므로
`conftest.synthetic_observation()` 이 `FaceObservation` 을 직접 만들어 준다 —
셀피 없이 도는 테스트라는 성질(`test/CLAUDE.md`)을 그대로 유지한다.

## 동작점에 대하여

`texture` 와 `noise` 를 분리한 이유가 여기서 드러난다. 질감 없이 흐림 게이트
(`min_blur_vol = 60`)를 통과시키려면 센서 노이즈를 0.010 까지 올려야 하는데,
그러면 깨끗한 얼굴에서 병변이 24~38개 검출된다. 실제 피부처럼 **무채색 질감은 많고
색 노이즈는 적은** 동작점(`CLEAN`)에서는 오검출이 0 이다.
"""

import numpy as np
import pytest

from conftest import (disc, hemoglobin_direction, inject, synthetic_face_linear,
                      synthetic_observation)
from skin_detector import imageio as sio, pipeline
from skin_detector.core.types import (ALL_REGIONS, CANONICAL_SIZE, ReasonCode,
                                      RegionId, Verdict)

# 흐림 게이트를 통과하면서 색 노이즈는 낮은 동작점. 실제 피부의 성질이다.
CLEAN = dict(texture=0.05, noise=0.004)
# 질감이 없으면 흐림 게이트에 걸린다 — 게이트 실패 경로 검사에 쓴다.
BLURRY = dict(texture=0.0, noise=0.004)

# 좌우 규약: RegionId.CHEEK_R = 피험자의 오른뺨 = 이미지 왼쪽(작은 x). CONTRACT 부수규약.
SUBJECT_RIGHT_CHEEK = (250.0, 430.0)


@pytest.fixture(scope="module")
def cfg_pair():
    cfg = sio.load_config()
    return cfg, sio.config_hash(cfg)


def analyze(linear, cfg_pair, **obs_kw):
    cfg, chash = cfg_pair
    obs = synthetic_observation(linear, **obs_kw)
    # observation 을 주면 rgb_original 은 쓰이지 않는다 (랜드마크·워프를 건너뛴다).
    return pipeline.analyze(np.zeros((4, 4, 3), np.uint8), cfg, chash,
                            model_path=None, observation=obs)


def scored(result):
    return {r.region: r for r in result.regions if r.area_fraction is not None}


def spread(result):
    vals = [r.area_fraction for r in result.regions if r.area_fraction is not None]
    return max(vals) - min(vals)


# ── 전 구간이 돌아가는가 ──────────────────────────────────────────────

def test_pipeline_runs_end_to_end(cfg_pair):
    """깨끗한 얼굴 한 장이 게이트를 통과해 5개 부위 점수까지 나온다."""
    ana = analyze(synthetic_face_linear(**CLEAN), cfg_pair)
    res = ana.result

    assert res.quality.passed, res.quality.reason
    assert res.quality.reason is ReasonCode.OK
    assert [r.region for r in res.regions] == list(ALL_REGIONS)
    assert res.algo_version == pipeline.ALGO_VERSION
    assert res.config_hash == cfg_pair[1]
    assert res.baseline is not None
    assert res.lesion_detection_enabled is True
    assert res.ita_deg is not None
    # 중간 맵은 오버레이·튜닝용으로 함께 돌아온다.
    assert ana.deviation.shape == (CANONICAL_SIZE, CANONICAL_SIZE)
    assert ana.skin.any()


def test_every_region_is_measured_on_a_clean_face(cfg_pair):
    """가림이 없으면 다섯 부위 모두 실제 수치가 나와야 한다."""
    res = analyze(synthetic_face_linear(**CLEAN), cfg_pair).result
    assert set(scored(res)) == set(ALL_REGIONS)
    for r in res.regions:
        assert r.coverage > 0.5, (r.region, r.coverage)


# ── 오검출: 깨끗한 얼굴을 깨끗하다고 하는가 ───────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_clean_face_has_no_false_lesions(cfg_pair, seed):
    """병변을 하나도 넣지 않았으면 검출도 없어야 한다.

    `max_lesions = 200` 이라 검출기가 잡음에 반응하면 상한까지 치솟는다.
    """
    res = analyze(synthetic_face_linear(seed=seed, **CLEAN), cfg_pair).result
    assert res.quality.passed
    assert len(res.lesions) < 5, "깨끗한 피부에서 병변 {}개".format(len(res.lesions))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clean_face_is_not_called_red(cfg_pair, seed):
    """홍반을 0 주입했으면 어떤 부위도 '붉음'이 되면 안 된다."""
    res = analyze(synthetic_face_linear(seed=seed, **CLEAN), cfg_pair).result
    red = [r.region.value for r in res.regions
           if r.erythema_verdict is Verdict.AFFECTED]
    assert red == [], "색소가 균일한 얼굴에서 '붉음' 판정: {}".format(red)


def test_shading_alone_does_not_create_redness(cfg_pair):
    """조명 구배만으로 '붉음'이 만들어지면 안 된다.

    로그 비율은 음영을 **평균에서** 정확히 소거하지만 **분산은 소거하지 않는다.**
    전역 tau 를 쓰면 그늘진 부위가 자동으로 붉게 나온다 — 실제로 관측된 실패
    (이마가 항상 '붉음')이고, `measure.region_noise_scale` 이 그 대응이다.
    """
    lit = analyze(synthetic_face_linear(shading=False, **CLEAN), cfg_pair).result
    shaded = analyze(synthetic_face_linear(shading=True, **CLEAN), cfg_pair).result

    assert [r.region.value for r in shaded.regions
            if r.erythema_verdict is Verdict.AFFECTED] == []
    # 음영이 만드는 부위 간 편차가 평탄 조명 대비 지나치게 커지면 안 된다.
    assert spread(shaded) < spread(lit) + 0.05, (
        "음영이 부위 간 편차를 {:.4f} -> {:.4f} 로 벌린다"
        .format(spread(lit), spread(shaded)))


# ── 검출: 넣은 것을 넣은 자리에서 찾는가 ──────────────────────────────

def test_injected_erythema_is_localized_and_side_correct(cfg_pair):
    """피험자 오른뺨에 주입하면 CHEEK_R 만 올라야 한다.

    좌우 규약(CONTRACT 부수규약)을 동시에 고정한다: `CHEEK_R` = 피험자의 오른뺨
    = **이미지 왼쪽(작은 x)**. 좌우가 뒤집히면 이 테스트가 잡는다.
    """
    base = synthetic_face_linear(**CLEAN)
    blob = disc(SUBJECT_RIGHT_CHEEK[0], SUBJECT_RIGHT_CHEEK[1], 90.0, 0.05)
    dosed = inject(base, blob, hemoglobin_direction())

    before = scored(analyze(base, cfg_pair).result)
    after = scored(analyze(dosed, cfg_pair).result)

    assert (after[RegionId.CHEEK_R].area_fraction
            > before[RegionId.CHEEK_R].area_fraction + 0.10)
    assert after[RegionId.CHEEK_R].erythema_verdict is Verdict.AFFECTED
    # 반대쪽 뺨은 실질적으로 그대로여야 한다.
    assert abs(after[RegionId.CHEEK_L].area_fraction
               - before[RegionId.CHEEK_L].area_fraction) < 0.10
    assert after[RegionId.CHEEK_L].erythema_verdict is Verdict.NORMAL


def test_erythema_response_is_monotonic(cfg_pair):
    """주입량을 늘리면 점수도 단조 증가해야 한다."""
    base = synthetic_face_linear(**CLEAN)
    vh = hemoglobin_direction()
    scores = []
    for amount in (0.0, 0.02, 0.05, 0.10):
        blob = disc(SUBJECT_RIGHT_CHEEK[0], SUBJECT_RIGHT_CHEEK[1], 90.0, amount)
        res = analyze(inject(base, blob, vh), cfg_pair).result
        scores.append(scored(res)[RegionId.CHEEK_R].score_ordinal_0_100)
    assert scores == sorted(scores), scores
    assert scores[-1] > scores[0]


# ── 게이트: 못 잰 것을 0 이라고 하지 않는가 ───────────────────────────

def test_gate_failure_returns_null_not_zero(cfg_pair):
    """게이트에 걸리면 모든 수치가 `None` 이어야 한다 — 0 이 아니다 (계약 4번)."""
    res = analyze(synthetic_face_linear(**BLURRY), cfg_pair).result

    assert not res.quality.passed
    assert res.quality.reason is ReasonCode.BLURRY
    assert len(res.regions) == len(ALL_REGIONS)
    for r in res.regions:
        assert r.area_fraction is None
        assert r.intensity is None
        assert r.median_d is None
        assert r.score_ordinal_0_100 is None
        assert r.lesion_count is None
        assert r.erythema_verdict is Verdict.NOT_MEASURED
        assert r.lesion_verdict is Verdict.NOT_MEASURED
        assert r.reason is ReasonCode.BLURRY


def test_low_resolution_disables_lesions_but_keeps_diffuse(cfg_pair):
    """IPD 가 모자라면 병변만 끄고 미만성은 계속 보고한다."""
    cfg, _ = cfg_pair
    ipd = cfg.quality.lesion_min_ipd_px - 10.0
    res = analyze(synthetic_face_linear(**CLEAN), cfg_pair, native_ipd_px=ipd).result

    assert res.quality.passed
    assert res.lesion_detection_enabled is False
    assert res.lesion_reason is ReasonCode.LESION_DISABLED_LOW_RESOLUTION
    assert res.lesions == []
    for r in res.regions:
        assert r.lesion_verdict is Verdict.NOT_MEASURED
        assert r.lesion_count is None
        assert r.area_fraction is not None      # 미만성은 살아 있다


def test_too_small_face_is_rejected_outright(cfg_pair):
    """`min_native_ipd_px` 미만이면 분석 자체를 거부한다."""
    cfg, _ = cfg_pair
    ipd = cfg.quality.min_native_ipd_px - 10.0
    res = analyze(synthetic_face_linear(**CLEAN), cfg_pair, native_ipd_px=ipd).result
    assert not res.quality.passed
    assert res.quality.reason is ReasonCode.LOW_RESOLUTION


def test_multiple_faces_is_rejected(cfg_pair):
    res = analyze(synthetic_face_linear(**CLEAN), cfg_pair, n_faces=2).result
    assert not res.quality.passed
    assert res.quality.reason is ReasonCode.MULTIPLE_FACES


# ── 좌표계: 역변환이 맞는가 ───────────────────────────────────────────

def test_lesions_to_original_is_identity_for_identity_affine(cfg_pair):
    """합성 관측의 affine 은 항등이므로 역변환도 항등이어야 한다.

    오버레이가 어긋나는 버그는 대부분 이 왕복에서 나온다 (`src/.../CLAUDE.md`).
    """
    base = synthetic_face_linear(**CLEAN)
    blob = disc(SUBJECT_RIGHT_CHEEK[0], SUBJECT_RIGHT_CHEEK[1], 7.0, 0.05)
    ana = analyze(inject(base, blob, hemoglobin_direction()), cfg_pair)

    assert ana.result.lesions, "왕복을 검사하려면 병변이 하나는 있어야 한다"
    back = pipeline.lesions_to_original(ana)
    assert len(back) == len(ana.result.lesions)
    for les, o in zip(ana.result.lesions, back):
        assert o["x"] == pytest.approx(les.x, abs=1e-6)
        assert o["y"] == pytest.approx(les.y, abs=1e-6)
        assert o["radius_px"] == pytest.approx(les.radius_px, rel=1e-6)


def test_lesions_to_original_is_empty_without_lesions(cfg_pair):
    ana = analyze(synthetic_face_linear(**CLEAN), cfg_pair)
    assert pipeline.lesions_to_original(ana) == []


def test_papule_detection_has_a_working_amplitude_range(cfg_pair):
    """구진이 검출되는 진폭 구간을 고정한다 — **상한이 존재한다.**

    헤모글로빈은 빛을 흡수하므로 진하게 넣을수록 그 자리가 어두워진다. 진폭이
    0.10 을 넘으면 국소 dL* 가 `darkness_reject_dl`(-8.0)을 밑돌아 **색소 병변으로
    오인되어 버려진다.** 즉 아주 강한 염증성 병변은 조용히 사라질 수 있다.

    임상적으로 의미 있는 구간(0.02~0.07)에서는 정상 검출된다. 이 테스트는 그 구간을
    보장하고, 동시에 상한이 존재한다는 사실을 문서가 아니라 코드로 남긴다.
    """
    base = synthetic_face_linear(**CLEAN)
    vh = hemoglobin_direction()

    def count(amount):
        blob = disc(SUBJECT_RIGHT_CHEEK[0], SUBJECT_RIGHT_CHEEK[1], 7.0, amount)
        return analyze(inject(base, blob, vh), cfg_pair).result

    for amount in (0.02, 0.04, 0.07):
        res = count(amount)
        assert len(res.lesions) == 1, (amount, len(res.lesions))
        assert res.lesions[0].region is RegionId.CHEEK_R

    strong = count(0.10)
    assert strong.lesions == []
    assert strong.lesion_reject_count.get("dark_pigmented") == 1, (
        "상한의 원인이 darkness 게이트가 아니게 되면 이 테스트를 다시 봐야 한다")


@pytest.mark.parametrize("texture,noise", [(0.0, 0.010), (0.0, 0.030), (0.05, 0.015)])
def test_noisy_capture_does_not_produce_a_lesion_storm(cfg_pair, texture, noise):
    """색 노이즈가 큰 사진에서도 병변이 쏟아지면 안 된다.

    회귀 테스트다. 병변 임계가 절대 상수(`peak_min_d`)뿐이던 때는 흐림 게이트를
    통과하는 가장 깨끗한 사진에서도 깨끗한 피부에 24~51개가 잡혔다. DoG 응답을
    정합 이득 0.149 로 나누므로 잡음이 6.7배 증폭되기 때문이다.

    `peak_min_k` 가 극대점 분포에서 잡음 수준을 직접 재서 임계를 올린다.
    """
    res = analyze(synthetic_face_linear(texture=texture, noise=noise), cfg_pair).result
    assert res.quality.passed, res.quality.reason
    assert len(res.lesions) < 5, (
        "texture={} noise={} 인 깨끗한 피부에서 병변 {}개"
        .format(texture, noise, len(res.lesions)))
