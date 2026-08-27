"""흐림 · 2단계 기준선 · 합성 홍반 복원 · DoG 병변 · null 전파."""

import numpy as np
import pytest

from conftest import inject, synthetic_face_linear, synthetic_landmarks
from skin_detector.core import color, mask as M, measure
from skin_detector.core.config import (
    BaselineConfig, LesionConfig, MaskConfig, QualityConfig, ScoreConfig,
)
from skin_detector.core.types import (
    ALL_REGIONS, CANONICAL_SIZE, ReasonCode, RegionId, Verdict,
)

FULL = np.ones((CANONICAL_SIZE, CANONICAL_SIZE), dtype=bool)


# ── 흐림 ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sigma", [3.6, 4.9, 6.6, 9.0, 80.0])
def test_box_approximation_sigma_within_5_percent(sigma):
    """박스 3회 근사의 실효 sigma 오차 상한을 고정한다.

    docstring 이 주장하는 ~4% 를 테스트가 붙잡아 둔다. 이 값이 커지면
    병변 크기 추정치가 그만큼 편향된다.
    """
    n = 513
    imp = np.zeros((n, n))
    imp[n // 2, n // 2] = 1.0
    out, _ = measure.normalized_blur(imp, np.ones((n, n), bool), sigma)
    row = out[n // 2, :]
    x = np.arange(n) - n // 2
    measured = np.sqrt((row * x * x).sum() / row.sum())
    assert abs(measured - sigma) / sigma < 0.05


def test_normalized_convolution_does_not_drag_edges_down():
    """**자주 틀리는 지점** (계획 §1.4).

    그냥 blur(v) 를 쓰면 마스크된 0 이 이웃을 끌어내려 눈·입술 주변이 전부
    가짜 저홍반이 된다. normalized convolution 은 상수 필드를 상수로 유지해야 한다.
    """
    v = np.ones((120, 120))
    valid = np.ones((120, 120), dtype=bool)
    valid[:, :50] = False
    out, ok = measure.normalized_blur(v, valid, 12.0)
    assert out[60, 55] == pytest.approx(1.0, abs=1e-6)
    assert out[60, 119] == pytest.approx(1.0, abs=1e-6)

    naive = measure._smooth(np.where(valid, v, 0.0), 12.0)
    naive = naive / naive.max()
    assert naive[60, 55] < 0.9, "naive blur 는 실제로 끌어내려야 픽스처가 유효하다"


def test_weighted_percentile_matches_unweighted_when_uniform():
    rng = np.random.default_rng(1)
    v = rng.normal(size=5000)
    w = np.ones_like(v)
    for q in (10, 25, 50, 90):
        assert measure.weighted_percentile(v, w, q) == pytest.approx(
            float(np.percentile(v, q)), abs=0.05)


def test_robust_sigma_resists_outliers():
    rng = np.random.default_rng(4)
    v = rng.normal(0.0, 0.01, 4000)
    v[:400] = 5.0
    assert measure.robust_sigma(v) == pytest.approx(0.01, rel=0.15)


# ── 통합 픽스처 ───────────────────────────────────────────────────────

def build(linear, landmarks, frame_valid=None):
    q = QualityConfig()
    e, m = color.chromophore_axes(color.optical_density(linear))
    w = color.snr_weights(linear)
    masks = M.build_masks(
        landmarks=landmarks, linear=linear, e=e, m=m,
        frame_valid=FULL if frame_valid is None else frame_valid,
        cfg=MaskConfig(), clip_level=q.clip_level, dark_level=q.dark_level)
    bcfg = BaselineConfig()
    eri, alpha, beta = measure.fit_eri(e, m, masks.skin, w, bcfg)
    cov = {r: masks.coverage(r) for r in ALL_REGIONS}
    base = measure.compute_baseline(eri, masks.skin, w, masks.regions, cov,
                                    alpha, beta, bcfg)
    d = measure.deviation_map(eri, base, masks.regions)
    return masks, eri, base, d, cov


# ── 합성 홍반 복원 (§6.2) — 지표가 선형인가 ──────────────────────────

def test_synthetic_erythema_recovery_is_linear(landmarks, v_hemoglobin):
    """주입한 Delta 와 복원된 d 가 선형이어야 한다.

    **선형성이 깨지면 지표 자체가 실패다** (계획 §7.1). 기울기·R^2 가 PoC PASS
    조건에 직접 들어간다.
    """
    lm = landmarks
    base_lin = synthetic_face_linear(noise=0.002)

    # 왼쪽 볼에만 주입한다 (얼굴 전체에 넣으면 자기참조가 흡수해 버린다 — §7.3-2).
    yy, xx = np.mgrid[0:CANONICAL_SIZE, 0:CANONICAL_SIZE]
    patch = ((xx - 560) ** 2 + (yy - 400) ** 2) < 70 ** 2

    deltas = [0.0, 0.02, 0.04, 0.08, 0.12]
    recovered = []
    for delta in deltas:
        amt = np.where(patch, delta, 0.0)
        lin = inject(base_lin, amt, v_hemoglobin)
        masks, _, _, d, _ = build(lin, lm)
        sel = patch & masks.skin
        recovered.append(float(np.median(d[sel])))

    x = np.array(deltas)
    y = np.array(recovered) - recovered[0]
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot

    # 기대 기울기: e 축 투영 = v_h[G] - v_h[R]
    expected = float(v_hemoglobin[1] - v_hemoglobin[0])
    assert r2 >= 0.99, "R^2={:.4f}".format(r2)
    assert slope == pytest.approx(expected, rel=0.15), \
        "기울기 {:.3f} vs 기대 {:.3f}".format(slope, expected)


def test_illumination_change_does_not_move_the_measurement(landmarks, v_hemoglobin):
    """같은 생리, 다른 조명 -> 같은 d 가 나와야 한다. §6.1 안정성 실험의 축소판이다."""
    yy, xx = np.mgrid[0:CANONICAL_SIZE, 0:CANONICAL_SIZE]
    patch = ((xx - 560) ** 2 + (yy - 400) ** 2) < 70 ** 2
    amt = np.where(patch, 0.06, 0.0)

    results = []
    for gain, scale in ((np.array([1.0, 1.0, 1.0]), 1.0),
                        (np.array([1.20, 0.95, 0.80]), 0.7),   # 텅스텐 + 어두움
                        (np.array([0.88, 1.00, 1.25]), 1.25)):  # 그늘(푸른빛) + 밝음
        lin = inject(synthetic_face_linear(noise=0.002), amt, v_hemoglobin)
        lin = np.clip(lin * gain.astype(np.float32) * scale, 1e-4, 1.0).astype(np.float32)
        masks, _, _, d, _ = build(lin, landmarks)
        results.append(float(np.median(d[patch & masks.skin])))

    spread = max(results) - min(results)
    assert spread < 0.004, "조명 간 편차 {:.4f} — 자기참조가 새고 있다".format(spread)


# ── 2단계 기준선 ──────────────────────────────────────────────────────

def test_baseline_reports_unavailable_when_too_few_reference_regions(landmarks):
    """참조 부위가 2개 미만이면 '통과'가 아니라 '확인 불가'여야 한다 (계획 §4.4).

    앞머리로 이마가 가려지는 셀피는 예외가 아니라 다수다. 이마 하나에 의존하면
    조명 구배 플래그가 조용히 무력화된다.
    """
    lin = synthetic_face_linear()
    masks, eri, _, _, cov = build(lin, landmarks)
    w = color.snr_weights(lin)
    starved = {r: 0.0 for r in ALL_REGIONS}          # 전 부위 coverage 0
    base = measure.compute_baseline(eri, masks.skin, w, masks.regions, starved,
                                    0.0, 0.0, BaselineConfig())
    assert base.b_alt is None
    assert base.n_ref_regions == 0
    assert base.illumination_gradient_flag is False   # '이상 없음'이 아니라 '판단 안 함'


def test_baseline_uses_multiple_reference_regions(landmarks):
    lin = synthetic_face_linear()
    masks, eri, base, _, cov = build(lin, landmarks)
    assert base.n_ref_regions >= 2
    assert base.b_alt is not None


def test_anatomical_prior_flag_is_honest(landmarks):
    """mu_r 유무가 플래그에 **양방향으로** 반영돼야 한다.

    `False` 만 단언하면 플래그가 `bool(mu_r)` 인 이상 절대 실패할 수 없는 테스트가
    된다. 상수를 재진술하는 대신, 준 경우와 안 준 경우를 모두 본다.
    """
    lin = synthetic_face_linear()
    masks, eri, base, _, cov = build(lin, landmarks)
    assert base.anatomical_prior_calibrated is False

    bcfg = BaselineConfig()
    w = color.snr_weights(lin)
    _eri, alpha, beta = measure.fit_eri(
        *color.chromophore_axes(color.optical_density(lin)),
        masks.skin, w, bcfg)
    mu_r = {r: 0.001 for r in ALL_REGIONS}
    calibrated = measure.compute_baseline(
        eri, masks.skin, w, masks.regions, cov, alpha, beta, bcfg, mu_r=mu_r)
    assert calibrated.anatomical_prior_calibrated is True


def test_mu_r_shifts_only_its_own_region(landmarks):
    lin = synthetic_face_linear()
    masks, eri, base, d0, _ = build(lin, landmarks)
    d1 = measure.deviation_map(eri, base, masks.regions, {RegionId.NOSE: 0.05})
    nose = masks.regions[RegionId.NOSE]
    cheek = masks.regions[RegionId.CHEEK_L]
    assert float(np.median(d0[nose] - d1[nose])) == pytest.approx(0.05, abs=1e-4)
    assert np.array_equal(d0[cheek], d1[cheek])


# ── 병변 검출 ─────────────────────────────────────────────────────────

def lesion_inputs(lin, landmarks):
    masks, eri, base, d, cov = build(lin, landmarks)
    scfg = ScoreConfig()
    _diffuse, band = measure.split_bands(d, masks.skin, scfg)
    e, m = color.chromophore_axes(color.optical_density(lin))
    sigma_lp = scfg.lowpass_sigma_frac * 320.0
    m_lp, _ = measure.normalized_blur(m, masks.skin, sigma_lp)
    lab = color.linear_to_lab(lin)
    l_lp, _ = measure.normalized_blur(lab[..., 0], masks.skin, sigma_lp)
    return masks, base, d, band, (m - m_lp), (lab[..., 0] - l_lp)


def blobs(centers, sigma_px, amplitude):
    yy, xx = np.mgrid[0:CANONICAL_SIZE, 0:CANONICAL_SIZE]
    amt = np.zeros((CANONICAL_SIZE, CANONICAL_SIZE))
    for (cx, cy) in centers:
        amt += amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                                  / (2.0 * sigma_px ** 2))
    return amt


def test_detects_injected_papules(landmarks, v_hemoglobin):
    """2~3mm 구진 크기의 헤모글로빈 블롭을 찾아야 한다 (1mm = 5.08px)."""
    centers = [(520, 380), (600, 430), (250, 400), (200, 460)]
    amt = blobs(centers, sigma_px=4.5, amplitude=0.09)
    lin = inject(synthetic_face_linear(noise=0.0015), amt, v_hemoglobin)
    masks, base, d, band, m_band, l_band = lesion_inputs(lin, landmarks)

    found, rejects = measure.detect_lesions(
        band, m_band, l_band, masks.skin, masks.edge_band, masks.regions,
        iris_px=M.iris_diameter_px(landmarks), cfg=LesionConfig())

    hits = 0
    for (cx, cy) in centers:
        if any((l.x - cx) ** 2 + (l.y - cy) ** 2 < 20 ** 2 for l in found):
            hits += 1
    assert hits >= 3, "4개 중 {}개만 찾았다 (기각: {})".format(hits, rejects)


def test_lesion_size_estimate_is_in_the_right_ballpark(landmarks, v_hemoglobin):
    amt = blobs([(560, 400)], sigma_px=4.5, amplitude=0.10)
    lin = inject(synthetic_face_linear(noise=0.001), amt, v_hemoglobin)
    masks, base, d, band, m_band, l_band = lesion_inputs(lin, landmarks)
    found, _ = measure.detect_lesions(
        band, m_band, l_band, masks.skin, masks.edge_band, masks.regions,
        iris_px=M.iris_diameter_px(landmarks), cfg=LesionConfig())
    near = [l for l in found if (l.x - 560) ** 2 + (l.y - 400) ** 2 < 20 ** 2]
    assert near, "블롭을 못 찾아 크기 검증을 할 수 없다"
    assert near[0].radius_mm is not None
    assert 0.7 <= near[0].radius_mm <= 3.0, near[0].radius_mm


def test_melanin_blob_is_not_detected_as_lesion(landmarks, v_melanin, v_hemoglobin):
    """**모반을 거르는 것이 2차원 색소 평면의 직접적 배당금이다.**

    같은 위치·같은 크기라도 멜라닌 방향으로 움직인 블롭은 병변이 되면 안 된다.
    a* 하나만 봤으면 불가능하다.

    방어선은 둘이고 순서가 있다.
      1차: `ERI = e - beta*m` 회귀가 상류에서 멜라닌 기여를 제거한다. 대부분 여기서 끝난다.
      2차: 회귀가 다 못 지운 경우 cascade 의 부호 규칙이 잡는다 (아래 별도 테스트).
    여기서는 **결과**만 본다 — 어느 방어선이 막았는지를 요구하지 않는다.
    """
    amt = blobs([(560, 400)], sigma_px=4.5, amplitude=0.16)

    lin_mel = inject(synthetic_face_linear(noise=0.001), amt, v_melanin)
    masks, _, _, band_mel, m_band, l_band = lesion_inputs(lin_mel, landmarks)
    found, rejects = measure.detect_lesions(
        band_mel, m_band, l_band, masks.skin, masks.edge_band, masks.regions,
        iris_px=M.iris_diameter_px(landmarks), cfg=LesionConfig())
    near = [l for l in found if (l.x - 560) ** 2 + (l.y - 400) ** 2 < 20 ** 2]
    assert not near, "멜라닌 블롭이 병변으로 잡혔다 (기각 통계: {})".format(rejects)

    # 같은 진폭의 헤모글로빈은 홍반 대역에 뚜렷이 남아야 한다.
    # 그래야 위 결과가 "멜라닌만 지워졌다"는 뜻이 된다 (둘 다 못 보는 게 아니라).
    lin_hb = inject(synthetic_face_linear(noise=0.001),
                    blobs([(560, 400)], 4.5, 0.16), v_hemoglobin)
    _, _, _, band_hb, _, _ = lesion_inputs(lin_hb, landmarks)
    assert abs(band_hb[400, 560]) > 5.0 * abs(band_mel[400, 560]), \
        "멜라닌 {:.5f} vs 헤모글로빈 {:.5f} — 분리가 안 되고 있다".format(
            band_mel[400, 560], band_hb[400, 560])


def test_cascade_rejects_positive_melanin_response():
    """2차 방어선 단위 검사: 부호 규칙이 실제로 동작하는가.

    **판별자는 dm 의 크기가 아니라 부호다.** 실측 방향 벡터에서
    헤모글로빈은 dm<0, 멜라닌은 dm>0 이다. `|dm| > |de|` 로 기각하면
    헤모글로빈(|-0.494| > 0.447)까지 걸려 진짜 병변을 전부 버린다.
    """
    n = 256
    yy, xx = np.mgrid[0:n, 0:n]
    blob = np.exp(-((xx - 128) ** 2 + (yy - 128) ** 2) / (2 * 4.5 ** 2))
    skin = np.ones((n, n), dtype=bool)
    regions = {RegionId.CHEEK_L: skin}
    flat = np.zeros((n, n))
    cfg = LesionConfig()

    def run(m_band):
        return measure.detect_lesions(
            0.10 * blob, m_band, flat, skin, np.zeros((n, n), bool), regions,
            iris_px=59.4, cfg=cfg)

    # dm > 0 (멜라닌) -> 기각
    found, rejects = run(+0.30 * blob)
    assert not found and rejects.get("melanin_nevus", 0) >= 1

    # dm < 0 (헤모글로빈) -> 같은 크기여도 통과
    found, rejects = run(-0.30 * blob)
    assert found, "헤모글로빈 부호인데 기각됐다: {}".format(rejects)


def test_no_lesions_on_clean_skin(landmarks):
    """깨끗한 피부에서 오검출이 거의 없어야 한다 (합성 기준 — 실사진 상한이 아니다)."""
    lin = synthetic_face_linear(noise=0.003)
    masks, _, _, band, m_band, l_band = lesion_inputs(lin, landmarks)
    found, _ = measure.detect_lesions(
        band, m_band, l_band, masks.skin, masks.edge_band, masks.regions,
        iris_px=M.iris_diameter_px(landmarks), cfg=LesionConfig())
    assert len(found) <= 2, "깨끗한 피부에서 {}개 오검출".format(len(found))


# ── 점수화와 null 전파 ────────────────────────────────────────────────

def test_low_coverage_yields_null_not_zero(landmarks):
    """**측정 실패는 None 이다. 0.0 이 아니다** (계약 4번).

    0 을 반환하면 '측정 못 함'이 '아주 정상'으로 둔갑한다.
    """
    lin = synthetic_face_linear()
    masks, _, base, d, cov = build(lin, landmarks)
    starved = {r: 0.1 for r in ALL_REGIONS}
    scores = measure.score_regions(
        d, masks.regions, starved, [], base, ScoreConfig(),
        min_coverage=0.5, lesion_enabled=True)
    for s in scores:
        assert s.score_ordinal_0_100 is None
        assert s.area_fraction is None
        assert s.erythema_verdict == Verdict.NOT_MEASURED
        assert s.reason != ReasonCode.OK


def test_forehead_gets_its_own_reason_code(landmarks):
    """이마가 가려진 것과 그냥 화소가 모자란 것은 사용자에게 다른 안내가 나가야 한다."""
    lin = synthetic_face_linear()
    masks, _, base, d, cov = build(lin, landmarks)
    starved = {r: 0.1 for r in ALL_REGIONS}
    scores = measure.score_regions(
        d, masks.regions, starved, [], base, ScoreConfig(),
        min_coverage=0.5, lesion_enabled=True,
        forehead_occluded_reason=ReasonCode.FOREHEAD_OCCLUDED)
    fore = [s for s in scores if s.region == RegionId.FOREHEAD][0]
    assert fore.reason == ReasonCode.FOREHEAD_OCCLUDED


def test_lesion_verdict_is_not_measured_when_disabled(landmarks):
    """저해상도로 병변을 껐으면 '없음'이 아니라 '측정 안 함'이다."""
    lin = synthetic_face_linear()
    masks, _, base, d, cov = build(lin, landmarks)
    scores = measure.score_regions(
        d, masks.regions, cov, [], base, ScoreConfig(),
        min_coverage=0.5, lesion_enabled=False)
    measured = [s for s in scores if s.reason == ReasonCode.OK]
    assert measured
    for s in measured:
        assert s.lesion_verdict == Verdict.NOT_MEASURED
        assert s.lesion_count is None


def test_injected_region_is_judged_affected(landmarks, v_hemoglobin):
    """붉은기 축 판정이 실제로 작동하는가 — §7.2 주 지표의 최소 단위."""
    yy, xx = np.mgrid[0:CANONICAL_SIZE, 0:CANONICAL_SIZE]
    patch = ((xx - 560) ** 2 + (yy - 400) ** 2) < 95 ** 2
    lin = inject(synthetic_face_linear(noise=0.002), np.where(patch, 0.10, 0.0),
                 v_hemoglobin)
    masks, _, base, d, cov = build(lin, landmarks)
    scores = measure.score_regions(d, masks.regions, cov, [], base, ScoreConfig(),
                                   min_coverage=0.5, lesion_enabled=True)
    by = {s.region: s for s in scores}
    assert by[RegionId.CHEEK_L].erythema_verdict == Verdict.AFFECTED
    assert by[RegionId.CHEEK_R].erythema_verdict == Verdict.NORMAL
