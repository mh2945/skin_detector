"""유일한 오케스트레이터. 계획 §4 의 [0]~[9] 를 순서대로 엮는다.

여기 말고 다른 곳에서 core/ 함수를 직접 조립하지 않는다 —
web/ 도 scripts/ 도 이 파일만 호출한다 (계약 9번).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import __version__, face as face_mod
from .core import color, mask as mask_mod, measure, quality
from .core.config import Config
from .core.types import (
    ALL_REGIONS,
    AnalysisResult,
    CaptureMeta,
    QualityReport,
    ReasonCode,
    RegionId,
    RegionScore,
    mm_per_px,
)

ALGO_VERSION = "v" + __version__


@dataclass
class Analysis:
    """결과 + 렌더링에 필요한 중간 맵.

    JSON 에 나가는 것은 `result` 뿐이다. 맵들은 오버레이와 튜닝 UI 용이고
    디스크에 남지 않는다.
    """

    result: AnalysisResult
    observation: Optional[face_mod.FaceObservation] = None
    deviation: Optional[np.ndarray] = None       # d(x)
    diffuse: Optional[np.ndarray] = None         # 저주파 성분
    skin: Optional[np.ndarray] = None
    masks: Optional[mask_mod.MaskSet] = None


def _failed(
    capture_id: str,
    cfg_hash: str,
    report: QualityReport,
    meta: Optional[CaptureMeta],
    native_ipd: Optional[float] = None,
) -> Analysis:
    """게이트에 걸렸을 때. 모든 부위는 `NOT_MEASURED` 다 — 0 이 아니다."""
    regions = [RegionScore(region=r, coverage=0.0, reason=report.reason)
               for r in ALL_REGIONS]
    return Analysis(result=AnalysisResult(
        capture_id=capture_id,
        algo_version=ALGO_VERSION,
        config_hash=cfg_hash,
        quality=report,
        regions=regions,
        native_ipd_px=native_ipd,
        capture=meta,
    ))


def analyze(
    rgb_original: np.ndarray,
    cfg: Config,
    cfg_hash: str,
    model_path: Path,
    meta: Optional[CaptureMeta] = None,
    mu_r: Optional[Dict[RegionId, float]] = None,
    observation: Optional[face_mod.FaceObservation] = None,
) -> Analysis:
    """원본 uint8 RGB 한 장 -> 분석 결과.

    Args:
        observation: 이미 만들어 둔 정규 프레임 (튜닝 시 캐시 재사용 — §2.6).
                     주면 랜드마크·워프를 건너뛰고 색 단계만 다시 돈다.
        mu_r: Level 2 부위별 해부학 오프셋. 없으면 0 이고 리포트에
              `anatomical_prior_calibrated: false` 로 정직하게 나간다.
    """
    capture_id = meta.capture_id if meta else ""

    # ── [0]~[2] 정규 프레임 ──
    if observation is None:
        observation = face_mod.detect(rgb_original, model_path)
    if observation is None:
        return _failed(capture_id, cfg_hash,
                       QualityReport(False, ReasonCode.NO_FACE, {"n_faces": 0.0}),
                       meta)

    obs = observation

    # ── [5] 색 변환 — 정규 프레임에서만 한다 (계획 §2.2) ──
    linear = color.srgb_to_linear(obs.canonical_rgb)
    density = color.optical_density(linear)
    e, m = color.chromophore_axes(density)
    weights = color.snr_weights(linear)

    # ── [3] 품질 게이트 — 먼저 걸리는 것이 이긴다 ──
    report = quality.evaluate(
        linear=linear,
        face_poly=obs.face_poly,
        frame_valid=obs.frame_valid,
        native_ipd_px=obs.native_ipd_px,
        yaw_deg=obs.yaw_deg,
        pitch_deg=obs.pitch_deg,
        roll_deg=obs.roll_deg,
        landmark_confidence=obs.confidence,
        n_faces=obs.n_faces,
        cfg=cfg.quality,
    )
    if not report.passed:
        failed = _failed(capture_id, cfg_hash, report, meta, obs.native_ipd_px)
        failed.observation = obs
        return failed

    # ── [4] 마스킹 ──
    masks = mask_mod.build_masks(
        landmarks=obs.landmarks,
        linear=linear,
        e=e, m=m,
        frame_valid=obs.frame_valid,
        cfg=cfg.mask,
        clip_level=cfg.quality.clip_level,
        dark_level=cfg.quality.dark_level,
        face_poly=obs.face_poly,
    )
    skin = masks.skin
    coverage = {r: masks.coverage(r) for r in ALL_REGIONS}

    if int(skin.sum()) < 2048:
        report = QualityReport(False, ReasonCode.LOW_COVERAGE, report.metrics)
        failed = _failed(capture_id, cfg_hash, report, meta, obs.native_ipd_px)
        failed.observation = obs
        failed.masks = masks
        return failed

    # ── [5] ERI (얼굴별 멜라닌 회귀 제거) ──
    eri, alpha, beta = measure.fit_eri(e, m, skin, weights, cfg.baseline)

    # ── [6] 2단계 기준선 -> 편차맵 ──
    baseline = measure.compute_baseline(
        eri=eri, skin=skin, weights=weights,
        region_masks=masks.regions, region_coverage=coverage,
        alpha=alpha, beta=beta, cfg=cfg.baseline, mu_r=mu_r,
    )
    d = measure.deviation_map(eri, baseline, masks.regions, mu_r)

    # ── [7] 대역 분리 ──
    diffuse, lesion_band = measure.split_bands(d, skin, cfg.score)

    # ── [8] 병변 ──
    lesion_enabled = obs.native_ipd_px >= cfg.quality.lesion_min_ipd_px
    lesions: List = []
    lesion_rejects: Dict[str, int] = {}
    lesion_reason = ReasonCode.OK
    iris_px = mask_mod.iris_diameter_px(obs.landmarks)

    if lesion_enabled:
        sigma_lp = cfg.score.lowpass_sigma_frac * 320.0
        m_lp, _ = measure.normalized_blur(m, skin, sigma_lp)
        m_band = (m - m_lp).astype(np.float32)

        lab = color.linear_to_lab(linear)
        l_star = lab[..., 0]
        l_lp, _ = measure.normalized_blur(l_star, skin, sigma_lp)
        l_band = (l_star - l_lp).astype(np.float32)

        lesions, lesion_rejects = measure.detect_lesions(
            lesion_band=lesion_band,
            melanin_band=m_band,
            lightness=l_band,
            skin=skin,
            edge_band=masks.edge_band,
            region_masks=masks.regions,
            iris_px=iris_px,
            cfg=cfg.lesion,
        )
    else:
        # 저해상도 입력이 섞여 들어와도 쓰레기 출력이 안 나가게 한다.
        lesion_reason = ReasonCode.LESION_DISABLED_LOW_RESOLUTION

    # ── [8] 부위 점수 ──
    forehead_reason = (ReasonCode.FOREHEAD_OCCLUDED
                       if coverage[RegionId.FOREHEAD] < cfg.baseline.min_coverage
                       else ReasonCode.OK)
    regions = measure.score_regions(
        d=d,
        masks_by_region=masks.regions,
        coverage_by_region=coverage,
        lesions=lesions,
        baseline=baseline,
        cfg=cfg.score,
        min_coverage=cfg.baseline.min_coverage,
        lesion_enabled=lesion_enabled,
        forehead_occluded_reason=forehead_reason,
    )

    # 조명 구배 교차검증 — 참조 부위가 2개 미만이면 '통과'가 아니라 '확인 불가'다.
    if baseline.n_ref_regions < cfg.baseline.min_ref_regions:
        illum = "unavailable"
    elif baseline.illumination_gradient_flag:
        illum = "flagged"
    else:
        illum = "ok"

    lab_all = color.linear_to_lab(linear)
    ita = float(np.median(color.ita_degrees(lab_all)[skin])) if skin.any() else None

    result = AnalysisResult(
        capture_id=capture_id,
        algo_version=ALGO_VERSION,
        config_hash=cfg_hash,
        quality=report,
        regions=regions,
        lesions=lesions,
        baseline=baseline,
        native_ipd_px=float(obs.native_ipd_px),
        iris_diameter_px=float(iris_px),
        mm_per_px=float(mm_per_px(iris_px)),
        ita_deg=ita,
        lesion_detection_enabled=lesion_enabled,
        lesion_reason=lesion_reason,
        illumination_check=illum,
        mask_reject_fraction=masks.reject_fraction,
        lesion_reject_count=lesion_rejects,
        capture=meta,
    )

    return Analysis(
        result=result,
        observation=obs,
        deviation=d,
        diffuse=diffuse,
        skin=skin,
        masks=masks,
    )


def lesions_to_original(analysis: Analysis) -> List[Dict[str, float]]:
    """병변 좌표를 원본 좌표계로 되돌린다.

    core/ 는 정규 프레임만 안다 (계약 3번). 원본 좌표는 여기서 만든다 —
    sample2 의 "분석 이미지 좌표로 오버레이를 그리면 어긋난다" 교훈.
    """
    if analysis.observation is None or not analysis.result.lesions:
        return []
    inv = face_mod.invert_affine(analysis.observation.affine)
    pts = np.array([[l.x, l.y] for l in analysis.result.lesions], dtype=np.float64)
    orig = face_mod.apply_affine(pts, inv)
    scale = float(np.sqrt(abs(np.linalg.det(inv[:, :2]))))
    return [
        {"x": float(o[0]), "y": float(o[1]),
         "radius_px": float(l.radius_px * scale),
         "peak_d": float(l.peak_d), "confidence": float(l.confidence)}
        for o, l in zip(orig, analysis.result.lesions)
    ]
