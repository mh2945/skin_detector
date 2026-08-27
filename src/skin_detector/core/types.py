"""공용 자료형 · 상수 · 사유 코드.

이 모듈은 numpy 조차 import 하지 않는다 (stdlib 만).
계약: docs/CONTRACT.md 1번.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ── 정규 프레임 규격 (계획 §2.3) ──────────────────────────────────────
# core/ 안의 모든 좌표는 예외 없이 이 프레임 기준이다. 계약 3번.
CANONICAL_SIZE = 768        # 정규 프레임 한 변 (px)
CANONICAL_IPD = 320.0       # 정규 프레임에서의 동공 간 거리 (px)

# 해부학 상수 — 개인차가 거의 없어 물리적 자(ruler)로 쓸 수 있다.
IRIS_DIAMETER_MM = 11.7     # 성인 홍채 지름. IPD(58~70mm)보다 훨씬 안정적이다.
TYPICAL_IPD_MM = 63.0       # mm 환산 실패 시의 폴백에만 쓴다.


def mm_per_px(iris_diameter_px: float) -> float:
    """정규 프레임에서의 mm/px. 홍채를 자로 쓴다 (계획 §2.4)."""
    if iris_diameter_px <= 0.0:
        return TYPICAL_IPD_MM / CANONICAL_IPD
    return IRIS_DIAMETER_MM / iris_diameter_px


class RegionId(str, Enum):
    """§7.2 의 관측 단위. 5부위 고정 — 늘리려면 평가 프로토콜부터 바꿔야 한다."""

    FOREHEAD = "forehead"
    CHEEK_L = "cheek_l"
    CHEEK_R = "cheek_r"
    NOSE = "nose"
    CHIN = "chin"


ALL_REGIONS: Tuple[RegionId, ...] = (
    RegionId.FOREHEAD,
    RegionId.CHEEK_L,
    RegionId.CHEEK_R,
    RegionId.NOSE,
    RegionId.CHIN,
)

# b_alt (조명 구배 교차검증) 참조 부위 우선순위 — 계획 §4.4.
# 이마 하나에 의존하면 앞머리에 가려 상시 null 이 되어 게이트가 조용히 무력화된다.
BASELINE_REF_REGIONS: Tuple[RegionId, ...] = (
    RegionId.FOREHEAD,
    RegionId.CHIN,
    RegionId.CHEEK_L,
    RegionId.CHEEK_R,
)


class ReasonCode(str, Enum):
    """게이트가 막은 이유. **먼저 걸린 것이 이긴다** (계획 §4.1).

    값은 그대로 JSON 에 나가고 web/ 의 안내 문장 테이블 키가 된다 (§5.3).
    """

    OK = "ok"

    # ── 얼굴 자체 ──
    NO_FACE = "no_face"
    MULTIPLE_FACES = "multiple_faces"
    LOW_LANDMARK_CONFIDENCE = "low_landmark_confidence"

    # ── 기하 ──
    POSE_OUT_OF_RANGE = "pose_out_of_range"
    LOW_RESOLUTION = "low_resolution"
    FACE_OUT_OF_FRAME = "face_out_of_frame"

    # ── 광학 ──
    BLURRY = "blurry"
    OVEREXPOSED = "overexposed"
    UNDEREXPOSED = "underexposed"
    NON_UNIFORM_ILLUMINATION = "non_uniform_illumination"

    # ── 부위 단위 (전역 실패가 아니라 그 부위만 null) ──
    FOREHEAD_OCCLUDED = "forehead_occluded"
    LOW_COVERAGE = "low_coverage"

    # ── 파이프라인 상태 ──
    ILLUMINATION_CHECK_UNAVAILABLE = "illumination_check_unavailable"
    LESION_DISABLED_LOW_RESOLUTION = "lesion_disabled_low_resolution"


# 전역 게이트: 걸리면 이미지 전체를 판정 불가로 만든다.
# 부위 단위 사유는 여기 없다 — 그건 해당 부위만 null 로 만든다.
FATAL_REASONS = frozenset({
    ReasonCode.NO_FACE,
    ReasonCode.MULTIPLE_FACES,
    ReasonCode.LOW_LANDMARK_CONFIDENCE,
    ReasonCode.POSE_OUT_OF_RANGE,
    ReasonCode.LOW_RESOLUTION,
    ReasonCode.FACE_OUT_OF_FRAME,
    ReasonCode.BLURRY,
    ReasonCode.OVEREXPOSED,
    ReasonCode.UNDEREXPOSED,
    ReasonCode.NON_UNIFORM_ILLUMINATION,
})


class Verdict(str, Enum):
    """부위별 2단계 판정 — §7.2 의 붉은기 축 / 트러블 축.

    NOT_MEASURED 는 0 이 아니다. 계약 4번.
    """

    NORMAL = "normal"
    AFFECTED = "affected"
    NOT_MEASURED = "not_measured"


@dataclass
class QualityReport:
    """게이트 체인 결과. 먼저 걸린 사유 하나만 담는다."""

    passed: bool
    reason: ReasonCode
    # 게이트를 통과했든 아니든 측정된 값은 전부 남긴다 — 튜닝과 디버깅의 근거.
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class RegionScore:
    """부위 하나의 측정 결과.

    측정 실패는 `None` 이다. `0.0` 이 아니다 — 계약 4번.
    """

    region: RegionId
    coverage: float                      # 유효 피부 화소 비율 [0,1]
    reason: ReasonCode = ReasonCode.OK

    # 물리적으로 의미 있는 값 (log10 비율). 이것이 "진짜 측정"이다.
    area_fraction: Optional[float] = None   # d > tau 인 화소 비율
    intensity: Optional[float] = None       # mean(d | d > tau)
    median_d: Optional[float] = None        # 부위 중앙 편차

    # 순서형 편의 척도 — 물리량이 아니다. 필드명이 그걸 강제한다.
    score_ordinal_0_100: Optional[float] = None

    # §7.2 의 두 축
    erythema_verdict: Verdict = Verdict.NOT_MEASURED
    lesion_verdict: Verdict = Verdict.NOT_MEASURED

    lesion_count: Optional[int] = None


@dataclass
class Lesion:
    """개별 병변 하나. 좌표는 정규 프레임 기준이다 (계약 3번).

    원본 좌표 역변환은 render/report 단계에서만 한다.
    """

    x: float
    y: float
    radius_px: float
    radius_mm: Optional[float]
    peak_d: float
    region: Optional[RegionId]
    confidence: float


@dataclass
class BaselineInfo:
    """§4.4 2단계 기준선의 산출 내역. 왜 그 점수가 나왔는지의 근거."""

    b_face: float                          # Level 1 전역 오프셋
    sigma_face: float                      # 1.4826 * MAD
    tau: float                             # k * sigma_face
    beta: float                            # 멜라닌 회귀 기울기
    alpha: float                           # 멜라닌 회귀 절편
    b_alt: Optional[float] = None          # 참조 부위 교차검증값
    n_ref_regions: int = 0                 # b_alt 에 기여한 부위 수 (2 미만이면 unavailable)
    illumination_gradient_flag: bool = False
    anatomical_prior_calibrated: bool = False


@dataclass
class CaptureMeta:
    """촬영 시점의 사실. `data/store/<id>/capture.json` 에 그대로 들어간다 (§5.4).

    **한 번 쓰면 수정하지 않는다** (계약 10번). 알고리즘이 바뀌어도 이건 안 바뀐다.

    `awb_gains` 는 대개 `None` 이다 — 수동 카메라 앱이 채널별 게인을 EXIF 에 온전히
    남기지 않기 때문이다 (계획 §3.1). 모르는 것을 0 으로 채우지 않는다.
    """

    capture_id: str = ""
    capture_path: str = "upload"           # "browser" | "upload" — 절대 섞지 않는다
    source_name: str = ""
    width: int = 0
    height: int = 0

    # EXIF 에서 읽히는 것
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    iso: Optional[float] = None
    exposure_time_s: Optional[float] = None
    focal_length_mm: Optional[float] = None
    orientation: Optional[int] = None
    datetime_original: Optional[str] = None

    # 파일명 규약에서 파싱하는 것 (§3.1). 실패하면 None + 경고.
    session: Optional[str] = None
    lighting: Optional[str] = None
    wb_kelvin: Optional[int] = None
    torch: Optional[bool] = None
    shot_index: Optional[int] = None
    filename_parsed: bool = False

    # 채널별 WB 게인. 수동 앱에서는 거의 항상 None 이다.
    awb_gains: Optional[List[float]] = None

    # 확장 여지만 열어둔다 (§5.4). 본인 외 데이터가 들어오면 정책이 선행 조건.
    subject_id: str = "self"
    consent: str = "n/a"

    ingested_at: Optional[str] = None


@dataclass
class AnalysisResult:
    """이미지 한 장의 분석 결과 전체. `results/<version>_<hash>.json` 이 된다.

    `algo_version` + `config_hash` 가 없으면 **어떤 설정에서 나온 숫자인지 알 수 없어
    수백 번의 튜닝 이력이 통째로 무의미해진다** (§5.4 ②).
    """

    capture_id: str
    algo_version: str
    config_hash: str

    quality: QualityReport
    regions: List[RegionScore] = field(default_factory=list)
    lesions: List[Lesion] = field(default_factory=list)
    baseline: Optional[BaselineInfo] = None

    native_ipd_px: Optional[float] = None
    iris_diameter_px: Optional[float] = None
    mm_per_px: Optional[float] = None
    ita_deg: Optional[float] = None            # 피부톤 계층화 — 조건부 해석용

    lesion_detection_enabled: bool = False
    lesion_reason: ReasonCode = ReasonCode.OK
    illumination_check: str = "ok"             # ok | flagged | unavailable

    mask_reject_fraction: Dict[str, float] = field(default_factory=dict)
    lesion_reject_count: Dict[str, int] = field(default_factory=dict)

    capture: Optional[CaptureMeta] = None

    def region(self, rid: RegionId) -> Optional[RegionScore]:
        for r in self.regions:
            if r.region == rid:
                return r
        return None
