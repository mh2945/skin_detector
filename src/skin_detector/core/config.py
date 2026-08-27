"""모든 임계값의 단일 출처 (계약 5번).

여기 있는 기본값은 `config/default.toml` 과 **같아야 한다**.
toml 이 진짜 출처이고 이 dataclass 는 그것을 담는 그릇이다 —
`test/test_contract.py` 가 두 값이 어긋나지 않았는지 검사한다.

core/ 는 파일을 읽지 않는다 (계약 2번). toml 로딩은 `imageio.load_config()` 가 한다.
"""

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class QualityConfig:
    """§4.1 품질 게이트. 먼저 걸리는 게이트가 이긴다."""

    min_native_ipd_px: float = 90.0      # 이보다 작으면 분석 자체를 거부
    lesion_min_ipd_px: float = 150.0     # 이보다 작으면 병변만 끄고 미만성은 보고
    max_yaw_deg: float = 20.0
    max_pitch_deg: float = 20.0
    max_roll_deg: float = 25.0
    min_landmark_confidence: float = 0.5
    min_blur_vol: float = 60.0           # 정규 프레임에서의 Laplacian 분산.
                                         # 원본에서 재면 해상도 간 비교가 안 된다 — 흔한 버그.
    clip_level: float = 250.0 / 255.0    # 이 위는 포화 -> 비율 무의미
    dark_level: float = 25.0 / 255.0     # 이 아래는 로그 비율 분산 폭발
    max_clipped_fraction: float = 0.05
    max_dark_fraction: float = 0.30
    max_illum_ratio: float = 4.0         # P90(Y)/P10(Y). 세로 프레임은 상하 구배가 흔하다.
    min_face_in_frame: float = 0.90      # 정규 프레임 중 실제 원본에서 온 화소 비율


@dataclass
class MaskConfig:
    """§4.2-4.3 마스킹. 값은 전부 IPD 대비 비율이라 해상도에 자동 적응한다."""

    face_erode_frac: float = 0.03        # 머리카락/배경 누출 차단
    exclusion_dilate_frac: float = 0.02  # 눈·눈썹·입술 주변 여유
    edge_band_frac: float = 0.03         # 제외 폴리곤 경계 밴드 — DoG 가 스텝 에지를 좋아한다
    forehead_up_frac: float = 0.60       # 눈썹 위로 이 만큼까지 이마로 본다
    nose_half_width_frac: float = 0.20
    cheek_inner_gap_frac: float = 0.18
    mahalanobis_max: float = 3.0         # (e,m) 평면 색도 게이트. 하드코딩 피부색 박스를 쓰지 않는 이유.
    # 색도 분포의 최소 표준편차. **없으면 게이트가 병변 자체를 잘라낸다** —
    # 복색이 아주 균일한 얼굴에서는 적합된 가우시안이 임의로 좁아져 실제 홍반이
    # Mahalanobis 밖으로 밀려난다. 이 값 아래의 차이는 어차피 노이즈 수준이다.
    chroma_min_sigma: float = 0.030
    specular_mad_k: float = 3.0          # min(R,G,B) 가 이만큼 튀면 정반사로 보고 제외


@dataclass
class BaselineConfig:
    """§4.4 2단계 자기참조 기준선. 이 프로젝트에서 가장 민감한 값들."""

    percentile: float = 25.0             # b_face = P25(ERI). Phase 1 에서 P10/P25/P50 비교
    tau_k: float = 2.0                   # tau = k * sigma_face (MAD 기반)
    b_alt_delta_max: float = 0.020       # |b_face - b_alt| 가 이보다 크면 조명 구배 플래그
    min_ref_regions: int = 2             # b_alt 산출에 필요한 최소 참조 부위 수
    min_coverage: float = 0.50           # 이보다 낮으면 그 부위 점수는 null (0 아님)
    melanin_trim: float = 0.20           # robust 회귀 절사 비율
    melanin_iters: int = 5

    # ── 멜라닌 회귀 제약 (테스트로 발견한 두 가지 실패를 막는다) ──
    #
    # 1) e = D_G - D_R 과 m = D_B - D_G 는 **G 채널 노이즈를 공유**한다.
    #    G 의 노이즈가 e 에는 +1, m 에는 -1 로 들어가므로 순수 노이즈만으로
    #    beta ~= -0.5 의 가짜 기울기가 생긴다. 저역통과한 뒤 적합해 이걸 없앤다 —
    #    어차피 제거 대상인 멜라닌 구배는 저주파(태닝선·눈 주위 착색)다.
    #
    # 2) **멜라닌은 e 와 m 을 같은 방향으로 움직인다** (기울기 +0.33).
    #    헤모글로빈은 반대다 (-0.91). 따라서 **음수 beta 는 물리적으로 멜라닌일 수
    #    없고**, 자유 적합을 허용하면 회귀가 홍반을 멜라닌으로 오인해 빼버린다.
    #    범위를 물리적으로 가능한 구간에 가둔다.
    melanin_fit_sigma_frac: float = 0.08   # 적합 전 저역통과 sigma (IPD 대비)
    melanin_beta_min: float = 0.0
    melanin_beta_max: float = 0.6


@dataclass
class LesionConfig:
    """§2.4 · §4.5 병변 검출. sigma 는 정규 IPD 320 기준 px."""

    sigma_min_px: float = 3.6            # 2mm 구진 (1mm = 5.08px, r/sqrt(2))
    sigma_ratio: float = 1.36
    n_scales: int = 4                    # -> 3.6, 4.9, 6.6, 9.0
    peak_min_d: float = 0.012            # 이 아래 피크는 노이즈로 본다
    nms_radius_frac: float = 0.025       # IPD 대비 비최대 억제 반경
    anisotropy_max: float = 3.0          # 구조 텐서 l1/l2. 구진은 등방성, 수염은 길쭉하다.
    melanin_reject_ratio: float = 1.0    # **부호 있는** dm 이 이 배수를 넘으면 모반 -> 기각.
                                         # 헤모글로빈은 dm<0 이라 절대 안 걸린다.
    darkness_reject_dl: float = -8.0     # 국소 dL* 가 이보다 어두우면 색소 병변
    max_lesions: int = 200


@dataclass
class ScoreConfig:
    """§4.4 region 점수 + §7.2 2단계 판정."""

    lowpass_sigma_frac: float = 0.25     # 미만성 저역통과 sigma (IPD 대비)
    w_area: float = 0.6
    w_intensity: float = 0.4
    area_x0: float = 0.15                # norm(x) = 1 - exp(-x/x0)
    intensity_x0: float = 0.030
    erythema_area_threshold: float = 0.12   # 이 이상이면 그 부위는 '붉음'
    lesion_count_threshold: int = 1         # 이 이상이면 그 부위는 '트러블 있음'


@dataclass
class Config:
    """전체 설정. `imageio.load_config()` 가 toml 을 얹어 만든다."""

    quality: QualityConfig = None
    mask: MaskConfig = None
    baseline: BaselineConfig = None
    lesion: LesionConfig = None
    score: ScoreConfig = None

    def __post_init__(self) -> None:
        if self.quality is None:
            self.quality = QualityConfig()
        if self.mask is None:
            self.mask = MaskConfig()
        if self.baseline is None:
            self.baseline = BaselineConfig()
        if self.lesion is None:
            self.lesion = LesionConfig()
        if self.score is None:
            self.score = ScoreConfig()

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """config_hash 산출과 리포트 기록용 (§5.4 ②)."""
        out: Dict[str, Dict[str, Any]] = {}
        for f in fields(self):
            section = getattr(self, f.name)
            out[f.name] = {sf.name: getattr(section, sf.name)
                           for sf in fields(section)}
        return out


_SECTIONS = {
    "quality": QualityConfig,
    "mask": MaskConfig,
    "baseline": BaselineConfig,
    "lesion": LesionConfig,
    "score": ScoreConfig,
}


def from_dict(data: Dict[str, Any]) -> Config:
    """toml 에서 읽은 dict 를 Config 로. 모르는 키는 조용히 넘기지 않고 막는다.

    오타 난 임계값이 조용히 무시되면 튜닝 결과를 신뢰할 수 없게 된다.
    """
    kwargs = {}
    for name, cls in _SECTIONS.items():
        raw = data.get(name, {}) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                "config [{}] 에 알 수 없는 키: {}".format(name, sorted(unknown)))
        kwargs[name] = cls(**raw)
    unknown_sections = set(data) - set(_SECTIONS)
    if unknown_sections:
        raise ValueError("config 에 알 수 없는 섹션: {}".format(sorted(unknown_sections)))
    return Config(**kwargs)
