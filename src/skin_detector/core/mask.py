"""478 랜드마크 인덱스 테이블 · 폴리곤 · 피부/정반사/그림자 게이트.

계약: numpy + stdlib 만.
좌표는 전부 정규 프레임(768x768, IPD=320) 기준이다 (계약 3번).

**좌/우 규약은 이 파일 한 곳에서만 정한다** (sample2 의 점수 방향 규약 교훈):
MediaPipe 의 left/right 는 **피사체 기준**이다. 정면 사진에서 피사체의 오른쪽은
이미지의 **왼쪽(작은 x)** 에 나타난다. `RegionId.CHEEK_R` = 피사체의 오른쪽 볼 =
이미지 왼쪽. 이 문장을 뒤집지 말 것 — 확인된 사실이지 추측이 아니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import MaskConfig
from .types import ALL_REGIONS, CANONICAL_IPD, RegionId

# ── 랜드마크 인덱스 (MediaPipe FaceLandmarker 478점) ──────────────────
# mediapipe.solutions 가 1.0.x 에서 제거됐으므로 이 테이블은 우리가 소유한다.
# Kotlin/Swift 포팅 때도 같은 숫자를 그대로 쓴다 (계획 §3).

FACE_OVAL: Tuple[int, ...] = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)

# 피사체 왼쪽 눈 (이미지 오른쪽)
EYE_L: Tuple[int, ...] = (
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
)
# 피사체 오른쪽 눈 (이미지 왼쪽)
EYE_R: Tuple[int, ...] = (
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
)

# 눈썹: 아래선을 먼저, 위선을 역순으로 이어 폴리곤이 되게 배열했다.
BROW_L: Tuple[int, ...] = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
BROW_R: Tuple[int, ...] = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)

LIPS_OUTER: Tuple[int, ...] = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    409, 270, 269, 267, 0, 37, 39, 40, 185,
)

# 홍채 — IPD 와 mm 환산이 여기서 나온다. MediaPipe 가 대체 불가인 진짜 이유.
IRIS_R_CENTER = 468          # 피사체 오른쪽
IRIS_R_RING: Tuple[int, ...] = (469, 470, 471, 472)
IRIS_L_CENTER = 473          # 피사체 왼쪽
IRIS_L_RING: Tuple[int, ...] = (474, 475, 476, 477)

# 기하 기준점
NOSE_TIP = 1
NOSE_BASE = 2
NOSTRIL_R = 98
NOSTRIL_L = 327
MOUTH_CORNER_R = 61
MOUTH_CORNER_L = 291
CHIN_BOTTOM = 152

N_LANDMARKS = 478


# ── 폴리곤 · 형태학 (pure numpy, C++ 로 그대로 옮겨쓸 수 있는 형태) ────

def polygon_mask(poly_xy: np.ndarray, h: int, w: int) -> np.ndarray:
    """다각형 내부 마스크. 교차수(crossing number) 판정.

    반복문은 정점 개수(<=36)만큼만 돌고 픽셀은 전부 벡터 연산이다 — 계약 6번.
    """
    poly = np.asarray(poly_xy, dtype=np.float64)
    n = poly.shape[0]
    yy = np.arange(h, dtype=np.float64)[:, None]
    xx = np.arange(w, dtype=np.float64)[None, :]
    inside = np.zeros((h, w), dtype=bool)

    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if y0 == y1:
            continue
        straddles = (y0 > yy) != (y1 > yy)
        x_cross = (x1 - x0) * (yy - y0) / (y1 - y0) + x0
        inside ^= straddles & (xx < x_cross)
    return inside


def _shift(a: np.ndarray, dy: int, dx: int, fill: bool) -> np.ndarray:
    """내용을 (dy, dx) 만큼 옮기고 빈 곳을 fill 로 채운다."""
    h, w = a.shape
    out = np.full((h, w), fill, dtype=a.dtype)
    ys0, ys1 = max(0, -dy), h - max(0, dy)
    yd0, yd1 = max(0, dy), h - max(0, -dy)
    xs0, xs1 = max(0, -dx), w - max(0, dx)
    xd0, xd1 = max(0, dx), w - max(0, -dx)
    if ys1 > ys0 and xs1 > xs0:
        out[yd0:yd1, xd0:xd1] = a[ys0:ys1, xs0:xs1]
    return out


_NEIGHBORS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if not (dy == 0 and dx == 0))


def binary_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    """3x3 팽창 반복. 원형이 아닌 팔각형 근사지만 C++ 이식이 자명하다."""
    out = mask
    for _ in range(max(0, int(iterations))):
        acc = out.copy()
        for dy, dx in _NEIGHBORS:
            acc |= _shift(out, dy, dx, False)
        out = acc
    return out


def binary_erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    """3x3 침식 반복.

    이미지 경계 밖은 `True` 로 채운다 — 프레임 가장자리 때문에 마스크가 깎이면 안 된다.
    진짜 프레임 밖(9:16 크롭이 넘어간 영역)은 `frame_valid` 가 따로 처리한다.
    """
    out = mask
    for _ in range(max(0, int(iterations))):
        acc = out.copy()
        for dy, dx in _NEIGHBORS:
            acc &= _shift(out, dy, dx, True)
        out = acc
    return out


def _frac_to_iters(frac: float) -> int:
    """IPD 대비 비율 -> 3x3 반복 횟수."""
    return int(round(max(0.0, frac) * CANONICAL_IPD))


def _poly(lm: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    return lm[list(idx), :2]


def _disc(cx: float, cy: float, r: float, h: int, w: int) -> np.ndarray:
    yy = np.arange(h, dtype=np.float64)[:, None]
    xx = np.arange(w, dtype=np.float64)[None, :]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * r)


# ── 색도 게이트 (얼굴 자체에서 학습한다) ──────────────────────────────

def chromaticity_gate(
    e: np.ndarray,
    m: np.ndarray,
    seed: np.ndarray,
    max_maha: float,
    min_sigma: float = 0.030,
    n_iter: int = 2,
) -> np.ndarray:
    """(e, m) 평면 2-D 가우시안 적합 -> Mahalanobis 거리 게이트.

    **하드코딩 YCbCr 피부색 박스를 쓰지 않는 이유**: 피부톤이 조금만 벗어나면
    무너진다. 얼굴 자체에서 분포를 학습하면 모든 피부톤에서 자동 동작한다.
    머리카락·수염·배경은 이 분포에서 크게 벗어나므로 함께 걸린다.

    **공분산에 바닥을 깔지 않으면 이 게이트가 병변 자체를 잘라낸다.** 복색이 아주
    균일한 얼굴에서는 적합된 가우시안이 임의로 좁아져 실제 홍반이 분포 밖으로
    밀려난다. `min_sigma` 아래의 색도 차이는 어차피 노이즈 수준이다.

    Args:
        seed: 고신뢰 피부 화소 (볼 중앙). 여기서 분포를 추정한다.
        min_sigma: 각 축 표준편차의 하한.
    Returns:
        bool 마스크 — True 면 피부 분포 안쪽.
    """
    pts = np.stack([e, m], axis=-1)
    floor = np.eye(2) * (min_sigma * min_sigma)
    sel = seed
    mu = np.zeros(2)
    cov = np.eye(2)

    for _ in range(max(1, n_iter)):
        sample = pts[sel]
        if sample.shape[0] < 64:
            break
        mu = sample.mean(axis=0)
        cov = np.cov(sample, rowvar=False) + floor
        inv = np.linalg.inv(cov)
        d = pts - mu
        maha2 = np.einsum("...i,ij,...j->...", d, inv, d)
        sel = seed & (maha2 <= max_maha * max_maha)

    inv = np.linalg.inv(cov)
    d = pts - mu
    maha2 = np.einsum("...i,ij,...j->...", d, inv, d)
    return maha2 <= (max_maha * max_maha)


def specular_gate(min_channel: np.ndarray, face: np.ndarray, k: float) -> np.ndarray:
    """정반사 제외 마스크. True = 정반사 아님 (통과).

    min(R,G,B) 의 얼굴 내 robust 분포에서 위로 튀는 화소를 제외한다.
    자기참조라 피부톤·노출에 자동 적응한다.
    """
    vals = min_channel[face]
    if vals.size < 64:
        return np.ones_like(min_channel, dtype=bool)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad
    if sigma <= 1e-9:
        return np.ones_like(min_channel, dtype=bool)
    return min_channel <= (med + k * sigma)


# ── 결과 묶음 ─────────────────────────────────────────────────────────

@dataclass
class MaskSet:
    """마스킹 결과 일체."""

    skin: np.ndarray                          # 유효 피부 (모든 게이트 통과)
    face: np.ndarray                          # 얼굴 폴리곤 (침식 후, 게이트 전)
    regions: Dict[RegionId, np.ndarray]       # 부위별 유효 피부
    edge_band: np.ndarray                     # 제외 폴리곤 경계 밴드 — 병변 기각용
    seed: np.ndarray                          # 색도 분포 추정에 쓴 고신뢰 패치
    geo_area: Dict[RegionId, int] = field(default_factory=dict)  # 게이트 전 기하 면적
    reject_fraction: Dict[str, float] = field(default_factory=dict)

    def coverage(self, region: RegionId) -> float:
        """부위의 유효 화소 비율. 분모는 **게이트 전** 기하 면적이다.

        분모를 게이트 후로 잡으면 다 걸러진 부위도 coverage 1.0 이 되어
        "측정 못 함"이 "정상"으로 둔갑한다.
        """
        valid = self.regions.get(region)
        if valid is None:
            return 0.0
        denom = int(self.geo_area.get(region, 0))
        if denom <= 0:
            return 0.0
        return float(int(valid.sum())) / float(denom)


def build_masks(
    landmarks: np.ndarray,
    linear: np.ndarray,
    e: np.ndarray,
    m: np.ndarray,
    frame_valid: np.ndarray,
    cfg: MaskConfig,
    clip_level: float,
    dark_level: float,
    face_poly: Optional[np.ndarray] = None,
) -> MaskSet:
    """랜드마크 + 색 -> 부위별 유효 피부 마스크.

    순서가 중요하다. 배경 화소가 통계를 오염시키기 전에 기하로 먼저 자르고,
    그 다음 색·정반사로 거른다. 세로 사진에서 얼굴은 프레임의 20~35% 뿐이다.
    """
    h, w = e.shape
    lm = np.asarray(landmarks, dtype=np.float64)

    # 1) 얼굴 폴리곤 -> 침식 (머리카락/배경 누출 차단)
    if face_poly is None:
        face_poly = polygon_mask(_poly(lm, FACE_OVAL), h, w)
    face = binary_erode(face_poly, _frac_to_iters(cfg.face_erode_frac))

    # 2) 제외 영역: 눈 · 눈썹 · 입술 · 콧구멍
    dil = _frac_to_iters(cfg.exclusion_dilate_frac)
    exclude = np.zeros((h, w), dtype=bool)
    for idx in (EYE_L, EYE_R, BROW_L, BROW_R, LIPS_OUTER):
        exclude |= polygon_mask(_poly(lm, idx), h, w)
    exclude = binary_dilate(exclude, dil)

    nostril_r = 0.035 * CANONICAL_IPD
    for ni in (NOSTRIL_R, NOSTRIL_L):
        exclude |= _disc(lm[ni, 0], lm[ni, 1], nostril_r, h, w)

    # DoG 는 강한 스텝 에지를 제일 좋아한다 -> 경계 밴드를 따로 들고 있다가 병변에서 기각
    edge_band = binary_dilate(exclude, _frac_to_iters(cfg.edge_band_frac)) & ~exclude

    base = face & ~exclude & frame_valid

    # 3) 광학 게이트
    max_ch = np.max(linear, axis=-1)
    min_ch = np.min(linear, axis=-1)
    not_clipped = max_ch < clip_level          # 포화 -> 비율 무의미
    not_dark = max_ch >= dark_level            # 로그 비율 분산 폭발 (콧구멍도 여기서 걸린다)
    not_specular = specular_gate(min_ch, base, cfg.specular_mad_k)

    optical_ok = base & not_clipped & not_dark & not_specular

    # 4) 기하학적 부위 분할
    geo = _region_bands(lm, h, w, cfg)
    for r in geo:
        geo[r] &= face_poly                    # 부위는 얼굴 밖으로 나가지 않는다

    # 5) 색도 게이트 — 볼 중앙 고신뢰 패치에서 분포를 학습
    seed = (geo[RegionId.CHEEK_L] | geo[RegionId.CHEEK_R]) & optical_ok
    seed = binary_erode(seed, _frac_to_iters(0.02))
    if int(seed.sum()) < 256:
        seed = optical_ok                      # 볼이 안 잡히면 얼굴 전체로 후퇴
    is_skin = chromaticity_gate(e, m, seed, cfg.mahalanobis_max, cfg.chroma_min_sigma)

    skin = optical_ok & is_skin

    regions = {r: (geo[r] & skin) for r in ALL_REGIONS}
    geo_area = {r: int(geo[r].sum()) for r in ALL_REGIONS}

    denom = float(max(1, int(base.sum())))
    rejects = {
        "clipped": float(int((base & ~not_clipped).sum())) / denom,
        "dark": float(int((base & ~not_dark).sum())) / denom,
        "specular": float(int((base & ~not_specular).sum())) / denom,
        "non_skin_chroma": float(int((optical_ok & ~is_skin).sum())) / denom,
    }

    return MaskSet(
        skin=skin,
        face=face,
        regions=regions,
        edge_band=edge_band,
        seed=seed,
        geo_area=geo_area,
        reject_fraction=rejects,
    )


def _region_bands(
    lm: np.ndarray, h: int, w: int, cfg: MaskConfig
) -> Dict[RegionId, np.ndarray]:
    """정규 프레임의 기하로 5부위를 나눈다.

    프레임이 IPD 정규화 + roll 보정돼 있으므로 기하 구성이 안정적이다.
    랜드마크 윤곽을 부위마다 손으로 이어 붙이는 것보다 인덱스 오류에 강하다.
    """
    ipd = CANONICAL_IPD
    yy = np.arange(h, dtype=np.float64)[:, None]
    xx = np.arange(w, dtype=np.float64)[None, :]

    eye_y = 0.5 * (lm[IRIS_R_CENTER, 1] + lm[IRIS_L_CENTER, 1])
    nose_x = lm[NOSE_TIP, 0]
    nose_base_y = lm[NOSE_BASE, 1]
    brow_y = float(np.min(lm[list(BROW_L) + list(BROW_R), 1]))
    mouth_y = 0.5 * (lm[MOUTH_CORNER_R, 1] + lm[MOUTH_CORNER_L, 1])
    chin_y = lm[CHIN_BOTTOM, 1]

    inner = cfg.cheek_inner_gap_frac * ipd
    half_nose = cfg.nose_half_width_frac * ipd

    forehead = (yy < brow_y - 0.05 * ipd) & (yy > brow_y - cfg.forehead_up_frac * ipd)
    nose = (np.abs(xx - nose_x) < half_nose) & (yy > eye_y + 0.05 * ipd) & (yy < nose_base_y)

    cheek_y = (yy > eye_y + 0.12 * ipd) & (yy < mouth_y - 0.03 * ipd)
    cheek_r = cheek_y & (xx < nose_x - inner)     # 피사체 오른쪽 = 이미지 왼쪽
    cheek_l = cheek_y & (xx > nose_x + inner)

    chin = ((yy > mouth_y + 0.08 * ipd) & (yy < chin_y)
            & (np.abs(xx - nose_x) < 0.35 * ipd))

    bands = {
        RegionId.FOREHEAD: np.broadcast_to(forehead, (h, w)).copy(),
        RegionId.NOSE: np.broadcast_to(nose, (h, w)).copy(),
        RegionId.CHEEK_R: np.broadcast_to(cheek_r, (h, w)).copy(),
        RegionId.CHEEK_L: np.broadcast_to(cheek_l, (h, w)).copy(),
        RegionId.CHIN: np.broadcast_to(chin, (h, w)).copy(),
    }

    # **부위는 서로 배타적이어야 한다.** 겹치면 화소가 두 번 세어지고 부위별
    # 코호트 통계(mu_r)와 §7.2 부위-이미지 쌍 카운트가 조용히 틀어진다.
    #
    # 상수만 맞춰 두면(예: cheek_inner_gap >= nose_half_width) toml 튜닝 한 번에
    # 다시 깨진다. 우선순위대로 이미 배정된 화소를 빼서 **구조적으로** 막는다.
    # 순서: 코(가장 좁고 해부학적으로 뚜렷) -> 이마 -> 턱 -> 볼.
    assigned = np.zeros((h, w), dtype=bool)
    for rid in (RegionId.NOSE, RegionId.FOREHEAD, RegionId.CHIN,
                RegionId.CHEEK_R, RegionId.CHEEK_L):
        bands[rid] &= ~assigned
        assigned |= bands[rid]
    return bands


def iris_diameter_px(landmarks: np.ndarray) -> float:
    """정규 프레임에서의 홍채 지름 (양안 평균). mm 환산의 자.

    성인 홍채는 11.7 +- 0.5mm 로 개인차가 거의 없다 (IPD 는 58~70mm).
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    diams: List[float] = []
    for center, ring in ((IRIS_R_CENTER, IRIS_R_RING), (IRIS_L_CENTER, IRIS_L_RING)):
        c = lm[center, :2]
        r = np.linalg.norm(lm[list(ring), :2] - c[None, :], axis=1).mean()
        diams.append(2.0 * float(r))
    return float(np.mean(diams)) if diams else 0.0
