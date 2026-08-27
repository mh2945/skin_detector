"""2단계 자기참조 기준선 · 미만성/병변 대역 분리 · DoG 검출 · 점수화.

계약: numpy + stdlib 만.

가장 자주 틀리는 지점 두 개를 여기서 구조적으로 막는다.
1. 저역통과는 **반드시 normalized convolution** 이다. 마스크된 화소가 값을 0 쪽으로
   끌어내리면 부위 경계가 전부 가짜 저홍반이 된다.
2. 측정 실패는 `None` 이다. `0.0` 이 아니다 (계약 4번).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .color import robust_line_fit
from .config import BaselineConfig, LesionConfig, ScoreConfig
from .types import (
    ALL_REGIONS,
    BASELINE_REF_REGIONS,
    CANONICAL_IPD,
    BaselineInfo,
    Lesion,
    ReasonCode,
    RegionId,
    RegionScore,
    Verdict,
    mm_per_px,
)


# ── 흐림 연산 (박스 3회 = 가우시안 근사, cumsum 이라 O(N)) ─────────────

def _boxes_for_gauss(sigma: float, n: int = 3) -> List[int]:
    """주어진 sigma 를 n 회 박스 필터로 근사할 때의 창 너비들 (Kovesi).

    3회 통과면 중심극한정리로 가우시안 모양에 충분히 가깝다. 다만 창 너비가 정수라
    **실효 sigma 는 목표에서 최대 ~4% 어긋난다** (측정: 3.6->3.46, 9.0->8.83, 80->83.1).
    스케일 간 간격이 1.36 배라 스케일 선택에는 영향이 없고, 병변 크기 추정치에만
    ~4% 편향으로 들어온다. `test_measure.py` 가 이 오차 상한을 고정한다.

    FFT 가 아니라 이걸 고른 이유는 C++ 이식이 자명하기 때문이다 (계약 6번).
    """
    if sigma <= 0.0:
        return [1] * n
    w_ideal = math.sqrt(12.0 * sigma * sigma / n + 1.0)
    wl = int(math.floor(w_ideal))
    if wl % 2 == 0:
        wl -= 1
    wl = max(1, wl)
    wu = wl + 2
    denom = (-4.0 * wl - 4.0)
    m_ideal = (12.0 * sigma * sigma - n * wl * wl - 4.0 * n * wl - 3.0 * n) / denom
    m = int(round(m_ideal))
    m = max(0, min(n, m))
    return [wl if i < m else wu for i in range(n)]


def _box_sum_1d(a: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """누적합 기반 박스 합. 경계는 잘린 창을 그대로 쓴다.

    잘림은 normalized convolution 의 분자/분모에서 **정확히 상쇄**되므로
    따로 보정하지 않는다.
    """
    if radius <= 0:
        return a
    n = a.shape[axis]
    pad_shape = list(a.shape)
    pad_shape[axis] = 1
    zero = np.zeros(pad_shape, dtype=np.float64)
    c = np.concatenate([zero, np.cumsum(a, axis=axis, dtype=np.float64)], axis=axis)
    idx = np.arange(n)
    hi = np.minimum(idx + radius + 1, n)
    lo = np.maximum(idx - radius, 0)
    return np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)


def _smooth(a: np.ndarray, sigma: float) -> np.ndarray:
    """분리형 박스 3회 통과. 정규화하지 않은 합을 돌려준다."""
    out = a.astype(np.float64)
    for w in _boxes_for_gauss(sigma):
        r = (w - 1) // 2
        if r <= 0:
            continue
        out = _box_sum_1d(out, r, axis=0)
        out = _box_sum_1d(out, r, axis=1)
    return out


def normalized_blur(
    values: np.ndarray, valid: np.ndarray, sigma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """마스크를 존중하는 저역통과: blur(v * valid) / blur(valid).

    **그냥 blur(v) 를 쓰면 마스크된 화소(0)가 이웃 값을 끌어내려** 눈·입술·머리카락
    주변이 전부 가짜 저홍반이 된다. 계획 §1.4 에서 "자주 틀리는 지점"으로 표시한 곳.

    Returns:
        (평활값, 유효 마스크). 유효하지 않은 곳의 값은 0 이지만 마스크로 구분된다.
    """
    vf = valid.astype(np.float64)
    num = _smooth(np.where(valid, values, 0.0), sigma)
    den = _smooth(vf, sigma)
    ok = den > 1e-6
    out = np.zeros(values.shape, dtype=np.float32)
    np.divide(num, den, out=out, where=ok, casting="unsafe")
    return out, ok


# ── 가중 통계 ─────────────────────────────────────────────────────────

def weighted_percentile(
    values: np.ndarray, weights: np.ndarray, q: float
) -> float:
    """가중 백분위수. q 는 0~100."""
    v = np.asarray(values, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if v.size == 0:
        return float("nan")
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cw = np.cumsum(w)
    total = cw[-1]
    if total <= 0:
        return float(np.percentile(v, q))
    target = (q / 100.0) * total
    i = int(np.searchsorted(cw, target, side="left"))
    return float(v[min(i, v.size - 1)])


def robust_sigma(values: np.ndarray) -> float:
    """1.4826 * MAD. 얼굴 자체의 산포를 단위로 삼기 위한 값."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size < 8:
        return 0.0
    med = float(np.median(v))
    return 1.4826 * float(np.median(np.abs(v - med)))


# ── 1) ERI 산출 (얼굴별 멜라닌 회귀 제거) ─────────────────────────────

def fit_eri(
    e: np.ndarray,
    m: np.ndarray,
    skin: np.ndarray,
    weights: np.ndarray,
    cfg: BaselineConfig,
) -> Tuple[np.ndarray, float, float]:
    """피부 마스크 위에서 alpha, beta 를 적합하고 ERI 맵을 만든다.

    자유 적합을 그대로 쓰면 **두 가지 방식으로 홍반 신호를 잃는다.**

    1. `e = D_G - D_R` 와 `m = D_B - D_G` 는 G 채널 노이즈를 공유한다. G 의 노이즈가
       e 에는 +1, m 에는 -1 로 들어가므로 **순수 노이즈만으로 beta ~= -0.5** 가 적합된다.
       -> 적합 전에 저역통과한다. 제거 대상인 멜라닌 구배는 원래 저주파다.
    2. **멜라닌은 e 와 m 을 같은 방향으로 움직인다** (기울기 약 +0.33). 헤모글로빈은
       반대다 (약 -0.91). 음수 beta 는 물리적으로 멜라닌일 수 없으므로, 허용하면
       회귀가 홍반을 "멜라닌"으로 오인해 통째로 빼버린다.
       -> beta 를 물리적으로 가능한 구간으로 클램프한다.

    두 보정 모두 테스트로 발견한 실패를 막기 위한 것이고,
    `test_measure.py::test_synthetic_erythema_recovery_is_linear` 가 고정한다.
    """
    sigma = cfg.melanin_fit_sigma_frac * CANONICAL_IPD
    e_lp, _ = normalized_blur(e, skin, sigma)
    m_lp, _ = normalized_blur(m, skin, sigma)

    alpha, beta = robust_line_fit(
        m_lp[skin], e_lp[skin], weights[skin],
        n_iter=cfg.melanin_iters, trim=cfg.melanin_trim,
    )
    beta = float(min(max(beta, cfg.melanin_beta_min), cfg.melanin_beta_max))

    # alpha 는 클램프된 beta 에 맞춰 다시 잡는다. 그렇지 않으면 전역 오프셋이 어긋난다.
    # (어차피 b_face 가 오프셋을 다시 빼지만, ERI 자체를 보고할 때 일관성이 필요하다.)
    alpha = float(np.median(e[skin] - beta * m[skin]))

    eri = (e - (beta * m + alpha)).astype(np.float32)
    return eri, alpha, beta


# ── 2) 2단계 기준선 ───────────────────────────────────────────────────

def compute_baseline(
    eri: np.ndarray,
    skin: np.ndarray,
    weights: np.ndarray,
    region_masks: Dict[RegionId, np.ndarray],
    region_coverage: Dict[RegionId, float],
    alpha: float,
    beta: float,
    cfg: BaselineConfig,
    mu_r: Optional[Dict[RegionId, float]] = None,
) -> BaselineInfo:
    """계획 §4.4. Level 1 전역 오프셋 + Level 2 부위별 해부학 오프셋.

    **얼굴 전체 단일 기준선은 틀렸다.** 코와 볼은 이마·턱보다 정상적으로 혈관이
    많다 (Kang & Kim: 부위 eta^2=0.18 > 기기 eta^2=0.12). 전역 기준선만 쓰면
    모든 정상적인 코를 염증으로 판정한다.
    """
    vals = eri[skin]
    wts = weights[skin]

    b_face = weighted_percentile(vals, wts, cfg.percentile)
    sigma_face = robust_sigma(vals)
    tau = cfg.tau_k * sigma_face

    # 참조 부위 교차검증 — 이마 하나에 의존하지 않는다 (계획 §4.4).
    ref_vals: List[float] = []
    for r in BASELINE_REF_REGIONS:
        if region_coverage.get(r, 0.0) < cfg.min_coverage:
            continue
        mask = region_masks.get(r)
        if mask is None or not mask.any():
            continue
        ref_vals.append(float(np.median(eri[mask])))

    b_alt: Optional[float] = None
    grad_flag = False
    if len(ref_vals) >= cfg.min_ref_regions:
        b_alt = float(np.median(ref_vals))
        grad_flag = abs(b_face - b_alt) > cfg.b_alt_delta_max

    return BaselineInfo(
        b_face=float(b_face),
        sigma_face=float(sigma_face),
        tau=float(tau),
        beta=float(beta),
        alpha=float(alpha),
        b_alt=b_alt,
        n_ref_regions=len(ref_vals),
        illumination_gradient_flag=grad_flag,
        anatomical_prior_calibrated=bool(mu_r),
    )


def deviation_map(
    eri: np.ndarray,
    baseline: BaselineInfo,
    region_masks: Dict[RegionId, np.ndarray],
    mu_r: Optional[Dict[RegionId, float]] = None,
) -> np.ndarray:
    """d(x) = ERI(x) - b_face - mu_r(x). 부위별 해부학 오프셋까지 뺀 편차맵."""
    d = eri - baseline.b_face
    if mu_r:
        for r, mu in mu_r.items():
            mask = region_masks.get(r)
            if mask is not None and mu:
                d = np.where(mask, d - mu, d)
    return d.astype(np.float32)


# ── 3) 대역 분리 ──────────────────────────────────────────────────────

def split_bands(
    d: np.ndarray, valid: np.ndarray, cfg: ScoreConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """미만성(저주파) / 병변(고주파) 분리. 계획 §1.4."""
    sigma = cfg.lowpass_sigma_frac * CANONICAL_IPD
    diffuse, _ = normalized_blur(d, valid, sigma)
    lesion_band = (d - diffuse).astype(np.float32)
    return diffuse, lesion_band


# ── 4) 병변 검출 ──────────────────────────────────────────────────────

def _local_maxima(a: np.ndarray) -> np.ndarray:
    """8-이웃 지역 최대. 경계는 제외."""
    out = np.ones(a.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.full(a.shape, -np.inf, dtype=a.dtype)
            h, w = a.shape
            ys0, ys1 = max(0, -dy), h - max(0, dy)
            yd0, yd1 = max(0, dy), h - max(0, -dy)
            xs0, xs1 = max(0, -dx), w - max(0, dx)
            xd0, xd1 = max(0, dx), w - max(0, -dx)
            shifted[yd0:yd1, xd0:xd1] = a[ys0:ys1, xs0:xs1]
            out &= a >= shifted
    out[0, :] = False
    out[-1, :] = False
    out[:, 0] = False
    out[:, -1] = False
    return out


def _anisotropy(field: np.ndarray, x: int, y: int, r: int) -> float:
    """국소 구조 텐서의 고유값 비 l1/l2.

    구진은 등방성이고 수염/주름은 길쭉하다. 3 이상이면 선형 구조로 보고 기각한다.
    """
    h, w = field.shape
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    patch = field[y0:y1, x0:x1].astype(np.float64)
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return 1.0
    gy, gx = np.gradient(patch)
    jxx = float((gx * gx).sum())
    jyy = float((gy * gy).sum())
    jxy = float((gx * gy).sum())
    tr = jxx + jyy
    det = jxx * jyy - jxy * jxy
    disc = max(tr * tr / 4.0 - det, 0.0)
    l1 = tr / 2.0 + math.sqrt(disc)
    l2 = tr / 2.0 - math.sqrt(disc)
    if l2 <= 1e-12:
        return float("inf")
    return l1 / l2


def detect_lesions(
    lesion_band: np.ndarray,
    melanin_band: np.ndarray,
    lightness: np.ndarray,
    skin: np.ndarray,
    edge_band: np.ndarray,
    region_masks: Dict[RegionId, np.ndarray],
    iris_px: float,
    cfg: LesionConfig,
) -> Tuple[List[Lesion], Dict[str, int]]:
    """스케일 정규화 DoG + 교란 억제 cascade. 계획 §2.4 · §4.5.

    스케일 축 argmax 에서 **병변 크기 추정치가 공짜로** 나온다.

    Returns:
        (병변 리스트, 사유별 기각 개수). 기각 통계는 튜닝에 필수다.
    """
    sigmas = [cfg.sigma_min_px * (cfg.sigma_ratio ** k) for k in range(cfg.n_scales)]

    # DoG 를 **진폭 단위로 되돌린다.**
    #
    # 진폭 A, 폭 sigma_b 인 가우시안 블롭에 대한 DoG(sigma, k*sigma) 의 중심 응답은
    #     A * [ sb^2/(sb^2+s^2) - sb^2/(sb^2+k^2 s^2) ]
    # 이고, 정합 스케일(s = sb)에서 A * (1/2 - 1/(1+k^2)) 이다.
    # 이 상수로 나누면 응답의 최대값이 곧 **블롭 진폭 A(= d 와 같은 log10 비율 단위)** 가 된다.
    #
    # 흔히 쓰는 sigma^2 곱셈을 하지 않는 이유: 이 비율 형태는 이미 정합 스케일에서
    # 최대가 되므로 추가 정규화가 불필요하고, sigma^2 를 곱하면 `peak_min_d` 가
    # `d` 와 다른 단위가 되어 **임계값을 물리적으로 해석할 수 없게 된다.**
    # 튜닝이 이 프로젝트의 본체이므로 임계값은 해석 가능해야 한다.
    k2 = cfg.sigma_ratio ** 2
    matched_gain = 0.5 - 1.0 / (1.0 + k2)

    def dog_stack(band: np.ndarray) -> np.ndarray:
        blurred = []
        for s in sigmas + [sigmas[-1] * cfg.sigma_ratio]:
            b, _ = normalized_blur(band, skin, s)
            blurred.append(b)
        return np.stack(
            [(blurred[i] - blurred[i + 1]) / matched_gain for i in range(len(sigmas))],
            axis=0)

    stack = dog_stack(lesion_band)
    # 멜라닌·명도도 **같은 스케일의 같은 연산자**로 재야 비교가 공정하다.
    # 원시 밴드값과 DoG 응답을 비교하면 단위가 달라 기각 임계값이 무의미해진다.
    m_stack = dog_stack(melanin_band)
    l_stack = dog_stack(lightness)

    best = stack.max(axis=0)
    best_k = stack.argmax(axis=0)

    # 임계값은 **절대 하한과 잡음 상대 하한 중 큰 쪽**이다.
    #
    # `peak_min_d` 만 쓰면 잡음이 조금만 늘어도 후보가 폭증한다. DoG 는 정합 이득
    # 0.149 로 나누므로 잡음도 6.7배 증폭된다. 실측: 색 노이즈 0.010(흐림 게이트를
    # 겨우 통과하는 수준)에서 깨끗한 피부에 24~38개가 잡혔다.
    #
    # 잡음을 재는 모집단은 **극대점들**이지 화소 전체가 아니다. 후보가 되는 것은
    # 극대점뿐이고, 극대점 값은 주변 최대값이라 화소 분포보다 계통적으로 높다.
    # 화소 전체의 sigma 로 임계를 잡으면 항상 과소평가된다 (실측: 5 sigma 를 잡아도
    # 절대 하한 아래로 내려가 아무 효과가 없었다).
    #
    # 극대점은 얼굴당 ~1050개로 잡음 수준과 무관하게 거의 일정하다. 병변은 그 중
    # 소수라 median 과 MAD 를 흔들지 못하므로, 병변이 있는 얼굴에서도 이 추정은
    # 잡음 추정으로 유효하다. **단, 병변이 극단적으로 많은 얼굴에서는 임계가 올라가
    # 검출이 보수적으로 변한다** — 오검출보다 미검출이 낫다는 선택이다.
    lm_mask = _local_maxima(best) & skin
    peaks = best[lm_mask]
    noise_floor = 0.0
    if peaks.size >= 32:
        noise_floor = float(np.median(peaks)) + cfg.peak_min_k * robust_sigma(peaks)
    peak_min = max(cfg.peak_min_d, noise_floor)

    cand = lm_mask & (best > peak_min)
    ys, xs = np.nonzero(cand)
    if ys.size == 0:
        return [], {}

    order = np.argsort(-best[ys, xs])
    ys, xs = ys[order], xs[order]

    nms_r = cfg.nms_radius_frac * CANONICAL_IPD
    mm = mm_per_px(iris_px)

    rejects: Dict[str, int] = {}

    def _rej(key: str) -> None:
        rejects[key] = rejects.get(key, 0) + 1

    kept: List[Lesion] = []
    taken_xy: List[Tuple[float, float]] = []

    for y, x in zip(ys.tolist(), xs.tolist()):
        if len(kept) >= cfg.max_lesions:
            break

        # 비최대 억제
        too_close = False
        for (tx, ty) in taken_xy:
            if (tx - x) ** 2 + (ty - y) ** 2 < nms_r * nms_r:
                too_close = True
                break
        if too_close:
            continue

        # 제외 폴리곤 경계 밴드 — DoG 가 스텝 에지에서 잘 터진다
        if edge_band[y, x]:
            _rej("edge_band")
            continue

        k = int(best_k[y, x])
        sigma = sigmas[k]
        radius_px = sigma * math.sqrt(2.0)
        # peak 는 정합 DoG 응답 = 블롭 진폭 추정치다 (원시 화소값이 아니라).
        # 단일 화소 노이즈에 휘둘리지 않는다.
        peak = float(best[y, x])

        # 점/모반: 멜라닌 축으로 움직이면 혈액이 아니다.
        # 2차원 색소 평면을 만든 직접적 배당금 — a* 하나만 봤으면 불가능하다.
        #
        # **판별자는 dm 의 크기가 아니라 부호다.** 실측한 방향 벡터에서:
        #     헤모글로빈  de=+0.447, dm=-0.494   (m 을 내린다)
        #     멜라닌      de=+0.149, dm=+0.448   (m 을 올린다)
        # |dm| > |de| 로 기각하면 헤모글로빈도 (|-0.494| > 0.447) 함께 걸려
        # **진짜 병변을 전부 버린다.** 부호 있는 값으로 비교해야 한다.
        de = abs(peak)
        dm = float(m_stack[k, y, x])
        if dm > cfg.melanin_reject_ratio * de:
            _rej("melanin_nevus")
            continue

        # 색소 병변은 주변보다 어둡다 (홍반 구진은 그렇지 않다)
        if float(l_stack[k, y, x]) < cfg.darkness_reject_dl:
            _rej("dark_pigmented")
            continue

        # 수염/주름: 길쭉한 구조
        if _anisotropy(lesion_band, x, y, max(3, int(round(radius_px)))) > cfg.anisotropy_max:
            _rej("anisotropic")
            continue

        region: Optional[RegionId] = None
        for r in ALL_REGIONS:
            rm = region_masks.get(r)
            if rm is not None and rm[y, x]:
                region = r
                break

        conf = float(min(1.0, best[y, x] / max(cfg.peak_min_d * 4.0, 1e-9)))
        kept.append(Lesion(
            x=float(x), y=float(y),
            radius_px=float(radius_px),
            radius_mm=float(radius_px * mm) if mm > 0 else None,
            peak_d=peak,
            region=region,
            confidence=conf,
        ))
        taken_xy.append((float(x), float(y)))

    return kept, rejects


# ── 5) 부위 점수 ──────────────────────────────────────────────────────

def _norm(x: float, x0: float) -> float:
    """포화 정규화 1 - exp(-x/x0). 큰 값에서 완만해진다."""
    if x0 <= 0:
        return 0.0
    return 1.0 - math.exp(-max(0.0, x) / x0)


def region_noise_scale(
    lesion_band: np.ndarray,
    skin: np.ndarray,
    masks_by_region: Dict[RegionId, np.ndarray],
    clamp: float,
) -> Dict[RegionId, float]:
    """부위별 잡음 수준을 **경험적으로** 재서 얼굴 평균 대비 비율로 돌려준다.

    왜 필요한가 — `area_fraction = mean(d > tau)` 는 **산포 통계**인데 `tau` 는
    얼굴 전체에서 한 번 정해지는 **전역** 값이다. 로그 비율은 음영을 평균에서
    정확히 소거하지만(측정: 부위별 mean(d) 가 5자리까지 동일) **분산은 소거하지
    않는다** — 어두운 부위일수록 d 의 산포가 넓어진다 (측정: std(d) 가 밝기에
    반비례, 이마 0.00878 vs 턱 0.00665).

    그래서 전역 tau 를 쓰면 **그늘진 부위가 자동으로 '붉게' 나온다.** 색소가 완전히
    균일한 합성 얼굴에서 이마가 항상 '붉음'으로 판정되던 원인이 이것이다.

    잡음 수준은 모델링하지 않고 **고주파 대역에서 직접 잰다.** 깨끗한 피부에서
    `lesion_band` 는 사실상 순수 잡음이라 그 robust sigma 가 곧 국소 잡음 추정치다.
    센서 모델(샷 노이즈냐 리드 노이즈냐)을 맞출 필요가 없다는 것이 이 방식의 장점이다.
    `snr_weights` 로 가중해 보는 방법도 시도했지만 그 모델은 샷 노이즈 형태(var∝1/r)라
    효과가 없었다 — 실측 산포는 리드 노이즈 형태(var∝1/r²)에 가까웠다.

    조명이 평탄하면 모든 비율이 1.0 이 되어 **이 보정은 자동으로 무력화된다.**
    """
    face_sigma = robust_sigma(lesion_band[skin])
    out: Dict[RegionId, float] = {}
    for r in ALL_REGIONS:
        mask = masks_by_region.get(r)
        if mask is None or face_sigma <= 0.0:
            out[r] = 1.0
            continue
        local = robust_sigma(lesion_band[mask])
        if local <= 0.0:
            out[r] = 1.0
            continue
        lo = 1.0 / clamp if clamp > 0 else 1.0
        out[r] = float(min(max(local / face_sigma, lo), clamp if clamp > 0 else 1.0))
    return out


def score_regions(
    d: np.ndarray,
    masks_by_region: Dict[RegionId, np.ndarray],
    coverage_by_region: Dict[RegionId, float],
    lesions: List[Lesion],
    baseline: BaselineInfo,
    cfg: ScoreConfig,
    min_coverage: float,
    lesion_enabled: bool,
    forehead_occluded_reason: ReasonCode = ReasonCode.LOW_COVERAGE,
    noise_scale: Optional[Dict[RegionId, float]] = None,
) -> List[RegionScore]:
    """부위별 정량값 + §7.2 2단계 판정.

    coverage 가 모자라면 점수를 **null 로 두고 사유를 남긴다.** 0 을 반환하면
    "측정 못 함"이 "정상"으로 둔갑한다 — 계약 4번.

    `noise_scale` 을 주면 부위별 tau 를 그 비율로 조정한다 (`region_noise_scale`).
    주지 않으면 전역 tau 를 그대로 쓴다 — 기존 동작.
    """
    lesion_count: Dict[RegionId, int] = {}
    for les in lesions:
        if les.region is not None:
            lesion_count[les.region] = lesion_count.get(les.region, 0) + 1

    out: List[RegionScore] = []
    for r in ALL_REGIONS:
        mask = masks_by_region.get(r)
        cov = float(coverage_by_region.get(r, 0.0))

        if mask is None or cov < min_coverage or not mask.any():
            reason = (forehead_occluded_reason if r == RegionId.FOREHEAD
                      else ReasonCode.LOW_COVERAGE)
            out.append(RegionScore(region=r, coverage=cov, reason=reason))
            continue

        vals = d[mask]
        tau_r = baseline.tau * float((noise_scale or {}).get(r, 1.0))
        hot = vals > tau_r
        area = float(hot.mean())
        intensity = float(vals[hot].mean()) if bool(hot.any()) else 0.0
        median_d = float(np.median(vals))

        score = 100.0 * min(1.0, max(0.0,
                                     cfg.w_area * _norm(area, cfg.area_x0)
                                     + cfg.w_intensity * _norm(intensity, cfg.intensity_x0)))

        n_les = lesion_count.get(r, 0)
        erythema = (Verdict.AFFECTED if area >= cfg.erythema_area_threshold
                    else Verdict.NORMAL)
        if lesion_enabled:
            lesion_v = (Verdict.AFFECTED if n_les >= cfg.lesion_count_threshold
                        else Verdict.NORMAL)
        else:
            lesion_v = Verdict.NOT_MEASURED

        out.append(RegionScore(
            region=r,
            coverage=cov,
            reason=ReasonCode.OK,
            area_fraction=area,
            intensity=intensity,
            median_d=median_d,
            score_ordinal_0_100=score,
            erythema_verdict=erythema,
            lesion_verdict=lesion_v,
            lesion_count=(n_les if lesion_enabled else None),
        ))
    return out
