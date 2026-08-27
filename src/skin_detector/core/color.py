"""색 변환 · 색소 축 분리 · 홍반 지수.

계약: numpy + stdlib 만. cv2 · PIL · mediapipe 금지.

연산 순서는 고정이다 (계획 §1.3-2):
    sRGB -> 감마 선형화 -> 광학밀도 -> 채널 비율 -> 기준선 차감
이 순서를 바꾸면 음영 불변성이 깨진다.
"""

from typing import Optional, Tuple

import numpy as np

# 로그 폭발 방지 바닥값. 선형 반사율 기준.
# 품질 게이트가 어두운 화소를 이미 거르지만 core 는 방어적으로 동작해야 한다.
_FLOOR = 1e-4

# sRGB(D65) -> XYZ. IEC 61966-2-1.
_M_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)

_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)

# D65 백점 (2도 관측자).
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)


# ── 감마 · 광학밀도 ────────────────────────────────────────────────────

def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """sRGB -> 선형 반사율. **piecewise 구간을 반드시 지킨다** (계획 §1.3-1).

    검정 근처를 단순 거듭제곱으로 근사하면 채널별 상수항이 어긋나 자기참조가 깨진다.

    Args:
        img: uint8 [0,255] 또는 float [0,1], shape (..., 3)
    Returns:
        float32 선형값, 같은 shape
    """
    x = img.astype(np.float32)
    if img.dtype == np.uint8:
        x = x / np.float32(255.0)
    lo = x / np.float32(12.92)
    hi = np.power((x + np.float32(0.055)) / np.float32(1.055), np.float32(2.4))
    return np.where(x <= 0.04045, lo, hi).astype(np.float32)


def optical_density(linear: np.ndarray) -> np.ndarray:
    """선형 반사율 -> 광학밀도 D = -log10(R)."""
    return -np.log10(np.maximum(linear, _FLOOR)).astype(np.float32)


def chromophore_axes(density: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """광학밀도 -> (홍반 축 e, 멜라닌 축 m). 계획 §1.2.

        e = D_G - D_R = log10(R/G)   헤모글로빈 우세
        m = D_B - D_G = log10(G/B)   멜라닌 우세

    이 로그 비율이 **픽셀별 음영·노출·거리·플래시 감쇠를 정확히 소거**한다.
    화이트밸런스는 상수 오프셋으로만 남고, 그건 자기참조 기준선이 죽인다.
    """
    d_r = density[..., 0]
    d_g = density[..., 1]
    d_b = density[..., 2]
    e = d_g - d_r
    m = d_b - d_g
    return e, m


def snr_weights(linear: np.ndarray, read_noise: float = 0.003) -> np.ndarray:
    """픽셀별 1/분산 가중 (계획 §4.3-d).

        Var[log(R/G)] ~= (sigma_R/R)^2 + (sigma_G/G)^2

    포아송 + 읽기 노이즈를 가정한다. 어두운 화소의 로그 비율은 분산이 거대하므로
    이후 모든 퍼센타일·평균·회귀를 이 가중으로 돌린다.
    비용 대비 견고성 이득이 가장 큰 항목 중 하나다.
    """
    r = np.maximum(linear[..., 0], _FLOOR)
    g = np.maximum(linear[..., 1], _FLOOR)
    nr2 = read_noise * read_noise
    var_r = (r + nr2) / (r * r)
    var_g = (g + nr2) / (g * g)
    var = var_r + var_g
    return (1.0 / np.maximum(var, 1e-12)).astype(np.float32)


def specular_proxy(linear: np.ndarray) -> np.ndarray:
    """min(R,G,B) — 정반사 프록시 (계획 §4.3-b).

    피부 정반사는 표면(Fresnel) 반사라 **분광적으로 중립**이고 광원색을 띤다.
    무광 피부에서 min 채널은 낮은 B 를 따라가지만 정반사가 얹히면 함께 올라간다.

    주의: 이 값으로 `C - min(R,G,B)` 보정을 하고 싶어지지만 비율을 비선형으로
    바꿔버린다. **보정이 아니라 마스크로만 쓴다.**
    """
    return np.min(linear, axis=-1)


# ── 얼굴별 멜라닌 회귀 제거 ────────────────────────────────────────────

def robust_line_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_iter: int = 5,
    trim: float = 0.2,
) -> Tuple[float, float]:
    """반복 절사 가중 최소제곱으로 y = beta*x + alpha 적합.

    매 반복마다 잔차가 큰 `trim` 비율을 버리고 다시 적합한다.
    병변·정반사 같은 소수 이상치가 기울기를 끌고 가는 것을 막는다.

    Returns:
        (alpha, beta). 표본이 부족하면 (median(y), 0.0) 으로 후퇴한다.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    w = (np.ones_like(x) if weights is None
         else np.asarray(weights, dtype=np.float64).ravel())

    if x.size < 32:
        return (float(np.median(y)) if y.size else 0.0), 0.0

    keep = np.ones(x.size, dtype=bool)
    alpha = float(np.median(y))
    beta = 0.0

    for _ in range(n_iter):
        xk = x[keep]
        yk = y[keep]
        wk = w[keep]
        sw = float(wk.sum())
        if sw <= 0.0 or xk.size < 16:
            break
        mx = float((wk * xk).sum() / sw)
        my = float((wk * yk).sum() / sw)
        vxx = float((wk * (xk - mx) * (xk - mx)).sum())
        vxy = float((wk * (xk - mx) * (yk - my)).sum())
        beta = (vxy / vxx) if vxx > 1e-12 else 0.0
        alpha = my - beta * mx

        resid = np.abs(y - (beta * x + alpha))
        cutoff = float(np.quantile(resid, 1.0 - trim))
        new_keep = resid <= max(cutoff, 1e-12)
        if int(new_keep.sum()) < 16:
            break
        keep = new_keep

    return alpha, beta


def erythema_index(
    e: np.ndarray, m: np.ndarray, alpha: float, beta: float
) -> np.ndarray:
    """ERI = e - (beta*m + alpha). 계획 §1.2 주 지표.

    얼굴 한 장 안에서 e 와 m 의 지배적 공변 성분은 멜라닌 구배(태닝선·눈 주위
    착색·광노화)다. 얼굴마다 회귀로 빼면 고정 벡터 투영보다 피부톤·기기·조명 SPD
    변화를 잘 흡수한다.
    """
    return (e - (beta * m + alpha)).astype(np.float32)


# ── CIELAB 교차검증 ────────────────────────────────────────────────────

def linear_to_lab(linear: np.ndarray) -> np.ndarray:
    """선형 sRGB -> CIELAB (D65). 문헌 비교용 delta-a* 교차검증에 쓴다.

    ERI 와 delta-a* 의 **불일치 자체가 멜라닌/음영 오염의 진단 신호**다 (계획 §1.2).
    """
    shp = linear.shape
    flat = linear.reshape(-1, 3).astype(np.float64)
    xyz = flat @ _M_RGB2XYZ.T
    t = xyz / _WHITE_D65[None, :]

    delta = 6.0 / 29.0
    f = np.where(t > delta ** 3,
                 np.cbrt(np.maximum(t, 1e-12)),
                 t / (3.0 * delta * delta) + 4.0 / 29.0)

    lab = np.empty_like(f)
    lab[:, 0] = 116.0 * f[:, 1] - 16.0
    lab[:, 1] = 500.0 * (f[:, 0] - f[:, 1])
    lab[:, 2] = 200.0 * (f[:, 1] - f[:, 2])
    return lab.reshape(shp).astype(np.float32)


def ita_degrees(lab: np.ndarray) -> np.ndarray:
    """ITA = atan2(L*-50, b*) in degrees — 피부톤 계층화 지표.

    어두운 피부(FST V-VI)에서는 멜라닌이 녹색 대역 Hb 신호를 억눌러 e 채널 SNR 이
    붕괴한다 (계획 §7.3-3). 점수와 함께 출력해 **조용히 자신 있는 숫자를 반환하지
    않도록** 한다.
    """
    l_star = lab[..., 0]
    b_star = lab[..., 2]
    safe_b = np.where(np.abs(b_star) < 1e-6, 1e-6, b_star)
    return np.degrees(np.arctan2(l_star - 50.0, safe_b))


# ── 흡광계수 -> 채널 방향 벡터 (합성 병변 주입용) ─────────────────────

def _cie_gaussian(x: np.ndarray, mu: float, s1: float, s2: float) -> np.ndarray:
    """비대칭(piecewise) 가우시안 — CIE CMF 해석적 근사의 기본 단위."""
    s = np.where(x < mu, s1, s2)
    t = (x - mu) / s
    return np.exp(-0.5 * t * t)


def cie_xyz_bar(wavelength_nm: np.ndarray) -> np.ndarray:
    """CIE 1931 2도 색일치함수의 다엽 가우시안 근사 (Wyman et al. 2013).

    표를 파일로 들고 다니지 않기 위한 선택이다 — 외부 데이터 파일은 흡광계수 CSV
    하나로 유지한다 (계획 §5.2). 근사 오차는 합성 주입 방향 산출에 충분하다.

    Returns: shape (N, 3) — x_bar, y_bar, z_bar
    """
    w = np.asarray(wavelength_nm, dtype=np.float64)
    x = (1.056 * _cie_gaussian(w, 599.8, 37.9, 31.0)
         + 0.362 * _cie_gaussian(w, 442.0, 16.0, 26.7)
         - 0.065 * _cie_gaussian(w, 501.1, 20.4, 26.2))
    y = (0.821 * _cie_gaussian(w, 568.8, 46.9, 40.5)
         + 0.286 * _cie_gaussian(w, 530.9, 16.3, 31.1))
    z = (1.217 * _cie_gaussian(w, 437.0, 11.8, 36.0)
         + 0.681 * _cie_gaussian(w, 459.0, 26.0, 13.8))
    return np.stack([x, y, z], axis=-1)


def channel_response(wavelength_nm: np.ndarray) -> np.ndarray:
    """sRGB 채널별 유효 분광 응답. CMF 에 XYZ->RGB 행렬을 적용해 얻는다.

    실제 primaries 라 음수 lobe 가 생긴다. 가중 평균의 가중치로 쓰기 위해 0 에서
    자른다 — PoC 수준의 근사이고, 이 근사가 영향을 주는 곳은 합성 주입 방향뿐이다.

    Returns: shape (N, 3), 각 채널 합이 1 로 정규화됨.
    """
    resp = cie_xyz_bar(wavelength_nm) @ _M_XYZ2RGB.T
    resp = np.clip(resp, 0.0, None)
    total = resp.sum(axis=0, keepdims=True)
    return resp / np.maximum(total, 1e-12)


def dermal_pathlength(wavelength_nm: np.ndarray) -> np.ndarray:
    """진피 혈관층까지의 상대 유효 광로장. 파랑에서 0 에 가깝고 적색에서 1 이다.

    **이 가중이 없으면 v_h 가 물리적으로 뒤집힌다.** oxy-Hb 의 415nm Soret 밴드는
    흡광계수가 Q-band 보다 10배 크지만, 그 파장의 빛은 표피 산란·멜라닌 흡수에 막혀
    **혈관이 있는 깊이까지 도달하지 못한다.** 흡광계수만 곱하면 "헤모글로빈은 파랑을
    가장 많이 먹는다"는 결론이 나오고, 그러면 §1.1 의 구분자(파랑 채널)가 무너진다.

    경험적 시그모이드다. 산란 계수에서 유도한 값이 아니라 "파랑은 진피에 못 간다"는
    사실을 부드럽게 인코딩한 것이고, 쓰이는 곳은 합성 주입 방향 하나뿐이다.
    멜라닌은 표피에 있어 모든 파장이 닿으므로 이 가중을 적용하지 않는다.
    """
    w = np.asarray(wavelength_nm, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-(w - 520.0) / 40.0))


def absorber_rgb_direction(
    wavelength_nm: np.ndarray,
    extinction: np.ndarray,
    depth_weighted: bool = False,
) -> np.ndarray:
    """파장별 흡광계수 -> log-RGB 공간에서의 단위 이동 방향.

    농도가 dc 만큼 늘면 광학밀도가 채널마다 `dc * v[c]` 만큼 증가한다.
    합성 병변을 **물리적으로 올바르게** 주입하는 데 쓴다 (계획 §6.2) —
    붉은 원 알파 블렌딩은 비물리적이라 검출기를 실제보다 좋아 보이게 만든다.

    Args:
        depth_weighted: 진피 색소(헤모글로빈)면 True. 표피 색소(멜라닌)면 False.

    Returns: shape (3,), 최대 성분이 1 이 되도록 정규화된 방향 벡터.
    """
    w = np.asarray(wavelength_nm, dtype=np.float64)
    eps = np.asarray(extinction, dtype=np.float64)
    if depth_weighted:
        eps = eps * dermal_pathlength(w)
    resp = channel_response(w)                  # (N, 3)
    v = (resp * eps[:, None]).sum(axis=0)       # (3,)
    peak = float(np.max(np.abs(v)))
    return (v / peak if peak > 0 else v).astype(np.float32)
