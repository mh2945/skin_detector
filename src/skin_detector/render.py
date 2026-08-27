"""히트맵 · 병변 마커 · 부위 외곽선 오버레이.

core/ 밖이다. cv2 를 여기서만(그리고 face.py 에서만) 쓴다.

오버레이는 전부 **정규 프레임**에 그린다. 원본 위에 그리려면 좌표를 역변환해야
하고, 그 변환은 `pipeline.lesions_to_original()` 한 곳에만 있다.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core.types import ALL_REGIONS, RegionId, Verdict
from .pipeline import Analysis

# 부위 외곽선 색 (RGB). 판정 결과와 무관한 고정색 — 판정은 채움색으로 보여준다.
_REGION_COLOR = (90, 190, 255)
_LESION_COLOR = (255, 220, 40)
_AFFECTED_TINT = (255, 70, 70)


def heatmap(d: np.ndarray, skin: np.ndarray, tau: float,
            span: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """편차맵 -> 색상 + 알파.

    `tau` 아래는 완전히 투명하다. 임계 미만을 옅게라도 칠하면 "전체가 조금 붉다"는
    인상을 만들어 §7.3-2(레벨보다 대비를 본다)의 한계를 숨기게 된다.
    """
    import cv2

    hi = max(tau * span, 1e-6)
    t = np.clip((d - tau) / hi, 0.0, 1.0).astype(np.float32)
    t_u8 = (t * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(t_u8, cv2.COLORMAP_INFERNO)[:, :, ::-1]  # BGR->RGB
    alpha = np.where(skin & (d > tau), t * 0.72, 0.0).astype(np.float32)
    return colored.astype(np.uint8), alpha


def _blend(base: np.ndarray, layer: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return (base.astype(np.float32) * (1.0 - a)
            + layer.astype(np.float32) * a).astype(np.uint8)


def draw_region_outlines(img: np.ndarray, masks: Dict[RegionId, np.ndarray],
                         affected: Optional[Dict[RegionId, bool]] = None) -> np.ndarray:
    """부위 외곽선. '붉음' 판정된 부위는 굵게 그린다."""
    import cv2

    out = np.ascontiguousarray(img)
    for rid in ALL_REGIONS:
        mask = masks.get(rid)
        if mask is None or not mask.any():
            continue
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        thick = 2 if (affected or {}).get(rid) else 1
        color = _AFFECTED_TINT if (affected or {}).get(rid) else _REGION_COLOR
        cv2.drawContours(out, contours, -1, color, thick)
    return out


def draw_lesions(img: np.ndarray, lesions: Sequence) -> np.ndarray:
    """병변 마커. 반지름은 스케일 축 argmax 에서 나온 추정치다."""
    import cv2

    out = np.ascontiguousarray(img)
    for les in lesions:
        c = (int(round(les.x)), int(round(les.y)))
        r = max(3, int(round(les.radius_px)))
        cv2.circle(out, c, r, _LESION_COLOR, 1, lineType=cv2.LINE_AA)
        cv2.circle(out, c, 1, _LESION_COLOR, -1)
    return out


def render_overlay(analysis: Analysis, show_regions: bool = True,
                   show_lesions: bool = True) -> Optional[np.ndarray]:
    """정규 프레임 위의 완성 오버레이. 분석이 실패했으면 원본 프레임만 돌려준다."""
    obs = analysis.observation
    if obs is None:
        return None

    base = obs.canonical_rgb
    res = analysis.result

    if analysis.deviation is not None and analysis.skin is not None and res.baseline:
        layer, alpha = heatmap(analysis.deviation, analysis.skin, res.baseline.tau)
        base = _blend(base, layer, alpha)

    if show_regions and analysis.masks is not None:
        affected = {r.region: (r.erythema_verdict == Verdict.AFFECTED)
                    for r in res.regions}
        base = draw_region_outlines(base, analysis.masks.regions, affected)

    if show_lesions and res.lesions:
        base = draw_lesions(base, res.lesions)

    return base


def save_png(path, img: np.ndarray) -> None:
    import cv2
    cv2.imwrite(str(path), img[:, :, ::-1])


def contact_sheet(images: Sequence[np.ndarray], cols: int = 4,
                  tile: int = 256) -> np.ndarray:
    """여러 장을 한 판에. **튜닝에 필수다** — 한 장에서 튜닝하면 그 한 장에 오버핏한다."""
    import cv2

    if not images:
        return np.zeros((tile, tile, 3), dtype=np.uint8)
    cols = max(1, cols)
    rows = (len(images) + cols - 1) // cols
    sheet = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        small = cv2.resize(im, (tile, tile), interpolation=cv2.INTER_AREA)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = small
    return sheet
