"""파라미터 튜닝 UI. 슬라이더 + contact sheet.

    python scripts/tune.py --capture-id 20260901_143022_a4f2
    python scripts/tune.py --contact-sheet data/store

**contact sheet 병용은 선택이 아니라 필수다** (계획 §6.1).
한 장에서 튜닝하면 그 한 장에 오버핏한다.

랜드마크·워프·마스크는 캐시하고 색 단계만 다시 돌린다 (§2.6) — 정규 프레임
768x768 에서 ~30ms 라 슬라이더가 실시간으로 반응한다. 캐시 없이는 매번 ~90ms 다.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skin_detector import face as face_mod, imageio as sio, pipeline, render  # noqa: E402
from skin_detector.core.types import Verdict  # noqa: E402

WINDOW = "skin_detector tune"

# (라벨, 섹션, 필드, 최소, 최대, 슬라이더 눈금 수)
SLIDERS = [
    ("tau_k x100", "baseline", "tau_k", 0.5, 4.0, 350),
    ("P(baseline)", "baseline", "percentile", 5.0, 60.0, 55),
    ("maha x10", "mask", "mahalanobis_max", 1.0, 6.0, 50),
    ("beta_max x100", "baseline", "melanin_beta_max", 0.0, 1.0, 100),
    ("peak_min x1000", "lesion", "peak_min_d", 0.002, 0.060, 58),
    ("area_thr x100", "score", "erythema_area_threshold", 0.02, 0.50, 48),
]


def load_capture(store: Path, capture_id: str):
    cap = store / capture_id
    src = sio.find_original(cap)
    if src is None:
        raise SystemExit("원본을 찾을 수 없다: {}".format(cap))
    rgb, _ = sio.load_original(src)
    return rgb, sio.read_capture_meta(cap)


def apply_sliders(cfg, values):
    for (_, section, field, lo, hi, _), v in zip(SLIDERS, values):
        setattr(getattr(cfg, section), field, v)
    return cfg


def status_bar(result) -> str:
    if not result.quality.passed:
        return "REJECTED: " + result.quality.reason.value
    hot = sum(1 for r in result.regions if r.erythema_verdict == Verdict.AFFECTED)
    nul = sum(1 for r in result.regions if r.score_ordinal_0_100 is None)
    b = result.baseline
    return ("붉음 {}/5  측정불가 {}  병변 {}  tau={:.4f} beta={:.2f}  조명={}"
            .format(hot, nul, len(result.lesions),
                    b.tau if b else 0.0, b.beta if b else 0.0,
                    result.illumination_check))


def run_single(args) -> int:
    import cv2

    rgb, meta = load_capture(Path(args.store), args.capture_id)
    cfg = sio.load_config(Path(args.config))

    # ★ 캐시: 랜드마크 + 워프. 슬라이더를 움직여도 다시 계산하지 않는다.
    obs = face_mod.detect(rgb, Path(args.model))
    if obs is None:
        print("얼굴을 찾지 못했다.")
        return 1
    print("정규 프레임 캐시 완료. native_ipd={:.0f}px".format(obs.native_ipd_px))

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    for label, section, field, lo, hi, steps in SLIDERS:
        cur = getattr(getattr(cfg, section), field)
        pos = int(round((cur - lo) / (hi - lo) * steps))
        cv2.createTrackbar(label, WINDOW, pos, steps, lambda _v: None)

    print("슬라이더를 움직여 조정. s=현재 설정 저장, q=종료")
    last: Optional[Tuple] = None
    while True:
        vals = []
        for label, section, field, lo, hi, steps in SLIDERS:
            pos = cv2.getTrackbarPos(label, WINDOW)
            vals.append(lo + (hi - lo) * pos / max(1, steps))
        key_tuple = tuple(round(v, 6) for v in vals)

        if key_tuple != last:
            last = key_tuple
            apply_sliders(cfg, vals)
            cfg_hash = sio.config_hash(cfg)
            ana = pipeline.analyze(rgb, cfg, cfg_hash, Path(args.model),
                                   meta=meta, observation=obs)
            overlay = render.render_overlay(ana)
            if overlay is not None:
                canvas = np.ascontiguousarray(overlay[:, :, ::-1])
                cv2.putText(canvas, status_bar(ana.result), (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
                cv2.imshow(WINDOW, canvas)

        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            out = Path(args.config).with_name("tuned.toml")
            _write_toml(out, cfg)
            print("저장: {}  (default.toml 을 덮어쓰지 않는다)".format(out))

    cv2.destroyAllWindows()
    return 0


def _write_toml(path: Path, cfg) -> None:
    lines = ["# scripts/tune.py 가 저장한 설정.",
             "# default.toml 로 승격하려면 직접 복사한다 — 튜닝값이 자동으로",
             "# 진짜 출처를 덮어쓰면 어떤 값이 검증된 것인지 알 수 없게 된다.", ""]
    for section, values in cfg.to_dict().items():
        lines.append("[{}]".format(section))
        for k, v in values.items():
            lines.append("{} = {}".format(k, repr(v).replace("True", "true")
                                          .replace("False", "false")))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_contact_sheet(args) -> int:
    """여러 장을 한 판에. 한 장 오버핏을 막는 유일한 방법이다."""
    import cv2

    store = Path(args.contact_sheet)
    cfg = sio.load_config(Path(args.config))
    cfg_hash = sio.config_hash(cfg)
    caps = sorted(d for d in store.iterdir() if d.is_dir())[:args.limit]
    if not caps:
        print("store 가 비었다: {}".format(store))
        return 1

    tiles: List[np.ndarray] = []
    for cap in caps:
        src = sio.find_original(cap)
        if src is None:
            continue
        rgb, _ = sio.load_original(src)
        meta = sio.read_capture_meta(cap)
        ana = pipeline.analyze(rgb, cfg, cfg_hash, Path(args.model), meta=meta)
        overlay = render.render_overlay(ana)
        if overlay is None:
            continue
        tag = ("REJECT " + ana.result.quality.reason.value
               if not ana.result.quality.passed
               else "{} 붉음".format(sum(
                   1 for r in ana.result.regions
                   if r.erythema_verdict == Verdict.AFFECTED)))
        overlay = np.ascontiguousarray(overlay)
        cv2.putText(overlay, tag, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2, cv2.LINE_AA)
        tiles.append(overlay)

    if not tiles:
        print("렌더할 이미지가 없다.")
        return 1

    sheet = render.contact_sheet(tiles, cols=args.cols, tile=args.tile)
    out = Path(args.out) if args.out else ROOT / "out" / "contact_sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render.save_png(out, sheet)
    print("{}장 -> {}  (config_hash={})".format(len(tiles), out, cfg_hash))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-id")
    p.add_argument("--contact-sheet", help="이 store 폴더로 contact sheet 생성")
    p.add_argument("--store", default=str(ROOT / "data" / "store"))
    p.add_argument("--config", default=str(ROOT / "config" / "default.toml"))
    p.add_argument("--model", default=str(ROOT / "models" / "face_landmarker.task"))
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--out")
    args = p.parse_args()

    if args.contact_sheet:
        return run_contact_sheet(args)
    if args.capture_id:
        return run_single(args)
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
