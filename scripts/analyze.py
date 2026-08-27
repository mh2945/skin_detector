"""단건/배치 분석 + store 편입.

    python scripts/analyze.py --ingest data/incoming        # 원본을 store 규약으로
    python scripts/analyze.py --store data/store --out out/ # store 전체 분석
    python scripts/analyze.py --image a.heic --out out/     # 파일 하나만

결과 JSON 은 `data/store/<id>/results/<algo>_<hash>.json` 에 남고,
오버레이는 `--out` 에 떨어진다. 원본과 capture.json 은 절대 수정하지 않는다.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _console import setup_console  # noqa: E402

setup_console()          # cp949 리다이렉트에서 죽지 않게 한다. import 직후여야 한다.

from skin_detector import imageio as sio, pipeline, render  # noqa: E402
from skin_detector.core.types import Verdict  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"}


def summarize(result) -> str:
    q = result.quality
    if not q.passed:
        return "  판정 불가: {}".format(q.reason.value)
    lines = []
    for r in result.regions:
        if r.score_ordinal_0_100 is None:
            lines.append("  {:<9} 측정 못 함 ({})".format(r.region.value, r.reason.value))
            continue
        les = "-" if r.lesion_count is None else str(r.lesion_count)
        lines.append(
            "  {:<9} score {:5.1f}  area {:5.1%}  d~{:+.4f}  붉은기 {:<8} 트러블 {:<12} 병변 {}"
            .format(r.region.value, r.score_ordinal_0_100, r.area_fraction,
                    r.median_d, r.erythema_verdict.value, r.lesion_verdict.value, les))
    b = result.baseline
    if b:
        lines.append("  기준선 b_face={:+.4f} tau={:.4f} beta={:.3f} 참조부위={} 조명={}"
                     .format(b.b_face, b.tau, b.beta, b.n_ref_regions,
                             result.illumination_check))
        if not b.anatomical_prior_calibrated:
            lines.append("  주의: 부위별 해부학 prior 미보정 (anatomical_prior_calibrated=false)")
    if not result.lesion_detection_enabled:
        lines.append("  병변 검출 꺼짐: {}".format(result.lesion_reason.value))
    return "\n".join(lines)


def analyze_capture(cap_dir: Path, cfg, cfg_hash, model, out_dir):
    src = sio.find_original(cap_dir)
    if src is None:
        return None
    meta = sio.read_capture_meta(cap_dir)
    rgb, _ = sio.load_original(src)
    ana = pipeline.analyze(rgb, cfg, cfg_hash, model, meta=meta)
    sio.write_result(cap_dir, ana.result)

    if out_dir:
        overlay = render.render_overlay(ana)
        if overlay is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            render.save_png(out_dir / (meta.capture_id + "_overlay.png"), overlay)
    return ana


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ingest", type=Path, help="이 폴더의 원본을 store 로 편입")
    p.add_argument("--store", type=Path, default=ROOT / "data" / "store")
    p.add_argument("--image", type=Path, help="파일 하나만 (store 를 거치지 않는다)")
    p.add_argument("--out", type=Path, help="오버레이 출력 폴더")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "default.toml")
    p.add_argument("--model", type=Path,
                   default=ROOT / "models" / "face_landmarker.task")
    args = p.parse_args()

    cfg = sio.load_config(args.config)
    cfg_hash = sio.config_hash(cfg)
    print("config_hash={}  algo={}".format(cfg_hash, pipeline.ALGO_VERSION))

    if args.ingest:
        n, warn = 0, 0
        for src in sorted(args.ingest.rglob("*")):
            if src.suffix.lower() not in IMAGE_EXTS:
                continue
            cap_dir, meta, warnings = sio.ingest(src, args.store)
            n += 1
            for w in warnings:
                warn += 1
                print("  [경고] " + w)
            print("  편입 {} -> {}".format(src.name, meta.capture_id))
        print("편입 {}건 (파일명 규약 경고 {}건)".format(n, warn))
        if warn:
            print("경고가 있는 파일은 WB/ISO/torch 를 알 수 없어 §6 검증에 쓰면 데이터가 오염된다.")
        return 0

    if args.image:
        rgb, meta = sio.load_original(args.image)
        ana = pipeline.analyze(rgb, cfg, cfg_hash, args.model, meta=meta)
        print("{}".format(args.image.name))
        print(summarize(ana.result))
        if args.out:
            overlay = render.render_overlay(ana)
            if overlay is not None:
                args.out.mkdir(parents=True, exist_ok=True)
                render.save_png(args.out / (args.image.stem + "_overlay.png"), overlay)
        return 0

    caps = sorted(d for d in args.store.iterdir() if d.is_dir()) \
        if args.store.exists() else []
    if not caps:
        print("store 가 비었다: {}  (--ingest 로 먼저 편입한다)".format(args.store))
        return 1

    ok = 0
    for cap in caps:
        ana = analyze_capture(cap, cfg, cfg_hash, args.model, args.out)
        if ana is None:
            continue
        ok += 1
        print("{}".format(cap.name))
        print(summarize(ana.result))
    print("\n분석 {}/{}건".format(ok, len(caps)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
