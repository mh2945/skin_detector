"""store 전량 재채점 · 버전 간 비교. **목표 3의 실증부.**

    python scripts/reprocess.py --all
    python scripts/reprocess.py --compare v0.1.0_a4f2c9 v0.1.0_71be03

알고리즘이나 config 를 고친 뒤 과거 데이터 전부를 새 버전으로 다시 채점한다.
원본과 capture.json 은 손대지 않으므로(계약 10번) 몇 번을 다시 돌려도 안전하고,
결과는 `results/` 아래에 버전별로 **나란히 쌓인다** — 지우지 않는다.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _console import setup_console  # noqa: E402

setup_console()          # cp949 리다이렉트에서 죽지 않게 한다. import 직후여야 한다.

from skin_detector import imageio as sio, pipeline  # noqa: E402
from skin_detector.core.types import ALL_REGIONS  # noqa: E402


def reprocess_all(store: Path, cfg, cfg_hash, model) -> int:
    caps = sorted(d for d in store.iterdir() if d.is_dir())
    done = 0
    for cap in caps:
        src = sio.find_original(cap)
        if src is None:
            continue
        meta = sio.read_capture_meta(cap)
        rgb, _ = sio.load_original(src)
        ana = pipeline.analyze(rgb, cfg, cfg_hash, model, meta=meta)
        out = sio.write_result(cap, ana.result)
        done += 1
        print("  {} -> {}".format(cap.name, out.name))
    return done


def _load_version(store: Path, tag: str) -> Dict[str, dict]:
    """`<algo>_<confighash>` 태그에 해당하는 결과를 capture_id 별로 모은다."""
    out: Dict[str, dict] = {}
    for cap in sorted(d for d in store.iterdir() if d.is_dir()):
        f = cap / "results" / (tag + ".json")
        if f.exists():
            out[cap.name] = json.loads(f.read_text(encoding="utf-8"))
    return out


def compare(store: Path, tag_a: str, tag_b: str) -> int:
    a = _load_version(store, tag_a)
    b = _load_version(store, tag_b)
    common = sorted(set(a) & set(b))
    if not common:
        print("두 버전에 공통으로 존재하는 capture 가 없다.")
        print("  {}: {}건, {}: {}건".format(tag_a, len(a), tag_b, len(b)))
        return 1

    print("공통 {}건  ({} -> {})\n".format(len(common), tag_a, tag_b))
    print("{:<10} {:>10} {:>10} {:>9}   {:>12}".format(
        "부위", tag_a[-8:], tag_b[-8:], "차이", "판정 변화"))
    print("-" * 60)

    for rid in ALL_REGIONS:
        sa: List[float] = []
        sb: List[float] = []
        flips = 0
        for cid in common:
            ra = next((r for r in a[cid]["regions"] if r["region"] == rid.value), None)
            rb = next((r for r in b[cid]["regions"] if r["region"] == rid.value), None)
            if not ra or not rb:
                continue
            if ra["score_ordinal_0_100"] is not None and rb["score_ordinal_0_100"] is not None:
                sa.append(ra["score_ordinal_0_100"])
                sb.append(rb["score_ordinal_0_100"])
            if ra["erythema_verdict"] != rb["erythema_verdict"]:
                flips += 1
        if not sa:
            print("{:<10} {:>10} {:>10} {:>9}   {:>12}".format(
                rid.value, "-", "-", "-", "{}건".format(flips)))
            continue
        ma = sum(sa) / len(sa)
        mb = sum(sb) / len(sb)
        print("{:<10} {:>10.1f} {:>10.1f} {:>+9.1f}   {:>10}건".format(
            rid.value, ma, mb, mb - ma, flips))

    print("\n'판정 변화'는 붉은기 축 판정이 뒤집힌 이미지 수다.")
    print("§7.2 일치율에 직접 영향을 주므로 여기가 크게 움직이면 재평가가 필요하다.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", type=Path, default=ROOT / "data" / "store")
    p.add_argument("--all", action="store_true", help="전량 재채점")
    p.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"),
                   help="두 결과 버전 비교 (<algo>_<confighash>)")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "default.toml")
    p.add_argument("--model", type=Path,
                   default=ROOT / "models" / "face_landmarker.task")
    p.add_argument("--list", action="store_true", help="store 에 있는 결과 버전 나열")
    args = p.parse_args()

    if not args.store.exists():
        print("store 없음: {}".format(args.store))
        return 1

    if args.list:
        tags: Dict[str, int] = {}
        for cap in sorted(d for d in args.store.iterdir() if d.is_dir()):
            for f in (cap / "results").glob("*.json"):
                tags[f.stem] = tags.get(f.stem, 0) + 1
        if not tags:
            print("결과가 없다.")
            return 0
        for tag, n in sorted(tags.items()):
            print("  {:<24} {}건".format(tag, n))
        return 0

    if args.compare:
        return compare(args.store, args.compare[0], args.compare[1])

    if args.all:
        cfg = sio.load_config(args.config)
        cfg_hash = sio.config_hash(cfg)
        print("재채점 {} / config_hash={}".format(pipeline.ALGO_VERSION, cfg_hash))
        n = reprocess_all(args.store, cfg, cfg_hash, args.model)
        print("{}건 재채점 완료. 이전 버전 결과는 그대로 남아 있다.".format(n))
        return 0

    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
