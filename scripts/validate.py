"""검증 러너 (계획 §6). 서브커맨드별로 하나씩 답을 낸다.

    validate.py agreement --panel data/panel_ratings.csv   ★ 목표 1 관문
    validate.py stability                                  ★ Phase 1 관문 (조명 간 ICC)
    validate.py flashpair                                  플래시 on/off 쌍 차분 (L1 사전검증)
    validate.py synth                                      합성 Delta 복원 선형성
    validate.py negative                                   음성 대조군

**성능 수치는 경로 B(capture_path=upload)에서만 나온다** (계획 §3.1).
브라우저 캡처는 WB/ISO 통제가 없어 색 통제 실패와 알고리즘 실패를 구분할 수 없다.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skin_detector import face as face_mod, imageio as sio, pipeline  # noqa: E402
from skin_detector.core import color, measure  # noqa: E402
from skin_detector.core.types import ALL_REGIONS, RegionId  # noqa: E402


# ── 통계 ──────────────────────────────────────────────────────────────

def icc21(data: np.ndarray) -> Optional[float]:
    """ICC(2,1) 이원 랜덤·단일 측정·절대 일치.

    행 = 대상(여기서는 세션/반복), 열 = 평가자(여기서는 조명 조건).
    """
    x = np.asarray(data, dtype=np.float64)
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape
    if n < 2 or k < 2:
        return None
    grand = x.mean()
    ms_r = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    if abs(denom) < 1e-12:
        return None
    return float((ms_r - ms_e) / denom)


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> Optional[float]:
    """우연 일치를 제거한 일치도. **일치율 단독은 조작 가능하므로 항상 함께 본다.**"""
    if not a:
        return None
    labels = sorted(set(a) | set(b))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(a)
    conf = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        conf[idx[x], idx[y]] += 1
    po = np.trace(conf) / n
    pe = float((conf.sum(axis=0) * conf.sum(axis=1)).sum()) / (n * n)
    if abs(1.0 - pe) < 1e-12:
        return None
    return float((po - pe) / (1.0 - pe))


def no_information_rate(labels: Sequence[str]) -> float:
    """다수 클래스 비율 = **아무것도 안 하는 모델의 점수.**

    대부분의 부위가 '정상'이면 "전부 정상"이라고만 답해도 70% 를 넘긴다.
    일치율은 반드시 이 값보다 확실히 높아야 의미가 있다 (계획 §7.2).
    """
    if not labels:
        return 0.0
    counts: Dict[str, int] = defaultdict(int)
    for l in labels:
        counts[l] += 1
    return max(counts.values()) / len(labels)


def fmt_triplet(name: str, agree: float, nir: float, kappa: Optional[float],
                ceiling: Optional[float], n: int) -> str:
    """**일치율 단독 출력을 금지한다** (계획 §7.2). 세 값이 항상 같이 나간다."""
    k = "  n/a" if kappa is None else "{:+.2f}".format(kappa)
    c = "" if ceiling is None else "  (천장 {:.1%})".format(ceiling)
    verdict = ""
    if kappa is not None:
        if agree < nir + 0.10:
            verdict = "  <- NIR 대비 여유 부족"
        elif kappa < 0.4:
            verdict = "  <- kappa 미달 (쏠린 데이터에 편승)"
    return "{:<10} 일치율 {:.1%} / NIR {:.1%} / kappa {}{}  n={}{}".format(
        name, agree, nir, k, c, n, verdict)


# ── 결과 로딩 ─────────────────────────────────────────────────────────

def load_results(store: Path, upload_only: bool = True) -> Dict[str, dict]:
    """capture_id -> 가장 최근 결과 JSON."""
    out: Dict[str, dict] = {}
    if not store.exists():
        return out
    for cap in sorted(d for d in store.iterdir() if d.is_dir()):
        files = sorted((cap / "results").glob("*.json"))
        if not files:
            continue
        data = json.loads(files[-1].read_text(encoding="utf-8"))
        cm = data.get("capture") or {}
        if upload_only and cm.get("capture_path") != "upload":
            continue
        out[cap.name] = data
    return out


def _require(results: Dict[str, dict], what: str) -> bool:
    if results:
        return True
    print("경로 B(업로드) 결과가 없어 {} 단계를 진행할 수 없다.".format(what))
    print("  scripts/analyze.py --ingest <폴더> 로 편입한 뒤 분석을 먼저 돌린다.")
    print("  파일명 규약: <세션>_<조명>_<wb켈빈>_<iso>_<torch|noflash>_<n>.heic")
    return False


# ── agreement (목표 1 관문) ───────────────────────────────────────────

def cmd_agreement(args) -> int:
    """부위별 판정 일치율. 계획 §7.2.

    패널 CSV 형식:
        capture_id,region,rater,erythema,lesion
        20260901_143022_a4f2,cheek_l,kim,affected,normal
    """
    panel = Path(args.panel)
    if not panel.exists():
        print("패널 평가 파일이 없다: {}".format(panel))
        print("형식: capture_id,region,rater,erythema,lesion")
        print("평가자 3명이 **시스템 출력을 보지 않고** 독립적으로 매긴다 (앵커링 방지).")
        return 1

    rows = list(csv.DictReader(panel.open(encoding="utf-8")))
    results = load_results(Path(args.store), upload_only=not args.include_browser)
    if not _require(results, "일치율 평가"):
        return 1

    for axis in ("erythema", "lesion"):
        print("\n=== {} 축 ===".format("붉은기" if axis == "erythema" else "트러블"))

        by_pair: Dict[Tuple[str, str], Dict[str, str]] = defaultdict(dict)
        for r in rows:
            v = (r.get(axis) or "").strip()
            if v:
                by_pair[(r["capture_id"], r["region"])][r["rater"].strip()] = v

        truth: List[str] = []
        pred: List[str] = []
        raters_a: List[str] = []
        raters_b: List[str] = []

        for (cid, region), votes in sorted(by_pair.items()):
            if cid not in results:
                continue
            res = next((x for x in results[cid]["regions"]
                        if x["region"] == region), None)
            if res is None:
                continue
            sysv = res.get(axis + "_verdict")
            if sysv in (None, "not_measured"):
                continue        # 측정 못 한 부위는 채점 대상이 아니다 (0 이 아니다)

            counts: Dict[str, int] = defaultdict(int)
            for v in votes.values():
                counts[v] += 1
            majority = max(counts.items(), key=lambda kv: kv[1])[0]
            truth.append(majority)
            pred.append(sysv)

            names = sorted(votes)
            if len(names) >= 2:
                raters_a.append(votes[names[0]])
                raters_b.append(votes[names[1]])

        if not truth:
            print("  채점 가능한 부위-이미지 쌍이 없다.")
            continue

        agree = sum(1 for t, p in zip(truth, pred) if t == p) / len(truth)
        nir = no_information_rate(truth)
        kappa = cohens_kappa(truth, pred)

        ceiling = None
        if raters_a:
            ceiling = sum(1 for x, y in zip(raters_a, raters_b) if x == y) / len(raters_a)

        print("  " + fmt_triplet("시스템", agree, nir, kappa, ceiling, len(truth)))
        if raters_a:
            print("  " + fmt_triplet("평가자간", ceiling, no_information_rate(raters_a),
                                     cohens_kappa(raters_a, raters_b), None, len(raters_a)))
        if nir > 0.8:
            print("  ** NIR > 0.8 — 이 데이터셋으로는 목표 달성 여부를 판정하지 않는다 (§7.2).")
            print("     데이터를 더 모으거나 붉은/정상 균형을 맞춘다.")
    return 0


# ── stability (Phase 1 관문) ──────────────────────────────────────────

def _metric_table(store: Path, model: Path, cfg, cfg_hash
                  ) -> Dict[str, Dict[Tuple[str, str], Dict[RegionId, float]]]:
    """이미지마다 4개 지표를 부위별로 뽑는다. (session, lighting) 로 색인."""
    out: Dict[str, Dict[Tuple[str, str], Dict[RegionId, float]]] = {
        m: {} for m in ("raw_a", "delta_a", "raw_e", "ERI")}

    for cap in sorted(d for d in store.iterdir() if d.is_dir()):
        src = sio.find_original(cap)
        if src is None:
            continue
        meta = sio.read_capture_meta(cap)
        if meta.capture_path != "upload" or not meta.filename_parsed:
            continue
        rgb, _ = sio.load_original(src)
        ana = pipeline.analyze(rgb, cfg, cfg_hash, model, meta=meta)
        if not ana.result.quality.passed or ana.masks is None:
            continue

        obs = ana.observation
        linear = color.srgb_to_linear(obs.canonical_rgb)
        e, m = color.chromophore_axes(color.optical_density(linear))
        lab = color.linear_to_lab(linear)
        a_star = lab[..., 1]
        skin = ana.masks.skin
        a_ref = float(np.percentile(a_star[skin], 25))
        e_ref = float(np.percentile(e[skin], 25))

        key = ("{}_{}".format(meta.session, meta.shot_index), meta.lighting or "?")
        for name, field in (("raw_a", a_star), ("delta_a", a_star - a_ref),
                            ("raw_e", e), ("ERI", None)):
            vals: Dict[RegionId, float] = {}
            for rid in ALL_REGIONS:
                mk = ana.masks.regions[rid]
                if ana.masks.coverage(rid) < cfg.baseline.min_coverage or not mk.any():
                    continue
                if name == "ERI":
                    vals[rid] = float(np.median(ana.deviation[mk]))
                elif name == "raw_e":
                    vals[rid] = float(np.median(field[mk]) - e_ref)
                else:
                    vals[rid] = float(np.median(field[mk]))
            out[name][key] = vals
    return out


def cmd_stability(args) -> int:
    store = Path(args.store)
    cfg = sio.load_config(Path(args.config))
    cfg_hash = sio.config_hash(cfg)
    if not store.exists():
        print("store 없음")
        return 1

    table = _metric_table(store, Path(args.model), cfg, cfg_hash)
    lightings = sorted({k[1] for v in table.values() for k in v})
    subjects = sorted({k[0] for v in table.values() for k in v})

    if len(lightings) < 2 or len(subjects) < 2:
        print("조명 조건 {}종 / 반복 {}건 — ICC 를 내려면 각각 2 이상이어야 한다."
              .format(len(lightings), len(subjects)))
        print("계획 §6.1: 조명 4종 x 반복 2장 = 16장/세션, 여러 세션 반복.")
        return 1

    print("조명 {}종, 반복 {}건\n".format(len(lightings), len(subjects)))
    print("{:<10} {:>9} {:>9} {:>9} {:>9}".format("부위", "raw a*", "delta a*", "raw e", "ERI"))
    print("-" * 50)
    for rid in ALL_REGIONS:
        cells = []
        for name in ("raw_a", "delta_a", "raw_e", "ERI"):
            mat = np.full((len(subjects), len(lightings)), np.nan)
            for i, s in enumerate(subjects):
                for j, l in enumerate(lightings):
                    v = table[name].get((s, l), {}).get(rid)
                    if v is not None:
                        mat[i, j] = v
            val = icc21(mat)
            cells.append("     n/a" if val is None else "{:9.2f}".format(val))
        print("{:<10} {}".format(rid.value, "".join(cells)))

    print("\n**핵심은 raw 대비 정규화의 차이다** (계획 §6.1).")
    print("ERI >> raw a* 이면 자기참조가 실제로 일하고 있다는 뜻이다.")
    print("\n주의: 부위별 기준선(mu_r)은 부위마다 **상수** 오프셋이므로")
    print("      부위별 ICC 를 바꾸지 않는다. mu_r 이 개선하는 것은 ICC 가 아니라")
    print("      §7.2 판정 정확도다 — 두 지표를 혼동하지 말 것.")
    return 0


# ── flashpair ────────────────────────────────────────────────────────

def cmd_flashpair(args) -> int:
    """플래시 on/off 쌍 차분 (계획 §8 L1-2).

    두 장을 차분하면 주변광이 상쇄되고 **SPD 가 알려진 광원**만 남는다.
    성공하면 혼합 조명 문제(§7.3-7)를 소프트웨어만으로 직접 해결한다.
    """
    results = load_results(Path(args.store))
    if not _require(results, "플래시 쌍 분석"):
        return 1

    pairs: Dict[Tuple[str, str, int], Dict[bool, dict]] = defaultdict(dict)
    for cid, data in results.items():
        cm = data.get("capture") or {}
        if not cm.get("filename_parsed"):
            continue
        key = (cm.get("session"), cm.get("lighting"), cm.get("shot_index"))
        pairs[key][bool(cm.get("torch"))] = data

    complete = {k: v for k, v in pairs.items() if len(v) == 2}
    if not complete:
        print("torch/noflash 쌍이 없다. 같은 세션·조명·촬영번호로 두 장을 찍는다.")
        print("  ISO·셔터·WB 를 고정한 채 **토치만 토글**해야 차분이 성립한다 (§3.1).")
        return 1

    print("완성된 쌍 {}개\n".format(len(complete)))
    print("{:<10} {:>10} {:>10} {:>10}".format("부위", "무플래시", "토치", "차이"))
    print("-" * 44)
    for rid in ALL_REGIONS:
        off, on = [], []
        for v in complete.values():
            for torch, data in v.items():
                r = next((x for x in data["regions"] if x["region"] == rid.value), None)
                if r and r.get("median_d") is not None:
                    (on if torch else off).append(r["median_d"])
        if not off or not on:
            continue
        print("{:<10} {:>10.4f} {:>10.4f} {:>+10.4f}".format(
            rid.value, float(np.mean(off)), float(np.mean(on)),
            float(np.mean(on)) - float(np.mean(off))))
    print("\n차이가 조명 조건에 따라 **일정하면** 플래시 성분 분리가 잘 되는 것이다.")
    return 0


# ── synth ────────────────────────────────────────────────────────────

def cmd_synth(args) -> int:
    """합성 Delta 복원 선형성 (계획 §6.2). **선형성이 깨지면 지표 자체가 실패다.**"""
    store = Path(args.store)
    cfg = sio.load_config(Path(args.config))
    cfg_hash = sio.config_hash(cfg)
    results = load_results(store)
    if not _require(results, "합성 주입 검증"):
        return 1

    wl, hb, _ = sio.load_extinction_csv(ROOT / "data/spectra/hb_melanin_extinction.csv")
    v_h = color.absorber_rgb_direction(wl, hb, depth_weighted=True)
    print("v_h = {}  (G 가 최대여야 물리적으로 맞다)\n".format(np.round(v_h, 3)))

    deltas = [0.0, 0.02, 0.04, 0.08, 0.12]
    rows = []
    for cid in sorted(results)[:args.limit]:
        cap = store / cid
        src = sio.find_original(cap)
        if src is None:
            continue
        meta = sio.read_capture_meta(cap)
        rgb, _ = sio.load_original(src)
        obs = face_mod.detect(rgb, Path(args.model))
        if obs is None:
            continue

        base_lin = color.srgb_to_linear(obs.canonical_rgb)
        h, w = base_lin.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        patch = ((xx - w * 0.72) ** 2 + (yy - h * 0.52) ** 2) < (0.09 * w) ** 2

        rec = []
        for d in deltas:
            amt = np.where(patch, d, 0.0)
            lin = np.clip(base_lin * np.power(10.0, -amt[..., None] * v_h[None, None, :]),
                          1e-4, 1.0).astype(np.float32)
            u8 = np.clip(np.where(lin <= 0.0031308, lin * 12.92,
                                  1.055 * np.power(lin, 1 / 2.4) - 0.055) * 255.0,
                         0, 255).astype(np.uint8)
            probe = face_mod.FaceObservation(
                canonical_rgb=u8, frame_valid=obs.frame_valid, landmarks=obs.landmarks,
                face_poly=obs.face_poly, affine=obs.affine,
                native_ipd_px=obs.native_ipd_px, yaw_deg=obs.yaw_deg,
                pitch_deg=obs.pitch_deg, roll_deg=obs.roll_deg,
                confidence=obs.confidence, n_faces=obs.n_faces)
            ana = pipeline.analyze(rgb, cfg, cfg_hash, Path(args.model),
                                   meta=meta, observation=probe)
            if ana.deviation is None or ana.skin is None:
                rec = []
                break
            sel = patch & ana.skin
            rec.append(float(np.median(ana.deviation[sel])) if sel.any() else np.nan)

        if len(rec) == len(deltas) and not any(math.isnan(v) for v in rec):
            rows.append(rec)
            print("  {} 복원 {}".format(cid, [round(v - rec[0], 4) for v in rec]))

    if not rows:
        print("복원 가능한 이미지가 없다.")
        return 1

    y = np.array(rows) - np.array(rows)[:, :1]
    x = np.array(deltas)
    slope, intercept = np.polyfit(np.tile(x, len(rows)), y.ravel(), 1)
    pred = slope * np.tile(x, len(rows)) + intercept
    ss_res = float(((y.ravel() - pred) ** 2).sum())
    ss_tot = float(((y.ravel() - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    expected = float(v_h[1] - v_h[0])

    print("\n기울기 {:.3f} (기대 {:.3f})   R^2 {:.4f}   n={}장".format(
        slope, expected, r2, len(rows)))
    ok = (0.7 <= slope / expected <= 1.3) and r2 >= 0.9
    print("PASS 조건(§7.2): 상대기울기 0.7~1.3 이고 R^2 >= 0.9  ->  {}".format(
        "충족" if ok else "미충족"))
    return 0


# ── negative ─────────────────────────────────────────────────────────

def cmd_negative(args) -> int:
    """음성 대조군 (계획 §6.5). 전부 공짜다."""
    import cv2
    store = Path(args.store)
    cfg = sio.load_config(Path(args.config))
    cfg_hash = sio.config_hash(cfg)
    model = Path(args.model)

    print("=== 비얼굴 거부 ===")
    rng = np.random.default_rng(0)
    for name, img in (("난수 잡음", rng.integers(0, 255, (1600, 900, 3), dtype=np.uint8)),
                      ("단색", np.full((1600, 900, 3), 120, np.uint8))):
        ana = pipeline.analyze(img, cfg, cfg_hash, model)
        print("  {:<10} -> {} (점수를 내지 말고 거부해야 한다)".format(
            name, ana.result.quality.reason.value))

    results = load_results(store)
    if not results:
        print("\n(store 에 업로드 이미지가 없어 JPEG/리사이즈 drift 는 건너뛴다)")
        return 0

    cid = sorted(results)[0]
    src = sio.find_original(store / cid)
    rgb, meta = sio.load_original(src)
    base = pipeline.analyze(rgb, cfg, cfg_hash, model, meta=meta)
    ref = {r.region: r.median_d for r in base.result.regions}

    def drift(tag, img):
        ana = pipeline.analyze(img, cfg, cfg_hash, model, meta=meta)
        ds = [abs(r.median_d - ref[r.region])
              for r in ana.result.regions
              if r.median_d is not None and ref.get(r.region) is not None]
        print("  {:<16} 최대 drift {:.5f}".format(
            tag, max(ds) if ds else float("nan")))

    print("\n=== JPEG 재인코딩 drift ===")
    for q in (95, 75, 50):
        ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        drift("q={}".format(q), cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1])

    print("\n=== 리사이즈 drift (해상도 바닥 확인) ===")
    for s in (0.5, 0.25):
        small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        drift("{}x".format(s), small)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default=str(ROOT / "data" / "store"))
    p.add_argument("--config", default=str(ROOT / "config" / "default.toml"))
    p.add_argument("--model", default=str(ROOT / "models" / "face_landmarker.task"))
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("agreement", help="부위별 판정 일치율 (목표 1 관문)")
    a.add_argument("--panel", default=str(ROOT / "data" / "panel_ratings.csv"))
    a.add_argument("--include-browser", action="store_true",
                   help="브라우저 캡처도 포함 (성능 주장에는 쓰지 말 것 — §7.3-14)")
    a.set_defaults(func=cmd_agreement)

    s = sub.add_parser("stability", help="조명 간 ICC (Phase 1 관문)")
    s.set_defaults(func=cmd_stability)

    f = sub.add_parser("flashpair", help="플래시 on/off 쌍 차분")
    f.set_defaults(func=cmd_flashpair)

    y = sub.add_parser("synth", help="합성 Delta 복원 선형성")
    y.add_argument("--limit", type=int, default=5)
    y.set_defaults(func=cmd_synth)

    n = sub.add_parser("negative", help="음성 대조군")
    n.set_defaults(func=cmd_negative)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
