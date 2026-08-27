"""로컬 웹 서버. 촬영 안내 -> 품질 판정 -> 결과 근거 -> 피드백 (계획 §5.3).

**127.0.0.1 바인딩 고정. 이미지는 이 컴퓨터 밖으로 나가지 않는다.**

계약 9번: 여기서는 `pipeline` 만 호출한다. `core/` 를 직접 import 하지 않는다 —
웹이 코어를 우회해 조립하기 시작하면 그 로직은 모바일 포팅 때 통째로 사라진다.
사유 코드는 enum 이 아니라 **문자열**로 다룬다 (JSON 이 실어 나르는 형태 그대로).
"""

import base64
import io
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from skin_detector import imageio as sio, pipeline, render  # noqa: E402

STORE = ROOT / "data" / "store"
STATIC = Path(__file__).parent / "static"
MODEL = ROOT / "models" / "face_landmarker.task"
CONFIG = ROOT / "config" / "default.toml"

app = FastAPI(title="skin_detector", docs_url=None, redoc_url=None)


# ── 사유 코드 -> 사람이 읽을 안내 (목표 2의 실체) ────────────────────
#
# 품질 게이트가 ReasonCode 를 내는 구조는 이미 core/quality.py 에 있다.
# 여기 있는 것은 그 코드를 **재촬영 행동으로 번역**하는 표 하나뿐이다.
# §7.3 한계 중 혼합 광원·과노출·앞머리 가림·흐림은 알고리즘으로 못 고치고
# 재촬영으로만 고쳐진다 — 그래서 이 표가 성능의 일부다.

GUIDANCE: Dict[str, Dict[str, str]] = {
    "no_face": {
        "title": "얼굴을 찾지 못했습니다",
        "how": "얼굴 전체가 화면에 들어오도록 정면에서 다시 찍어주세요.",
        "retake": "yes"},
    "multiple_faces": {
        "title": "얼굴이 두 명 이상입니다",
        "how": "한 사람만 나오도록 다시 찍어주세요.",
        "retake": "yes"},
    "low_landmark_confidence": {
        "title": "얼굴 인식이 불안정합니다",
        "how": "조명을 밝히고 얼굴을 가리는 것이 없는지 확인한 뒤 다시 찍어주세요.",
        "retake": "yes"},
    "low_resolution": {
        "title": "해상도가 부족합니다",
        "how": "더 가까이에서, 또는 더 높은 화질로 촬영해 주세요.",
        "retake": "yes"},
    "face_out_of_frame": {
        "title": "얼굴이 화면 밖으로 잘렸습니다",
        "how": "세로 사진에서 흔합니다. 얼굴을 화면 가운데에 두고 다시 찍어주세요.",
        "retake": "yes"},
    "pose_out_of_range": {
        "title": "고개가 돌아가 있습니다",
        "how": "정면을 바라보고 다시 찍어주세요.",
        "retake": "yes"},
    "blurry": {
        "title": "초점이 흔들렸습니다",
        "how": "기기를 거치대에 고정하고 다시 찍어주세요.",
        "retake": "yes"},
    "overexposed": {
        "title": "너무 밝아 색이 날아갔습니다",
        "how": "조명을 줄이거나 한 걸음 물러나 주세요. 색이 포화되면 측정할 수 없습니다.",
        "retake": "yes"},
    "underexposed": {
        "title": "너무 어둡습니다",
        "how": "조명을 밝히거나 창을 마주 보고 다시 찍어주세요.",
        "retake": "yes"},
    "non_uniform_illumination": {
        "title": "한쪽에서만 빛이 들어옵니다",
        "how": "창을 마주 보거나 조명을 정면에 두고 다시 찍어주세요. "
               "측면광은 그림자를 붉은기로 오인하게 만듭니다.",
        "retake": "yes"},
    "forehead_occluded": {
        "title": "앞머리에 이마가 가렸습니다",
        "how": "머리를 넘기면 이마까지 측정됩니다. 다른 부위 결과는 그대로 유효합니다.",
        "retake": "no"},
    "low_coverage": {
        "title": "측정 가능한 피부가 부족합니다",
        "how": "가리는 것을 치우거나 조명을 고르게 하고 다시 찍어주세요.",
        "retake": "no"},
    "lesion_disabled_low_resolution": {
        "title": "개별 트러블 검출을 건너뜁니다",
        "how": "해상도가 낮아 붉은기만 측정합니다. 더 높은 화질로 찍으면 트러블도 표시됩니다.",
        "retake": "no"},
    "illumination_check_unavailable": {
        "title": "조명 균일성을 확인하지 못했습니다",
        "how": "참조 부위가 부족합니다. 결과는 참고용으로만 보세요.",
        "retake": "no"},
    "ok": {"title": "정상", "how": "", "retake": "no"},
}

REGION_LABEL = {
    "forehead": "이마", "cheek_l": "왼쪽 볼", "cheek_r": "오른쪽 볼",
    "nose": "코", "chin": "턱",
}

_cfg = None
_cfg_hash = ""


def cfg():
    global _cfg, _cfg_hash
    if _cfg is None:
        _cfg = sio.load_config(CONFIG)
        _cfg_hash = sio.config_hash(_cfg)
    return _cfg, _cfg_hash


def guidance_for(code: str) -> Dict[str, str]:
    return GUIDANCE.get(code, {
        "title": code, "how": "다시 촬영해 주세요.", "retake": "yes"})


def humanize(result: Dict[str, Any]) -> Dict[str, Any]:
    """결과 JSON 에 안내 문장과 '근거'를 덧붙인다. 숫자는 그대로 둔다."""
    q = result.get("quality", {})
    out: Dict[str, Any] = {
        "passed": bool(q.get("passed")),
        "guidance": guidance_for(q.get("reason", "ok")),
        "metrics": q.get("metrics", {}),
        "illumination_check": result.get("illumination_check"),
        "lesion_enabled": result.get("lesion_detection_enabled"),
        "ita_deg": result.get("ita_deg"),
        "algo_version": result.get("algo_version"),
        "config_hash": result.get("config_hash"),
        "regions": [],
        "lesions": result.get("lesions", []),
        "notes": [],
    }

    if not result.get("lesion_detection_enabled"):
        out["notes"].append(guidance_for(
            result.get("lesion_reason", "lesion_disabled_low_resolution")))
    if result.get("illumination_check") == "unavailable":
        out["notes"].append(guidance_for("illumination_check_unavailable"))
    elif result.get("illumination_check") == "flagged":
        out["notes"].append({
            "title": "조명 구배가 감지되었습니다",
            "how": "얼굴 안에서 밝기 차이가 큽니다. 정면 조명으로 다시 찍으면 더 정확합니다.",
            "retake": "no"})

    base = result.get("baseline") or {}
    if base and not base.get("anatomical_prior_calibrated", False):
        out["notes"].append({
            "title": "부위별 기준이 아직 보정되지 않았습니다",
            "how": "코와 볼은 원래 이마·턱보다 붉습니다. 사진이 더 쌓이면 이 차이를 "
                   "빼고 볼 수 있습니다. 지금은 부위 간 비교를 조심해서 보세요.",
            "retake": "no"})

    for r in result.get("regions", []):
        item = {
            "region": r["region"],
            "label": REGION_LABEL.get(r["region"], r["region"]),
            "measured": r.get("score_ordinal_0_100") is not None,
            "score_ordinal_0_100": r.get("score_ordinal_0_100"),
            "erythema_verdict": r.get("erythema_verdict"),
            "lesion_verdict": r.get("lesion_verdict"),
            "lesion_count": r.get("lesion_count"),
            # ── '왜 그렇게 나왔는지'의 근거. 접히는 영역에 그대로 노출한다. ──
            "evidence": {
                "coverage": r.get("coverage"),
                "area_fraction": r.get("area_fraction"),
                "intensity": r.get("intensity"),
                "median_d": r.get("median_d"),
            },
        }
        if not item["measured"]:
            # **0 점이 아니라 '측정 못 함 + 이유'** 로 표시한다 (계약 4번).
            item["reason"] = guidance_for(r.get("reason", "low_coverage"))
        out["regions"].append(item)

    if base:
        out["baseline"] = base
    return out


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    _, h = cfg()
    return {"ok": True, "algo_version": pipeline.ALGO_VERSION, "config_hash": h,
            "model_present": MODEL.exists()}


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...),
                  capture_path: str = Form("browser")):
    if capture_path not in ("browser", "upload"):
        raise HTTPException(400, "capture_path 는 browser 또는 upload 여야 한다")

    data = await image.read()
    if not data:
        raise HTTPException(400, "빈 파일")

    suffix = Path(image.filename or "capture.jpg").suffix.lower() or ".jpg"
    tmp_dir = Path(tempfile.mkdtemp(prefix="skin_"))
    # 파일명 규약(§3.1) 파싱은 원래 이름을 유지해야 동작한다.
    tmp = tmp_dir / (Path(image.filename or "capture").stem + suffix)
    tmp.write_bytes(data)

    warnings: List[str] = []
    try:
        cap_dir, meta, warnings = sio.ingest(tmp, STORE)
        meta.capture_path = capture_path
        rgb, _ = sio.load_original(tmp)
    finally:
        tmp.unlink(missing_ok=True)
        tmp_dir.rmdir()

    conf, chash = cfg()
    ana = pipeline.analyze(rgb, conf, chash, MODEL, meta=meta)
    sio.write_result(cap_dir, ana.result)

    payload = humanize(sio.to_jsonable(ana.result))
    payload["capture_id"] = meta.capture_id
    payload["capture_path"] = capture_path
    payload["warnings"] = warnings

    overlay = render.render_overlay(ana)
    if overlay is not None:
        import cv2
        ok, buf = cv2.imencode(".png", overlay[:, :, ::-1])
        if ok:
            payload["overlay"] = "data:image/png;base64," + \
                base64.b64encode(buf.tobytes()).decode("ascii")

    if capture_path == "browser":
        payload["notes"].append({
            "title": "간편 촬영 모드입니다",
            "how": "브라우저 촬영은 화이트밸런스·감도를 고정할 수 없습니다. "
                   "정확한 측정이 필요하면 카메라 앱으로 찍어 업로드해 주세요.",
            "retake": "no"})

    return JSONResponse(payload)


@app.post("/api/feedback")
async def feedback(capture_id: str = Form(...), verdict: str = Form(...),
                   region: str = Form(""), algo_version: str = Form("")):
    """결과 화면 피드백 (§5.4).

    약한 라벨이고 **부정 편향**이 있다 — 사용자는 틀렸을 때 더 자주 누른다.
    비율을 정확도로 읽으면 안 되고 실패 사례 수집기로만 쓴다 (§7.3-15).
    """
    if verdict not in ("correct", "over_detected", "missed"):
        raise HTTPException(400, "verdict 가 올바르지 않다")
    cap_dir = STORE / capture_id
    if not cap_dir.exists():
        raise HTTPException(404, "capture 를 찾을 수 없다")
    sio.append_feedback(cap_dir, {
        "verdict": verdict, "region": region or None,
        "algo_version": algo_version or pipeline.ALGO_VERSION})
    return {"ok": True}


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
