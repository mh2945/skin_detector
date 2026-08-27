"""웹 계층 검증 — 얇은 껍데기가 실제로 얇고, 안내 문장이 실제로 붙는가.

`web/server.py` 270줄도 이 파일이 생기기 전까지 한 번도 실행되지 않았다.

**신규 의존성은 없다.** FastAPI `TestClient` 가 이미 설치된 환경에서 동작한다.

store 는 반드시 `tmp_path` 로 갈아끼운다 — 테스트가 사용자의 실제
`data/store/` 에 사진을 쌓으면 안 된다 — 그곳은 불변 원시 레이어다.

MediaPipe 는 합성 얼굴을 얼굴로 인식하지 않으므로 분석 결과는 `no_face` 가 된다.
그것으로 충분하다 — 검사 대상은 색 과학이 아니라 **업로드 -> 편입 -> 분석 -> 안내**
경로 자체이고, 게이트 실패 경로야말로 목표 2(촬영 안내)의 핵심이다.
"""

import io
import json
import time

import pytest
from fastapi.testclient import TestClient

from conftest import synthetic_face_linear, to_uint8_srgb
from skin_detector.core.types import FATAL_REASONS, ReasonCode

from web.server import GUIDANCE, app, guidance_for


@pytest.fixture
def client(tmp_path, monkeypatch):
    """store 를 tmp 로 갈아끼운 클라이언트. 실제 data/store 를 절대 건드리지 않는다."""
    monkeypatch.setattr("web.server.STORE", tmp_path)
    return TestClient(app)


def png_bytes(name="capture.png"):
    from PIL import Image
    arr = to_uint8_srgb(synthetic_face_linear(texture=0.05, noise=0.004))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return name, buf.getvalue(), "image/png"


# ── GUIDANCE 표: 목표 2 의 자산 ───────────────────────────────────────

def test_guidance_covers_every_reason_code():
    """새 `ReasonCode` 가 생기면 안내 문장도 반드시 따라와야 한다.

    규칙으로 선언되어 있었지만 **강제하는 장치가 없었다.**
    빠지면 사용자는 왜 실패했는지 모른 채 같은 실수를 반복한다.
    """
    codes = {c.value for c in ReasonCode}
    assert set(GUIDANCE) == codes, {
        "missing": sorted(codes - set(GUIDANCE)),
        "extra": sorted(set(GUIDANCE) - codes),
    }


def test_every_guidance_entry_is_complete():
    for code, entry in GUIDANCE.items():
        assert set(entry) >= {"title", "how", "retake"}, code
        assert entry["retake"] in ("yes", "no"), code
        if code != ReasonCode.OK.value:
            assert entry["title"].strip(), code
            assert entry["how"].strip(), code


def test_fatal_reasons_ask_for_a_retake():
    """전역 게이트 실패는 재촬영으로만 고쳐진다 — 그렇게 안내해야 한다.

    반대로 부위 한정·파이프라인 상태 사유는 재촬영을 요구하면 안 된다.
    """
    for code in ReasonCode:
        entry = GUIDANCE[code.value]
        if code in FATAL_REASONS:
            assert entry["retake"] == "yes", code
        else:
            assert entry["retake"] == "no", code


def test_guidance_for_unknown_code_degrades_gracefully():
    entry = guidance_for("something_new_from_the_future")
    assert entry["retake"] == "yes"
    assert entry["how"].strip()


# ── 엔드포인트 ────────────────────────────────────────────────────────

def test_health_reports_version_and_model(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["algo_version"].startswith("v")
    assert len(body["config_hash"]) == 6
    assert isinstance(body["model_present"], bool)


def test_index_is_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "html" in r.headers["content-type"]


def test_rejects_unknown_capture_path(client):
    r = client.post("/api/analyze", files={"image": png_bytes()},
                    data={"capture_path": "telepathy"})
    assert r.status_code == 400


def test_rejects_empty_upload(client):
    r = client.post("/api/analyze",
                    files={"image": ("empty.png", b"", "image/png")},
                    data={"capture_path": "upload"})
    assert r.status_code == 400


def test_rejects_unknown_feedback_verdict(client):
    r = client.post("/api/feedback",
                    data={"capture_id": "whatever", "verdict": "shrug"})
    assert r.status_code == 400


# ── 경로 A / 경로 B 를 섞지 않는가 (회귀 테스트) ──────────────────────

@pytest.mark.parametrize("path", ["browser", "upload"])
def test_capture_path_is_persisted_to_disk(client, tmp_path, path):
    """`capture.json` 에 실제 촬영 경로가 기록되어야 한다.

    회귀 테스트다. 예전에는 `ingest()` 가 capture.json 을 먼저 쓰고 **그 뒤에**
    `meta.capture_path` 를 대입해서, 브라우저 캡처가 디스크에는 `"upload"` 로
    남았다. capture.json 은 불변(계약 10번)이라 되돌릴 수도 없고,
    `validate.py` 의 upload 전용 필터가 조용히 오염된다 —
    `docs/VALIDATION.md` §0 이 "절대 섞지 않는다"고 못박은 그 혼입이다.
    """
    r = client.post("/api/analyze", files={"image": png_bytes()},
                    data={"capture_path": path})
    assert r.status_code == 200, r.text
    capture_id = r.json()["capture_id"]

    on_disk = json.loads((tmp_path / capture_id / "capture.json")
                         .read_text(encoding="utf-8"))
    assert on_disk["capture_path"] == path, (
        "응답은 {} 인데 디스크에는 {} 로 기록됐다"
        .format(path, on_disk["capture_path"]))


def test_browser_capture_carries_a_caveat_note(client):
    r = client.post("/api/analyze", files={"image": png_bytes()},
                    data={"capture_path": "browser"})
    assert r.status_code == 200
    titles = [n["title"] for n in r.json()["notes"]]
    assert any("간편" in t for t in titles), titles


# ── 결과 표현: 미측정은 0 이 아니다 ───────────────────────────────────

def test_gate_failure_is_explained_not_scored(client):
    """합성 이미지는 얼굴로 인식되지 않는다. 그때 점수가 아니라 **이유**가 나가야 한다."""
    r = client.post("/api/analyze", files={"image": png_bytes()},
                    data={"capture_path": "upload"})
    assert r.status_code == 200
    body = r.json()

    assert body["passed"] is False
    # 안내 문장이 실제로 붙어 나간다 (목표 2).
    assert body["guidance"]["title"].strip()
    assert body["guidance"]["how"].strip()
    assert body["guidance"]["retake"] == "yes"

    # 못 잰 부위는 0 점이 아니라 null + 사유여야 한다 (계약 4번).
    assert body["regions"], "부위 항목은 실패해도 5개가 나와야 한다"
    for region in body["regions"]:
        assert region["measured"] is False, region
        assert region["score_ordinal_0_100"] is None, region
        assert region["reason"]["how"].strip(), region
        assert region["label"].strip()


def test_ingest_is_idempotent_for_the_same_bytes(client, tmp_path):
    """같은 사진을 두 번 올려도 capture_id 가 같고 디렉터리가 늘지 않는다.

    **초 경계를 일부러 넘긴다.** capture_id 의 시각 부분은 EXIF 가 없으면 파일
    mtime 에서 오는데, 브라우저 캡처는 EXIF 없는 PNG 라 업로드마다 시각이 달라진다.
    두 업로드가 같은 1초 안에 들어가면 이 버그가 보이지 않는다 — 실제로 그렇게
    통과했다가 초 경계를 넘긴 실행에서 실패하는 flaky 테스트였다.
    """
    name, data, ct = png_bytes()
    first = client.post("/api/analyze", files={"image": (name, data, ct)},
                        data={"capture_path": "upload"}).json()["capture_id"]
    time.sleep(1.1)
    second = client.post("/api/analyze", files={"image": (name, data, ct)},
                         data={"capture_path": "upload"}).json()["capture_id"]
    assert first == second, "같은 사진이 서로 다른 캡처로 쌓인다"
    assert len([p for p in tmp_path.iterdir() if p.is_dir()]) == 1


def test_feedback_is_appended(client, tmp_path):
    capture_id = client.post("/api/analyze", files={"image": png_bytes()},
                             data={"capture_path": "upload"}).json()["capture_id"]
    for verdict in ("correct", "over_detected"):
        r = client.post("/api/feedback",
                        data={"capture_id": capture_id, "verdict": verdict})
        assert r.status_code == 200, r.text

    entries = json.loads((tmp_path / capture_id / "feedback.json")
                         .read_text(encoding="utf-8"))
    assert [e["verdict"] for e in entries] == ["correct", "over_detected"]


def test_feedback_for_unknown_capture_is_404(client):
    r = client.post("/api/feedback",
                    data={"capture_id": "20990101_000000_deadbeef",
                          "verdict": "correct"})
    assert r.status_code == 404
