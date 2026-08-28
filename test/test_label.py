"""라벨링 도구(scripts/label.py)의 규약 검증.

GUI 는 테스트하지 않는다 — 사람이 클릭하는 부분이다. 대신 **라벨 파일이 지켜야 하는
규약**을 고정한다. 라벨은 사람의 시간이 들어간 자산이고 알고리즘보다 오래 살아야 하므로,
스키마가 조용히 어긋나면 이미 찍은 라벨이 전부 무효가 된다.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import label as lab  # noqa: E402
from skin_detector.core.types import ALL_REGIONS  # noqa: E402


def test_label_tool_covers_every_region():
    """부위를 늘리면 라벨 도구도 같이 늘어나야 한다.

    `types.py` 에 부위를 추가했는데 `REGION_KEYS` 를 안 고치면, 그 부위는 **영원히
    `unlabelable` 로 남는다.** 라벨러는 누를 키가 없다는 사실을 알아채지 못하고,
    그 부위의 학습·평가 데이터만 조용히 비어 있게 된다.
    """
    mapped = {rid for rid, _ in lab.REGION_KEYS.values()}
    declared = {r.value for r in ALL_REGIONS}
    assert mapped == declared, (
        "REGION_KEYS 와 ALL_REGIONS 가 어긋난다. 누를 키가 없는 부위: {}"
        .format(sorted(declared - mapped)))


def test_region_default_is_unlabelable_not_clean():
    """계약 4번(null != zero)을 라벨에도 적용한다.

    기본값이 `clean` 이면 **라벨러가 보지도 않은 부위가 "깨끗함"으로 학습된다.**
    앞머리에 가린 이마가 정상 피부의 근거로 쓰이는 순간 모델은 가려짐을 정상으로 배운다.
    """
    lb = lab.empty_label("cap1", "a.jpg", 100, 200, "mh")
    assert set(lb["regions"]) == {r.value for r in ALL_REGIONS}
    assert set(lb["regions"].values()) == {"unlabelable"}
    assert lb["lesions"] == []


def test_unlabelable_is_a_distinct_state():
    """`clean` · `trouble` · `unlabelable` 셋은 서로 접히면 안 된다."""
    assert len(set(lab.REGION_STATES)) == 3
    assert "clean" in lab.REGION_STATES and "unlabelable" in lab.REGION_STATES


def test_labelers_do_not_overwrite_each_other(tmp_path):
    """LIMITS §1 — 평가셋은 평가자 3명의 **독립** 라벨이 있어야 다수결이 성립한다.

    한 경로에 덮어쓰면 독립성이 사라지고, 그러면 천장(평가자 간 일치율)을 잴 수 없다.
    """
    paths = {lab.label_path(tmp_path, who, "cap1") for who in ("mh", "a", "b")}
    assert len(paths) == 3


def test_save_load_round_trip(tmp_path):
    """저장한 라벨이 그대로 돌아온다. 좌표는 **원본 좌표**다."""
    path = lab.label_path(tmp_path, "mh", "cap1")
    lb = lab.empty_label("cap1", "a.jpg", 4000, 3000, "mh")
    lb["lesions"].append({"x": 1204.0, "y": 883.0, "type": "papule",
                          "confidence": "certain"})
    lb["regions"]["forehead"] = "trouble"
    lab.save_label(path, lb)

    back = lab.load_label(path, "cap1", "a.jpg", 4000, 3000, "mh")
    assert back["lesions"] == lb["lesions"]
    assert back["regions"]["forehead"] == "trouble"
    assert back["image_size"] == [4000, 3000]
    assert back["labeled_at"] is not None

    # 원본 해상도 좌표여야 한다 — 정규 프레임(768)으로 저장하면 랜드마커가 바뀌는 순간 죽는다.
    assert back["lesions"][0]["x"] > 768


def test_load_fills_regions_added_later(tmp_path):
    """스키마가 늘어나도 예전 라벨 파일이 죽지 않는다."""
    path = lab.label_path(tmp_path, "mh", "cap1")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "capture_id": "cap1", "lesions": [], "regions": {"forehead": "trouble"},
    }), encoding="utf-8")

    back = lab.load_label(path, "cap1", "a.jpg", 100, 100, "mh")
    assert back["regions"]["forehead"] == "trouble"
    assert back["regions"]["nose"] == "unlabelable"     # 빠진 키는 0 이 아니라 미측정


def test_lesion_type_cycle_starts_at_none():
    """type 은 v1 학습에 쓰이지 않는다 (LIMITS §3-1). 기본값이 None 이어야
    라벨러가 아무것도 안 눌러도 '모름'으로 남는다 — 임의의 타입이 찍히면 안 된다."""
    assert lab.LESION_TYPES[0] is None


@pytest.mark.parametrize("key,rid", [(k, v[0]) for k, v in lab.REGION_KEYS.items()])
def test_cheek_hint_states_image_side(key, rid):
    """CONTRACT 부수 규약: 좌/우는 **피사체 기준**이고 화면에서는 뒤집혀 보인다.

    이 힌트가 UI 에 없으면 라벨러는 반드시 좌우를 뒤집어 찍고, 그 오염은 학습이
    끝난 뒤에야 드러난다.
    """
    _, hint = lab.REGION_KEYS[key]
    if rid in ("cheek_l", "cheek_r"):
        assert "화면" in hint, "볼은 화면 기준 방향을 함께 표시해야 한다: " + hint
