"""병변 점 라벨링 도구 (ROADMAP L3).

    python scripts/label.py --labeler mh                      # store 전체를 순회
    python scripts/label.py --labeler mh --image a.heic       # 파일 하나만
    python scripts/label.py --labeler mh --only-unlabeled     # 아직 안 찍은 것만

라벨은 `data/labels/<labeler>/<capture_id>.json` 에 **원본 좌표**로 남는다.

**왜 원본 좌표인가** — 정규 프레임(768x768, IPD=320)으로 저장하면 `face.py` 의 랜드마커
버전이나 정규 프레임 정의가 바뀌는 순간 **이미 찍은 라벨이 전부 무효가 된다.** 라벨은
사람의 시간이 들어간 자산이고 알고리즘보다 오래 살아야 한다. 계약 10번이 원본을
불변으로 잡은 것과 같은 논리다. 정규 프레임 투영은 학습 시점에 `face.py` 가 한다.

**왜 라벨러별로 폴더를 나누는가** — LIMITS §1 이 평가셋을 "평가자 3명이 독립적으로 매긴
다수결"로 정의했다. 한 파일에 덮어쓰면 독립성이 사라진다.

**결과 오버레이를 절대 띄우지 않는다** — LIMITS §1 의 "시스템 출력은 보여주지 않는다
(앵커링 방지)". 이 스크립트는 `pipeline` 을 import 조차 하지 않는다.

조작:
    좌클릭          병변 추가 (certain)
    shift+좌클릭    병변 추가 (unsure — 학습에서 제외되고 평가에서 무시된다)
    우클릭          가장 가까운 병변 삭제
    t               마지막 병변의 type 순환 (none/papule/pustule/comedone/pih)
    1~5             부위 순환 clean -> trouble -> unlabelable
    s               저장
    n / b           저장하고 다음 / 이전
    q               저장하고 종료
    확대·이동은 matplotlib 툴바를 쓴다 (툴바가 켜져 있으면 클릭은 무시된다).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _console import setup_console  # noqa: E402

setup_console()          # cp949 리다이렉트에서 죽지 않게 한다. import 직후여야 한다.

from skin_detector import imageio as sio  # noqa: E402
from skin_detector.core.types import ALL_REGIONS  # noqa: E402

SCHEMA_VERSION = 1
LABEL_VERSION = "v1"

# 부위 상태 3값. `clean` 과 `unlabelable` 은 다르다 — 계약 4번(null != zero).
# 앞머리에 가린 이마를 `clean` 으로 찍으면 모델이 "가려진 것 = 깨끗함"을 배운다.
REGION_STATES = ("clean", "trouble", "unlabelable")

# type 은 v1 모델이 학습에 쓰지 않는다 (LIMITS §3-1: 활성 염증 vs PIE 는 원리적 구분 불가).
# 그래도 찍어 두면 나중에 쓸 수 있으므로 필드는 남긴다. 기본값은 None 이다.
LESION_TYPES = (None, "papule", "pustule", "comedone", "pih")

# 좌/우는 피사체 기준이다 (CONTRACT 부수 규약). 정면 사진에서 피사체의 오른쪽 볼은
# **이미지의 왼쪽**에 나타난다. 이 힌트가 없으면 라벨러가 반드시 뒤집어 찍는다.
REGION_KEYS = {
    "1": ("forehead", "이마"),
    "2": ("cheek_l", "좌볼 = 화면 오른쪽"),
    "3": ("cheek_r", "우볼 = 화면 왼쪽"),
    "4": ("nose", "코"),
    "5": ("chin", "턱"),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"}


def label_path(labels_root: Path, labeler: str, capture_id: str) -> Path:
    return Path(labels_root) / labeler / (capture_id + ".json")


def empty_label(capture_id: str, source_name: str, width: int, height: int,
                labeler: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "capture_id": capture_id,
        "source_name": source_name,
        "image_size": [int(width), int(height)],
        "labeler": labeler,
        "label_version": LABEL_VERSION,
        "labeled_at": None,
        "lesions": [],
        # 부위 기본값은 `unlabelable` 이다. 라벨러가 명시적으로 판단하기 전에는
        # "깨끗함"이 아니라 "아직 안 봤음"이어야 한다 — 계약 4번.
        "regions": {r.value: "unlabelable" for r in ALL_REGIONS},
    }


def load_label(path: Path, capture_id: str, source_name: str,
               width: int, height: int, labeler: str) -> dict:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        base = empty_label(capture_id, source_name, width, height, labeler)
        base.update(data)
        # 부위 키가 빠져 있으면 채운다 (스키마가 늘어난 경우).
        for r in ALL_REGIONS:
            base["regions"].setdefault(r.value, "unlabelable")
        return base
    return empty_label(capture_id, source_name, width, height, labeler)


def save_label(path: Path, label: dict) -> None:
    label["labeled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")


class Labeler:
    """이미지 한 장의 라벨링 세션. matplotlib 이벤트만 다룬다."""

    def __init__(self, rgb, label: dict, path: Path, title: str):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.label = label
        self.path = path
        self.action = "quit"          # next | prev | quit

        self.fig, self.ax = plt.subplots(figsize=(10, 12))
        self.ax.imshow(rgb)
        self.ax.set_axis_off()
        self.title = title
        self.points = self.ax.plot([], [], "o", mfc="none", mec="#00ff88",
                                   mew=1.6, ms=14)[0]
        self.points_unsure = self.ax.plot([], [], "o", mfc="none", mec="#ffcc00",
                                          mew=1.6, ms=14, ls="")[0]

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()

    # ── 그리기 ────────────────────────────────────────────────
    def redraw(self) -> None:
        certain = [(l["x"], l["y"]) for l in self.label["lesions"]
                   if l.get("confidence") != "unsure"]
        unsure = [(l["x"], l["y"]) for l in self.label["lesions"]
                  if l.get("confidence") == "unsure"]
        self.points.set_data([p[0] for p in certain], [p[1] for p in certain])
        self.points_unsure.set_data([p[0] for p in unsure], [p[1] for p in unsure])

        regions = self.label["regions"]
        rtxt = "  ".join(
            "{}:{}={}".format(k, name.split(" ")[0], regions[rid][:3])
            for k, (rid, name) in sorted(REGION_KEYS.items()))
        self.ax.set_title(
            "{}\n병변 {}개 (certain {} / unsure {})\n{}"
            .format(self.title, len(self.label["lesions"]),
                    len(certain), len(unsure), rtxt),
            fontsize=9)
        self.fig.canvas.draw_idle()

    # ── 이벤트 ────────────────────────────────────────────────
    def _toolbar_active(self) -> bool:
        """확대·이동 중에는 클릭이 라벨이 아니다. 이 검사가 없으면 줌 할 때마다 점이 찍힌다."""
        tb = getattr(self.fig.canvas, "toolbar", None)
        return bool(getattr(tb, "mode", ""))

    def on_click(self, event) -> None:
        if event.inaxes is not self.ax or self._toolbar_active():
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        if event.button == 1:
            unsure = bool(event.key and "shift" in str(event.key))
            self.label["lesions"].append({
                "x": round(x, 1), "y": round(y, 1),
                "type": None,
                "confidence": "unsure" if unsure else "certain",
            })
        elif event.button == 3:
            self._delete_nearest(x, y)
        self.redraw()

    def _delete_nearest(self, x: float, y: float) -> None:
        if not self.label["lesions"]:
            return
        d2 = [(l["x"] - x) ** 2 + (l["y"] - y) ** 2 for l in self.label["lesions"]]
        i = min(range(len(d2)), key=d2.__getitem__)
        # 아무 데나 우클릭했다고 멀리 있는 점이 지워지면 안 된다.
        if d2[i] <= (0.02 * max(self.label["image_size"])) ** 2:
            self.label["lesions"].pop(i)

    def on_key(self, event) -> None:
        k = (event.key or "").lower()

        if k in REGION_KEYS:
            rid, _ = REGION_KEYS[k]
            cur = self.label["regions"][rid]
            nxt = REGION_STATES[(REGION_STATES.index(cur) + 1) % len(REGION_STATES)]
            self.label["regions"][rid] = nxt
            self.redraw()
        elif k == "t" and self.label["lesions"]:
            last = self.label["lesions"][-1]
            cur = last.get("type")
            last["type"] = LESION_TYPES[
                (LESION_TYPES.index(cur) + 1) % len(LESION_TYPES)]
            print("  마지막 병변 type -> {}".format(last["type"]))
        elif k == "s":
            save_label(self.path, self.label)
            print("  저장: {}".format(self.path))
        elif k in ("n", "right"):
            self.action = "next"
            self.plt.close(self.fig)
        elif k in ("b", "left"):
            self.action = "prev"
            self.plt.close(self.fig)
        elif k == "q":
            self.action = "quit"
            self.plt.close(self.fig)

    def run(self) -> str:
        self.plt.show()
        save_label(self.path, self.label)      # 창을 어떻게 닫든 잃지 않는다
        return self.action


def collect_targets(args) -> list:
    """(원본 경로, capture_id, source_name) 목록."""
    if args.image:
        return [(args.image, None, args.image.name)]

    store = args.store
    if not store.exists():
        return []
    out = []
    for cap in sorted(d for d in store.iterdir() if d.is_dir()):
        src = sio.find_original(cap)
        if src is None:
            continue
        out.append((src, cap.name, src.name))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labeler", required=True,
                   help="라벨러 식별자. LIMITS §1 의 3인 독립 라벨링에 필수")
    p.add_argument("--store", type=Path, default=ROOT / "data" / "store")
    p.add_argument("--labels", type=Path, default=ROOT / "data" / "labels")
    p.add_argument("--image", type=Path, help="파일 하나만 (store 를 거치지 않는다)")
    p.add_argument("--only-unlabeled", action="store_true",
                   help="아직 라벨 파일이 없는 것만")
    args = p.parse_args()

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib 이 필요하다:  pip install -e .[label]")
        return 1

    targets = collect_targets(args)
    if not targets:
        print("라벨할 이미지가 없다: {}".format(args.image or args.store))
        print("먼저 편입한다:  python scripts/analyze.py --ingest data/incoming")
        return 1

    if args.only_unlabeled:
        targets = [t for t in targets
                   if not label_path(args.labels, args.labeler,
                                     t[1] or Path(t[0]).stem).exists()]
        if not targets:
            print("전부 라벨링돼 있다.")
            return 0

    print("라벨러={}  대상 {}장  ->  {}"
          .format(args.labeler, len(targets), args.labels / args.labeler))
    print("좌클릭 추가 · shift+좌클릭 unsure · 우클릭 삭제 · 1~5 부위 · t type "
          "· s 저장 · n 다음 · b 이전 · q 종료")

    i = 0
    while 0 <= i < len(targets):
        src, capture_id, source_name = targets[i]
        rgb, meta = sio.load_original(src)
        cid = capture_id or meta.capture_id
        path = label_path(args.labels, args.labeler, cid)
        label = load_label(path, cid, source_name,
                           rgb.shape[1], rgb.shape[0], args.labeler)

        title = "[{}/{}] {}".format(i + 1, len(targets), source_name)
        action = Labeler(rgb, label, path, title).run()

        if action == "next":
            i += 1
        elif action == "prev":
            i = max(0, i - 1)
        else:
            break

    print("라벨 저장 위치: {}".format(args.labels / args.labeler))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
