"""이미지 로드 · EXIF · 9:16 축소 · store 편입 · JSON 리포트.

core/ 밖이다. PIL · tomllib 를 여기서만 쓴다.

**핵심 규칙 (계획 §2.2): 원본은 uint8 로만 다룬다.**
2268x4032 를 float64 로 올리면 배열 하나가 219MB 다. 색 연산은 정규 프레임에서만 한다.
"""

import hashlib
import json
import re
import shutil
import tomllib
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ExifTags

from .core.config import Config, from_dict
from .core.types import AnalysisResult, CaptureMeta

# 검출기에 넣기 전 긴 변을 이 크기로 줄인다.
# MediaPipe 내부 detector 가 128~192px 텐서로 축소하므로 9:16 원본을 그대로 넣으면
# 레터박싱까지 겹쳐 얼굴이 20px 수준이 된다.
DETECT_LONG_EDGE = 1536

_EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}

# <세션>_<조명>_<wb켈빈>_<iso>_<torch|noflash>_<n>
_FILENAME_RE = re.compile(
    r"^(?P<session>[A-Za-z0-9]+)_(?P<lighting>[A-Za-z0-9]+)_"
    r"(?P<wb>\d{3,5})_(?P<iso>\d{2,6})_(?P<torch>torch|noflash)_(?P<n>\d+)$"
)


# ── 설정 ──────────────────────────────────────────────────────────────

def load_config(path: Optional[Path] = None) -> Config:
    """config/default.toml -> Config. toml 이 임계값의 진짜 출처다 (계약 5번)."""
    if path is None:
        return Config()
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return from_dict(data)


def config_hash(cfg: Config) -> str:
    """설정 내용 해시. 결과 JSON 에 박아 어떤 설정에서 나온 숫자인지 못 박는다."""
    blob = json.dumps(cfg.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:6]


# ── 이미지 ────────────────────────────────────────────────────────────

def _register_heif() -> bool:
    try:
        import pillow_heif  # type: ignore
        pillow_heif.register_heif_opener()
        return True
    except Exception:
        return False


_HEIF_OK = _register_heif()


def read_exif(img: Image.Image) -> Dict[str, Any]:
    """EXIF 에서 읽히는 것만 읽는다. 없는 값은 채워 넣지 않는다."""
    out: Dict[str, Any] = {}
    try:
        raw = img.getexif()
    except Exception:
        return out
    if not raw:
        return out

    def get(name: str) -> Any:
        tag = _EXIF_TAGS.get(name)
        return raw.get(tag) if tag is not None else None

    out["camera_make"] = get("Make")
    out["camera_model"] = get("Model")
    out["orientation"] = get("Orientation")
    out["datetime_original"] = get("DateTimeOriginal") or get("DateTime")

    try:
        ifd = raw.get_ifd(0x8769)  # ExifIFD
    except Exception:
        ifd = {}
    for name, key in (("iso", "ISOSpeedRatings"),
                      ("exposure_time_s", "ExposureTime"),
                      ("focal_length_mm", "FocalLength")):
        tag = _EXIF_TAGS.get(key)
        val = ifd.get(tag) if tag is not None else None
        if val is None:
            continue
        try:
            out[name] = float(val)
        except (TypeError, ValueError):
            pass

    return {k: v for k, v in out.items() if v is not None}


def parse_capture_filename(stem: str) -> Optional[Dict[str, Any]]:
    """§3.1 업로드 파일명 규약 파싱.

    수동 앱이 WB 게인을 EXIF 에 안 남기므로 **파일명이 유일한 진실**이다.
    실패하면 None 을 돌려주고, 호출부는 조용히 넘어가지 말고 경고해야 한다 —
    조용히 넘어가면 검증 데이터가 오염된다.
    """
    m = _FILENAME_RE.match(stem)
    if not m:
        return None
    return {
        "session": m.group("session"),
        "lighting": m.group("lighting"),
        "wb_kelvin": int(m.group("wb")),
        "iso": float(m.group("iso")),
        "torch": m.group("torch") == "torch",
        "shot_index": int(m.group("n")),
    }


def load_original(path: Path) -> Tuple[np.ndarray, CaptureMeta]:
    """원본을 EXIF 방향 보정만 적용해 uint8 RGB 로 읽는다.

    리사이즈하지 않는다 — 워프는 원본 화소에서 직접 뜬다 ("행렬은 싸게, 픽셀은 원본에서").
    """
    path = Path(path)
    with Image.open(path) as im:
        exif = read_exif(im)
        im = _apply_orientation(im, exif.get("orientation"))
        rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)

    meta = CaptureMeta(
        source_name=path.name,
        width=int(rgb.shape[1]),
        height=int(rgb.shape[0]),
        camera_make=exif.get("camera_make"),
        camera_model=exif.get("camera_model"),
        iso=exif.get("iso"),
        exposure_time_s=exif.get("exposure_time_s"),
        focal_length_mm=exif.get("focal_length_mm"),
        orientation=exif.get("orientation"),
        datetime_original=exif.get("datetime_original"),
    )

    parsed = parse_capture_filename(path.stem)
    if parsed:
        meta.session = parsed["session"]
        meta.lighting = parsed["lighting"]
        meta.wb_kelvin = parsed["wb_kelvin"]
        meta.torch = parsed["torch"]
        meta.shot_index = parsed["shot_index"]
        if meta.iso is None:
            meta.iso = parsed["iso"]
        meta.filename_parsed = True

    meta.capture_id = _capture_id(path, meta)
    return rgb, meta


def _apply_orientation(im: Image.Image, orientation: Optional[int]) -> Image.Image:
    """EXIF 방향 적용. 세로 사진은 거의 항상 태그가 있다."""
    if not orientation or orientation == 1:
        return im
    ops = {
        2: (Image.FLIP_LEFT_RIGHT,),
        3: (Image.ROTATE_180,),
        4: (Image.FLIP_TOP_BOTTOM,),
        5: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_90),
        6: (Image.ROTATE_270,),
        7: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_270),
        8: (Image.ROTATE_90,),
    }
    for op in ops.get(int(orientation), ()):
        im = im.transpose(op)
    return im


def _capture_id(path: Path, meta: CaptureMeta) -> str:
    """내용 해시 기반 id. 같은 파일을 다시 넣어도 같은 id 가 나온다(멱등)."""
    h = hashlib.sha1(path.read_bytes()).hexdigest()[:8]
    stamp = meta.datetime_original
    if stamp:
        try:
            dt = datetime.strptime(str(stamp), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            dt = datetime.fromtimestamp(path.stat().st_mtime)
    else:
        dt = datetime.fromtimestamp(path.stat().st_mtime)
    return "{}_{}".format(dt.strftime("%Y%m%d_%H%M%S"), h)


def downscale_for_detection(rgb: np.ndarray, long_edge: int = DETECT_LONG_EDGE
                            ) -> Tuple[np.ndarray, float]:
    """검출용 축소본과 축소 배율.

    랜드마크는 축소본에서 뽑고 좌표만 원본으로 되돌린다. 오차 1px @1536 은
    원본 2.6px = IPD 450 대비 0.6% 라 워프 행렬 추정엔 충분하다.
    """
    h, w = rgb.shape[:2]
    scale = float(long_edge) / float(max(h, w))
    if scale >= 1.0:
        return rgb, 1.0
    import cv2  # core 밖에서만 쓴다
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, scale


# ── 흡광계수 CSV (유일한 외부 데이터 파일) ────────────────────────────

def load_extinction_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """wavelength_nm, hbo2, melanin 3열 CSV 를 읽는다.

    합성 병변 주입을 물리적으로 정확하게 만드는 데만 쓴다 (§5.2 · §6.2).
    """
    rows: List[Tuple[float, float, float]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue  # 헤더 줄
    if not rows:
        raise ValueError("흡광계수 CSV 를 읽지 못했다: {}".format(path))
    arr = np.array(rows, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2]


# ── 불변 원시 레이어 (§5.4) ───────────────────────────────────────────

def ingest(src: Path, store: Path) -> Tuple[Path, CaptureMeta, List[str]]:
    """원본을 `data/store/<capture_id>/` 규약으로 편입한다.

    원본과 capture.json 은 **한 번 쓰고 수정하지 않는다** (계약 10번).
    이미 있으면 덮어쓰지 않고 그대로 둔다 — 재편입은 멱등이다.

    Returns:
        (capture_dir, meta, warnings)
    """
    src = Path(src)
    _rgb, meta = load_original(src)
    warnings: List[str] = []
    if not meta.filename_parsed:
        warnings.append(
            "파일명이 §3.1 규약과 다르다: {} — WB/ISO/torch 를 알 수 없어 "
            "검증 데이터로 쓰면 오염된다".format(src.name))

    cap_dir = Path(store) / meta.capture_id
    cap_dir.mkdir(parents=True, exist_ok=True)
    (cap_dir / "results").mkdir(exist_ok=True)

    dst = cap_dir / ("original" + src.suffix.lower())
    if not dst.exists():
        shutil.copy2(src, dst)

    meta_path = cap_dir / "capture.json"
    if not meta_path.exists():
        meta.ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta_path.write_text(
            json.dumps(to_jsonable(meta), ensure_ascii=False, indent=2),
            encoding="utf-8")
    else:
        meta = read_capture_meta(cap_dir)

    return cap_dir, meta, warnings


def read_capture_meta(cap_dir: Path) -> CaptureMeta:
    data = json.loads((Path(cap_dir) / "capture.json").read_text(encoding="utf-8"))
    known = {f for f in CaptureMeta.__dataclass_fields__}
    return CaptureMeta(**{k: v for k, v in data.items() if k in known})


def find_original(cap_dir: Path) -> Optional[Path]:
    for p in sorted(Path(cap_dir).glob("original.*")):
        return p
    return None


def result_path(cap_dir: Path, result: AnalysisResult) -> Path:
    return (Path(cap_dir) / "results"
            / "{}_{}.json".format(result.algo_version, result.config_hash))


def write_result(cap_dir: Path, result: AnalysisResult) -> Path:
    """결과를 버전 태그된 파일로 남긴다. 기존 버전을 지우지 않는다."""
    out = result_path(cap_dir, result)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    return out


def append_feedback(cap_dir: Path, entry: Dict[str, Any]) -> None:
    """결과 화면 피드백을 append 한다 (§5.4).

    약한 라벨이고 **부정 편향**이 있다 (틀렸을 때 더 자주 누른다).
    정확도로 읽으면 안 되고 실패 사례 수집기로만 쓴다 (§7.3-15).
    """
    path = Path(cap_dir) / "feedback.json"
    log: List[Dict[str, Any]] = []
    if path.exists():
        try:
            log = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log = []
    entry = dict(entry)
    entry.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    log.append(entry)
    path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 직렬화 ────────────────────────────────────────────────────────────

def to_jsonable(obj: Any) -> Any:
    """dataclass · Enum · numpy 스칼라를 JSON 으로.

    `None` 은 `None` 으로 남긴다 — 0 으로 바꾸지 않는다 (계약 4번).
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {(k.value if isinstance(k, Enum) else str(k)): to_jsonable(v)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        return None if isinstance(val, float) and not np.isfinite(val) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
