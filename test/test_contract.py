"""계약 강제 (docs/CONTRACT.md). 이 파일이 깨지면 설계가 깨진 것이다.

sample2 가 net8.0 소스 링크로 강제했던 계층 규율의 Python 등가물이다.
규칙을 문서에만 적어두면 반드시 새어 나간다 — AST 로 막는다.
"""

import ast
import sys
import tomllib
from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "skin_detector"
sys.path.insert(0, str(ROOT / "src"))

from skin_detector.core import config as cfg_mod  # noqa: E402

# core/ 가 import 해도 되는 것: numpy + stdlib + core 내부.
CORE_ALLOWED_THIRD_PARTY = {"numpy"}

STDLIB = set(sys.stdlib_module_names)


def _imports(path: Path):
    """(모듈명, 줄번호) 목록. 상대 import 는 level>0 로 구분한다."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name.split(".")[0], node.lineno, 0
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            yield root, node.lineno, node.level


def _py_files(d: Path):
    return sorted(p for p in d.rglob("*.py") if p.name != "__init__.py")


@pytest.mark.parametrize("path", _py_files(SRC / "core"), ids=lambda p: p.name)
def test_core_imports_only_numpy_and_stdlib(path):
    """계약 1번. core/ 는 numpy + stdlib 만 쓴다.

    cv2 · mediapipe · PIL 이 들어오는 순간 모바일 포팅 대상이 오염되고,
    분석 로직을 의존성 없이 단위 테스트할 수 없게 된다.
    """
    bad = []
    for name, lineno, level in _imports(path):
        if level > 0:          # 상대 import = core 내부
            continue
        if name in STDLIB or name in CORE_ALLOWED_THIRD_PARTY:
            continue
        bad.append("{}:{} -> {}".format(path.name, lineno, name))
    assert not bad, (
        "core/ 는 numpy + stdlib 만 import 한다 (계약 1번). 위반: " + ", ".join(bad)
        + "\n이 기능이 정말 필요하면 core/ 밖으로 옮겨야 한다.")


def test_web_does_not_import_core_directly():
    """계약 9번. web/ 은 pipeline 만 호출한다.

    웹이 코어를 우회해 조립하기 시작하면 그 로직은 모바일 포팅 때 통째로 사라진다.
    """
    web = ROOT / "web"
    if not web.exists():
        pytest.skip("web/ 없음")
    bad = []
    for path in _py_files(web):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "core" in node.module.split("."):
                    bad.append("{}:{} -> {}".format(path.name, node.lineno, node.module))
    assert not bad, ("web/ 은 core/ 를 직접 import 하지 않는다 (계약 9번): "
                     + ", ".join(bad))


def test_toml_defaults_match_dataclass_defaults():
    """계약 5번. toml 이 임계값의 진짜 출처이고 dataclass 기본값은 그 사본이다.

    둘이 어긋나면 "설정을 고쳤는데 안 먹는다"가 조용히 발생하고,
    그 순간부터 모든 튜닝 결과를 믿을 수 없게 된다.
    """
    data = tomllib.loads((ROOT / "config" / "default.toml").read_text(encoding="utf-8"))
    loaded = cfg_mod.from_dict(data)
    default = cfg_mod.Config()

    mismatches = []
    for section in fields(default):
        a = getattr(loaded, section.name)
        b = getattr(default, section.name)
        for f in fields(b):
            va, vb = getattr(a, f.name), getattr(b, f.name)
            if isinstance(va, float) or isinstance(vb, float):
                same = abs(float(va) - float(vb)) <= 1e-12
            else:
                same = va == vb
            if not same:
                mismatches.append("{}.{}: toml={} dataclass={}".format(
                    section.name, f.name, va, vb))
    assert not mismatches, "\n".join(mismatches)


def test_toml_covers_every_threshold():
    """toml 에 빠진 임계값이 없어야 한다. 빠지면 코드 기본값이 조용히 이긴다."""
    data = tomllib.loads((ROOT / "config" / "default.toml").read_text(encoding="utf-8"))
    missing = []
    for section in fields(cfg_mod.Config()):
        present = set(data.get(section.name, {}))
        expected = {f.name for f in fields(getattr(cfg_mod.Config(), section.name))}
        for name in sorted(expected - present):
            missing.append("{}.{}".format(section.name, name))
    assert not missing, "config/default.toml 에 누락: " + ", ".join(missing)


def test_unknown_config_key_is_rejected():
    """오타 난 임계값이 조용히 무시되면 안 된다."""
    with pytest.raises(ValueError):
        cfg_mod.from_dict({"baseline": {"tau_kk": 2.0}})
    with pytest.raises(ValueError):
        cfg_mod.from_dict({"nonexistent_section": {}})


def test_only_one_external_data_file():
    """계획 §5.2. 외부 데이터 파일은 흡광계수 CSV 하나뿐이다.

    파일이 늘기 시작하면 모바일 포팅에서 전부 번들해야 한다.
    """
    spectra = sorted(p.name for p in (ROOT / "data" / "spectra").glob("*")
                     if p.is_file())
    assert spectra == ["hb_melanin_extinction.csv"], spectra
