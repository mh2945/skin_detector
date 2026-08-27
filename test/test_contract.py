"""계약 강제 (docs/CONTRACT.md). 이 파일이 깨지면 설계가 깨진 것이다.

sample2 가 net8.0 소스 링크로 강제했던 계층 규율의 Python 등가물이다.
규칙을 문서에만 적어두면 반드시 새어 나간다 — AST 로 막는다.
"""

import ast
import re
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
        # `from x.core import y` 와 `import skin_detector.core.color` 를 **둘 다** 본다.
        # ImportFrom 만 보면 후자가 그대로 새어 나간다.
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if "core" in name.split("."):
                    bad.append("{}:{} -> {}".format(path.name, node.lineno, name))
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


# ── 의존성 선언이 두 곳에 있다: 그 둘이 어긋나지 않게 한다 ────────────

def _pyproject_requirements():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    proj = data["project"]
    reqs = list(proj.get("dependencies", []))
    for extra in proj.get("optional-dependencies", {}).values():
        reqs += list(extra)
    names = set()
    for r in reqs:
        # "uvicorn[standard]>=0.27" -> "uvicorn"
        name = re.split(r"[<>=!\[;\s]", r, 1)[0].strip().lower()
        if name:
            names.add(name.replace("_", "-"))
    return names


def _setup_script_packages():
    """setup_env.ps1 의 `pip install` 줄에서 패키지 이름만 뽑는다.

    PowerShell 의 줄 이음 문자는 백틱이다. 백틱으로 끝나는 줄은 다음 줄까지 이어붙인다.
    """
    lines = (ROOT / "scripts" / "setup_env.ps1").read_text(
        encoding="utf-8").splitlines()
    body = None
    for i, line in enumerate(lines):
        # 주석에도 "pip install" 이 나온다 (Store 스텁 함정 설명). 실행 줄만 본다.
        if line.lstrip().startswith("#"):
            continue
        if "pip install" not in line or "--upgrade pip" in line:
            continue
        chunk, j = line, i
        while chunk.rstrip().endswith("`"):
            j += 1
            chunk = chunk.rstrip().rstrip("`") + " " + lines[j]
        body = chunk
        break
    assert body is not None, "setup_env.ps1 의 pip install 줄을 찾지 못했다"

    names = set()
    for tok in body.replace('"', " ").split():
        tok = tok.strip()
        if (not tok or tok.startswith(("-", "&", "$"))
                or tok in ("pip", "install", "-m")):
            continue
        name = re.split(r"[<>=!\[]", tok, maxsplit=1)[0].strip().lower()
        if name and name.replace("-", "").replace(".", "").isalnum():
            names.add(name.replace("_", "-"))
    return names


def test_setup_script_installs_what_pyproject_declares():
    """의존성이 두 곳에 선언되어 있다 — 어긋나면 여기서 잡는다.

    실제로 환경을 만드는 것은 `setup_env.ps1` 이고 `pyproject.toml` 은 문서에
    가깝다. 둘이 갈라지면 "문서에는 있는데 안 깔리는" 패키지가 생기고,
    그 상태는 새 클론에서만 드러나므로 발견이 늦다.

    버전 핀까지는 강제하지 않는다 (setup 스크립트는 의도적으로 핀이 없다).
    **이름 집합**만 일치하면 된다.
    """
    declared = _pyproject_requirements()
    installed = _setup_script_packages()
    missing = declared - installed
    extra = installed - declared
    assert not missing, "pyproject 에는 있는데 setup_env.ps1 이 안 깐다: {}".format(
        sorted(missing))
    assert not extra, "setup_env.ps1 만 까는 패키지: {}".format(sorted(extra))
