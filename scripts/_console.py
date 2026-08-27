"""콘솔 인코딩 안전망. 스크립트 4종이 최상단에서 부른다.

한국어 Windows 의 기본 로케일은 **cp949** 다. 파이썬은 진짜 콘솔에 붙어 있을 때는
`WriteConsoleW` 로 유니코드를 그대로 쓰지만, **출력이 파이프나 파일로 리다이렉트되면**
로케일 인코딩으로 되돌아간다. 이때 cp949 가 못 담는 문자(예: em-dash `—`)가 하나라도
섞이면 `UnicodeEncodeError` 로 **프로세스가 죽는다.**

이 프로젝트의 `.py` 소스에는 em-dash 가 82곳 있고, argparse 는 모듈 docstring 을
`--help` 로 그대로 출력한다. 그래서 리다이렉트된 환경에서는:

    validate.py stability > log.txt   -> 판정을 출력하다 죽는다
    reprocess.py --help | head        -> 도움말조차 못 띄운다

CI·로그 수집·태스크 러너가 전부 리다이렉트 환경이므로, **검증 게이트가 결과를 남기지
못한다.** 문장을 고치는 대신 출력 경로를 고친다 — 어떤 문자가 새로 들어와도 안전하다.
"""

import sys


def setup_console() -> None:
    """stdout/stderr 이 어떤 문자에도 죽지 않게 만든다. 부작용은 이것뿐이다."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:          # 이미 감싸인 스트림(pytest capture 등)
            continue
        try:
            if stream.isatty():
                # 진짜 콘솔: 파이썬이 이미 유니코드로 쓴다. 인코딩은 건드리지 않는다 —
                # UTF-8 을 강제하면 cp949 터미널에서 한글이 깨진다.
                reconfigure(errors="replace")
            else:
                # 파이프·파일: 로케일 폴백이 일어나는 유일한 경로다. 로그는 UTF-8 이어야 한다.
                reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # 재설정이 불가능한 스트림이면 조용히 넘어간다. 안전망이 없는 것뿐이지
            # 없다고 해서 더 나빠지지는 않는다.
            pass
