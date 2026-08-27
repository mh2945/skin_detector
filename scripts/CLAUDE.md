# scripts/ — CLI 진입점

전부 `.venv/Scripts/python.exe` 로 부른다. 넷 다 `--store` · `--config` · `--model` 을
공통으로 받고 기본값은 `data/store` · `config/default.toml` · `models/face_landmarker.task` 다.

| 스크립트 | 하는 일 |
|---|---|
| `setup_env.ps1` | Python 3.12 탐색/설치 → venv → 의존성 → 모델 다운로드 → 자체 검증. **멱등** |
| `analyze.py` | `--ingest <폴더>` 편입 · `--out` 전량 분석 · `--image` 단건 |
| `validate.py` | `agreement` `stability` `flashpair` `synth` `negative` |
| `reprocess.py` | `--list` · `--all` 재채점 · `--compare TAG_A TAG_B` |
| `tune.py` | `--capture-id` 슬라이더 · `--contact-sheet` 격자 |

## 환경 함정 두 가지 — 여기서 대부분 막힌다

**① Store 스텁.** `python.exe` 가 0바이트 Microsoft Store 스텁인 경우가 흔하다.
버전을 출력하지 않고 `pip install` 이 **조용히 실패한다.**
→ **venv 를 쓰면 완전히 우회된다.** `.venv/Scripts/python.exe` 는 실제 인터프리터의 복사본이라
별칭이 끼어들 여지가 없다. 전역 `python` 을 쓰고 싶으면
*설정 → 앱 → 고급 앱 설정 → 앱 실행 별칭* 에서 꺼야 한다.

**② 모델 번들 부재.** `models/face_landmarker.task`(3.7MB)는 mediapipe 휠에 들어 있지 않고
gitignore 대상이라 **새로 체크아웃하면 없다.** 없으면 이 스크립트 넷이 전부 죽는다.
**그런데 `pytest` 는 86개 그대로 통과한다** — 테스트는 합성 픽스처를 쓰고 랜드마커를 아예
부르지 않기 때문이다.

> **"테스트는 되는데 스크립트만 안 된다"면 이것부터 확인한다.**
> `setup_env.ps1` 이 받아 오고, 멱등이라 다시 돌려도 안전하다.

## `analyze.py --ingest` — 데이터가 들어오는 유일한 문

원본을 `data/store/<capture_id>/` 규약으로 편입한다. `capture_id` 는 **내용 해시 기반**이라
같은 파일을 두 번 넣어도 중복이 생기지 않는다.

**파일명이 유일한 촬영 메타다.** 파싱 실패하면 `null` 로 들어가고 **경고만 한다** —
경고를 무시하고 검증에 쓰면 데이터가 조용히 오염된다. 상세는 `data/CLAUDE.md`.

## `validate.py` — 세 값을 떼어 쓸 수 없다

`agreement` 는 **일치율 · NIR · κ 를 한 줄로만** 출력한다. 일치율 단독 출력은 **코드로
막혀 있다.** 대부분의 부위가 `정상`이면 "전부 정상"이라고만 답해도 70% 를 넘기기 때문이다.

```
붉은기 : 일치율 74.5% / NIR 62.0% / kappa +0.46  (천장 81.0%)  n=200
트러블 : 일치율 69.0% / NIR 71.0% / kappa +0.11  n=200  <- NIR 대비 여유 부족
```

두 번째 줄이 함정의 모습이다 — 69% 는 그럴듯해 보이지만 **아무것도 안 하는 모델(71%)보다
낮다.** κ 0.11 이 그 사실을 드러낸다.

**기본적으로 `capture_path == "upload"`(경로 B)만 읽는다.** `--include-browser` 는 있지만
그 결과로 성능을 주장하면 안 된다.

`stability` 의 ICC 가 **프로젝트 전체의 go/no-go** 다. 아무것도 0.5 를 못 넘으면
그 위에 아무것도 짓지 말고 멈춰서 `docs/ROADMAP.md` L1/L4 를 재검토한다.

## `reprocess.py` — 목표 3의 실증

원본과 `capture.json` 을 건드리지 않으므로 **몇 번을 돌려도 안전하다.** 결과는 버전별로
나란히 쌓인다. `--compare` 는 부위별 평균 점수 변화와 **판정이 뒤집힌 이미지 수**를 낸다.
후자가 크게 움직이면 일치율을 다시 재야 한다.

## `tune.py` — contact sheet 는 선택이 아니다

`--capture-id` 하나로만 튜닝하면 **그 한 장에 오버핏한다.** 슬라이더를 움직일 때마다
`--contact-sheet` 로 여러 장을 동시에 확인한다.

## 빈 store 에서도 죽지 않는다

넷 다 데이터가 없으면 무엇을 먼저 하라는 문장을 내고 정상 종료한다.
스택 트레이스를 뱉지 않는다.
