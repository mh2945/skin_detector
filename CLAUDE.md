# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

얼굴 사진에서 **미만성 홍조**와 **염증성 여드름 병변**을 검출하는 룰 기반 시스템.
학습 데이터 없이 색 과학만으로 동작한다. 1차 배포는 로컬 웹 UI, 그 다음이 모바일 온디바이스다.

**목표 3가지** (사용자 확정):
1. 붉은기·트러블 검출 성공률 70~80% → **부위별 판정 일치율**로 조작적 정의 (`docs/LIMITS.md` §1)
2. 촬영부터 결과까지 상세한 안내 → 웹 UI 의 `ReasonCode`→문장 표 (`web/server.py` `GUIDANCE`)
3. 지속적 데이터 확보로 고도화 가능한 구조 → 불변 원시 레이어 + 버전 태그 (`data/store/`)

**세 목표는 사슬로 이어진다.** 촬영 안내가 `docs/LIMITS.md` §3 한계의 절반(혼합 광원·과노출·
가림·흐림)을 캡처 단계에서 제거해 목표 1의 일치율을 올리고, 같은 UI 가 목표 3의 통로가 된다.
**안내는 부가 기능이 아니라 성능의 일부다.**

## 현재 상태 — 숫자를 인용하기 전에 읽을 것

구현은 전 구간이 존재하고 **86개 테스트가 통과한다.** 하지만 **실사진으로 검증된 것은 하나도 없다.**
`docs/LIMITS.md` §2 의 실측 칸은 대부분 _미측정_ 이고 채워진 값은 전부 합성 데이터다.
**합성은 실사보다 쉬우므로 상한으로만 읽어야 하고, 이 숫자로 성능을 주장하는 문장을 쓰면 안 된다.**

다음 블로커는 코드가 아니라 데이터다. `docs/VALIDATION.md` §0 규약대로 **경로 B** 로 찍은
부트스트랩 셋(조명 4종 × {토치, 무플래시} × 2장 = 16장/세션)이 `data/store/` 에 들어와야
`validate.py stability` 가 프로젝트 전체의 go/no-go 를 낸다.

## 개발 명령

```bash
# 최초 1회 — venv + 의존성 + FaceLandmarker 모델(3.7MB) 다운로드까지 한 번에
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

# 테스트 (이미지 불필요 — 합성 얼굴 픽스처로 전 구간 검증)
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pytest test/test_contract.py -v      # 계층 규칙 강제
.venv/Scripts/python.exe -m pytest test/test_measure.py::test_synthetic_erythema_recovery_is_linear -v
.venv/Scripts/python.exe -m pytest -q -k melanin                 # 이름으로 필터

# 사진 편입 -> 분석
.venv/Scripts/python.exe scripts/analyze.py --ingest data/incoming  # 원본을 store 규약으로 편입
.venv/Scripts/python.exe scripts/analyze.py --out out/              # store 전량 + 오버레이
.venv/Scripts/python.exe scripts/analyze.py --image <path> --out out/   # store 를 거치지 않고 한 장

# 검증 (docs/VALIDATION.md 참조)
.venv/Scripts/python.exe scripts/validate.py agreement --panel data/panel_ratings.csv  # 목표 1 관문
.venv/Scripts/python.exe scripts/validate.py stability    # Phase 1 관문 = 프로젝트 go/no-go
.venv/Scripts/python.exe scripts/validate.py flashpair    # L1 광학 개입 사전 검증
.venv/Scripts/python.exe scripts/validate.py synth        # 합성 Δ 복원 선형성
.venv/Scripts/python.exe scripts/validate.py negative     # 음성 대조군

# 재처리 — 목표 3의 실증. 원본이 불변이므로 몇 번을 돌려도 안전하다
.venv/Scripts/python.exe scripts/reprocess.py --list
.venv/Scripts/python.exe scripts/reprocess.py --all
.venv/Scripts/python.exe scripts/reprocess.py --compare v0.1.0_ed33b1 v0.2.0_71be03

# 웹 UI (localhost 전용)
.venv/Scripts/python.exe -m uvicorn web.server:app --host 127.0.0.1 --port 8000

# 튜닝 (contact sheet 병용 필수 — 한 장 오버핏 방지)
.venv/Scripts/python.exe scripts/tune.py --capture-id <id>
.venv/Scripts/python.exe scripts/tune.py --contact-sheet data/store
```

**Windows 함정**: `python.exe` 가 0바이트 Microsoft Store 스텁인 경우가 흔하다.
버전을 출력하지 않고 `pip install` 이 조용히 실패한다. **venv 를 쓰면 완전히 우회된다** —
`.venv/Scripts/python.exe` 를 직접 부르면 별칭이 끼어들 여지가 없다.

**모델 번들이 없으면 스크립트 4개가 전부 죽는데 `pytest` 는 그대로 통과한다.**
진단 순서는 `scripts/CLAUDE.md` 참조.

## 아키텍처

### 계층 의존성 규칙 (협상 불가 — `docs/CONTRACT.md`)

```
web/  ──→ pipeline.py ──→ core/        (numpy + stdlib 만)
                    └──→ face.py · imageio.py · render.py   (cv2 · mediapipe · PIL)
```

`core/` 는 **numpy + stdlib 만** import 한다. `web/` 은 `pipeline` 만 호출하고 `core/` 를 직접
import 하지 않는다. **둘 다 `test/test_contract.py` 가 AST 로 강제한다.**
이 규칙 덕에 ① `core/` 를 그대로 C++ 로 옮겨 Android/iOS 공유 코어로 쓸 수 있고
② 셀피 한 장 없이 86개 테스트가 돈다.

### 데이터 흐름 (사진 1장의 여정)

```
원본 uint8 2268x4032 (9:16)
  ↓ EXIF 방향 -> 긴 변 1536px 축소 -> MediaPipe -> 478 landmarks
  ↓ 랜드마크를 원본 좌표로 역스케일 -> 홍채 2점 -> similarity 행렬
  ↓ warpAffine(원본 uint8 -> 768x768)          ★ 리샘플링 1회, 원본 화소에서 직접
정규 프레임 uint8 768x768 (IPD=320) -> float32
  ↓ 선형화 -> 광학밀도 -> (e, m) -> 품질 게이트 -> 마스킹
  ↓ ERI = e - (beta*m + alpha) -> 2단계 기준선 -> 편차맵 d
  ↓ 대역 분리 -> 미만성 / DoG 병변 -> 부위 점수
AnalysisResult (JSON)
```

**원본을 float 로 올리지 않는다.** 2268×4032 를 float64 로 올리면 배열 하나가 219MB 다.
색 연산은 정규 프레임에서만 한다 — **"행렬은 싸게, 픽셀은 원본에서."**

### 디렉터리별 상세 규칙

작업하는 디렉터리의 `CLAUDE.md` 가 함께 로드된다. 그 파일이 해당 계층의 유일한 출처다.

| 파일 | 다루는 것 |
|---|---|
| `src/skin_detector/core/CLAUDE.md` | ★ 색 과학 · 게이트 순서 · null≠zero · 좌우 규약 · 임계값 |
| `src/skin_detector/CLAUDE.md` | 어댑터 계층 · 좌표계 · 9:16 고해상도 처리 |
| `web/CLAUDE.md` | 얇은 껍데기 규칙 · `GUIDANCE` 표 · 미측정 표시 |
| `test/CLAUDE.md` | 합성 픽스처 · 계약 테스트 · 물리적으로 올바른 주입 |
| `scripts/CLAUDE.md` | CLI 4종 · 환경 함정 |
| `data/CLAUDE.md` | 불변 원시 레이어 · 파일명 규약 · 개인정보 |

## 알려진 한계 (코드 수정 시 인지할 것)

`docs/LIMITS.md` §3 에 15개가 있다. 특히 자주 잊는 것:

- **활성 염증 vs PIE 구분 불가.** 튜닝으로 해결되는 문제가 아니다.
- **절대 홍반 측정 불가.** 자기참조는 *레벨*보다 *대비*를 본다. 얼굴 전체가 고르게 붉으면
  그 붉음이 부분적으로 스스로 정규화되어 사라진다.
- **브라우저 캡처(경로 A)로 성능을 주장하지 않는다.** WB/ISO 통제가 없어
  색 통제 실패와 알고리즘 실패를 구분할 수 없다. 성능 수치는 경로 B(업로드)에서만 나온다.
- **피드백은 부정 편향이 있다.** 틀렸을 때 더 자주 누른다. 정확도로 읽지 말 것.
- **검증 피험자 1인.** 피부톤·증상 다양성 일반화 근거가 원천적으로 없다.
- **컬러차트 미사용.** 안정적이면서 동시에 계통적으로 틀려 있어도 알아낼 방법이 없다.

## 참조 자료

`sample1/`, `sample2/` 는 **타 프로젝트(안면인식 데모)의 문서 복사본**이다.
이 프로젝트의 설정이 아니다 — 계층 규율과 게이트 체인 패턴만 계승했다.

`model/` 의 Alchera FaceSDK DLL·모델은 **쓰지 않는다.** 그 SDK 에는 피부 관련 기능이
전혀 없어(스캔 0건) 배제해도 잃는 것이 없다. 삭제하지는 않되 여기에 배선하지 않는다.
