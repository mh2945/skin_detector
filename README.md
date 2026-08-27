# skin_detector

얼굴 사진에서 **붉은기(미만성 홍조)** 와 **트러블(염증성 병변)** 을 검출하는 룰 기반 도구.
학습 데이터 없이 색 과학만으로 동작한다.

> **의료기기가 아니며 진단을 제공하지 않습니다.** 색만으로는 주사·지루피부염·접촉피부염을
> 구분할 수 없습니다. 피부 증상이 걱정되면 피부과 전문의와 상담하세요.

---

## 빠른 시작

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.venv\Scripts\python.exe -m uvicorn web.server:app --host 127.0.0.1 --port 8000
```

브라우저에서 <http://127.0.0.1:8000> 을 연다. **사진은 이 컴퓨터 밖으로 나가지 않는다.**

## 어떻게 동작하나

혈액(헤모글로빈)과 색소(멜라닌)는 **둘 다 얼굴을 붉게 만든다.** 이것이 문제의 정체다.
구분하려면 색을 2차원으로 봐야 한다:

```
e = log₁₀(R/G)   홍반 축        헤모글로빈  Δe=+0.447, Δm=−0.494
m = log₁₀(G/B)   멜라닌 축      멜라닌      Δe=+0.149, Δm=+0.448
```

**둘 다 e 를 올리지만 m 의 부호가 반대다.** 이 차이가 붉은기와 점·기미를 가른다.

로그 비율을 쓰면 **음영·노출·거리가 수학적으로 소거된다**(실측 1e-5 이내).
화이트밸런스는 상수 오프셋으로만 남고, 그건 얼굴 자체를 기준선으로 삼아 없앤다.
그래서 컬러차트 없이도 조명이 바뀐 사진끼리 비교할 수 있다.

상세: [`docs/COLOR_SCIENCE.md`](docs/COLOR_SCIENCE.md)

## 무엇을 내놓나

- **부위별 판정** — 이마·좌볼·우볼·코·턱 각각에 붉은기(정상/붉음)와 트러블(없음/있음)
- **개별 병변 위치와 크기** (mm) — 홍채 지름 11.7mm 를 자로 쓴다
- **히트맵 오버레이**
- **왜 그렇게 나왔는지의 근거** — 면적 비율, 편차 `d`, 기준선, 조명 균일성, 피부톤 ITA°

점수 0~100 은 **순서형 편의 척도**이지 물리량이 아니다. 그래서 JSON 필드명이
`score_ordinal_0_100` 이다. 물리적으로 의미 있는 값은 `d`(log₁₀ 비율)와 면적 비율뿐이다.

**측정하지 못한 부위는 0점이 아니라 "측정 못 함 + 이유"** 로 나온다.

## 정확한 측정을 원한다면

브라우저 촬영은 화이트밸런스·감도를 고정할 수 없어 정확도가 낮다.
수동 카메라 앱(iOS Halide/ProCam, Android Open Camera)에서 **WB 켈빈 · ISO · 셔터를 고정**하고
HDR·뷰티보정을 끈 뒤 찍어 업로드한다. 거치대 필수.

파일명 규약을 지키면 촬영 설정이 자동으로 기록된다:

```
s01_window_5500_100_torch_1.heic
세션_조명_WB켈빈_ISO_토치여부_번호
```

상세: [`docs/VALIDATION.md`](docs/VALIDATION.md) §0

## 구조

```
docs/     CONTRACT(계층 규칙) · COLOR_SCIENCE(물리) · LIMITS(한계) · ROADMAP · VALIDATION
test/     86개. 셀피 한 장 없이 합성 얼굴 픽스처로 전 구간 검증
scripts/  setup_env.ps1 · analyze.py · tune.py · validate.py · reprocess.py
src/skin_detector/
  core/     ★ numpy + stdlib 만. 그대로 C++ 로 옮겨 모바일 공유 코어가 된다
  face.py   MediaPipe 어댑터 + 정규 프레임 워프 (원본 -> 768x768 변환은 여기 한 곳뿐)
  pipeline.py  유일한 오케스트레이터
web/      FastAPI + 정적 프론트. 127.0.0.1 고정
config/default.toml       모든 임계값의 단일 출처
data/store/               불변 원시 레이어 — 원본은 절대 수정하지 않는다
data/spectra/*.csv        유일한 외부 데이터 파일 (흡광계수)
```

## 데이터가 쌓일수록 좋아지는 구조

원본과 촬영 메타는 **한 번 쓰면 수정하지 않는다.** 결과에는 `algo_version` + `config_hash` 가
박힌다. 알고리즘을 고치면 과거 전량을 다시 채점할 수 있다:

```bash
python scripts/reprocess.py --all
python scripts/reprocess.py --compare v0.1.0_ed33b1 v0.2.0_71be03
```

부위별 점수 변화와 **판정이 뒤집힌 이미지 수**가 나온다. 프레임워크가 아니라 규약 두 개다.

## 지금 어디까지 왔나

합성 데이터로 색 과학의 전제가 성립함을 확인했다 — 밝기·음영 불변, 조명색 상수 오프셋,
두 색소 60° 이상 분리, 합성 Δ 복원 R² ≥ 0.99.

**실사진으로는 아직 아무것도 검증되지 않았다.** 위 숫자는 전부 상한으로만 읽어야 한다.
현실적 기대치와 원리적 한계 15가지는 [`docs/LIMITS.md`](docs/LIMITS.md) 에 미리 선언해 뒀다.
