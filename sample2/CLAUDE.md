> ⚠ **타 프로젝트 문서 — 이 프로젝트(skin_detector)의 설정이 아닙니다.**
>
> 안면인식 데모(EtusDetectSample)의 CLAUDE.md 복사본이고, 참조하는 `docs/`·소스가
> 여기에 없어 링크가 끊겨 있습니다. skin_detector 의 규칙은 최상위 `CLAUDE.md` 와
> `docs/CONTRACT.md` 에 있습니다.
>
> 이 문서에서 **계승한 것은 세 가지뿐**입니다:
> 계층 의존성 규율(외부 의존 0인 분석 코어), 게이트 체인 우선순위(먼저 걸리는 것이
> 이긴다), 그리고 "NaN = 미측정이지 정상이 아니다"라는 원칙.
>
> 여기 적힌 임계값·SDK 버전·빌드 명령은 skin_detector 와 무관합니다.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

FASMH-94 PoC — Alchera FaceSDK 1.17.7 기반 Windows WinForms 데모(`EtusDetectSample`). 자습실 책상 카메라로
학생의 **졸음**·**이석**을 감지하고, 안면인식만으로 뽑을 수 있는 데이터(눈꺼풀 거리, occlusion, head pose,
PERCLOS 등)를 한 화면에서 보여준다. 판정 로직(`Analysis/`)은 SDK 기능이 아니라 이 저장소에서 신규 구현한 것이다.

**이름이 세 갈래로 다르다** — 헷갈리지 말 것:
- 리포지토리/프로젝트 폴더: `EtusDetectSample`
- C# 네임스페이스: `Etoos.DetectSample`
- 빌드 산출물 어셈블리명: `EtoosDetectSample` ("Eto**o**s", o가 하나 더 많다)

## 빌드 · 테스트 명령

앱 본체(`src/EtusDetectSample`, net48)는 **Windows x64에서만** 빌드된다. 판정 로직(`Analysis/` 등)은
net8.0으로 소스 링크되어 **macOS/Linux/Windows 어디서든** 단위 테스트가 돈다. 개발은 macOS, 실기 검증은
Windows에서 하는 것이 이 프로젝트의 기본 흐름이다(`docs/WINDOWS_TEST_GUIDE.md` 참조).

```bash
# 단위 테스트 (플랫폼 무관) — Analysis/Alerts/Logging/Config 계층만 검증
dotnet test tests/EtusDetectSample.Tests/EtusDetectSample.Tests.csproj

# 단일 테스트 클래스/메서드
dotnet test tests/EtusDetectSample.Tests/EtusDetectSample.Tests.csproj --filter "FullyQualifiedName~EyeStateGateTests"
```

```bat
:: Windows 전용 — 앱 빌드 전 최초 1회(또는 FaceSDK 재배포 시) 런타임 배치
cd tools
setup-runtime.bat
:: natives\ 에 AlcheraFaceSDK.dll / AlcheraFaceSDKCS.dll / AlcheraEncryptCS.dll / opencv_world455.dll 추출
:: + models\ (license.cer 포함, ~260MB)를 bin\x64\{Debug,Release}\models\ 로 복사
```

앱 빌드는 `EtusDetectSample.sln`을 VS2022로 열어 **구성 `x64`**로 빌드(AnyCPU는
`BadImageFormatException`으로 크래시 — native DLL 3종이 x64 전용). `tests/EtusDetectSample.Tests`는
이 sln에 포함되어 있지 않다(net8.0 vs net48이라 별도 프로젝트).

- `EtusDetectSample.sln`은 `EtusDetectSample.csproj` 하나만 포함한다.
- 테스트 프로젝트는 앱 소스를 `<Compile Include="..\..\src\...\*.cs" Link="...">`로 **파일 링크**해서
  재컴파일한다(ProjectReference 아님) — net48 프로젝트를 net8.0에서 직접 참조할 수 없기 때문. 새 파일을
  `Analysis/`·`Alerts/`·`Logging/`에 추가하면 테스트 csproj를 안 고쳐도 와일드카드로 자동 포함된다. 단
  `Config/`는 `AppSettings.cs` 한 파일만 명시적으로 링크한다(`AppSettingsLoader.cs`는 `System.Configuration`
  의존이라 링크 대상에서 제외).
- 실기 수동 검증 시나리오(졸음/이석/게이트/엎드림, 임계값 재보정 절차 포함)는 `docs/WINDOWS_TEST_GUIDE.md`에
  런북 형태로 있다.

## 아키텍처

### 계층 의존성 규칙 (협상 불가 — `docs/CONTRACT.md` 0절)

`Analysis/`, `Alerts/`, `Logging/`, `Config/`는 **외부 의존 0**이다. WinForms, OpenCvSharp, FaceSDK를
`using`하지 않는다 — 이 규칙 덕에 net8.0 소스 링크 단위 테스트가 성립한다. 새 기능을 이 네 폴더에 추가할 때
`System.*` 외 참조가 필요해지면, 그 기능은 이 폴더 밖(`Worker/`, `Sdk/` 등)으로 옮겨야 한다.

`src/EtusDetectSample/Sdk/FaceSDK.cs`는 SDK zip 원본이다 — **한 글자도 수정 금지**(SDK 재배포 시 그대로
덮어쓸 수 있어야 함). SDK 관련 수정은 대신 `Sdk/FaceEngine.cs`(래퍼)에서 한다.

전체 코드는 **C# 7.3 문법만** 허용한다(net48 + 소스 링크된 net8.0 양쪽 컴파일 조건). 금지: `record`,
init-only 프로퍼티, target-typed `new()`, switch expression, nullable reference type, 범위 연산자(`..`),
`using` 선언문(declaration form). `var`, 식 본문 멤버, 문자열 보간은 허용.

### 데이터 흐름 (프레임 1장의 여정)

```
CameraCapture(Mat) → byte[] BGR (720x1280 세로로 회전/크롭/리사이즈됨 — 캡처 원본이 아님)
      ↓
FaceEngine.Analyze(bgr,w,h) → FrameObservation
      ↓
EyeStateGate.Evaluate(obs)  → EyeDecision   (상태 없는 순수 함수)
      ↓
SeatStateMachine.Push(obs, eye) → SeatSnapshot + AlertEvent[]   (시간축 판정)
      ↓
AlertDispatcher / FrameCsvLogger / MainForm(BeginInvoke)
```

`FrameObservation`의 box/landmark 좌표는 **분석 이미지(720x1280) 기준**이다 — 캡처 원본 좌표로 오버레이를
그리면 어긋난다.

### 게이트 우선순위 패턴 (`Analysis/EyeStateGate.cs`)

`EyeStateGate.Evaluate()`는 상태 없는 게이트 체인이다. **먼저 걸리는 조건이 이긴다**, 뒤 조건은 평가조차
안 됨:

```
SdkError → NoFace → LandmarkConf → Pose → Occlusion(coarse) → FineOcclusion → EyelidValid → 눈 개폐
```

이 순서 때문에 나중 단계 신호(예: 정상적인 눈 개폐값)가 있어도 앞 게이트가 먼저 걸리면 `Unknown`으로
확정되고 뒷값은 버려진다. `UnknownReason`(왜 걸렸는지)은 `MainForm.IsBehaviorReason()`이
행동성(`PoseOutOfRange`/`EyeOccluded`/`FineOccluded`/`AsymmetricEye` — "집중 흐트러짐 의심"으로 표시)과
시스템성(`SdkError`/`LowLandmarkConfidence`/`EyelidUnavailable`/`NoFace` — "판정 불가"로 표시)으로 나뉘어
UI 배지 색·문구가 갈린다. 단, 우측 누적 통계(`SessionStats`)의 "판정 불가" 시간은 이 구분과 무관하게
`SeatState.Unknown` 전체를 합산한다.

**점수 방향 규약** (`EyeStateGate.IsOccluded()` 한 곳에서만 결정): Occlusion/FineOcclusion은 **높을수록
가려짐**(`score > max` → 가려짐). Mask는 반대(`face_mask_confidence`가 **낮을수록** 착용)라 `IsOccluded()`를
타지 않고 `Rst.IsFaceMasked()`를 별도로 쓴다. 두 방향을 헷갈려서 부등호를 뒤집지 말 것 — 확인된 사실이지
추측이 아니다.

**NaN = "미측정"**, "안 가려짐"이 아니다. `Mask`/`Occlusion`/`FineOcclusion` 속성은 매 프레임이 아니라
`AppSettings.AttrIntervalFrames`(기본 5프레임) 주기로만 갱신되므로 대부분의 프레임에서 NaN이다. NaN
비교는 항상 false이고, 그 결과 게이트를 자동 통과하는 것이 **의도된 설계**다(미측정 프레임마다 Unknown이
되면 시간 판정이 불가능해짐). 반대로 이 주기성 때문에 FineOcclusion 값은 coarse Occlusion보다 최대
`AttrIntervalFrames / TargetAnalyzeFps`초만큼 갱신이 늦다.

`FaceEngine._fineOcclusionAvailable`은 한 방향 스위치다 — `DetectFineOcclusion` 호출이 한 번이라도
실패(모델 미포함 등)하면 세션 내내 영구 비활성되고, 이후 프레임은 이 게이트를 조용히 건너뛴다(앱이
죽지 않는 것이 계약 조건).

### 스레딩 계약

워커 스레드 **1개**가 `VideoCapture`와 `FaceSDK` 인스턴스를 독점 소유한다(SDK 스레드 안전성 미문서화 +
continuous tracking 캐시가 프레임 순서에 의존). UI→워커는 `lock`+`Queue`. 워커→UI는 **`BeginInvoke`만**
(`Invoke`는 종료 시 데드락). 종료 플래그는 `volatile bool`. UI 스레드에서 SDK를 직접 호출하지 않는다.

### 시간축은 항상 `MonotonicSec`

판정 로직은 `FrameObservation.MonotonicSec`(Stopwatch 기반)만 쓴다. `DateTime.Now`/`UtcNow`는 로그
표기 외에는 금지 — 시스템 시각이 바뀌어도 판정이 흔들리면 안 되기 때문.

### 설정 시스템 (`Config/AppSettings.cs`)

모든 임계값은 `AppSettings` 필드 하나에 모여 있고 `App.config`의 `<appSettings>`에서 읽는다
(`AppSettingsLoader` → `AppSettings.Clamp()`로 범위 보정). 배포된 `EtoosDetectSample.exe.config`를
직접 편집하면 **재빌드 없이** 현장 재보정이 가능하다. `SettingsForm.cs`는 이 중 8개 판정 게이트 값을
실행 중 실시간으로 조정하는 UI다 — 컨트롤 값이 바뀌면 `AppController`가 들고 있는 그 `AppSettings`
인스턴스 필드를 즉시 mutate하고(별도 "적용" 배선 없음), "저장" 버튼을 눌러야 `App.config` 파일에
영구 반영된다. `Clamp()`의 min/max가 이 UI의 `NumericUpDown` 범위와 동일해야 한다는 암묵적 불변조건이
있다.

### 계층 소유권 (여러 명이 동시에 작업할 때)

`docs/CONTRACT.md`는 "WinDev-Sdk"(`Sdk/`, `Capture/`, `Worker/`)와 "WinDev-Logic"(`Analysis/`)처럼
작업자별 소유 폴더를 전제로 쓰여 있다. 다른 에이전트/사람과 동시에 작업할 가능성이 있으면 이 문서의
0절부터 확인할 것 — 소유 파일 밖을 건드리지 않는 것이 이 프로젝트의 명시적 규칙이다.

## 알려진 한계 (코드/UI 수정 시 인지할 것)

- 안면 인식 기반이다. 객체 인식이 아니라서 **엎드림과 이석을 원리적으로 구분 못 한다** —
  `EnableSlumpHeuristic`으로 "눈 감고 있다가 얼굴 사라짐 = 엎드림"으로 완화.
- `StudySec`는 "깨어 있는 시간"이지 실제 공부 시간이 아니다.
- C# wrapper는 MultiFace를 지원하지 않는다 — 프레임당 대표 얼굴 1개, 단일 좌석 1인 기준 데모.
- 모든 게이트 임계값은 SDK 데모(근접 셀피) 기준을 물려받은 **잠정치**다. 현장 재보정이 전제.
