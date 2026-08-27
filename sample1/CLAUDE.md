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

## What this is

`FaceSDKSample` — a .NET Framework 4.8 WinForms demo app for the Alchera Face SDK. One project
(`face-sdk-wrapper-samples.csproj`, assembly `FaceSDKSample`), no tests, not a git repository.
It exercises the passive-liveness → best-shot → BGR-liveness → face-compare pipeline against a
USB webcam or an IP camera, running each stage either locally (`ClientOnly`) or against a
FaceServer REST endpoint (`FaceServer`).

`docs/FASMH-97-changes.md` documents the current in-flight change set (IP-cam model profiles,
threshold update, error-code detail). Read it before touching those areas.

## Build

Build the **`.sln`**, never the `.csproj` alone — `PreBuildEvent` is `dotnet clean "$(SolutionDir)"`
and `SolutionDir` is undefined in a csproj-only build (fails with MSB1009/MSB3073).

Use the **VS BuildTools** MSBuild. The .NET Framework MSBuild under `C:\Windows\Microsoft.NET\`
rejects `<LangVersion>7.3</LangVersion>` with CS1617.

```bash
MSBUILD='C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\MSBuild\Current\Bin\MSBuild.exe'
"$MSBUILD" face-sdk-wrapper-samples.sln //p:Configuration=Debug //p:Platform=x64 //t:Rebuild
```

(Locate MSBuild on other machines with `vswhere.exe -latest -products * -find MSBuild\**\Bin\MSBuild.exe`.)

- Only **Debug** configurations exist (`Any CPU`, `x64`, `x86`); `Any CPU` maps to `x64`. There is no Release.
- **`FaceSDKSample.exe` must not be running.** Compilation succeeds but the obj→bin copy fails with
  MSB3021/MSB3027. Also kill any attached `Visual Studio Remote Debugger`.
- Threshold constants are `const` and get inlined at every call site, so value changes require
  `/t:Rebuild` — an incremental build silently keeps the old numbers.
- There are 4 long-standing warnings (`DemoClientServer.cs(216)`, `FaceSDKSVR.cs(1180)`,
  `FaceSDKUtil.cs(80)`, `FaceSDKUtil.cs(456)`). Don't treat them as regressions.

## Run

Run `bin\x64\Debug\FaceSDKSample.exe` **with the output directory as the working directory** —
`Program.Main` passes `Directory.GetCurrentDirectory()` to `Sample.Create()`, and the native SDK
resolves `models` relative to the process cwd.

The output directory needs three things that MSBuild does **not** produce:

1. **`models\`** (~620 MB) including `models\license.cer`. Activate with `LicenseGen.exe` from
   `FaceSDK\lib\cpu\windows-x86_64\`.
2. **Six LibTorch-family native DLLs** copied by hand from `FaceSDK\lib\cpu\windows-x86_64\`:
   `torch_cpu`, `c10`, `fbgemm`, `libiomp5md`, `asmjit`, `uv`. Missing any of them breaks the
   `AlcheraFaceSDKCS.dll → AlcheraFaceSDK.dll → torch_cpu.dll/c10.dll` chain and surfaces as
   `DllNotFoundException: AlcheraFaceSDKCS.dll` (Win32 error 126 — the *dependency* is missing,
   not the named DLL). `.gitignore` excludes `*.dll` and `bin/`, and the csproj has no copy step,
   so **this recurs in every fresh clone or worktree**.
3. **`AlcheraFaceSDKCS.dll` and `AlcheraEncryptCS.dll`** from
   `FaceSDK\lib\cpu\windows-x86_64\csharp\`, plus **`AlcheraFaceSDK.dll`** and
   **`opencv_world455.dll`** from `FaceSDK\lib\cpu\windows-x86_64\`. `AlcheraEncryptCS.dll` is
   referenced by a `HintPath` pointing *into* `bin\x64\Debug\`, so it must be there before the
   build, not just before the run.

There is no test project. Verification is manual through the UI; for compiled-in constants you can
read the built assembly by reflection instead of launching the app.

## Architecture

### Startup and ownership

`Program.Main` → `Sample.inst.Create(cwd)` → `Application.Run(new Form1())`.

`Sample` (partial: `sample/Sample.cs`, `sample/SampleDemoMan.cs`) is a singleton service locator
holding the two long-lived objects, reachable as `Sample.fsdk` and `Sample.cam`:

- `SampleFaceSDKCtx` — wraps `FaceSDK.Instance()` + `Initialize(models)`. Created once; recreating
  is a no-op.
- `CamCtx` — the currently open capture. `OpenCapture()` disposes any previous one first.

`Form1` enforces a three-step gate through button enabling: **Load FaceSDK → open a capture
(USB or IP cam) → start the demo.** Each step enables the next; nothing works out of order.

### The demo thread and its message queue

`Demo` (`sample/Demo.cs`) is an abstract background-thread host: subclass overrides `Run()`,
exits cooperatively via `SetExit()` / `IsRunning()` / `Join()`.

`DemoClientServer` is the only implementation, split across four partial files:

| File | Role |
|---|---|
| `sample/DemoClientServer.cs` | the `Run()` loop and the `DemoStage` state machine |
| `sample/DemoClientServerCtx.cs` | `RunContext` / `RecContext` — per-run state, rebuilt on `Reset` |
| `sample/DemoClientServerMsg.cs` | `DispatchMessage()` — the inbound message switch |
| `sample/DemoClientServerUI.cs` | overlay rendering helpers |

`DemoStage`: `Init → DoPassiveLivenessAndCollectLivenessBestShot → BGRImageLivenessMulitframe →
FaceCompare → FinalResult → Reset`. `Reset` reallocates `RunContext`, clears the SDK's BGR-liveness
cache, and loops back.

**All UI → demo communication goes through the queue.** `UserPrm.PostMsg(id, prm0, prm1, prm2)` from
the WinForms thread; the demo thread drains it with `DispatchMessage(ctx)` at the top of every
iteration. Message ids are bare integers: `1000` property update, `1100` clear captured frames,
`2000` append UI status text, `3000` stage result, `4000` start recording, `9999` exit.
Demo → UI goes the other way through the `ui_onstart_` / `ui_onloop_` / `ui_onexit_` actions, which
must `BeginInvoke` onto the UI thread (see `Form1.cs:397`). Never touch a control from the demo thread.

### Configuration: PropertyGrid, not a config file

Runtime knobs live in `FaceSDKSample.SampleProp` as plain classes bound to WinForms `PropertyGrid`s
(`Prop` base is at `sample/SamplePropSettings.cs:20`):

- `Settings` (`sample/SamplePropSettings.cs`) — capture resolution/index/flip, server connection
- `DemoClientServerProp` (`sample/SamplePropDemo.cs`) — liveness and compare thresholds and toggles
- `SamplePropFaceInfo` (`sample/SamplePropFaceInfo.cs`)

The `[LIV-xx]` / `[CAP-xx]` prefixes visible in the UI are **`Category(...)` label text, not
identifiers** — searching for "LIV-75" finds the group, not a constant. Grep by property name
(`LIV_THRESHOLD_BGR_IMAGE_LIVENESS_MIN`) or by the category string.

Property edits reach the demo thread via `Form1.PPG_*_PropertyValueChanged` →
`Sample.PostDemoPropEvent` → message `1000`.

**Threshold defaults vs. runtime values.** Defaults live as `const` in
`Alchera.FaceSDK.LivenessThreshold` (`FaceSDKPassiveLiv.cs`) and `FaceSDK.Params` (`FaceSDK.cs`);
the PropertyGrid properties merely use them as initializers. The value actually in effect is the
PropertyGrid one, so **demo code must read `prop_demo.*`, never the `const` directly** — reading the
constant silently ignores whatever the user set (this bug existed at `DemoClientServer.cs:848`).
`App.config` / `Settings.settings` are unused; the only persisted state is `ipcam_models.json`.

### Capture

`CamCtx` (`sample/CamCtx.cs`) is an abstract template: `OnOpenCapture` / `OnCaptureFrame` / `OnTick` /
`OnCloseCapture` / `OnIsErr` / `GetInfo` / `Exec(key, ...)`. It doubles as the home for the static
OpenCvSharp drawing and image helpers the demo's overlay renderer uses (`DrawText`, `DrawLogBox`,
`MakeBGRFromImg`, `RenderCicleHoleMat`, `CamResResolver`, …).

- `CamCtx_USBCam` — OpenCvSharp `VideoCapture` by device index, backend selectable (ANY/MSMF/DSHOW/FFMPEG).
- `CamCtx_IPCam` — two protocols selected by which ctor overload is used: `RTSP` (`VideoCapture` +
  `VideoCaptureAPIs.FFMPEG`) or `MJPEG_WS` (MJPEG frames over a WebSocket).

IP camera protocol clients are in `sample/CamHelper/`, both shaped the same way — an `ERR` enum,
`SetLastErr`/`GetLastErrDesc()` formatted as `"{int}({ENUM_NAME}), {desc}"`, and `SetLogLevel()`:

- `IPCam335N_RTSPClient` (`IPCam335N_RTSP.cs`) — generic RTSP, works with any camera.
- `IPCam335N_MJpegWSClient` (`IPCam335N_MJpeg.cs`) — **vendor-specific to the 335N**: it sends
  `{"start_video":{"ch":N}}` over the WebSocket and expects two acks. Camera models added through
  the UI therefore only really work over RTSP.

`sample/CamHelper/IPCam335N.cs` and `IPCam335N-260517-01.back` are **not in the csproj** and
duplicate `IPCam335N_MJpegWSClient`. Adding either to the project produces CS0101.

### IP camera model profiles

`sample/IPCamModelProfile.cs` holds `IPCamModelProfile` (name, per-protocol default port, URL
template with `{ip}`/`{port}` tokens, `keep_user_input`) and `IPCamModelStore` (JSON load/save,
built-in defaults). Profiles are edited in-app through `sample/IPCamModelEditDlg.cs` (`[+]`/`[-]`
next to the Model combo in `Form1`) and persisted to **`ipcam_models.json` beside the exe** — the
same exe-relative convention as `models\` and `imgs\compare\`. A missing or unparsable file falls
back to the built-in defaults; a parse failure is reported, not swallowed.

`CamCtx.cap_dev_model_` is display-only — nothing reads it, so model names never reach the capture
layer.

### FaceServer client

`FaceSDKSVR.cs` (namespace `Alchera.FaceSDK.FaceServer`) is the REST layer: `Api` / `Conn` /
`RestConn`, with `LivenessMulitFrameApi` (`POST /liveness/multiframe/v2`) and `FaceCompareApi`
(`POST /compare`), plus optional RSA+AES payload encryption via `AlcheraEncryptCS.dll`
(`PayloadEncRSAAES`). Whether a stage calls the server is per-stage: the `api-type` properties
(`LIV_BGR_IMG_LIV_API_TYPE`, `COMPARE_API_TYPE`) select `ClientOnly` / `FaceServer` / `None`.

## Conventions and traps

- **Vendored SDK wrappers.** `FaceSDK.cs`, `FaceSDKPassiveLiv.cs`, `FaceSDKSVR.cs`, `FaceSDKUtil.cs`
  at the repo root are copies of vendor SDK sources; byte-identical originals sit under
  `FaceSDK\lib\cpu\windows-x86_64\csharp\`. Keep edits minimal and surgical, and consider mirroring
  them into the vendor copy for consistency (it has no build effect — those files are not compiled).
- **Line endings are deliberately mixed.** The four vendored root files are **CRLF**; the project's
  own code (`Form1*.cs`, `sample/**`, the csproj) is **LF**. `sed -i` strips CR, which turns a
  one-line change into a whole-file diff — restore CRLF byte-wise afterwards, or use the Edit tool
  on those four files. Most files are UTF-8 **with BOM**; match the file you're editing.
- **C# 7.3 only.** No switch expressions, nullable reference types, `using var`, target-typed `new`,
  or default interface members.
- **Naming.** Private fields are `snake_case_` with a trailing underscore, locals are `snake_case`,
  methods and types are `PascalCase`. Comments are Korean or English depending on the surrounding block.
- **Hand-coded dialogs.** `FeatureExplorerDlg.cs`, `FeatureExpCompareDlg.cs`, and
  `IPCamModelEditDlg.cs` build their controls in the constructor with **no `.Designer.cs` and no
  `.resx`**. Follow that pattern for new dialogs; only `Form1` uses the WinForms designer.
- **New source files must be added to the csproj by hand** (`<Compile Include="..." />`, plus
  `<SubType>Form</SubType>` for a Form) — this is a non-SDK-style project with no globbing.
- **Reentrancy in `Form1` IP-cam handlers.** `cap_src_ipcam_on_host_ip_changed()` assigns
  `.Text` on controls whose `TextChanged` calls back into it; it is guarded by `updating_ip_cam_url_`.
  Keep the guard when adding fields there.
- **`_backup_20260826/`** is a manual pre-change snapshot. Since the repo is not under version
  control it is the only copy of the prior state — don't delete it without asking.
- **`sample/bin/`** (~682 MB) is a stray duplicate output tree from an old csproj-only build. Not
  used by anything.
