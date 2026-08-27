"use strict";

const $ = (id) => document.getElementById(id);
let stream = null;
let lastCaptureId = null;
let lastAlgo = "";

// ── 초기화 ────────────────────────────────────────────────────────────
fetch("/api/health").then(r => r.json()).then(h => {
  $("version").textContent =
    `${h.algo_version} · config ${h.config_hash}` +
    (h.model_present ? "" : " · 랜드마커 모델 없음 (scripts/setup_env.ps1 실행 필요)");
}).catch(() => {});

// ── A. 브라우저 촬영 ──────────────────────────────────────────────────
$("btn-cam").onclick = async () => {
  try {
    // 가능한 최고 해상도를 요청한다. 실제로 무엇을 받았는지는 아래에서 확인한다 —
    // 브라우저 캡처는 기기마다 해상도가 크게 다르다.
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 3840 }, height: { ideal: 2160 } },
      audio: false,
    });
    $("preview").srcObject = stream;
    await $("preview").play();
    $("btn-shot").disabled = false;
    $("btn-cam").textContent = "카메라 켜짐";
  } catch (e) {
    alert("카메라를 열지 못했습니다: " + e.message);
  }
};

$("btn-shot").onclick = () => {
  const v = $("preview");
  const c = document.createElement("canvas");
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0);
  c.toBlob(b => send(b, "capture.png", "browser"), "image/png");
};

// ── B. 업로드 ─────────────────────────────────────────────────────────
$("file").onchange = (e) => {
  const f = e.target.files[0];
  if (f) send(f, f.name, "upload");
};

$("btn-retry").onclick = () => {
  $("step-result").hidden = true;
  window.scrollTo({ top: 0, behavior: "smooth" });
};

// ── 전송 ──────────────────────────────────────────────────────────────
async function send(blob, name, capturePath) {
  $("busy").hidden = false;
  $("step-result").hidden = true;
  try {
    const fd = new FormData();
    fd.append("image", blob, name);
    fd.append("capture_path", capturePath);
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    render(await res.json());
  } catch (e) {
    alert("분석 실패: " + e.message);
  } finally {
    $("busy").hidden = true;
  }
}

// ── 렌더 ──────────────────────────────────────────────────────────────
function render(d) {
  lastCaptureId = d.capture_id;
  lastAlgo = d.algo_version || "";
  $("step-result").hidden = false;
  $("fb-done").hidden = true;

  // [3] 품질 게이트에 걸렸으면 점수를 보여주지 않고 재촬영을 유도한다.
  // 이것이 이 UI 의 핵심이다 — 혼합 광원·과노출·가림은 알고리즘으로 못 고친다.
  if (!d.passed) {
    $("gate").hidden = false;
    $("result").hidden = true;
    $("gate-title").textContent = d.guidance.title;
    $("gate-how").textContent = d.guidance.how;
    return;
  }
  $("gate").hidden = true;
  $("result").hidden = false;

  if (d.overlay) $("overlay").src = d.overlay;

  // 부위별 카드
  $("regions").innerHTML = d.regions.map(r => {
    if (!r.measured) {
      // **0 점이 아니라 '측정 못 함 + 이유'.** 0 으로 표시하면
      // "측정 못 함"이 "아주 정상"으로 둔갑한다.
      return `<div class="region unmeasured">
        <div class="rname">${r.label}</div>
        <div class="rscore">측정 못 함</div>
        <div class="rwhy">${r.reason ? r.reason.title : ""}<br>
          <span class="dim">${r.reason ? r.reason.how : ""}</span></div>
      </div>`;
    }
    const hot = r.erythema_verdict === "affected";
    const les = r.lesion_verdict === "not_measured"
      ? '<span class="dim">트러블 미측정</span>'
      : (r.lesion_verdict === "affected"
        ? `트러블 ${r.lesion_count}개`
        : '<span class="dim">트러블 없음</span>');
    return `<div class="region ${hot ? "hot" : ""}">
      <div class="rname">${r.label}</div>
      <div class="rscore">${r.score_ordinal_0_100.toFixed(0)}</div>
      <div class="rverdict">${hot ? "붉음" : "정상"} · ${les}</div>
    </div>`;
  }).join("");

  // 주의 사항
  $("notes").innerHTML = (d.notes || []).map(n =>
    `<div class="note-box"><b>${n.title}</b><br>${n.how}</div>`).join("");

  // 근거 — 숫자를 숨기지 않는다. 점수만 주면 신뢰가 생기지 않는다.
  const b = d.baseline || {};
  const rows = d.regions.filter(r => r.measured).map(r => `
    <tr><td>${r.label}</td>
        <td>${pct(r.evidence.coverage)}</td>
        <td>${pct(r.evidence.area_fraction)}</td>
        <td>${num(r.evidence.median_d, 4)}</td>
        <td>${num(r.evidence.intensity, 4)}</td></tr>`).join("");
  $("detail-body").innerHTML = `
    <p class="dim">점수 0~100 은 <b>순서형 편의 척도</b>이지 물리량이 아닙니다.
       물리적으로 의미 있는 값은 <code>d</code>(log10 비율)와 면적 비율뿐입니다.</p>
    <table>
      <tr><th>부위</th><th>측정 가능 면적</th><th>붉은 면적</th>
          <th>중앙 편차 d</th><th>붉은 부위 강도</th></tr>
      ${rows}
    </table>
    <ul class="kv">
      <li>기준선 b_face: <code>${num(b.b_face, 4)}</code>
          (얼굴에서 가장 덜 붉은 피부를 원점으로 삼습니다)</li>
      <li>임계값 tau: <code>${num(b.tau, 4)}</code>
          = ${num(b.sigma_face, 4)} × k — 얼굴 자체의 산포 단위입니다</li>
      <li>멜라닌 기울기 beta: <code>${num(b.beta, 3)}</code></li>
      <li>조명 균일성 확인: <code>${d.illumination_check}</code>
          (참조 부위 ${b.n_ref_regions ?? 0}개)</li>
      <li>부위별 해부학 보정: <code>${b.anatomical_prior_calibrated ? "적용" : "미적용"}</code></li>
      <li>피부톤 ITA: <code>${num(d.ita_deg, 1)}°</code>
          — 어두운 피부일수록 신호 대 잡음비가 낮아집니다</li>
      <li>초점(VoL) <code>${num(d.metrics.blur_vol, 0)}</code> ·
          포화 <code>${pct(d.metrics.clipped_fraction)}</code> ·
          조명비 <code>${num(d.metrics.illum_ratio, 2)}</code></li>
      <li>버전: <code>${d.algo_version} / ${d.config_hash}</code>
          — 이 숫자가 있어야 나중에 재채점 결과와 비교할 수 있습니다</li>
    </ul>`;

  if (d.warnings && d.warnings.length) {
    $("notes").innerHTML += d.warnings.map(w =>
      `<div class="note-box warn">${w}</div>`).join("");
  }
}

const num = (v, n) => (v === null || v === undefined) ? "—" : Number(v).toFixed(n);
const pct = (v) => (v === null || v === undefined) ? "—" : (v * 100).toFixed(1) + "%";

// ── [5] 피드백 ────────────────────────────────────────────────────────
document.querySelectorAll(".feedback button").forEach(btn => {
  btn.onclick = async () => {
    if (!lastCaptureId) return;
    const fd = new FormData();
    fd.append("capture_id", lastCaptureId);
    fd.append("verdict", btn.dataset.v);
    fd.append("algo_version", lastAlgo);
    await fetch("/api/feedback", { method: "POST", body: fd });
    $("fb-done").hidden = false;
  };
});
