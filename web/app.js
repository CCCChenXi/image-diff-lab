/* 图像变化检测实验台 —— 前端零构建，只依赖浏览器原生 API。 */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const CHANGE_LABELS = {
  none: "无变化",
  added: "新增",
  removed: "消失",
  moved: "位置变化",
  content: "内容变化",
  unknown: "无法判定",
};

const MAX_IMAGE = 12 * 1024 * 1024;

const state = {
  meta: null,
  samples: [],
  before: null,
  after: null,
  selected: new Set(),
  result: null,
  busy: false,
};

let toastTimer = 0;
let loadingTimer = 0;
let hintTimer = 0;

/* ---------------------------------------------------------------- utils */
function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function formatMs(ms) {
  const n = Number(ms);
  if (!Number.isFinite(n)) return "—";
  if (n >= 1000) return `${(n / 1000).toFixed(2)} s`;
  return `${n.toFixed(1)} ms`;
}

function formatValue(value) {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    if (Number.isInteger(value)) return value.toLocaleString("zh-CN");
    return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }
  if (typeof value === "boolean") return value ? "是" : "否";
  return String(value ?? "—");
}

function toast(message, isError = false) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.toggle("error", isError);
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    node.hidden = true;
  }, isError ? 6000 : 3400);
}

/* ---------------------------------------------------------------- images */
function readImageFile(file) {
  return new Promise((resolve, reject) => {
    if (!file || !file.type.startsWith("image/")) {
      reject(new Error("请选择 PNG / JPG / WebP 图片"));
      return;
    }
    if (file.size > MAX_IMAGE) {
      reject(new Error(`「${file.name}」超过 12MB，请先压缩`));
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("读取图片失败"));
    reader.onload = () => {
      const dataUrl = String(reader.result);
      const probe = new Image();
      probe.onload = () =>
        resolve({
          dataUrl,
          name: file.name || "粘贴的图片",
          size: file.size,
          width: probe.naturalWidth,
          height: probe.naturalHeight,
        });
      probe.onerror = () => reject(new Error("浏览器无法解码这张图片"));
      probe.src = dataUrl;
    };
    reader.readAsDataURL(file);
  });
}

async function fetchAsImage(url, filename) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`示例图加载失败（HTTP ${res.status}）`);
  const blob = await res.blob();
  return readImageFile(new File([blob], filename, { type: blob.type || "image/png" }));
}

function setImage(side, image) {
  state[side] = image;
  const zone = $(`#dz-${side}`);
  const preview = $(`#preview-${side}`);
  const empty = $(".dz-empty", zone);
  const wrap = $(".dz-preview", zone);
  preview.src = image.dataUrl;
  $("#name-" + side).textContent = image.name;
  $("#meta-" + side).textContent = `${image.width} × ${image.height} · ${formatBytes(image.size)}`;
  empty.hidden = true;
  wrap.hidden = false;
  zone.classList.add("has-image");
  syncRunButton();
}

function clearImage(side) {
  state[side] = null;
  const zone = $(`#dz-${side}`);
  const preview = $(`#preview-${side}`);
  const empty = $(".dz-empty", zone);
  const wrap = $(".dz-preview", zone);
  preview.removeAttribute("src");
  empty.hidden = false;
  wrap.hidden = true;
  zone.classList.remove("has-image");
  syncRunButton();
}

function syncRunButton() {
  const canRun = Boolean(state.before && state.after && state.selected.size > 0 && !state.busy);
  $("#run").disabled = !canRun;
  $("#reset").hidden = !(state.before || state.after || state.result);
}

/* ---------------------------------------------------------------- upload UI */
function bindDropzone(side) {
  const zone = $(`#dz-${side}`);
  const input = $(`#file-${side}`);

  const pick = () => input.click();

  // 文件输入自己的 click 会冒泡到 dropzone；不拦住会递归触发 pick()，
  // Chrome 对嵌套的文件选择调用会直接忽略，表现就是“点了没反应”。
  input.addEventListener("click", (event) => event.stopPropagation());

  zone.addEventListener("click", (event) => {
    // 交给 <label for> 原生打开文件选择器；preventDefault 会把它也取消掉
    if (event.target.closest("label[for]")) return;
    if (event.target === input) return;
    const action = event.target.closest("[data-action]")?.dataset.action;
    if (action === "clear") {
      event.stopPropagation();
      clearImage(side);
      return;
    }
    event.preventDefault();
    pick();
  });

  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      pick();
    }
  });

  input.addEventListener("change", async () => {
    const file = input.files?.[0];
    input.value = "";
    if (!file) return;
    try {
      const image = await readImageFile(file);
      setImage(side, image);
    } catch (err) {
      toast(err.message, true);
    }
  });

  ["dragenter", "dragover"].forEach((type) => {
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      event.stopPropagation();
      zone.classList.add("dragover");
    });
  });
  ["dragleave", "dragend"].forEach((type) => {
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
    });
  });
  zone.addEventListener("drop", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    zone.classList.remove("dragover");
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    try {
      const image = await readImageFile(file);
      setImage(side, image);
    } catch (err) {
      toast(err.message, true);
    }
  });
}

function bindPaste() {
  document.addEventListener("paste", async (event) => {
    const items = [...(event.clipboardData?.items || [])];
    const item = items.find((it) => it.type.startsWith("image/"));
    if (!item) return;
    event.preventDefault();
    const file = item.getAsFile();
    if (!file) return;
    const side = !state.before ? "before" : !state.after ? "after" : "after";
    try {
      const image = await readImageFile(file);
      image.name = `粘贴图-${side === "before" ? "1" : "2"}.png`;
      setImage(side, image);
      toast(`已粘贴到「${side === "before" ? "第一张图" : "第二张图"}」`);
    } catch (err) {
      toast(err.message, true);
    }
  });
}

/* ---------------------------------------------------------------- meta */
async function loadMeta() {
  const res = await fetch("/api/meta");
  const data = await res.json();
  if (!data.ok) throw new Error(data.error || "读取元信息失败");
  state.meta = data;
  state.selected = new Set(
    data.methods.filter((m) => m.available !== false).map((m) => m.key)
  );
  renderMethods();
  updateEngineStatus(data);
  return data;
}

function renderMethods() {
  const host = $("#method-pick");
  const methods = state.meta?.methods || [];
  host.innerHTML = methods
    .map((m) => {
      const available = m.available !== false;
      const reason = m.unavailableReason || "";
      const note = available
        ? `${esc(m.cost || "")} · ${esc(m.strength || "")}`
        : "需自备权重 · 见 docs/yolo-byom.md";
      return `
      <label class="method-chip ${available ? "" : "disabled"}" style="--mc:${esc(m.color)}"
             title="${esc(available ? m.description : reason)}">
        <input type="checkbox" value="${esc(m.key)}" ${available ? "checked" : "disabled"}>
        <span class="chip-box" aria-hidden="true"></span>
        <span class="chip-text">
          <strong>${esc(m.name)}${available ? "" : `<em>可选</em>`}</strong>
          <small>${note}</small>
        </span>
      </label>`;
    })
    .join("");

  $$("input[type=checkbox]", host).forEach((input) => {
    input.addEventListener("change", () => {
      if (input.checked) state.selected.add(input.value);
      else state.selected.delete(input.value);
      syncRunButton();
    });
  });
}

function updateEngineStatus(meta) {
  const node = $("#engine-status");
  if (!meta) {
    node.className = "status-pill offline";
    $(".label", node).textContent = "服务离线";
    return;
  }
  const ocrMeta = meta.methods?.find((m) => m.key === "ocr");
  const yoloMeta = meta.methods?.find((m) => m.key === "yolo");
  const ocr = ocrMeta?.status;
  const yolo = yoloMeta?.status;
  const yoloAvailable = yoloMeta?.available !== false;
  const expected = 1 + (yoloAvailable ? 1 : 0);
  const ready = (ocr?.ready ? 1 : 0) + (yoloAvailable && yolo?.ready ? 1 : 0);
  node.className = "status-pill" + (ready === expected ? " ready" : "");
  let text =
    ready === expected
      ? yoloAvailable
        ? "模型就绪"
        : "模型就绪 · YOLO 可选"
      : `模型预热中 ${ready}/${expected}`;
  if (!yoloAvailable) text = ready >= 1 ? "模型就绪 · YOLO 可选" : "YOLO 未配置（可选）";
  $(".label", node).textContent = text;
}

async function pollStatus(attempt = 0) {
  try {
    const res = await fetch("/api/meta", { cache: "no-store" });
    const data = await res.json();
    if (!data.ok) throw new Error("meta failed");
    updateEngineStatus(data);
    const ocr = data.methods?.find((m) => m.key === "ocr")?.status;
    const yoloMeta = data.methods?.find((m) => m.key === "yolo");
    const yolo = yoloMeta?.status;
    const yoloAvailable = yoloMeta?.available !== false;
    if (!(ocr?.ready && (!yoloAvailable || yolo?.ready)) && attempt < 20) {
      setTimeout(() => pollStatus(attempt + 1), 3000);
    }
  } catch {
    if (attempt < 3) setTimeout(() => pollStatus(attempt + 1), 3000);
    else updateEngineStatus(null);
  }
}

/* ---------------------------------------------------------------- samples */
async function loadSamples() {
  const res = await fetch("/api/samples");
  const data = await res.json();
  if (!data.ok || !data.samples?.length) return;
  state.samples = data.samples;
  const host = $("#samples");
  host.hidden = false;
  host.innerHTML =
    `<span class="samples-label">载入示例</span>` +
    data.samples
      .map(
        (s) =>
          `<button type="button" class="sample-btn" data-sample="${esc(s.id)}" title="${esc(s.note)}">${esc(s.label)}</button>`
      )
      .join("");

  $$(".sample-btn", host).forEach((btn) => {
    btn.addEventListener("click", async () => {
      const sample = state.samples.find((s) => s.id === btn.dataset.sample);
      if (!sample) return;
      btn.disabled = true;
      const old = btn.textContent;
      btn.textContent = "载入中…";
      try {
        await applySample(sample);
        toast(`已载入「${sample.label}」：${sample.note}`);
      } catch (err) {
        toast(err.message, true);
      } finally {
        btn.disabled = false;
        btn.textContent = old;
      }
    });
  });
}

async function applySample(sample) {
  const [before, after] = await Promise.all([
    fetchAsImage(sample.before, `${sample.id}-before.png`),
    fetchAsImage(sample.after, `${sample.id}-after.png`),
  ]);
  setImage("before", before);
  setImage("after", after);
}

// 调试 / 截图用：?sample=added&run=1&methods=pixel_diff,red_hsv,ssim
async function maybeDemo() {
  const params = new URLSearchParams(location.search);
  const id = params.get("sample");
  if (!id) return;
  const sample = state.samples.find((s) => s.id === id);
  if (!sample) return;
  await applySample(sample);

  const methods = params.get("methods");
  if (methods) {
    const wanted = new Set(methods.split(",").map((s) => s.trim()).filter(Boolean));
    const available = new Set(
      (state.meta?.methods || []).filter((m) => m.available !== false).map((m) => m.key)
    );
    state.selected = new Set([...wanted].filter((k) => available.has(k)));
    $$("#method-pick input[type=checkbox]").forEach((input) => {
      input.checked = state.selected.has(input.value);
    });
    syncRunButton();
  }

  if (params.get("run") === "1") await runCompare();
}

/* ---------------------------------------------------------------- run */
function setBusy(busy) {
  state.busy = busy;
  const btn = $("#run");
  $(".btn-label", btn).textContent = busy ? "分析中…" : "开始分析";
  syncRunButton();
}

function startLoading() {
  $("#loading").hidden = false;
  const hints = [
    "解码两张图片并对齐尺寸…",
    "OpenCV 像素差分：定位变化区域…",
    "HSV 红色检测：比较红点数量与位置…",
    "SSIM：计算整体结构相似度…",
    "OCR 文字识别：首次调用可能需要 10–20 秒…",
    "YOLO：检测 UI 元素并比较元素内部…",
    "汇总判定，结算 AOP 切面耗时…",
  ];
  let index = 0;
  $("#loading-hint").textContent = hints[0];
  const started = performance.now();
  clearInterval(loadingTimer);
  loadingTimer = setInterval(() => {
    $("#loading-timer").textContent = `${((performance.now() - started) / 1000).toFixed(1)}s`;
  }, 100);
  clearInterval(hintTimer);
  hintTimer = setInterval(() => {
    index = (index + 1) % hints.length;
    $("#loading-hint").textContent = hints[index];
  }, 2800);
}

function stopLoading() {
  $("#loading").hidden = true;
  clearInterval(loadingTimer);
  clearInterval(hintTimer);
}

async function runCompare() {
  if (state.busy) return;
  if (!state.before || !state.after) {
    toast("请先选择两张图片", true);
    return;
  }
  if (!state.selected.size) {
    toast("至少选择一种检测方法", true);
    return;
  }

  setBusy(true);
  startLoading();
  const started = performance.now();
  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        before: state.before.dataUrl,
        after: state.after.dataUrl,
        methods: [...state.selected],
      }),
    });
    const data = await res.json().catch(() => ({ ok: false, error: "服务端返回的不是 JSON" }));
    if (!res.ok || !data.ok) {
      throw new Error(data.error || `请求失败（HTTP ${res.status}）`);
    }
    data.clientMs = performance.now() - started;
    state.result = data;
    renderResult(data);
    $("#results").hidden = false;
    requestAnimationFrame(() => {
      $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
    });
  } catch (err) {
    toast(err.message || "分析失败", true);
  } finally {
    stopLoading();
    setBusy(false);
  }
}

/* ---------------------------------------------------------------- render */
function stateMeta(response) {
  const map = new Map();
  for (const m of state.meta?.methods || []) map.set(m.key, m);
  return map;
}

function renderResult(data) {
  const metaMap = stateMeta(data);
  const timing = data.timing || {};
  const images = data.images || {};

  const aligned = images.aligned ? ` · 已对齐到 ${images.width}×${images.height}` : "";
  $("#result-sub").textContent =
    `${data.requestId} · 服务端 ${formatMs(timing.totalMs)} · 往返 ${formatMs(data.clientMs)}` +
    `${aligned}${images.warning ? " · ⚠ 宽高比不一致" : ""} · 单次对比（无真值，不计算准确率）`;

  renderVerdict(data.consensus || {});
  renderTiming(data.methods || [], timing, data.clientMs);
  renderCards(data.methods || [], metaMap);
  renderAspect(timing.aspect || {}, timing);
  bindZoom();
  syncRunButton();
}

function renderVerdict(consensus) {
  const host = $("#verdict");
  const hasChange = consensus.hasChange;
  const cls = hasChange === true ? "changed" : hasChange === false ? "stable" : "failed";
  const title = hasChange === true ? "检测到变化" : hasChange === false ? "未检测到变化" : "无法综合判断";
  const icon =
    cls === "changed"
      ? `<svg viewBox="0 0 24 24" fill="none"><path d="M12 9v4m0 4h.01M10.3 3.9 2.4 18a1.5 1.5 0 0 0 1.3 2.2h16.6a1.5 1.5 0 0 0 1.3-2.2L13.7 3.9a1.5 1.5 0 0 0-2.6 0Z" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="none"><path d="m5 12.5 4.2 4.2L19 7" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  const votes = consensus.votes || { change: 0, noChange: 0 };
  const agreement = Math.round((consensus.agreement || 0) * 100);
  const typeVotes = Object.entries(consensus.typeVotes || {})
    .filter(([, n]) => n > 0)
    .map(([k, n]) => `<span class="type-pill">${esc(CHANGE_LABELS[k] || k)} ×${n}</span>`)
    .join("");

  host.className = `panel verdict ${cls}`;
  host.innerHTML = `
    <div class="verdict-top">
      <span class="verdict-icon">${icon}</span>
      <div>
        <div class="verdict-label">CONSENSUS · 多数投票</div>
        <h3>${title}</h3>
      </div>
    </div>
    <div class="verdict-type">${esc(consensus.changeTypeLabel || "—")}</div>
    <p class="summary">${esc(consensus.summary || "")}</p>
    <div class="card-state" style="padding:0;margin-top:12px">${typeVotes}</div>
    <div class="agreement">
      <div class="agreement-head" title="没有真值时无法计算准确率；一致度只表示方法之间的投票一致程度">
        <span>方法一致度 ${agreement}% <small>（≠ 准确率）</small></span>
        <span>${votes.change} 个判变化 · ${votes.noChange} 个判无变化</span>
      </div>
      <div class="agreement-track"><div class="agreement-fill" data-w="${agreement}%"></div></div>
    </div>`;

  requestAnimationFrame(() => {
    const fill = $(".agreement-fill", host);
    if (fill) fill.style.width = fill.dataset.w;
  });
}

function renderTiming(methods, timing, clientMs) {
  const host = $("#timing-bars");
  const total = Number(timing.totalMs) || 0;
  $("#total-ms").textContent = `${formatMs(total)} 总耗时`;

  const rows = [...methods].sort((a, b) => b.elapsedMs - a.elapsedMs);
  const max = Math.max(1, ...rows.map((m) => Number(m.elapsedMs) || 0));
  host.innerHTML = rows
    .map((m) => {
      const color = (state.meta?.methods || []).find((x) => x.key === m.key)?.color || "var(--muted)";
      const width = Math.max(2, Math.min(100, ((Number(m.elapsedMs) || 0) / max) * 100));
      return `
        <div class="timing-row" style="--mc:${esc(color)}">
          <span class="t-name"><span class="t-dot"></span>${esc(m.name)}</span>
          <span class="t-track"><span class="t-fill" data-w="${width.toFixed(1)}%"></span></span>
          <span class="t-val">${formatMs(m.elapsedMs)}</span>
        </div>`;
    })
    .join("");

  const overhead = Math.max(0, total - rows.reduce((sum, m) => sum + (Number(m.elapsedMs) || 0), 0));
  $("#timing-note").textContent =
    `服务端按顺序串行调用，避免多个模型争抢 CPU/GPU；总耗时含图片解码、对齐与调度开销 ${formatMs(overhead)}。` +
    ` 客户端往返 ${formatMs(clientMs)}。`;

  requestAnimationFrame(() => {
    $$(".t-fill", host).forEach((fill) => {
      fill.style.width = fill.dataset.w;
    });
  });
}

function renderCards(methods, metaMap) {
  const host = $("#cards");
  host.innerHTML = methods.map((m, index) => renderCard(m, metaMap, index)).join("");
}

function renderCard(method, metaMap, index) {
  const meta = metaMap.get(method.key) || {};
  const color = meta.color || "#888";
  const unavailable = Boolean(method.details && method.details.unavailable);
  const verdictClass = unavailable ? "unavailable" : !method.ok ? "failed" : method.hasChange ? "changed" : "stable";
  const verdictText = unavailable ? "未配置" : !method.ok ? "执行失败" : method.hasChange ? "发生变化" : "未变化";
  const conf = Math.round((Number(method.confidence) || 0) * 100);

  const baseMetrics = method.metrics || [];
  const metrics = baseMetrics.some((m) => m.key === "regions")
    ? baseMetrics
    : [...baseMetrics, { key: "regions", label: "证据区域", value: (method.regions || []).length, unit: "处" }];

  const metricHtml = unavailable
    ? ""
    : metrics
        .map(
          (m) => `
      <div class="metric">
        <span>${esc(m.label)}</span>
        <strong>${esc(formatValue(m.value))}${m.unit ? `<small>${esc(m.unit)}</small>` : ""}</strong>
      </div>`
        )
        .join("");

  const overlay = method.overlay
    ? `<div class="card-overlay">
         <img src="${method.overlay}" alt="${esc(method.name)} 叠加层" data-caption="${esc(method.name)} · ${esc(method.changeTypeLabel)}">
         <span class="overlay-hint">点击放大</span>
       </div>`
    : "";

  const error = !method.ok && !unavailable
    ? `<div class="card-error">${esc(method.error || method.summary || "未知错误")}</div>`
    : "";

  const unavailableNote = unavailable
    ? `<div class="card-unavailable">
         <strong>YOLO 是可选方法（BYOM）</strong>
         <p>${esc(method.summary || "模型权重不随项目提供。")}</p>
         <p>把训练好的 <code>best.pt</code> 放到 <code>models/</code>，或设置 <code>YOLO_WEIGHTS</code> 后重启服务。详见 <code>docs/yolo-byom.md</code>。</p>
       </div>`
    : "";

  const detailsJson = JSON.stringify(method.details || {}, null, 2);

  return `
    <article class="method-card ${unavailable ? "is-unavailable" : ""}" style="--mc:${esc(color)};animation-delay:${index * 45}ms">
      <div class="card-head">
        <div class="method-ident">
          <span class="method-dot"></span>
          <div>
            <h4>${esc(method.name)}</h4>
            <small>${esc(method.key)}</small>
          </div>
        </div>
        <span class="card-time">${unavailable ? "—" : formatMs(method.elapsedMs)}</span>
      </div>
      <div class="card-state">
        <span class="verdict-pill ${verdictClass}">${verdictText}</span>
        <span class="type-pill">${unavailable ? "可选方法" : esc(method.changeTypeLabel || CHANGE_LABELS[method.changeType] || "—")}</span>
        <span class="conf-pill">${unavailable ? "需自备权重" : `置信度 ${conf}%`}</span>
      </div>
      <p class="card-summary">${esc(method.summary || "")}</p>
      ${unavailableNote}
      ${overlay}
      ${error}
      ${metricHtml ? `<div class="card-metrics">${metricHtml}</div>` : ""}
      <details class="card-details">
        <summary>原始指标 / details</summary>
        <pre>${esc(detailsJson)}</pre>
      </details>
    </article>`;
}

function renderAspect(aspect, timing) {
  const records = aspect.records || [];
  const traceId = aspect.traceId || "—";
  $("#pointcut").textContent = aspect.pointcut || "execution(*.detect(..))";
  $("#trace-id").textContent = `trace ${traceId}`;
  $("#aspect-meta").textContent =
    `${records.length} 条记录 · 服务总计 ${formatMs(timing.totalMs)}`;

  const host = $("#aspect-log");
  host.innerHTML = [...records]
    .sort((a, b) => (a.seq || 0) - (b.seq || 0))
    .map((r) => {
      const isService = String(r.target || "").includes("CompareService");
      const time = r.startedAt ? String(r.startedAt).slice(11, 23) : "—";
      return `
        <div class="log-row ${isService ? "service" : ""}">
          <span class="l-seq">#${esc(r.seq)}</span>
          <span class="l-target" title="${esc(r.target)}">${esc(r.target)}</span>
          <span class="l-ms">${formatMs(r.elapsedMs)}</span>
          <span class="l-status ${r.status === "error" ? "error" : ""}">${esc(r.status)}</span>
          <span class="l-time">${esc(time)}</span>
        </div>`;
    })
    .join("");
}

/* ---------------------------------------------------------------- lightbox */
function bindZoom() {
  $$(".card-overlay img[data-caption]").forEach((img) => {
    img.addEventListener("click", () => {
      $("#lightbox-img").src = img.src;
      $("#lightbox-cap").textContent = img.dataset.caption || "";
      $("#lightbox").hidden = false;
      document.body.style.overflow = "hidden";
    });
  });
}

function closeLightbox() {
  $("#lightbox").hidden = true;
  $("#lightbox-img").removeAttribute("src");
  document.body.style.overflow = "";
}

/* ---------------------------------------------------------------- export/reset */
function exportJson() {
  if (!state.result) return;
  const copy = { ...state.result };
  delete copy.clientMs;
  const text = JSON.stringify(copy, null, 2);
  const blob = new Blob([text], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `image-diff-${state.result.requestId || "result"}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("已导出本次结果 JSON");
}

function resetAll() {
  clearImage("before");
  clearImage("after");
  state.result = null;
  $("#results").hidden = true;
  stopLoading();
  toast("已清空");
}

/* ---------------------------------------------------------------- init */
async function init() {
  bindDropzone("before");
  bindDropzone("after");
  bindPaste();

  $("#run").addEventListener("click", runCompare);
  $("#reset").addEventListener("click", resetAll);
  $("#export").addEventListener("click", exportJson);
  $("#lightbox").addEventListener("click", (event) => {
    if (event.target.id === "lightbox" || event.target.closest(".lightbox-close")) closeLightbox();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeLightbox();
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      if (!state.busy && state.before && state.after) runCompare();
    }
  });

  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  try {
    await loadMeta();
    pollStatus();
  } catch (err) {
    updateEngineStatus(null);
    toast(`无法连接后端：${err.message}`, true);
  }

  try {
    await loadSamples();
    await maybeDemo();
  } catch {
    /* 示例图可选，失败不阻塞主流程 */
  }

  syncRunButton();
}

init();
