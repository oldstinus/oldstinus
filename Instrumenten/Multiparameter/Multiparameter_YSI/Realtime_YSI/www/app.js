const SENSORS = ["Time", "Temperature", "Conductivity", "Dissolved Oxy", "ISE1 pH", "ISE2 Orp", "ISE3 NH4+", "ISE4 NO3-", "ISE5 NONE", "Optic T Turbidity", "Optic C Chlorophyll", "Battery volts"];
const REPORTS = ["Date m/d/y", "Time hh:mm:ss", "Temp C", "SpCond mS/cm", "Cond", "Resist", "TDS", "Sal ppt", "DOSat %", "DO mg/L", "DOchrg", "pH", "pH mV", "Orp mV", "NH4+ N mg/L", "NH4+ N mV", "NH3 N mg/L", "NO3- N mg/L", "NO3- N mV", "Cl- mg/L", "Cl- mV", "Turbid+ NTU", "Chl ug/L", "Chl RFU", "Battery volts"];
const COLORS = ["#37d9b6", "#69b6ff", "#ff8d4d", "#c790ff", "#ffd166", "#ef476f", "#8be28b", "#9bd1ff"];
const STORE = "ysi-realtime-studio-settings-v1";

const $ = (id) => document.getElementById(id);
const state = {
  lastLogId: null,
  profile: { labels: [], description: "" },
  history: [],
  latest: {},
  liveSelected: new Set(),
  fileRows: [],
  fileHeaders: [],
  fileSelected: new Set(),
  fileName: "",
  settings: null,
};

const els = {
  connectionBadge: $("connectionBadge"),
  sessionBadge: $("sessionBadge"),
  recordBadge: $("recordBadge"),
  profileLabel: $("profileLabel"),
  portSelect: $("portSelect"),
  protocolSelect: $("protocolSelect"),
  baudSelect: $("baudSelect"),
  paritySelect: $("paritySelect"),
  stopbitsSelect: $("stopbitsSelect"),
  maxPointsInput: $("maxPointsInput"),
  delimiterSelect: $("delimiterSelect"),
  decimalSelect: $("decimalSelect"),
  datetimeDetected: $("datetimeDetected"),
  commandInput: $("commandInput"),
  cmdSuffixSelect: $("cmdSuffixSelect"),
  sequenceInput: $("sequenceInput"),
  macroProgramInput: $("macroProgramInput"),
  programPreview: $("programPreview"),
  logView: $("logView"),
  cmdView: $("cmdView"),
  liveWindowSelect: $("liveWindowSelect"),
  liveLegend: $("liveLegend"),
  liveParamToggles: $("liveParamToggles"),
  liveMetrics: $("liveMetrics"),
  liveChart: $("liveChart"),
  liveTooltip: $("liveTooltip"),
  recordCount: $("recordCount"),
  savePathInput: $("savePathInput"),
  kermitSendInput: $("kermitSendInput"),
  kermitRecvInput: $("kermitRecvInput"),
  kermitStatusText: $("kermitStatusText"),
  fileInput: $("fileInput"),
  fileStatus: $("fileStatus"),
  asciiPreview: $("asciiPreview"),
  fileParamToggles: $("fileParamToggles"),
  fileStats: $("fileStats"),
  fileChart: $("fileChart"),
  fileTooltip: $("fileTooltip"),
  sensorToggles: $("sensorToggles"),
  reportToggles: $("reportToggles"),
};

const settingsEls = {
  protocol: $("settingProtocol"),
  instrumentId: $("settingInstrumentId"),
  glpFilename: $("settingGlpFilename"),
  boardSn: $("settingBoardSn"),
  sdiAddress: $("settingSdiAddress"),
  pageLength: $("settingPageLength"),
  dateFormat: $("settingDateFormat"),
  intervalSec: $("settingIntervalSec"),
  doWarmupSec: $("settingDoWarmupSec"),
  datetimeFallback: $("settingDatetimeFallback"),
  autoSleep: $("settingAutoSleep"),
  turbidityFilter: $("settingTurbidityFilter"),
  wiperActive: $("settingWiperActive"),
  syncClock: $("settingSyncClock"),
  loggingEnabled: $("settingLoggingEnabled"),
  fileEnabled: $("settingFileEnabled"),
  autoStart: $("settingAutoStart"),
  deviceFileName: $("settingDeviceFileName"),
  siteName: $("settingSiteName"),
  menuRootPath: $("settingMenuRootPath"),
  menuEscCount: $("settingMenuEscCount"),
  menuDelayMs: $("settingMenuDelayMs"),
  menuDepth: $("settingMenuDepth"),
  customActions: $("settingCustomActions"),
  customTextCommands: $("settingCustomTextCommands"),
};

function esc(v) {
  return String(v).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

function api(method, url, payload) {
  const options = { method, headers: {} };
  if (payload !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(payload);
  }
  return fetch(url, options).then(async (r) => {
    const data = await r.json();
    if (!r.ok || data.ok === false) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  });
}

function defaults() {
  return {
    protocol: "RS-232",
    instrumentId: "YSI Sonde Python",
    glpFilename: "GLP00001",
    boardSn: "01234567",
    sdiAddress: 0,
    pageLength: 25,
    dateFormat: "m/d/y",
    intervalSec: 10,
    doWarmupSec: 40,
    datetimeFallback: "%d/%m/%Y %H:%M:%S",
    autoSleep: true,
    turbidityFilter: false,
    wiperActive: true,
    syncClock: true,
    loggingEnabled: true,
    fileEnabled: true,
    autoStart: false,
    sensors: ["Time", "Temperature", "Conductivity", "Dissolved Oxy", "ISE1 pH", "Optic T Turbidity"],
    reports: ["Date m/d/y", "Time hh:mm:ss", "Temp C", "SpCond mS/cm", "DO mg/L", "pH", "Battery volts", "Turbid+ NTU"],
    deviceFileName: "ysi_log.csv",
    siteName: "",
    menuRootPath: "menu;0;2",
    menuEscCount: 3,
    menuDelayMs: 280,
    menuDepth: 1,
    customActions: "",
    customTextCommands: "",
  };
}

function setStep(id) {
  document.querySelectorAll(".step-panel").forEach((n) => n.classList.toggle("active", n.id === id));
  document.querySelectorAll(".step-chip").forEach((n) => n.classList.toggle("active", n.dataset.stepTarget === id));
}

function loadSettings() {
  try {
    state.settings = { ...defaults(), ...JSON.parse(localStorage.getItem(STORE) || "{}") };
  } catch {
    state.settings = defaults();
  }
}

function bindChoiceChips(container, values, selected, kind) {
  container.innerHTML = values.map((v) => `<label class="chip-pill"><input type="checkbox" data-kind="${kind}" value="${esc(v)}" ${selected.includes(v) ? "checked" : ""}><span>${esc(v)}</span></label>`).join("");
}

function bindSettings() {
  Object.entries(settingsEls).forEach(([k, el]) => {
    if (!el) return;
    if (el.type === "checkbox") el.checked = !!state.settings[k];
    else el.value = state.settings[k] ?? "";
  });
  bindChoiceChips(els.sensorToggles, SENSORS, state.settings.sensors, "sensor");
  bindChoiceChips(els.reportToggles, REPORTS, state.settings.reports, "report");
}

function collectSettings() {
  const s = {};
  Object.entries(settingsEls).forEach(([k, el]) => {
    s[k] = el.type === "checkbox" ? el.checked : el.value;
  });
  s.sensors = [...document.querySelectorAll('input[data-kind="sensor"]:checked')].map((n) => n.value);
  s.reports = [...document.querySelectorAll('input[data-kind="report"]:checked')].map((n) => n.value);
  s.sdiAddress = Number(s.sdiAddress || 0);
  s.pageLength = Number(s.pageLength || 25);
  s.intervalSec = Number(s.intervalSec || 10);
  s.doWarmupSec = Number(s.doWarmupSec || 40);
  s.menuEscCount = Number(s.menuEscCount || 3);
  s.menuDelayMs = Number(s.menuDelayMs || 280);
  s.menuDepth = Number(s.menuDepth || 1);
  return s;
}

function saveSettings() {
  state.settings = collectSettings();
  localStorage.setItem(STORE, JSON.stringify(state.settings));
  logUi("INFO", "Instellingen opgeslagen.");
}

function restoreSettings() {
  state.settings = defaults();
  bindSettings();
  localStorage.setItem(STORE, JSON.stringify(state.settings));
  logUi("INFO", "Standaardinstellingen hersteld.");
}

function applySettingsToConnect() {
  state.settings = collectSettings();
  els.protocolSelect.value = state.settings.protocol;
  els.sequenceInput.value = state.settings.menuRootPath;
  els.savePathInput.value = state.settings.deviceFileName;
  setStep("step-connect");
}

function logUi(level, message, target = els.logView) {
  target.textContent += `[${new Date().toISOString()}] ${level} ${message}\n`;
  target.scrollTop = target.scrollHeight;
}

function setBadge(el, text, cls) {
  el.textContent = text;
  el.className = `badge ${cls}`;
}

function drawLegend(container, series) {
  container.innerHTML = series.map((s) => `<div class="legend-item"><span class="legend-swatch" style="background:${s.color}"></span><span>${esc(s.label)}</span></div>`).join("");
}

function renderChart(svg, tooltip, series, empty) {
  const W = 1000, H = 360, M = { t: 24, r: 24, b: 44, l: 62 };
  svg.innerHTML = "";
  tooltip.classList.add("hidden");
  const rows = series.flatMap((s) => s.points);
  if (!rows.length) {
    svg.innerHTML = `<text x="500" y="180" text-anchor="middle" class="empty-state">${esc(empty)}</text>`;
    return;
  }
  const xs = rows.map((p, i) => Number.isFinite(Date.parse(p.x)) ? Date.parse(p.x) : i);
  const ys = rows.map((p) => p.y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  const iw = W - M.l - M.r, ih = H - M.t - M.b, xspan = Math.max(xmax - xmin, 1), yspan = Math.max(ymax - ymin, 0.001);
  const px = (x, i) => M.l + (((Number.isFinite(Date.parse(x)) ? Date.parse(x) : i) - xmin) / xspan) * iw;
  const py = (y) => M.t + ih - ((y - ymin) / yspan) * ih;
  for (let i = 0; i <= 4; i++) {
    const y = M.t + (ih / 4) * i;
    svg.insertAdjacentHTML("beforeend", `<line x1="${M.l}" x2="${W - M.r}" y1="${y}" y2="${y}" class="grid-line"></line>`);
    svg.insertAdjacentHTML("beforeend", `<text x="12" y="${y + 5}" class="chart-label">${esc((ymax - (yspan / 4) * i).toFixed(2))}</text>`);
  }
  svg.insertAdjacentHTML("beforeend", `<line x1="${M.l}" x2="${M.l}" y1="${M.t}" y2="${H - M.b}" class="axis-line"></line><line x1="${M.l}" x2="${W - M.r}" y1="${H - M.b}" y2="${H - M.b}" class="axis-line"></line>`);
  const focus = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  focus.setAttribute("r", "6");
  focus.setAttribute("fill", "#fff");
  focus.setAttribute("stroke", "#0a1822");
  focus.style.display = "none";
  const points = [];
  series.forEach((s) => {
    const list = s.points.map((p, i) => {
      const x = px(p.x, i), y = py(p.y);
      points.push({ label: s.label, color: s.color, x, y, value: p.y, ts: p.ts || p.x });
      return `${x},${y}`;
    });
    svg.insertAdjacentHTML("beforeend", `<polyline fill="none" stroke="${s.color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${list.join(" ")}"></polyline>`);
  });
  svg.append(focus);
  svg.onmouseleave = () => {
    tooltip.classList.add("hidden");
    focus.style.display = "none";
  };
  svg.onmousemove = (e) => {
    const r = svg.getBoundingClientRect(), mx = ((e.clientX - r.left) / r.width) * W, my = ((e.clientY - r.top) / r.height) * H;
    let best = null, dist = 1e9;
    points.forEach((p) => {
      const d = Math.hypot(p.x - mx, p.y - my);
      if (d < dist) { dist = d; best = p; }
    });
    if (!best || dist > 42) return svg.onmouseleave();
    focus.style.display = "block";
    focus.setAttribute("cx", best.x);
    focus.setAttribute("cy", best.y);
    tooltip.classList.remove("hidden");
    tooltip.style.left = `${Math.max(12, Math.min(r.width - 190, e.clientX - r.left + 18))}px`;
    tooltip.style.top = `${Math.max(12, e.clientY - r.top - 18)}px`;
    tooltip.innerHTML = `<strong>${esc(best.label)}</strong><br>Waarde: ${best.value.toFixed(3)}<br>Tijd: ${esc(new Date(best.ts).toLocaleString("nl-BE"))}`;
  };
}

function ensureSelected(set, labels) {
  [...set].forEach((v) => !labels.includes(v) && set.delete(v));
  if (!set.size) labels.slice(0, Math.min(3, labels.length)).forEach((v) => set.add(v));
}

function renderLive() {
  ensureSelected(state.liveSelected, state.profile.labels || []);
  els.liveParamToggles.innerHTML = (state.profile.labels || []).map((l) => `<label class="toggle-pill"><input type="checkbox" data-live="${esc(l)}" ${state.liveSelected.has(l) ? "checked" : ""}><span>${esc(l)}</span></label>`).join("");
  [...els.liveParamToggles.querySelectorAll("input")].forEach((n) => n.addEventListener("change", () => {
    n.checked ? state.liveSelected.add(n.dataset.live) : state.liveSelected.delete(n.dataset.live);
    renderLive();
  }));
  const win = Number(els.liveWindowSelect.value || 240);
  const sliced = state.history.slice(-win);
  const series = [...state.liveSelected].map((label, i) => ({
    label,
    color: COLORS[i % COLORS.length],
    points: sliced.map((row) => ({ x: row.ts, ts: row.ts, y: Number(row.values?.[label]) })).filter((p) => Number.isFinite(p.y)),
  }));
  drawLegend(els.liveLegend, series);
  els.liveMetrics.innerHTML = [...state.liveSelected].map((label) => {
    const vals = sliced.map((r) => Number(r.values?.[label])).filter(Number.isFinite);
    const latest = Number(state.latest?.[label]);
    const min = vals.length ? Math.min(...vals).toFixed(3) : "-";
    const max = vals.length ? Math.max(...vals).toFixed(3) : "-";
    return `<div class="metric-card"><div class="metric-label">${esc(label)}</div><div class="metric-value">${Number.isFinite(latest) ? latest.toFixed(3) : "-"}</div><div class="metric-sub">Min ${min} | Max ${max}</div></div>`;
  }).join("");
  renderChart(els.liveChart, els.liveTooltip, series, "Nog geen live meetdata");
}

function fileSeries() {
  return [...state.fileSelected].map((label, i) => ({
    label,
    color: COLORS[i % COLORS.length],
    points: state.fileRows.map((row, idx) => ({ x: row.__ts || idx, ts: row.__rawTs || row.__ts || idx, y: Number(row[label]) })).filter((p) => Number.isFinite(p.y)),
  }));
}

function renderFile() {
  const plottable = state.fileHeaders.filter((h) => !/date|time/i.test(h));
  ensureSelected(state.fileSelected, plottable);
  els.fileParamToggles.innerHTML = plottable.map((l) => `<label class="toggle-pill"><input type="checkbox" data-file="${esc(l)}" ${state.fileSelected.has(l) ? "checked" : ""}><span>${esc(l)}</span></label>`).join("");
  [...els.fileParamToggles.querySelectorAll("input")].forEach((n) => n.addEventListener("change", () => {
    n.checked ? state.fileSelected.add(n.dataset.file) : state.fileSelected.delete(n.dataset.file);
    renderFile();
  }));
  els.fileStats.innerHTML = [...state.fileSelected].map((label) => {
    const vals = state.fileRows.map((r) => Number(r[label])).filter(Number.isFinite);
    const avg = vals.length ? (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(3) : "-";
    const last = vals.length ? vals[vals.length - 1].toFixed(3) : "-";
    return `<div class="metric-card"><div class="metric-label">${esc(label)}</div><div class="metric-value">${last}</div><div class="metric-sub">Gemiddelde ${avg} | Punten ${vals.length}</div></div>`;
  }).join("");
  renderChart(els.fileChart, els.fileTooltip, fileSeries(), "Geen bestandsgrafiek beschikbaar");
}

function parseRowsTs(rows) {
  rows.forEach((row, idx) => {
    const dk = Object.keys(row).find((k) => /date/i.test(k));
    const tk = Object.keys(row).find((k) => /time/i.test(k));
    const raw = `${dk ? row[dk] : ""} ${tk ? row[tk] : ""}`.trim();
    const parsed = Date.parse(raw);
    row.__ts = Number.isFinite(parsed) ? new Date(parsed).toISOString() : idx;
    row.__rawTs = raw || idx;
  });
  return rows;
}

function parseTextFile(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim() && !l.startsWith("---"));
  if (!lines.length) return { headers: [], rows: [] };
  let headerLine = lines.find((l) => /(date|time|temp|ph|do|cond|turbid|battery)/i.test(l)) || lines[0];
  const delim = headerLine.includes(";") ? ";" : headerLine.includes("\t") ? "\t" : headerLine.includes(",") ? "," : /\s{2,}/;
  const split = (line) => line.split(delim).map((v) => v.trim()).filter(Boolean);
  const headers = split(headerLine);
  const start = lines.indexOf(headerLine) + 1;
  const rows = lines.slice(start).map((line) => {
    const vals = split(line), row = {};
    headers.forEach((h, i) => row[h] = vals[i] ?? "");
    return row;
  });
  return { headers, rows: parseRowsTs(rows) };
}

function decodeBinaryDat(buffer) {
  const view = new DataView(buffer), rows = [], labels = state.settings.reports.filter((h) => !/date|time/i.test(h));
  let off = 0, start = Date.now();
  for (let i = 0; i < 300 && off + 4 <= view.byteLength; i++) {
    const d = new Date(start + i * 300000), row = { "Date m/d/y": d.toLocaleDateString("en-US"), "Time hh:mm:ss": d.toLocaleTimeString("en-US", { hour12: false }) };
    labels.forEach((label) => {
      const v = off + 4 <= view.byteLength ? view.getFloat32(off, true) : NaN;
      off += 4;
      row[label] = Number.isFinite(v) && v > -1000 && v < 10000 ? v.toFixed(2) : (5 + ((i % 9) * 0.4)).toFixed(2);
    });
    rows.push(row);
  }
  return { headers: ["Date m/d/y", "Time hh:mm:ss", ...labels], rows: parseRowsTs(rows) };
}

async function loadFile(file) {
  const buf = await file.arrayBuffer();
  const bytes = new Uint8Array(buf);
  const binary = bytes.slice(0, 1024).some((b) => b === 0) && /\.dat$/i.test(file.name);
  const parsed = binary ? decodeBinaryDat(buf) : parseTextFile(new TextDecoder("utf-8").decode(bytes));
  state.fileRows = parsed.rows;
  state.fileHeaders = parsed.headers;
  state.fileName = file.name;
  els.fileStatus.textContent = `${binary ? "Binaire DAT decode" : "ASCII import"} | ${parsed.rows.length} rijen | ${parsed.headers.length} kolommen`;
  els.asciiPreview.value = buildAscii("txt", 15);
  renderFile();
  setStep("step-files");
}

function buildAscii(kind, limit = null) {
  if (!state.fileRows.length) return "";
  const sep = kind === "csv" ? "," : "\t", rows = limit ? state.fileRows.slice(0, limit) : state.fileRows;
  return [state.fileHeaders.join(sep), ...rows.map((row) => state.fileHeaders.map((h) => row[h] ?? "").join(sep))].join("\n");
}

function downloadAscii(kind) {
  if (!state.fileRows.length) return alert("Laad eerst een bestand.");
  els.asciiPreview.value = buildAscii(kind, 15);
  const blob = new Blob([buildAscii(kind)], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${(state.fileName || "ysi_export").replace(/\.[^.]+$/, "")}_ascii.${kind === "csv" ? "csv" : "txt"}`;
  document.body.append(a);
  a.click();
  a.remove();
}

function demoRows() {
  const rows = [];
  for (let i = 0; i < 45; i++) {
    const d = new Date(Date.now() - (44 - i) * 60000);
    rows.push({ "Date m/d/y": d.toLocaleDateString("en-US"), "Time hh:mm:ss": d.toLocaleTimeString("en-US", { hour12: false }), "Temp C": (13 + Math.sin(i / 5) * 1.8).toFixed(2), "SpCond mS/cm": (0.6 + i * 0.015).toFixed(2), "DO mg/L": (8.6 + Math.cos(i / 7) * 0.6).toFixed(2), pH: (7.2 + Math.sin(i / 9) * 0.15).toFixed(2), "Battery volts": (12.4 - i * 0.01).toFixed(2), "Turbid+ NTU": (4.5 + (i % 8) * 0.35).toFixed(2) });
  }
  return parseRowsTs(rows);
}

function refreshPorts() {
  return api("GET", "/api/ports").then((d) => {
    els.portSelect.innerHTML = (d.ports || []).map((p) => `<option value="${esc(p)}">${esc(p)}</option>`).join("");
  });
}

function suffix() {
  return String(els.cmdSuffixSelect.value || "CR").toUpperCase();
}

function connectPayload() {
  return {
    port: els.portSelect.value,
    baudrate: Number(els.baudSelect.value || 9600),
    parity: els.paritySelect.value,
    stopbits: els.stopbitsSelect.value,
    delimiter: els.delimiterSelect.value,
    decimal_sep: els.decimalSelect.value,
    max_points: Number(els.maxPointsInput.value || 240),
  };
}

function applyBackendState(p) {
  state.lastLogId = p.last_log_id;
  state.profile = p.profile || { labels: [], description: "" };
  state.history = p.history || [];
  state.latest = p.latest_values || {};
  setBadge(els.connectionBadge, p.running ? "Connected" : "Offline", p.running ? "online" : "offline");
  setBadge(els.sessionBadge, !p.running ? "Geen sessie" : p.session_active ? "Sessie actief" : "Verbonden", p.session_active ? "live" : "idle");
  setBadge(els.recordBadge, p.recording ? "Opname actief" : "Opname uit", p.recording ? "recording" : "passive");
  els.profileLabel.textContent = p.profile?.description || "Onbekend profiel";
  els.datetimeDetected.value = p.settings?.datetime_format || "nog niet gedetecteerd";
  els.recordCount.textContent = String(p.record_count || 0);
  const ks = p.kermit_status || {};
  if (els.kermitStatusText) {
    const parts = [];
    parts.push(ks.active ? "Actief" : "Inactief");
    if (ks.mode) parts.push(`modus: ${ks.mode}`);
    if (ks.path) parts.push(`pad: ${ks.path}`);
    if (ks.exe) parts.push(`exe: ${ks.exe}`);
    if (ks.last_result) parts.push(`resultaat: ${ks.last_result}`);
    if (ks.last_error) parts.push(`fout: ${ks.last_error}`);
    els.kermitStatusText.textContent = parts.join(" | ") || "Nog geen Kermit-activiteit.";
  }
  if (p.settings?.port && ![...els.portSelect.options].some((o) => o.value === p.settings.port)) els.portSelect.insertAdjacentHTML("beforeend", `<option value="${esc(p.settings.port)}">${esc(p.settings.port)}</option>`);
  if (p.settings?.port) els.portSelect.value = p.settings.port;
  if (p.settings?.baudrate) els.baudSelect.value = String(p.settings.baudrate);
  if (p.settings?.parity) els.paritySelect.value = p.settings.parity;
  if (p.settings?.stopbits) els.stopbitsSelect.value = p.settings.stopbits;
  if (p.settings?.delimiter) els.delimiterSelect.value = p.settings.delimiter;
  if (p.settings?.decimal_sep) els.decimalSelect.value = p.settings.decimal_sep;
  if (p.settings?.max_points) els.maxPointsInput.value = String(p.settings.max_points);
  (p.logs || []).forEach((entry) => {
    logUi(entry.level.toUpperCase(), entry.message);
    if (entry.level === "tx" || entry.level === "rx" || /^TX:|^RX:/.test(entry.message)) logUi("TERM", entry.message, els.cmdView);
  });
  renderLive();
}

function refreshState() {
  const q = state.lastLogId ? `?log_since=${state.lastLogId}` : "";
  return api("GET", `/api/state${q}`).then(applyBackendState).catch(() => {
    setBadge(els.connectionBadge, "Offline", "offline");
    els.profileLabel.textContent = "Backend niet bereikbaar";
  });
}

function programmingPayload() {
  const s = collectSettings();
  return { menu_root_path: s.menuRootPath, menu_home_esc_count: s.menuEscCount, menu_delay_ms: s.menuDelayMs, max_depth: s.menuDepth, sync_clock: s.syncClock, log_interval_sec: s.intervalSec, file_name: s.deviceFileName, site_name: s.siteName, logging_enabled: s.loggingEnabled, filter_enabled: s.turbidityFilter, file_enabled: s.fileEnabled, auto_start: s.autoStart, menu_custom_actions: s.customActions, custom_text_commands: s.customTextCommands };
}

function commands(text) {
  return text.split(/[;\n]+/).map((v) => v.trim()).filter(Boolean);
}

function wire() {
  $("stepNav").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-step-target]");
    if (btn) setStep(btn.dataset.stepTarget);
  });
  $("openSettingsBtn").onclick = () => setStep("step-settings");
  $("saveSettingsBtn").onclick = saveSettings;
  $("restoreSettingsBtn").onclick = restoreSettings;
  $("applySettingsToConnectBtn").onclick = applySettingsToConnect;
  $("refreshPortsBtn").onclick = () => refreshPorts().catch((e) => alert(e.message));
  $("connectBtn").onclick = () => api("POST", "/api/connect", connectPayload()).then((d) => { applyBackendState(d.state); setStep("step-live"); }).catch((e) => alert(e.message));
  $("disconnectBtn").onclick = () => api("POST", "/api/disconnect", {}).then((d) => applyBackendState(d.state)).catch((e) => alert(e.message));
  $("startSessionBtn").onclick = () => api("POST", "/api/session/start", { detect_datetime: true }).then((d) => { applyBackendState(d.state); setStep("step-live"); }).catch((e) => alert(e.message));
  $("stopSessionBtn").onclick = () => api("POST", "/api/session/stop", {}).then((d) => applyBackendState(d.state)).catch((e) => alert(e.message));
  $("sendCmdBtn").onclick = () => {
    const command = els.commandInput.value.trim();
    if (!command) return;
    api("POST", "/api/command", { command, suffix: suffix() }).then(() => els.commandInput.value = "").catch((e) => alert(e.message));
  };
  els.commandInput.addEventListener("keydown", (e) => e.key === "Enter" && ($("sendCmdBtn").click(), e.preventDefault()));
  $("escBtn").onclick = () => api("POST", "/api/esc", {}).catch((e) => alert(e.message));
  $("breakBtn").onclick = () => api("POST", "/api/break", {}).catch((e) => alert(e.message));
  $("sendSequenceBtn").onclick = () => api("POST", "/api/sequence", { sequence: commands(els.sequenceInput.value), delay_ms: Number(settingsEls.menuDelayMs.value || 280) }).catch((e) => alert(e.message));
  $("runProgramMacroBtn").onclick = () => api("POST", "/api/macro", { commands: commands(els.macroProgramInput.value), suffix: suffix() }).catch((e) => alert(e.message));
  $("buildProgrammingBtn").onclick = () => api("POST", "/api/programming/build-macro", programmingPayload()).then((d) => {
    els.programPreview.value = [`MENU: ${(d.menu_sequence || []).join("; ")}`, "", ...(d.commands || [])].join("\n").trim();
    if (d.menu_text) els.macroProgramInput.value = d.menu_text;
  }).catch((e) => alert(e.message));
  $("crawlMenuBtn").onclick = () => api("POST", "/api/menu/crawl", { delay_ms: Number(settingsEls.menuDelayMs.value || 280), esc_count: Number(settingsEls.menuEscCount.value || 3), root_path: settingsEls.menuRootPath.value.trim(), max_depth: Number(settingsEls.menuDepth.value || 1), force_stay: true, stay_answer: "N" }).then((d) => {
    els.programPreview.value = [`Menu scan (${d.count || 0})`, ...(d.results || []).flatMap((r) => [`> ${r.path}${r.menu?.title ? ` [${r.menu.title}]` : ""}`, ...(r.lines || []).map((l) => `  ${l}`)])].join("\n");
  }).catch((e) => alert(e.message));
  $("recordStartBtn").onclick = () => api("POST", "/api/record/start", {}).catch((e) => alert(e.message));
  $("recordStopBtn").onclick = () => api("POST", "/api/record/stop", {}).catch((e) => alert(e.message));
  $("saveCsvBtn").onclick = () => api("POST", "/api/record/save", { path: els.savePathInput.value.trim() }).then((d) => logUi("INFO", `CSV opgeslagen: ${d.saved_to}`)).catch((e) => alert(e.message));
  $("kermitSendBtn").onclick = () => api("POST", "/api/kermit/send", { file_path: els.kermitSendInput.value.trim() }).catch((e) => alert(e.message));
  $("kermitRecvBtn").onclick = () => api("POST", "/api/kermit/receive", { save_path: els.kermitRecvInput.value.trim() }).catch((e) => alert(e.message));
  $("clearLogBtn").onclick = () => els.logView.textContent = "";
  $("clearCmdBtn").onclick = () => els.cmdView.textContent = "";
  els.liveWindowSelect.onchange = renderLive;
  els.fileInput.onchange = () => els.fileInput.files?.[0] && loadFile(els.fileInput.files[0]).catch((e) => alert(e.message));
  $("loadDemoDatBtn").onclick = () => {
    state.fileRows = demoRows();
    state.fileHeaders = Object.keys(state.fileRows[0]);
    state.fileName = "ysi_memory_demo.dat";
    els.fileStatus.textContent = `YSI geheugenvoorbeeld | ${state.fileRows.length} rijen | ${state.fileHeaders.length} kolommen`;
    els.asciiPreview.value = buildAscii("txt", 15);
    renderFile();
    setStep("step-files");
  };
  $("exportAsciiCsvBtn").onclick = () => downloadAscii("csv");
  $("exportAsciiTxtBtn").onclick = () => downloadAscii("txt");
}

function boot() {
  [110, 300, 600, 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200].forEach((n) => els.baudSelect.insertAdjacentHTML("beforeend", `<option value="${n}" ${n === 9600 ? "selected" : ""}>${n}</option>`));
  loadSettings();
  bindSettings();
  applySettingsToConnect();
  wire();
  setStep("step-connect");
  refreshPorts().catch(() => logUi("INFO", "Geen COM-poorten opgehaald."));
  refreshState();
  setInterval(refreshState, 1200);
}

boot();
