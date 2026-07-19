const API_BASE = window.location.origin.includes("file://") ? "https://revive-ai-backend.onrender.com" : "";

function req(id) {
  const el = document.getElementById(id);
  if (!el) {
    console.error(`[EcoShaadi] Missing required element: #${id}. Check that index.html matches the original markup (nothing commented out).`);
  }
  return el;
}

/* ---------------- Mobile nav ---------------- */

const navToggle = req("navToggle");
const navLinks = req("navLinks");

if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => {
    const isOpen = navLinks.classList.toggle("open");
    navToggle.setAttribute("aria-expanded", String(isOpen));
  });
  navLinks.querySelectorAll("a").forEach((a) =>
    a.addEventListener("click", () => {
      navLinks.classList.remove("open");
      navToggle.setAttribute("aria-expanded", "false");
    })
  );
}

/* ---------------- Ambient floating flower petals ---------------- */

const PETAL_SYMBOLS = ["#fl-marigold", "#fl-rose", "#fl-jasmine", "#fl-leaf"];
const PETAL_COLORS = ["var(--magenta)", "var(--marigold)", "var(--gold)", "var(--leaf)"];

function spawnPetals(containerId, count) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reducedMotion) return;

  const isMobile = window.innerWidth < 640;
  const actualCount = isMobile ? Math.round(count * 0.5) : count;

  for (let i = 0; i < actualCount; i++) {
    const svgNS = "http://www.w3.org/2000/svg";
    const el = document.createElementNS(svgNS, "svg");
    const use = document.createElementNS(svgNS, "use");
    use.setAttributeNS("http://www.w3.org/1999/xlink", "href", PETAL_SYMBOLS[i % PETAL_SYMBOLS.length]);
    use.setAttribute("href", PETAL_SYMBOLS[i % PETAL_SYMBOLS.length]);
    el.appendChild(use);
    el.setAttribute("viewBox", "0 0 100 100");
    el.classList.add("petal");

    const size = 16 + Math.random() * 26;
    const driftDuration = 9 + Math.random() * 8;
    const appearDelay = Math.random() * 5;
    const restOpacity = (0.18 + Math.random() * 0.28).toFixed(2);

    el.style.left = `${Math.random() * 100}%`;
    el.style.top = `${Math.random() * 100}%`;
    el.style.width = `${size}px`;
    el.style.height = `${size}px`;
    el.style.color = PETAL_COLORS[i % PETAL_COLORS.length];
    el.style.setProperty("--delay", `${appearDelay}s`);
    el.style.setProperty("--rest-opacity", restOpacity);
    el.style.animationDuration = `0.9s, ${driftDuration}s`;

    container.appendChild(el);
  }
}

spawnPetals("petalField", 22);

function req2(id) { return document.getElementById(id); }

const dropzone = req("dropzone");
const fileInput = req("file-input");
const previewGrid = req("preview-grid");
const analyzeBtn = req("analyze-btn");
const formHint = req("form-hint");
const uploadForm = req("upload-form");

const processingOverlay = req("processing");
const processingStageEl = req("processing-stage");
const processingFill = req("processing-fill");

const resultsSection = req("results");
const garlandBeads = document.querySelectorAll(".bead");

const processingSlowEl = req("processing-slow");
const cancelBtn = req("cancel-btn");

console.log("[EcoShaadi] app.js loaded. API_BASE =", API_BASE || "(same origin)");

let selectedFiles = [];
let lastResult = null;
let currentController = null;
let slowTimer = null;

/* ---------------- File selection ---------------- */

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);
dropzone.addEventListener("drop", (e) => {
  addFiles(e.dataTransfer.files);
});

fileInput.addEventListener("change", () => addFiles(fileInput.files));

function addFiles(fileList) {
  for (const f of fileList) {
    if (f.type.startsWith("image/") || f.type.startsWith("video/")) {
      selectedFiles.push(f);
    }
  }
  renderPreviews();
}

function renderPreviews() {
  previewGrid.innerHTML = "";
  if (selectedFiles.length === 0) {
    previewGrid.classList.add('is-hidden');
    analyzeBtn.disabled = true;
    formHint.textContent = "Add at least one photo or video to begin.";
    return;
  }

  previewGrid.classList.remove('is-hidden');
  analyzeBtn.disabled = false;
  formHint.textContent = `${selectedFiles.length} file${selectedFiles.length > 1 ? "s" : ""} ready for the AI agents.`;

  selectedFiles.forEach((file, index) => {
    const wrapper = document.createElement("div");
    wrapper.className = "preview-item";

    if (file.type.startsWith("image/")) {
      const img = document.createElement("img");
      img.src = URL.createObjectURL(file);
      wrapper.appendChild(img);
    } else {
      const chip = document.createElement("div");
      chip.className = "video-chip";
      chip.textContent = "🎬";
      wrapper.appendChild(chip);
    }

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "preview-remove";
    removeBtn.setAttribute("aria-label", `Remove ${file.name}`);
    removeBtn.textContent = "×";
    removeBtn.addEventListener("click", () => {
      selectedFiles.splice(index, 1);
      renderPreviews();
    });
    wrapper.appendChild(removeBtn);

    previewGrid.appendChild(wrapper);
  });
}

/* ---------------- Submit / processing animation ---------------- */

const STAGES = [
  { key: "vision", label: "Vision Agent is looking at your photos…" },
  { key: "classification", label: "Classification Agent is sorting each item…" },
  { key: "recommendation", label: "Recommendation Agent is finding the best receivers…" },
  { key: "logistics", label: "Logistics Agent is planning the pickup…" },
  { key: "sustainability", label: "Sustainability Agent is calculating impact…" },
  { key: "report", label: "Report Agent is compiling your results…" },
];

let stageTimer = null;

function startStageAnimation() {
  let i = 0;
  processingOverlay.classList.remove('is-hidden');
  processingSlowEl.classList.add('is-hidden');
  garlandBeads.forEach((b) => b.classList.remove("active", "done"));

  slowTimer = setTimeout(() => {
    processingSlowEl.classList.remove('is-hidden');
  }, 15000);

  const setStage = (index) => {
    STAGES.forEach((s, idx) => {
      const bead = document.querySelector(`.bead[data-stage="${s.key}"]`);
      if (!bead) return;
      bead.classList.toggle("active", idx === index);
      bead.classList.toggle("done", idx < index);
    });
    processingStageEl.textContent = STAGES[index].label;
    processingFill.style.width = `${((index + 1) / STAGES.length) * 100}%`;
  };

  setStage(0);
  stageTimer = setInterval(() => {
    i += 1;
    if (i >= STAGES.length - 1) {
      setStage(STAGES.length - 1);
      clearInterval(stageTimer);
      return;
    }
    setStage(i);
  }, 1400);
}

function finishStageAnimation() {
  if (stageTimer) clearInterval(stageTimer);
  if (slowTimer) clearTimeout(slowTimer);
  garlandBeads.forEach((b) => {
    b.classList.remove("active");
    b.classList.add("done");
  });
  processingFill.style.width = "100%";
  setTimeout(() => {
    processingOverlay.classList.add('is-hidden');
  }, 350);
}

// Safari (and some other browsers) can restore this page from the
// back/forward cache with old JS state intact. Force a clean slate so a
// previous file selection can never silently resubmit.
window.addEventListener("pageshow", (event) => {
  if (event.persisted) {
    if (currentController) currentController.abort();
    if (slowTimer) clearTimeout(slowTimer);
    if (stageTimer) clearInterval(stageTimer);
    selectedFiles = [];
    fileInput.value = "";
    renderPreviews();
    resultsSection.classList.add('is-hidden');
    processingOverlay.classList.add('is-hidden');
  }
});

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  console.log("[EcoShaadi] submit fired. selectedFiles:", selectedFiles.length);

  if (selectedFiles.length === 0) {
    console.warn("[EcoShaadi] No files selected — aborting submit.");
    return;
  }

  const formData = new FormData();
  selectedFiles.forEach((f) => formData.append("files", f));

  currentController = new AbortController();
  startStageAnimation();

  try {
    console.log("[EcoShaadi] sending POST to", `${API_BASE}/api/analyze`);
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      body: formData,
      signal: currentController.signal,
    });
    console.log("[EcoShaadi] response status:", res.status);

    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed (${res.status})`);
    }

    const data = await res.json();
    console.log("[EcoShaadi] response data:", data);
    finishStageAnimation();
    lastResult = data;
    renderResults(data);
  } catch (err) {
    console.error("[EcoShaadi] request failed:", err);
    finishStageAnimation();
    if (err.name === "AbortError") return; // user cancelled, no alert needed
    alert(`Something went wrong: ${err.message}\n\nMake sure the backend is running and your GROQ_API_KEY is set.`);
  } finally {
    currentController = null;
  }
});

cancelBtn.addEventListener("click", () => {
  if (currentController) currentController.abort();
  finishStageAnimation();
});

/* ---------------- Results rendering ---------------- */

function renderResults(data) {
  resultsSection.classList.remove('is-hidden');

  // Detected waste
  const wasteList = document.getElementById("waste-list");
  wasteList.innerHTML = "";
  const classificationEntries = Object.entries(data.classification || {});

  if (classificationEntries.length === 0) {
    wasteList.innerHTML = `<p class="empty-note">No recognizable wedding waste was detected in these files. Try a clearer, closer photo.</p>`;
  } else {
    classificationEntries.forEach(([name, info]) => {
      const row = document.createElement("div");
      row.className = "waste-row";
      row.innerHTML = `
        <div>
          <div class="name">${escapeHtml(name)}</div>
          <div class="qty">${info.quantity_kg} kg</div>
        </div>
        <span class="badge badge-${info.category}">${info.category}</span>
      `;
      wasteList.appendChild(row);
    });
  }

  // Receivers
  const receiverList = document.getElementById("receiver-list");
  receiverList.innerHTML = "";
  const recommendations = Object.entries(data.recommendations || {});

  if (recommendations.length === 0) {
    receiverList.innerHTML = `<p class="empty-note">No matching receiver found in the demo directory for this waste yet.</p>`;
  } else {
    recommendations.forEach(([waste, rec]) => {
      const row = document.createElement("div");
      row.className = "receiver-row";
      row.innerHTML = `
        <div class="top">
          <span class="receiver-name">${escapeHtml(rec.receiver)}</span>
          <span class="badge badge-${rec.category}">${rec.category}</span>
        </div>
        <div class="meta">
          <span>For ${escapeHtml(waste)}</span>
          <span>${rec.distance_km} km away</span>
          <a href="tel:${rec.contact}">${rec.contact}</a>
        </div>
      `;
      receiverList.appendChild(row);
    });
  }

  // Logistics
  const logisticsBody = document.getElementById("logistics-body");
  const plan = data.pickup_plan || {};
  if (!plan.stops || plan.stops.length === 0) {
    logisticsBody.innerHTML = `<p class="empty-note">No pickup needed yet.</p>`;
  } else {
    logisticsBody.innerHTML = `
      <div class="logistics-summary">
        <span>🚚 ${escapeHtml(plan.vehicle || "Mini Truck")}</span>
        <span>${plan.estimated_distance} km total</span>
      </div>
      <ol class="logistics-stops">
        ${plan.stops.map((s) => `<li>${escapeHtml(s.waste)} → ${escapeHtml(s.receiver)} (${s.distance_km} km)</li>`).join("")}
      </ol>
    `;
  }

  // Sustainability
  const metrics = data.sustainability_metrics || { waste_diverted_kg: 0, co2_saved_kg: 0 };
  document.getElementById("impact-diverted").textContent = metrics.waste_diverted_kg;
  document.getElementById("impact-co2").textContent = metrics.co2_saved_kg;

  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

/* ---------------- Reset ---------------- */

document.getElementById("reset-btn").addEventListener("click", () => {
  selectedFiles = [];
  fileInput.value = "";
  renderPreviews();
  resultsSection.classList.add('is-hidden');
  document.getElementById("upload").scrollIntoView({ behavior: "smooth" });
});

/* ---------------- Certificate PDF ---------------- */

// document.getElementById("download-report").addEventListener("click", () => {
//   if (!lastResult) return;
//   const { jsPDF } = window.jspdf;
//   const doc = new jsPDF();

//   const metrics = lastResult.sustainability_metrics || {};
//   doc.setFont("times", "bold");
//   doc.setFontSize(22);
//   doc.text("ReVive AI — Sustainability Certificate", 105, 25, { align: "center" });

//   doc.setFont("times", "normal");
//   doc.setFontSize(12);
//   doc.text(`Generated: ${new Date().toLocaleString()}`, 105, 34, { align: "center" });

//   let y = 50;
//   doc.setFontSize(14);
//   doc.text("Detected Waste", 20, y);
//   doc.setFontSize(11);
//   y += 8;
//   Object.entries(lastResult.classification || {}).forEach(([name, info]) => {
//     doc.text(`- ${name}: ${info.quantity_kg} kg (${info.category})`, 24, y);
//     y += 7;
//   });

//   y += 6;
//   doc.setFontSize(14);
//   doc.text("Recommended Receivers", 20, y);
//   doc.setFontSize(11);
//   y += 8;
//   Object.entries(lastResult.recommendations || {}).forEach(([waste, rec]) => {
//     doc.text(`- ${waste} -> ${rec.receiver} (${rec.distance_km} km, ${rec.contact})`, 24, y);
//     y += 7;
//   });

//   y += 10;
//   doc.setFontSize(16);
//   doc.setFont("times", "bold");
//   doc.text(`Waste diverted from landfill: ${metrics.waste_diverted_kg ?? 0} kg`, 20, y);
//   y += 9;
//   doc.text(`CO2 emissions avoided: ${metrics.co2_saved_kg ?? 0} kg`, 20, y);

//   doc.save("ReVive-sustainability-certificate.pdf");
// });