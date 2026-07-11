(() => {
  const $ = (sel) => document.querySelector(sel);

  const tabs = document.querySelectorAll(".tab");
  const panes = {
    upload: $("#pane-upload"),
    form: $("#pane-form"),
  };

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("is-active"));
      tab.classList.add("is-active");
      Object.values(panes).forEach((p) => p.classList.remove("is-active"));
      panes[tab.dataset.tab].classList.add("is-active");
    });
  });

  const dropzone = $("#dropzone");
  const fileInput = $("#file-input");
  const fileName = $("#file-name");

  // Dropzone is a <label> wrapping the file input — browser already opens
  // the picker on click. Do not call fileInput.click() again (double dialog).
  fileInput?.addEventListener("change", () => {
    fileName.textContent = fileInput.files[0]?.name || "No file selected";
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone?.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("is-drag");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropzone?.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("is-drag");
    });
  });
  dropzone?.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    fileName.textContent = file.name;
  });

  const empty = $("#results-empty");
  const layout = $("#results-layout");
  const tbody = $("#score-tbody");
  const errBox = $("#results-error");
  const inputWarn = $("#results-input-warning");
  const pipelineNote = $("#results-pipeline-note");
  const loading = $("#results-loading");
  const meta = $("#results-meta");
  const filters = $("#filters");
  const searchInput = $("#search-id");
  const bandFilter = $("#filter-band");
  const sortBy = $("#sort-by");
  const filterCount = $("#filter-count");
  const detailPlaceholder = $("#detail-placeholder");
  const detailContent = $("#detail-content");
  const inputDrawer = document.querySelector(".input-drawer");

  let allResults = [];
  let selectedId = null;

  function setLoading(on) {
    loading.classList.toggle("hidden", !on);
    if (on) {
      empty.classList.add("hidden");
      errBox.classList.add("hidden");
      inputWarn?.classList.add("hidden");
      pipelineNote?.classList.add("hidden");
      layout.classList.add("hidden");
      filters.classList.add("hidden");
    }
  }

  function showError(msg) {
    errBox.textContent = msg;
    errBox.classList.remove("hidden");
    inputWarn?.classList.add("hidden");
    pipelineNote?.classList.add("hidden");
    empty.classList.add("hidden");
    layout.classList.add("hidden");
    filters.classList.add("hidden");
  }

  function bandClass(band) {
    const b = (band || "").toLowerCase();
    if (b === "good") return "band-good";
    if (b === "bad") return "band-bad";
    return "band-neutral";
  }

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function renderSignalList(title, items, cls) {
    if (!items?.length) return "";
    const lis = items.map((t) => `<li>${escapeHtml(t)}</li>`).join("");
    return `<div class="signal-block ${cls}"><h4>${title}</h4><ul>${lis}</ul></div>`;
  }

  function renderDrivers(title, drivers, direction) {
    if (!drivers?.length) return "";
    const items = drivers
      .filter((d) => Math.abs(Number(d.shap_value)) >= 0.001)
      .map((d) => {
        const shap = Number(d.shap_value);
        const missing = d.value_missing || d.value_display === "not in input";
        return `
      <div class="driver ${direction}">
        <strong>${escapeHtml(d.label || d.feature)}</strong>
        <span>${missing ? "Missing in input" : `Value ${escapeHtml(String(d.value_display ?? "—"))}`}
          · contribution ${shap > 0 ? "+" : ""}${shap.toFixed(4)}</span>
      </div>`;
      })
      .join("");
    if (!items) return "";
    return `<div><h4>${title}</h4>${items}</div>`;
  }

  function showDetail(row) {
    if (!row) {
      detailPlaceholder.classList.remove("hidden");
      detailContent.classList.add("hidden");
      detailContent.innerHTML = "";
      return;
    }
    selectedId = row.LandLordID;
    detailPlaceholder.classList.add("hidden");
    detailContent.classList.remove("hidden");
    const q = Number(row.PredictedQualityScore ?? 0).toFixed(1);
    const pct = row.PercentileRank != null ? Number(row.PercentileRank).toFixed(1) : "—";
    const band = String(row.QualityBand || "unknown");
    const conf = String(row.ConfidenceLevel || "—");
    const main = row.MainReason || row.ExplanationSummary || "";
    const final = row.FinalInterpretation || "";
    const posSig = row.PositiveSignals || [];
    const negSig = row.NegativeSignals || [];

    detailContent.innerHTML = `
      <h3>${escapeHtml(row.LandLordID || "—")}</h3>
      <div class="detail-metrics">
        <span class="pill ${bandClass(band)}">Prediction: ${escapeHtml(band)}</span>
        <span class="pill">Quality ${q} / 100</span>
        <span class="pill">Percentile ${pct}</span>
        <span class="pill">Confidence ${escapeHtml(conf)}</span>
      </div>

      <section class="narrative">
        <h4>Main reason</h4>
        <p>${escapeHtml(main)}</p>
        ${renderSignalList("Positive signals", posSig, "up")}
        ${renderSignalList("Negative signals", negSig, "down")}
        <h4>Final interpretation</h4>
        <p>${escapeHtml(final)}</p>
      </section>

      <details class="tech-shap">
        <summary>Technical SHAP details</summary>
        <div class="drivers">
          ${renderDrivers("Features that supported a higher score", row.TopPositiveDrivers, "up")}
          ${renderDrivers("Features that weighed the score down", row.TopNegativeDrivers, "down")}
        </div>
      </details>
    `;
    tbody.querySelectorAll("tr").forEach((tr) => {
      tr.classList.toggle("is-selected", tr.dataset.id === String(selectedId));
    });
  }

  function filteredSorted() {
    const q = (searchInput.value || "").trim().toLowerCase();
    const band = bandFilter.value;
    let rows = allResults.slice();

    if (q) {
      rows = rows.filter((r) => String(r.LandLordID || "").toLowerCase().includes(q));
    }
    if (band !== "all") {
      rows = rows.filter((r) => String(r.QualityBand || "").toLowerCase() === band);
    }

    const sort = sortBy.value;
    rows.sort((a, b) => {
      if (sort === "score-asc") return (a.PredictedScore ?? 0) - (b.PredictedScore ?? 0);
      if (sort === "score-desc") return (b.PredictedScore ?? 0) - (a.PredictedScore ?? 0);
      if (sort === "id-asc") {
        return String(a.LandLordID || "").localeCompare(String(b.LandLordID || ""));
      }
      if (sort === "band") {
        return String(a.QualityBand || "").localeCompare(String(b.QualityBand || ""));
      }
      return 0;
    });
    return rows;
  }

  function renderTable() {
    const rows = filteredSorted();
    filterCount.textContent = `Showing ${rows.length} of ${allResults.length}`;
    tbody.innerHTML = rows
      .map((r) => {
        const score = Number(r.PredictedScore ?? 0).toFixed(4);
        const q = Number(r.PredictedQualityScore ?? 0).toFixed(1);
        const pct = r.PercentileRank != null ? Number(r.PercentileRank).toFixed(1) : "—";
        const selected = String(r.LandLordID) === String(selectedId) ? "is-selected" : "";
        return `
        <tr data-id="${escapeHtml(r.LandLordID || "")}" class="${selected}">
          <td class="id-cell">${escapeHtml(r.LandLordID || "—")}</td>
          <td><span class="pill ${bandClass(r.QualityBand)}">${escapeHtml(r.QualityBand || "—")}</span></td>
          <td>${score}</td>
          <td>${q}</td>
          <td>${pct}</td>
          <td>${escapeHtml(r.ConfidenceLevel || "—")}</td>
        </tr>`;
      })
      .join("");

    tbody.querySelectorAll("tr").forEach((tr) => {
      tr.addEventListener("click", () => {
        const row = allResults.find((r) => String(r.LandLordID) === tr.dataset.id);
        showDetail(row);
      });
    });

    if (!rows.length) {
      showDetail(null);
      return;
    }
    // Keep selection if still visible; else select first
    const stillVisible = rows.find((r) => String(r.LandLordID) === String(selectedId));
    showDetail(stillVisible || rows[0]);
  }

  function renderResults(payload) {
    empty.classList.add("hidden");
    errBox.classList.add("hidden");
    layout.classList.remove("hidden");
    filters.classList.remove("hidden");

    if (payload.pipeline_note && payload.pipeline?.pipeline_applied) {
      pipelineNote.textContent = payload.pipeline_note;
      pipelineNote.classList.remove("hidden");
    } else if (payload.pipeline_note && !payload.input_warning) {
      pipelineNote.textContent = payload.pipeline_note;
      pipelineNote.classList.remove("hidden");
    } else {
      pipelineNote?.classList.add("hidden");
    }

    if (payload.input_warning) {
      inputWarn.textContent = payload.input_warning;
      inputWarn.classList.remove("hidden");
    } else {
      inputWarn?.classList.add("hidden");
    }

    const pipe = payload.pipeline || {};
    meta.textContent =
      `Scored ${payload.n_scored} of ${payload.n_input} row(s)` +
      (payload.truncated ? ` (truncated to ${payload.max_rows})` : "") +
      (pipe.input_kind ? ` · input=${pipe.input_kind}` : "") +
      (pipe.pipeline_applied ? " · FE pipeline on" : " · direct features") +
      ` · ${payload.model_type || "model"} ${payload.model_version || ""}`;

    allResults = payload.results || [];
    selectedId = allResults[0]?.LandLordID ?? null;

    // Collapse input so results dominate the screen
    if (inputDrawer && allResults.length > 1) {
      inputDrawer.open = false;
    }

    renderTable();
    $("#results-stage")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  [searchInput, bandFilter, sortBy].forEach((el) => {
    el?.addEventListener("input", renderTable);
    el?.addEventListener("change", renderTable);
  });

  async function postScore(url, body, isFormData) {
    setLoading(true);
    try {
      const res = await fetch(url, {
        method: "POST",
        body,
        headers: isFormData ? undefined : { "Content-Type": "application/json" },
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const detail = data.detail;
        const msg = Array.isArray(detail)
          ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
          : detail || res.statusText || "Request failed";
        throw new Error(msg);
      }
      renderResults(data);
    } catch (err) {
      showError(err.message || String(err));
      meta.textContent = "Scoring failed.";
    } finally {
      setLoading(false);
    }
  }

  $("#upload-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = fileInput.files?.[0];
    if (!file) {
      showError("Choose a CSV or Parquet file first.");
      return;
    }
    const fd = new FormData();
    fd.append("file", file);
    fd.append("top_k", $("#upload-top-k").value || "5");
    await postScore("/api/score/file", fd, true);
  });

  $("#manual-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const record = {};
    for (const [k, v] of fd.entries()) {
      if (v === "" || v == null) continue;
      const num = Number(v);
      record[k] =
        Number.isFinite(num) &&
        String(v).trim() !== "" &&
        !["PreferredIndustry", "OriginCity", "OriginCountry", "LandLordID"].includes(k)
          ? num
          : v;
    }
    ["PreferredIndustry", "OriginCity", "OriginCountry", "LandLordID"].forEach((k) => {
      if (fd.get(k)) record[k] = String(fd.get(k));
    });
    if (!Object.keys(record).length) {
      showError("Enter at least some landlord feature values.");
      return;
    }
    const top_k = Number($("#form-top-k").value || 5);
    await postScore(
      "/api/score/json",
      JSON.stringify({ landlords: [record], top_k }),
      false
    );
  });
})();
