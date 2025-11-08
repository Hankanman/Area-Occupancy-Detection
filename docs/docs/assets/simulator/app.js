/* eslint-disable no-undef */
(function () {
  const DEFAULT_API_BASE_URL = "https://aod-simulator.onrender.com";
  const LOCAL_API_BASE_URL = "http://127.0.0.1:5000";
  const STORAGE_KEY = "aodSimulatorApiBaseUrl";

  const state = {
    apiBaseUrl: DEFAULT_API_BASE_URL,
    autoUpdateInterval: null,
    probabilityChart: null,
    probabilityHistory: [],
    simulationData: null,
  };

  document.body.classList.add("aod-simulator-mode");

  function formatPercent(value, digits = 1) {
    return `${(value * 100).toFixed(digits)}%`;
  }

  function formatDecimal(value, digits = 2) {
    return value.toFixed(digits);
  }

  function setDisplayText(element, text) {
    if (!element) {
      return;
    }

    element.textContent = text;
    element.dataset.displayValue = text;
  }

  function getRangeBaseLabel(slider, fallback = "") {
    if (!slider) {
      return fallback;
    }

    if (!slider.dataset.baseLabel) {
      slider.dataset.baseLabel =
        slider.dataset.label ||
        slider.getAttribute("data-label") ||
        slider.label ||
        fallback;
    }

    return slider.dataset.baseLabel;
  }

  function setRangeLabelWithValue(
    slider,
    value,
    fallbackLabel = "",
    formatter = (val) => formatPercent(val)
  ) {
    if (!slider) {
      return;
    }

    const baseLabel = getRangeBaseLabel(slider, fallbackLabel);
    slider.label = `${baseLabel}: ${formatter(value)}`;
  }

  function getEventValue(event) {
    if (
      event?.detail &&
      Object.prototype.hasOwnProperty.call(event.detail, "value")
    ) {
      return event.detail.value;
    }
    return event?.target?.value;
  }

  function normalizeBaseUrl(url) {
    if (!url) {
      return DEFAULT_API_BASE_URL;
    }

    try {
      const normalized = new URL(url);
      return normalized.toString().replace(/\/$/, "");
    } catch (error) {
      return DEFAULT_API_BASE_URL;
    }
  }

  function isLocalDocsHost() {
    return ["localhost", "127.0.0.1"].includes(window.location.hostname);
  }

  function loadStoredBaseUrl() {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (!stored) {
        if (isLocalDocsHost()) {
          return LOCAL_API_BASE_URL;
        }
        return DEFAULT_API_BASE_URL;
      }

      return normalizeBaseUrl(stored);
    } catch (error) {
      if (isLocalDocsHost()) {
        return LOCAL_API_BASE_URL;
      }
      return DEFAULT_API_BASE_URL;
    }
  }

  function persistBaseUrl(url) {
    try {
      window.localStorage.setItem(STORAGE_KEY, url);
    } catch (error) {
      // ignore persistence errors (e.g., privacy mode)
    }
  }

  function updateApiBaseUrl(url, { persist = true } = {}) {
    const normalized = normalizeBaseUrl(url);
    state.apiBaseUrl = normalized;

    if (persist) {
      persistBaseUrl(normalized);
    }

    if (apiBaseDisplay) {
      apiBaseDisplay.textContent = normalized;
    }

    if (apiBaseInput) {
      apiBaseInput.value = normalized;
    }

    updateApiStatus("unknown", "Checking");
    verifyApiOnline();
  }

  function buildApiUrl(path) {
    const base = state.apiBaseUrl.replace(/\/$/, "");
    if (!path.startsWith("/")) {
      return `${base}/${path}`;
    }

    return `${base}${path}`;
  }

  async function apiFetch(path, options = {}) {
    const url = buildApiUrl(path);
    const response = await fetch(url, options);
    return response;
  }

  async function requestJson(path, options = {}) {
    const response = await apiFetch(path, options);
    const contentType = response.headers.get("content-type") ?? "";
    let payload = null;

    if (contentType.includes("application/json")) {
      try {
        payload = await response.json();
      } catch (error) {
        payload = null;
      }
    }

    if (!response.ok) {
      const statusMessage = options.errorMessage || `HTTP ${response.status}`;
      const message = payload?.error || statusMessage;
      throw new Error(message);
    }

    return payload ?? {};
  }

  function applySimulationPayload(payload, { resetHistory = false } = {}) {
    if (resetHistory) {
      state.probabilityHistory = [];
      initChart();
    }

    state.simulationData = payload;
    renderSimulation(payload);
    hideError();
  }

  async function executeSimulationUpdate(
    path,
    options = {},
    { resetHistory = false } = {}
  ) {
    const payload = await requestJson(path, options);
    applySimulationPayload(payload, { resetHistory });
    return payload;
  }

  function handleApiFailure(error, { silent = false } = {}) {
    const message = error instanceof Error ? error.message : String(error);
    if (!silent) {
      showError(message);
    }

    updateApiStatus("offline", "API Offline");
  }

  function updateApiStatus(stateClass, label) {
    if (!apiStatusBadge) {
      return;
    }

    apiStatusBadge.textContent = label;
    apiStatusBadge.classList.remove("online", "offline", "unknown");
    if (stateClass) {
      apiStatusBadge.classList.add(stateClass);
    }
  }

  async function verifyApiOnline() {
    try {
      await requestJson("/api/get-purposes");
      updateApiStatus("online", "API Online");
    } catch (error) {
      updateApiStatus("offline", "API Offline");
    }
  }

  // DOM references
  const rootEl = document.querySelector(".aod-simulator");
  const yamlInput = document.getElementById("yaml-input");
  const loadBtn = document.getElementById("load-btn");
  const errorMessage = document.getElementById("error-message");
  const simulationDisplay = document.getElementById("simulation-display");
  const areaName = document.getElementById("area-name");
  const probabilityValue = document.getElementById("probability-value");
  const probabilityFill = document.getElementById("probability-fill");
  const sensorsPlaceholder = document.getElementById("sensors-placeholder");
  const sensorsContainer = document.getElementById("sensors-container");
  const breakdownList = document.getElementById("breakdown-list");
  const globalPriorSlider = document.getElementById("global-prior-slider");
  const timePriorSlider = document.getElementById("time-prior-slider");
  const combinedPriorInput = document.getElementById("combined-prior-input");
  const finalPriorInput = document.getElementById("final-prior-input");
  const purposeSelect = document.getElementById("purpose-select");
  const halfLifeInput = document.getElementById("half-life-display");
  const apiBaseInput = document.getElementById("api-base-input");
  const apiBaseSave = document.getElementById("api-base-save");
  const apiBaseReset = document.getElementById("api-base-reset");
  const apiBaseDisplay = document.getElementById("api-base-display");
  const apiStatusBadge = document.getElementById("api-status-badge");

  const weightControls = {
    motion: {
      slider: document.getElementById("weight-motion-slider"),
      baseLabel: "Motion",
    },
    media: {
      slider: document.getElementById("weight-media-slider"),
      baseLabel: "Media",
    },
    appliance: {
      slider: document.getElementById("weight-appliance-slider"),
      baseLabel: "Appliance",
    },
    door: {
      slider: document.getElementById("weight-door-slider"),
      baseLabel: "Door",
    },
    window: {
      slider: document.getElementById("weight-window-slider"),
      baseLabel: "Window",
    },
    illuminance: {
      slider: document.getElementById("weight-illuminance-slider"),
      baseLabel: "Illuminance",
    },
    humidity: {
      slider: document.getElementById("weight-humidity-slider"),
      baseLabel: "Humidity",
    },
    temperature: {
      slider: document.getElementById("weight-temperature-slider"),
      baseLabel: "Temperature",
    },
  };

  function showError(message) {
    if (!errorMessage) {
      return;
    }
    errorMessage.textContent = message;
    errorMessage.classList.add("show");
  }

  function hideError() {
    if (!errorMessage) {
      return;
    }
    errorMessage.textContent = "";
    errorMessage.classList.remove("show");
  }

  function initChart() {
    const ctx = document.getElementById("probability-chart");
    if (state.probabilityChart) {
      state.probabilityChart.destroy();
    }

    if (!ctx) {
      return;
    }

    state.probabilityChart = new Chart(ctx, {
      type: "line",
      data: {
        labels: [],
        datasets: [
          {
            label: "Occupancy Probability",
            data: [],
            borderColor: "#667eea",
            backgroundColor: "rgba(102, 126, 234, 0.1)",
            borderWidth: 2,
            fill: true,
            tension: 0.4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        scales: {
          y: {
            beginAtZero: true,
            max: 1.0,
            ticks: {
              callback(value) {
                return `${(value * 100).toFixed(0)}%`;
              },
            },
          },
          x: {
            display: false,
          },
        },
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            callbacks: {
              label(context) {
                return `Probability: ${(context.parsed.y * 100).toFixed(2)}%`;
              },
            },
          },
        },
      },
    });
  }

  function addChartPoint(probability) {
    const now = new Date();
    const timeLabel = now.toLocaleTimeString();

    state.probabilityHistory.push({
      time: timeLabel,
      probability,
    });

    if (state.probabilityHistory.length > 30) {
      state.probabilityHistory.shift();
    }

    if (state.probabilityChart) {
      state.probabilityChart.data.labels = state.probabilityHistory.map(
        (p) => p.time
      );
      state.probabilityChart.data.datasets[0].data =
        state.probabilityHistory.map((p) => p.probability);
      state.probabilityChart.update("none");
    }
  }

  function setProbability(probability) {
    if (probabilityValue) {
      setDisplayText(probabilityValue, formatPercent(probability));
    }

    if (probabilityFill) {
      probabilityFill.style.width = `${(probability * 100).toFixed(2)}%`;
    }

    addChartPoint(probability);
  }

  function renderSensors(data) {
    if (!sensorsContainer) {
      return;
    }

    sensorsContainer.innerHTML = "";
    sensorsContainer.classList.toggle("empty", data.entities.length === 0);

    if (sensorsPlaceholder) {
      sensorsPlaceholder.classList.toggle("hidden", data.entities.length > 0);
    }

    data.entities.forEach((entity) => {
      sensorsContainer.appendChild(createSensorCard(entity));
    });
  }

  function createElementWithClass(tagName, className) {
    const element = document.createElement(tagName);
    if (className) {
      element.className = className;
    }

    return element;
  }

  function parseDetailSegments(details) {
    if (!details) {
      return [];
    }

    if (Array.isArray(details)) {
      return details.filter(Boolean);
    }

    return String(details)
      .split(/(?:\s*•\s*|\n+)/)
      .map((segment) => segment.trim())
      .filter(Boolean);
  }

  function buildSensorContent(entity) {
    const content = createElementWithClass("div", "sim-card-content");
    const header = createElementWithClass("div", "sim-card-header");

    const title = document.createElement("h3");
    title.textContent = entity.name;
    header.appendChild(title);

    const stateEl = document.createElement("div");
    stateEl.textContent = `State: ${entity.state_display}`;
    header.appendChild(stateEl);

    content.appendChild(header);

    const segments = parseDetailSegments(entity.details);
    if (segments.length > 0) {
      const row = createElementWithClass("div", "sim-card-row");
      segments.forEach((segment) => {
        const detailItem = document.createElement("div");
        detailItem.textContent = segment;
        row.appendChild(detailItem);
      });
      content.appendChild(row);
    }

    return content;
  }

  function findToggleActions(actions) {
    if (!Array.isArray(actions) || actions.length === 0) {
      return { onAction: null, offAction: null };
    }

    const onAction =
      actions.find(
        (action) =>
          typeof action.state === "string" &&
          action.state.toLowerCase() === "on"
      ) ?? actions.find((action) => action.active);
    const offAction = actions.find((action) => action !== onAction) ?? null;

    return { onAction, offAction };
  }

  function createToggleControl(entity, onAction, offAction) {
    if (!onAction || !offAction) {
      return null;
    }

    const toggle = document.createElement("sl-switch");
    toggle.textContent = onAction.label ?? "Active";
    toggle.checked = Boolean(onAction.active);
    toggle.disabled = !onAction || !offAction;

    let lastKnownChecked = toggle.checked;

    toggle.addEventListener("sl-change", async (event) => {
      const control = event.currentTarget;
      const isChecked = control.checked;
      const targetAction = isChecked ? onAction : offAction;

      control.disabled = true;

      try {
        await executeSimulationUpdate("/api/toggle", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            entity_id: entity.entity_id,
            state: targetAction.state,
          }),
        });
        lastKnownChecked = isChecked;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        showError(message);
        control.checked = lastKnownChecked;
      } finally {
        control.disabled = false;
      }
    });

    return toggle;
  }

  function createNumericInput(entity, initialValue = "") {
    const input = document.createElement("sl-input");
    input.type = "number";
    input.size = "medium";
    input.value = initialValue;
    input.placeholder = "--";
    input.step = "any";
    input.inputMode = "decimal";
    input.disabled = false;

    let lastKnownValue = input.value;
    let isUpdating = false;

    const setLoadingState = (loading) => {
      isUpdating = loading;
      input.disabled = loading;
    };

    const submitValue = async (rawValue) => {
      const numericValue = Number.parseFloat(rawValue);
      if (!Number.isFinite(numericValue)) {
        showError(`Invalid numeric value: ${rawValue}`);
        input.value = lastKnownValue;
        return;
      }

      const lastNumeric = Number.parseFloat(lastKnownValue);
      if (!Number.isNaN(lastNumeric) && lastNumeric === numericValue) {
        input.value = lastKnownValue;
        return;
      }

      setLoadingState(true);

      try {
        await executeSimulationUpdate("/api/update", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            entity_id: entity.entity_id,
            value: numericValue,
          }),
        });
        lastKnownValue = numericValue.toString();
        input.value = lastKnownValue;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        showError(message);
        input.value = lastKnownValue;
      } finally {
        setLoadingState(false);
      }
    };

    input.addEventListener("sl-change", (event) => {
      if (isUpdating) {
        return;
      }

      submitValue(event.currentTarget.value);
    });

    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        input.blur();
      }
    });

    return input;
  }

  function createActionButton(entity, action) {
    const button = document.createElement("sl-button");
    button.variant = action.active ? "primary" : "default";
    button.size = "small";
    button.textContent = action.label;

    button.addEventListener("click", async () => {
      try {
        await executeSimulationUpdate("/api/toggle", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            entity_id: entity.entity_id,
            state: action.state,
          }),
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        showError(message);
      }
    });

    return button;
  }

  function buildSensorActions(entity) {
    const actions = Array.isArray(entity.actions) ? entity.actions : [];
    const container = createElementWithClass("div", "sim-card-actions");

    const supportsBinaryToggle =
      !entity.is_numeric &&
      actions.length >= 2 &&
      actions.every((action) => typeof action.state === "string");

    if (supportsBinaryToggle) {
      const { onAction, offAction } = findToggleActions(actions);
      const toggle = createToggleControl(entity, onAction, offAction);
      if (toggle) {
        container.appendChild(toggle);
        return container;
      }
    }

    if (entity.is_numeric) {
      const currentValueAction = actions.find((action) => action.active);
      const input = createNumericInput(
        entity,
        currentValueAction?.value?.toString() ?? entity.state ?? ""
      );
      container.appendChild(input);
      return container;
    }

    if (actions.length > 0) {
      actions.forEach((action) => {
        const button = createActionButton(entity, action);
        container.appendChild(button);
      });

      if (container.children.length > 0) {
        return container;
      }
    }

    return null;
  }

  function createSensorCard(entity) {
    const card = createElementWithClass("div", "md-card sim-card entity");
    const actions = buildSensorActions(entity);

    if (actions) {
      card.appendChild(actions);
    }

    card.appendChild(buildSensorContent(entity));
    return card;
  }

  function renderBreakdown(data) {
    if (!breakdownList) {
      return;
    }

    breakdownList.innerHTML = "";

    data.breakdown.forEach((item) => {
      const breakdownItem = document.createElement("div");
      breakdownItem.className = "md-card sim-breakdown-card";

      const label = document.createElement("div");
      const labelStrong = document.createElement("strong");
      labelStrong.textContent = item.name;
      const labelSpan = document.createElement("span");
      labelSpan.textContent = item.description;
      label.append(labelStrong, labelSpan);

      const likelihood = document.createElement("div");
      const likelihoodValue = document.createElement("strong");
      setDisplayText(likelihoodValue, formatPercent(item.likelihood));
      const likelihoodLabel = document.createElement("span");
      likelihoodLabel.textContent = "Likelihood";
      likelihood.append(likelihoodValue, likelihoodLabel);

      const contribution = document.createElement("div");
      const contributionValue = document.createElement("strong");
      setDisplayText(contributionValue, formatPercent(item.contribution));
      const contributionLabel = document.createElement("span");
      contributionLabel.textContent = "Contribution";
      contribution.append(contributionValue, contributionLabel);

      breakdownItem.append(label, likelihood, contribution);
      breakdownList.appendChild(breakdownItem);
    });
  }

  function syncPriorControls(data) {
    const globalPriorValue = Number(data.global_prior.toFixed(2));
    if (globalPriorSlider) {
      globalPriorSlider.value = globalPriorValue;
    }
    setRangeLabelWithValue(globalPriorSlider, globalPriorValue, "Global Prior");

    const timePriorValue = Number(data.time_prior.toFixed(2));
    if (timePriorSlider) {
      timePriorSlider.value = timePriorValue;
    }
    setRangeLabelWithValue(timePriorSlider, timePriorValue, "Time Prior");

    const combinedPriorPercent = formatPercent(data.combined_prior);
    const finalPriorPercent = formatPercent(data.final_prior);

    if (combinedPriorInput) {
      combinedPriorInput.value = combinedPriorPercent;
    }
    if (finalPriorInput) {
      finalPriorInput.value = finalPriorPercent;
    }

    if (halfLifeInput) {
      halfLifeInput.value = `${Math.round(data.half_life)}s`;
    }

    Object.entries(weightControls).forEach(([key, control]) => {
      if (!data.weights || typeof data.weights[key] !== "number") {
        return;
      }

      const value = Number(data.weights[key].toFixed(2));
      if (control.slider) {
        control.slider.value = value;
      }
      setRangeLabelWithValue(
        control.slider,
        data.weights[key],
        control.baseLabel,
        formatDecimal
      );
    });
  }

  function renderSimulation(data) {
    if (!simulationDisplay) {
      return;
    }

    simulationDisplay.classList.remove("hidden");

    if (areaName) {
      areaName.textContent = data.area_name;
    }

    setProbability(data.probability);
    renderSensors(data);
    renderBreakdown(data);
    syncPriorControls(data);

    if (purposeSelect && typeof data.purpose === "string") {
      purposeSelect.value = data.purpose;
    }
  }

  function startAutoUpdate() {
    if (state.autoUpdateInterval) {
      clearInterval(state.autoUpdateInterval);
    }

    state.autoUpdateInterval = setInterval(async () => {
      if (!state.simulationData) {
        return;
      }

      try {
        const payload = await requestJson("/api/tick", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({}),
        });

        applySimulationPayload(payload);
      } catch (error) {
        handleApiFailure(error, { silent: true });
      }
    }, 1000);
  }

  function registerApiSlider({
    slider,
    clamp,
    endpoint,
    payloadBuilder,
    baseLabel,
    formatter = formatPercent,
  }) {
    if (!slider) {
      return;
    }

    const applyValue = (value) => {
      const numericValue = Number.parseFloat(value);
      const normalized = Number.isFinite(numericValue) ? numericValue : 0;
      const clampedValue = clamp(normalized);
      slider.value = clampedValue;
      setRangeLabelWithValue(slider, clampedValue, baseLabel, formatter);
      return clampedValue;
    };

    slider.addEventListener("sl-input", (event) => {
      applyValue(getEventValue(event));
    });

    slider.addEventListener("sl-change", async (event) => {
      const clampedValue = applyValue(getEventValue(event));

      if (!state.simulationData) {
        return;
      }

      try {
        await executeSimulationUpdate(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payloadBuilder(clampedValue)),
        });
      } catch (error) {
        handleApiFailure(error);
      }
    });

    applyValue(slider.value);
  }

  async function loadPurposes() {
    if (!purposeSelect) {
      return;
    }

    try {
      const payload = await requestJson("/api/get-purposes");
      purposeSelect.innerHTML = "";
      payload.purposes.forEach((purpose) => {
        const option = document.createElement("sl-option");
        option.value = purpose.value;
        option.textContent = purpose.label;
        purposeSelect.appendChild(option);
      });

      updateApiStatus("online", "API Online");
    } catch (error) {
      handleApiFailure(error);
    }
  }

  async function handleLoadSimulation() {
    const yamlText = (yamlInput?.value ?? "").trim();

    if (!yamlText) {
      showError("Please paste YAML analysis output");
      return;
    }

    try {
      const payload = await requestJson("/api/load", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ yaml: yamlText }),
      });

      applySimulationPayload(payload, { resetHistory: true });

      startAutoUpdate();
      updateApiStatus("online", "API Online");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      showError(`Error loading simulation: ${message}`);
      updateApiStatus("offline", "API Offline");
    }
  }

  async function handlePurposeChange(event) {
    const value = getEventValue(event) ?? "";

    if (!state.simulationData) {
      return;
    }

    try {
      await executeSimulationUpdate("/api/update-purpose", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ purpose: value }),
      });
    } catch (error) {
      handleApiFailure(error);
    }
  }

  function initPriorControls() {
    registerApiSlider({
      slider: globalPriorSlider,
      clamp: (v) => Math.min(Math.max(v, 0), 1),
      endpoint: "/api/update-priors",
      payloadBuilder: (value) => ({ global_prior: value }),
      baseLabel: "Global Prior",
    });

    registerApiSlider({
      slider: timePriorSlider,
      clamp: (v) => Math.min(Math.max(v, 0), 1),
      endpoint: "/api/update-priors",
      payloadBuilder: (value) => ({ time_prior: value }),
      baseLabel: "Time Prior",
    });
  }

  function initWeightControls() {
    Object.entries(weightControls).forEach(([type, control]) => {
      registerApiSlider({
        slider: control.slider,
        clamp: (v) => Math.min(Math.max(v, 0.01), 1),
        endpoint: "/api/update-weights",
        payloadBuilder: (value) => ({ weight_type: type, value: value }),
        baseLabel: control.baseLabel,
        formatter: formatDecimal,
      });
    });
  }

  function initApiControls() {
    if (!apiBaseInput || !apiBaseSave || !apiBaseReset) {
      return;
    }

    const handleApiBaseUpdate = (value) => {
      updateApiBaseUrl(value ?? apiBaseInput.value);
    };

    apiBaseSave.addEventListener("click", () => {
      handleApiBaseUpdate(apiBaseInput.value);
    });

    apiBaseReset.addEventListener("click", () => {
      updateApiBaseUrl(DEFAULT_API_BASE_URL);
    });

    apiBaseInput.addEventListener("sl-change", (event) => {
      handleApiBaseUpdate(getEventValue(event));
    });

    apiBaseInput.addEventListener("change", (event) => {
      handleApiBaseUpdate(getEventValue(event));
    });

    apiBaseInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        handleApiBaseUpdate(apiBaseInput.value);
      }
    });
  }

  function init() {
    if (!rootEl) {
      return;
    }

    state.apiBaseUrl = loadStoredBaseUrl();
    updateApiBaseUrl(state.apiBaseUrl, { persist: false });

    initChart();
    initPriorControls();
    initWeightControls();
    initApiControls();
    loadPurposes();

    if (loadBtn) {
      loadBtn.addEventListener("click", handleLoadSimulation);
    }

    if (purposeSelect) {
      purposeSelect.addEventListener("sl-change", handlePurposeChange);
    }
  }

  init();
})();
