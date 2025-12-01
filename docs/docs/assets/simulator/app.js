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
    simulation: null,
    purposes: [],
    isAnalyzing: false,
    availableAreas: [],
    selectedAreaName: null,
    yamlData: null, // Store raw YAML data for area switching
    pendingRequest: null, // Track current in-flight request
    requestQueue: [], // Queue for pending requests
    abortController: null, // AbortController for cancelling requests
  };

  document.body.classList.add("aod-simulator-mode");

  function deepClone(value) {
    if (value === undefined) {
      return undefined;
    }
    return JSON.parse(JSON.stringify(value));
  }

  function hasSimulation() {
    return Boolean(state.simulation?.request && state.simulation?.result);
  }

  function buildEntityLookup(entities = []) {
    const lookup = {};
    entities.forEach((entity, index) => {
      lookup[entity.entity_id] = { entity, index };
    });
    return lookup;
  }

  function cloneSimulationInput(input) {
    if (input === null || input === undefined) {
      return {};
    }
    return deepClone(input);
  }

  function getEntityInput(entityId) {
    if (!hasSimulation()) {
      return undefined;
    }
    return state.simulation.entityLookup?.[entityId]?.entity;
  }

  function setSimulation(
    simulationInput,
    result,
    { resetHistory = false } = {}
  ) {
    const simulationCopy = cloneSimulationInput(simulationInput);
    simulationCopy.entities ??= [];
    simulationCopy.area ??= {};
    simulationCopy.weights ??= {};

    state.simulation = {
      request: simulationCopy,
      result: null,
      entityLookup: buildEntityLookup(simulationCopy.entities),
    };

    updateSimulationResult(result, { resetHistory });
  }

  function replaceEntityInputs(entityInputs = [], weights = {}) {
    if (!hasSimulation()) {
      return;
    }
    const nextEntities = deepClone(entityInputs ?? []);
    state.simulation.request.entities = nextEntities;
    state.simulation.entityLookup = buildEntityLookup(nextEntities);

    if (weights && typeof weights === "object") {
      // Merge weights instead of replacing to preserve manually set weights
      state.simulation.request.weights = {
        ...(state.simulation.request.weights ?? {}),
        ...deepClone(weights),
      };
    }
  }

  function updateSimulationResult(result, { resetHistory = false } = {}) {
    if (!state.simulation || !result) {
      return;
    }

    if (Array.isArray(result.entity_inputs)) {
      replaceEntityInputs(result.entity_inputs, result.weights ?? {});
    }

    if (result.area && state.simulation.request.area) {
      const requestArea = state.simulation.request.area;
      if (result.area.half_life !== undefined) {
        requestArea.half_life = result.area.half_life;
      }
      if (result.area.purpose !== undefined) {
        requestArea.purpose = result.area.purpose;
      }
    }

    state.simulation.result = result;
    applySimulationResult(result, { resetHistory });
  }

  function prepareAnalyzePayload() {
    if (!hasSimulation()) {
      return {};
    }
    return deepClone(state.simulation.request);
  }

  async function processRequestQueue() {
    if (state.isAnalyzing || state.requestQueue.length === 0) {
      return;
    }

    // Sort queue: user requests first, then auto requests
    state.requestQueue.sort((a, b) => {
      if (a.priority === "user" && b.priority !== "user") return -1;
      if (a.priority !== "user" && b.priority === "user") return 1;
      return 0;
    });

    const nextRequest = state.requestQueue.shift();
    if (nextRequest) {
      state.pendingRequest = nextRequest;
      await analyzeSimulationInternal(
        nextRequest.options,
        nextRequest.abortSignal
      );
    }
  }

  async function analyzeSimulationInternal(options = {}, abortSignal = null) {
    const { resetHistory = false, silent = false } = options;

    if (!hasSimulation()) {
      return false;
    }

    state.isAnalyzing = true;
    try {
      const payload = prepareAnalyzePayload();
      const result = await requestJson(
        "/api/analyze",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        },
        abortSignal
      );
      updateSimulationResult(result, { resetHistory });
      return true;
    } catch (error) {
      // Don't show error if request was aborted
      if (error.name === "AbortError") {
        return false;
      }
      handleApiFailure(error, { silent });
      return false;
    } finally {
      state.isAnalyzing = false;
      state.pendingRequest = null;
      // Process next request in queue
      processRequestQueue();
    }
  }

  async function analyzeSimulation({
    resetHistory = false,
    silent = false,
    priority = "auto",
  } = {}) {
    if (!hasSimulation()) {
      return false;
    }

    // If user request and there's a pending request, cancel it
    if (priority === "user" && state.pendingRequest) {
      // Cancel in-flight auto-update request
      if (state.abortController) {
        state.abortController.abort();
        state.abortController = null;
      }
      // Abort the pending request's signal if it exists
      if (state.pendingRequest.abortSignal) {
        state.pendingRequest.abortSignal.abort();
      }
      // Remove the cancelled request from queue if it's there
      state.requestQueue = state.requestQueue.filter(
        (req) => req !== state.pendingRequest
      );
      // Clear pending request - the aborted request will clean up isAnalyzing in its finally block
      state.pendingRequest = null;
    }

    // If user request, cancel any queued auto-update requests
    if (priority === "user" && state.requestQueue.length > 0) {
      state.requestQueue.forEach((req) => {
        if (req.priority === "auto" && req.abortSignal) {
          req.abortSignal.abort();
        }
      });
      // Remove cancelled auto-update requests from queue
      state.requestQueue = state.requestQueue.filter(
        (req) => req.priority !== "auto"
      );
    }

    // If a request is in progress, queue this one
    if (state.isAnalyzing || state.pendingRequest) {
      const abortController = new AbortController();
      state.requestQueue.push({
        options: { resetHistory, silent, priority },
        abortSignal: abortController.signal,
        priority,
      });
      return true; // Return true to indicate request was queued
    }

    // No request in progress, proceed directly

    // Create abort controller for this request
    const abortController = new AbortController();
    if (priority === "auto") {
      state.abortController = abortController;
    }

    state.pendingRequest = {
      options: { resetHistory, silent, priority },
      abortSignal: abortController.signal,
      priority,
    };

    return await analyzeSimulationInternal(
      { resetHistory, silent, priority },
      abortController.signal
    );
  }

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
        return isLocalDocsHost() ? LOCAL_API_BASE_URL : DEFAULT_API_BASE_URL;
      }

      return normalizeBaseUrl(stored);
    } catch (error) {
      return isLocalDocsHost() ? LOCAL_API_BASE_URL : DEFAULT_API_BASE_URL;
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

  async function requestJson(path, options = {}, abortSignal = null) {
    // Merge abort signal into options if provided
    const fetchOptions = abortSignal
      ? { ...options, signal: abortSignal }
      : options;

    const response = await apiFetch(path, fetchOptions);
    const contentType = response.headers.get("content-type") ?? "";
    let payload = null;

    if (contentType.includes("application/json")) {
      try {
        payload = await response.json();
      } catch (error) {
        // If aborted, rethrow as AbortError
        if (error.name === "AbortError" || abortSignal?.aborted) {
          const abortError = new Error("Request aborted");
          abortError.name = "AbortError";
          throw abortError;
        }
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

  // DOM references
  const rootEl = document.querySelector(".aod-simulator");
  const yamlInput = document.getElementById("yaml-input");
  const loadBtn = document.getElementById("load-btn");
  const areaSelector = document.getElementById("area-selector");
  const errorMessage = document.getElementById("error-message");
  const simulationDisplay = document.getElementById("simulation-display");
  const areaName = document.getElementById("area-name");
  const probabilityValue = document.getElementById("probability-value");
  const probabilityFill = document.getElementById("probability-fill");
  const sensorsContainer = document.getElementById("sensors-container");
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

  function createMetricChip(label, value) {
    const chip = createElementWithClass("div", "sim-metric-chip");

    const valueEl = document.createElement("strong");
    setDisplayText(valueEl, value);

    const labelEl = document.createElement("span");
    labelEl.textContent = label;

    chip.append(valueEl, labelEl);
    return chip;
  }

  function createSensorMetricsRow(entity) {
    const hasContribution = typeof entity.contribution === "number";
    const hasLikelihood = typeof entity.likelihood === "number";
    const hasDecay =
      entity.decay && typeof entity.decay.decay_factor === "number";

    if (!hasContribution && !hasLikelihood && !hasDecay) {
      return null;
    }

    const row = createElementWithClass("div", "sim-card-metrics");

    if (hasContribution) {
      row.appendChild(
        createMetricChip(
          "Contribution",
          formatPercent(entity.contribution ?? 0)
        )
      );
    }

    if (hasLikelihood) {
      row.appendChild(
        createMetricChip("Likelihood", formatPercent(entity.likelihood ?? 0))
      );
    }

    if (hasDecay) {
      const decayFactor = entity.decay.decay_factor ?? 1;
      const isDecaying = entity.decay.is_decaying ?? false;
      const decayLabel = isDecaying ? "Decay Factor" : "Active";
      const decayValue = isDecaying ? formatPercent(decayFactor) : "100%";
      const decayChip = createMetricChip(decayLabel, decayValue);
      if (isDecaying) {
        decayChip.classList.add("sim-metric-chip--decaying");
      }
      row.appendChild(decayChip);
    }

    return row;
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

    // Display analysis error if present
    if (entity.analysis_error) {
      const errorBadge = document.createElement("div");
      errorBadge.className = "sim-error-badge";
      errorBadge.style.cssText =
        "display: inline-block; margin-left: 0.5rem; padding: 0.25rem 0.5rem; background-color: var(--md-typeset-a-color, #ff6b6b); color: white; border-radius: 0.25rem; font-size: 0.75rem; font-weight: bold;";
      errorBadge.textContent = "Error";
      errorBadge.title = `Analysis Error: ${entity.analysis_error}`;
      header.appendChild(errorBadge);
    }

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

    // Show analysis error in details if present
    if (entity.analysis_error) {
      const errorRow = createElementWithClass("div", "sim-card-row");
      errorRow.style.cssText =
        "color: var(--md-typeset-a-color, #ff6b6b); font-weight: 500;";
      const errorItem = document.createElement("div");
      errorItem.textContent = `Analysis Error: ${entity.analysis_error}`;
      errorRow.appendChild(errorItem);
      content.appendChild(errorRow);
    }

    const metricsRow = createSensorMetricsRow(entity);
    if (metricsRow) {
      content.appendChild(metricsRow);
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

  function applyEntityToggle(entity, targetAction) {
    const entityInput = getEntityInput(entity.entity_id);
    if (!entityInput) {
      throw new Error(`Entity input not found for ${entity.entity_id}`);
    }

    entityInput.previous_evidence = entity.evidence;
    entityInput.state = targetAction.state;

    const decay = {
      ...(entityInput.decay ?? {}),
      decay_start: new Date().toISOString(),
    };

    if (typeof targetAction.label === "string") {
      const label = targetAction.label.toLowerCase();
      if (label === "active") {
        decay.is_decaying = false;
      } else if (label === "inactive") {
        decay.is_decaying = true;
      }
    }

    entityInput.decay = decay;
  }

  function applyNumericUpdate(entity, numericValue) {
    const entityInput = getEntityInput(entity.entity_id);
    if (!entityInput) {
      throw new Error(`Entity input not found for ${entity.entity_id}`);
    }

    entityInput.previous_evidence = entity.evidence;
    entityInput.state = numericValue;
    entityInput.decay = {
      ...(entityInput.decay ?? {}),
      is_decaying: false,
      decay_start: new Date().toISOString(),
    };
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
      if (!hasSimulation()) {
        return;
      }

      const control = event.currentTarget;
      const isChecked = control.checked;
      const targetAction = isChecked ? onAction : offAction;

      control.disabled = true;

      try {
        applyEntityToggle(entity, targetAction);
        const success = await analyzeSimulation({ priority: "user" });
        if (success) {
          lastKnownChecked = isChecked;
        } else {
          control.checked = lastKnownChecked;
        }
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

      if (!hasSimulation()) {
        input.value = lastKnownValue;
        return;
      }

      setLoadingState(true);

      try {
        applyNumericUpdate(entity, numericValue);
        const success = await analyzeSimulation({ priority: "user" });
        if (success) {
          lastKnownValue = numericValue.toString();
          input.value = lastKnownValue;
        } else {
          input.value = lastKnownValue;
        }
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
      if (!hasSimulation()) {
        return;
      }

      try {
        applyEntityToggle(entity, action);
        const success = await analyzeSimulation({ priority: "user" });
        if (!success) {
          showError("Failed to update sensor state");
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        showError(message);
      }
    });

    return button;
  }

  function buildSensorActions(entity) {
    if (!hasSimulation()) {
      return null;
    }

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

  function renderSensors(result) {
    if (!sensorsContainer) {
      return;
    }

    const entities = Array.isArray(result.entities) ? result.entities : [];
    const entityDecay = result.entity_decay ?? {};

    sensorsContainer.innerHTML = "";
    sensorsContainer.classList.toggle("empty", entities.length === 0);

    const sorted = [...entities].sort((a, b) => {
      const aId = a.entity_id ?? "";
      const bId = b.entity_id ?? "";
      return aId.localeCompare(bId);
    });

    sorted.forEach((entity) => {
      // Attach decay data to entity for display
      const entityWithDecay = {
        ...entity,
        decay: entityDecay[entity.entity_id] ?? null,
      };
      sensorsContainer.appendChild(createSensorCard(entityWithDecay));
    });
  }

  function syncPriorControls(result) {
    if (!hasSimulation()) {
      return;
    }

    const requestArea = state.simulation.request.area ?? {};
    const priors = result.area?.priors ?? {};
    const weights = state.simulation.request.weights ?? {};

    const globalPriorValue = Number(requestArea.global_prior ?? 0);
    if (globalPriorSlider) {
      globalPriorSlider.value = globalPriorValue;
    }
    setRangeLabelWithValue(globalPriorSlider, globalPriorValue, "Global Prior");

    const timePriorValue = Number(requestArea.time_prior ?? 0);
    if (timePriorSlider) {
      timePriorSlider.value = timePriorValue;
    }
    setRangeLabelWithValue(timePriorSlider, timePriorValue, "Time Prior");

    const combinedPriorPercent = formatPercent(
      priors.combined ?? globalPriorValue
    );
    const finalPriorPercent = formatPercent(priors.final ?? globalPriorValue);

    if (combinedPriorInput) {
      combinedPriorInput.value = combinedPriorPercent;
    }
    if (finalPriorInput) {
      finalPriorInput.value = finalPriorPercent;
    }

    const halfLifeValue = Math.round(
      result.area?.half_life ?? requestArea.half_life ?? 0
    );
    if (halfLifeInput) {
      halfLifeInput.value = `${halfLifeValue}s`;
    }

    if (purposeSelect) {
      purposeSelect.value = requestArea.purpose ?? "";
    }

    Object.entries(weightControls).forEach(([key, control]) => {
      // Only update slider if weight exists in request weights
      // This preserves manually set weights even if server doesn't return them
      if (key in weights && control.slider) {
        const value = Number(weights[key]);
        control.slider.value = value;
        setRangeLabelWithValue(
          control.slider,
          value,
          control.baseLabel,
          formatDecimal
        );
      } else if (control.slider) {
        // If weight not in request, just update label with current slider value
        // Don't change the slider value itself to preserve user input
        setRangeLabelWithValue(
          control.slider,
          Number(control.slider.value),
          control.baseLabel,
          formatDecimal
        );
      }
    });
  }

  function renderSimulationResult(result) {
    if (!simulationDisplay || !result) {
      return;
    }

    simulationDisplay.classList.remove("hidden");

    const area = result.area ?? {};
    if (areaName) {
      areaName.textContent = area.name ?? "Area";
    }

    setProbability(result.probability ?? 0);
    renderSensors(result);
    syncPriorControls(result);
  }

  function applySimulationResult(result, { resetHistory = false } = {}) {
    if (resetHistory) {
      state.probabilityHistory = [];
      initChart();
    }

    renderSimulationResult(result);
    hideError();
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

  function stopAutoUpdate() {
    if (state.autoUpdateInterval) {
      clearInterval(state.autoUpdateInterval);
      state.autoUpdateInterval = null;
    }
  }

  function startAutoUpdate() {
    stopAutoUpdate();

    state.autoUpdateInterval = setInterval(async () => {
      if (!hasSimulation()) {
        return;
      }

      // Skip auto-update if user request is queued or in progress
      if (
        state.isAnalyzing &&
        state.pendingRequest &&
        state.pendingRequest.priority === "user"
      ) {
        return;
      }

      // Skip if there are user requests in queue
      const hasUserRequestInQueue = state.requestQueue.some(
        (req) => req.priority === "user"
      );
      if (hasUserRequestInQueue) {
        return;
      }

      await analyzeSimulation({ silent: true, priority: "auto" });
    }, 1000);
  }

  async function loadPurposes() {
    if (!purposeSelect) {
      return;
    }

    try {
      const payload = await requestJson("/api/get-purposes");
      state.purposes = Array.isArray(payload.purposes) ? payload.purposes : [];

      purposeSelect.innerHTML = "";
      state.purposes.forEach((purpose) => {
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
      // Store YAML data for area switching
      state.yamlData = yamlText;

      const payload = await requestJson("/api/load", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ yaml: yamlText }),
      });

      // Handle multiple areas if available
      if (payload.available_areas && Array.isArray(payload.available_areas)) {
        state.availableAreas = payload.available_areas;
        state.selectedAreaName =
          payload.selected_area_name || payload.available_areas[0] || null;
        updateAreaSelector();
      } else {
        // Old format: single area
        state.availableAreas = [];
        state.selectedAreaName = null;
        if (areaSelector) {
          areaSelector.disabled = true;
          areaSelector.innerHTML = "";
          const option = document.createElement("sl-option");
          option.value = "";
          option.textContent = "Single area (old format)";
          areaSelector.appendChild(option);
        }
      }

      setSimulation(payload.simulation, payload.result, { resetHistory: true });
      startAutoUpdate();
      updateApiStatus("online", "API Online");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      showError(`Error loading simulation: ${message}`);
      updateApiStatus("offline", "API Offline");
    }
  }

  function updateAreaSelector() {
    if (!areaSelector) {
      return;
    }

    areaSelector.innerHTML = "";

    if (state.availableAreas.length === 0) {
      areaSelector.disabled = true;
      const option = document.createElement("sl-option");
      option.value = "";
      option.textContent = "No areas available";
      areaSelector.appendChild(option);
      return;
    }

    areaSelector.disabled = false;
    state.availableAreas.forEach((areaName) => {
      const option = document.createElement("sl-option");
      option.value = areaName;
      option.textContent = areaName;
      areaSelector.appendChild(option);
    });

    if (state.selectedAreaName) {
      areaSelector.value = state.selectedAreaName;
    }
  }

  async function handleAreaChange(event) {
    const newAreaName = getEventValue(event);
    if (!newAreaName || newAreaName === state.selectedAreaName) {
      return;
    }

    if (!state.yamlData) {
      showError("No YAML data available. Please load simulation first.");
      return;
    }

    try {
      state.isAnalyzing = true;
      updateApiStatus("checking", "Loading area...");

      const payload = await requestJson("/api/load", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          yaml: state.yamlData,
          area_name: newAreaName,
        }),
      });

      state.selectedAreaName = newAreaName;
      setSimulation(payload.simulation, payload.result, { resetHistory: true });
      updateApiStatus("online", "API Online");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      showError(`Error switching area: ${message}`);
      updateApiStatus("offline", "API Offline");
      // Revert selector to previous value
      if (areaSelector && state.selectedAreaName) {
        areaSelector.value = state.selectedAreaName;
      }
    } finally {
      state.isAnalyzing = false;
    }
  }

  async function handlePurposeChange(event) {
    if (!hasSimulation()) {
      return;
    }

    const value = getEventValue(event) ?? "";
    const requestArea = state.simulation.request.area ?? {};
    requestArea.purpose = value;

    const matchedPurpose = state.purposes.find(
      (purpose) => purpose.value === value
    );
    if (matchedPurpose?.half_life !== undefined) {
      requestArea.half_life = matchedPurpose.half_life;
    }

    await analyzeSimulation({ resetHistory: true, priority: "user" });
  }

  function initPriorControls() {
    if (globalPriorSlider) {
      const clampPrior = (value) => Math.min(Math.max(value, 0), 1);

      const applyValue = (value, { runAnalysis = false } = {}) => {
        const numericValue = clampPrior(Number.parseFloat(value ?? "0"));
        if (hasSimulation()) {
          state.simulation.request.area.global_prior = numericValue;
        }
        globalPriorSlider.value = numericValue;
        setRangeLabelWithValue(globalPriorSlider, numericValue, "Global Prior");
        if (runAnalysis) {
          analyzeSimulation({ resetHistory: false, priority: "user" });
        }
      };

      globalPriorSlider.addEventListener("sl-input", (event) => {
        applyValue(getEventValue(event));
      });

      globalPriorSlider.addEventListener("sl-change", (event) => {
        applyValue(getEventValue(event), { runAnalysis: true });
      });
    }

    if (timePriorSlider) {
      const clampPrior = (value) => Math.min(Math.max(value, 0), 1);

      const applyValue = (value, { runAnalysis = false } = {}) => {
        const numericValue = clampPrior(Number.parseFloat(value ?? "0"));
        if (hasSimulation()) {
          state.simulation.request.area.time_prior = numericValue;
        }
        timePriorSlider.value = numericValue;
        setRangeLabelWithValue(timePriorSlider, numericValue, "Time Prior");
        if (runAnalysis) {
          analyzeSimulation({ resetHistory: false, priority: "user" });
        }
      };

      timePriorSlider.addEventListener("sl-input", (event) => {
        applyValue(getEventValue(event));
      });

      timePriorSlider.addEventListener("sl-change", (event) => {
        applyValue(getEventValue(event), { runAnalysis: true });
      });
    }
  }

  function initWeightControls() {
    Object.entries(weightControls).forEach(([type, control]) => {
      if (!control.slider) {
        return;
      }

      const clampWeight = (value) => Math.min(Math.max(value, 0.01), 1);

      const applyValue = (value, { runAnalysis = false } = {}) => {
        const numericValue = clampWeight(Number.parseFloat(value ?? "0"));
        if (hasSimulation()) {
          state.simulation.request.weights ??= {};
          state.simulation.request.weights[type] = numericValue;
        }
        control.slider.value = numericValue;
        setRangeLabelWithValue(
          control.slider,
          numericValue,
          control.baseLabel,
          formatDecimal
        );
        if (runAnalysis) {
          analyzeSimulation({ resetHistory: false, priority: "user" });
        }
      };

      control.slider.addEventListener("sl-input", (event) => {
        applyValue(getEventValue(event));
      });

      control.slider.addEventListener("sl-change", (event) => {
        applyValue(getEventValue(event), { runAnalysis: true });
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

  function initHelpDialog() {
    const helpDialogBtn = document.getElementById("help-dialog-btn");
    const helpDialog = document.getElementById("help-dialog");

    if (helpDialogBtn && helpDialog) {
      helpDialogBtn.addEventListener("click", () => {
        helpDialog.show();
      });
    }
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
    initHelpDialog();
    loadPurposes();

    if (loadBtn) {
      loadBtn.addEventListener("click", handleLoadSimulation);
    }

    if (areaSelector) {
      areaSelector.addEventListener("sl-change", handleAreaChange);
    }

    if (purposeSelect) {
      purposeSelect.addEventListener("sl-change", handlePurposeChange);
    }
  }

  init();
})();
