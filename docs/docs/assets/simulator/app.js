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
    lastRequestHash: null, // Hash of last request sent to API for change detection
    lastDecayFactors: {}, // Store last decay factors from API for local decay calculation
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

    const currentEntities = state.simulation.request.entities ?? [];
    const newEntityInputs = deepClone(entityInputs ?? []);

    // Create a lookup map of current entity inputs by entity_id
    const currentEntityMap = new Map();
    currentEntities.forEach((entity) => {
      if (entity?.entity_id) {
        currentEntityMap.set(entity.entity_id, entity);
      }
    });

    // Create a map of new entity inputs by entity_id
    const newEntityMap = new Map();
    newEntityInputs.forEach((entity) => {
      if (entity?.entity_id) {
        newEntityMap.set(entity.entity_id, entity);
      }
    });

    // Merge entities: preserve user-modified states, update other fields
    const mergedEntities = [];

    // Process entities from new API response
    newEntityInputs.forEach((newEntity) => {
      if (!newEntity?.entity_id) {
        return;
      }

      const currentEntity = currentEntityMap.get(newEntity.entity_id);
      if (currentEntity) {
        // Entity exists in current: merge, preserving state from current
        const mergedEntity = {
          ...newEntity, // Start with new entity (has updated decay, etc.)
          state: currentEntity.state, // Preserve user-modified state
          // Preserve previous_evidence if it was set by user action
          previous_evidence:
            currentEntity.previous_evidence !== undefined
              ? currentEntity.previous_evidence
              : newEntity.previous_evidence,
        };
        mergedEntities.push(mergedEntity);
      } else {
        // New entity: add as-is
        mergedEntities.push(newEntity);
      }
    });

    // Preserve entities that exist in current but not in new (edge case)
    currentEntities.forEach((currentEntity) => {
      if (
        currentEntity?.entity_id &&
        !newEntityMap.has(currentEntity.entity_id)
      ) {
        mergedEntities.push(deepClone(currentEntity));
      }
    });

    state.simulation.request.entities = mergedEntities;
    state.simulation.entityLookup = buildEntityLookup(mergedEntities);

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

    // Determine if this is a user-initiated request or auto-update
    // User requests should always update sensors, auto-updates can skip if input is focused
    // If pendingRequest is null (e.g., initial load), treat as user request to ensure sensors render
    const isUserRequest =
      state.pendingRequest?.priority === "user" || !state.pendingRequest;
    const skipSensors = !isUserRequest && isNumericInputFocused();

    applySimulationResult(result, { resetHistory, skipSensors });
  }

  function prepareAnalyzePayload() {
    if (!hasSimulation()) {
      return {};
    }
    return deepClone(state.simulation.request);
  }

  function getRequestHash() {
    if (!hasSimulation()) {
      return null;
    }

    const request = state.simulation.request;
    // Create a hashable object excluding decay_start times
    const hashableData = {
      area: {
        name: request.area?.name,
        purpose: request.area?.purpose,
        global_prior: request.area?.global_prior,
        time_prior: request.area?.time_prior,
        half_life: request.area?.half_life,
        threshold: request.area?.threshold,
      },
      weights: request.weights ?? {},
      entities: (request.entities ?? []).map((entity) => ({
        entity_id: entity.entity_id,
        type: entity.type,
        state: entity.state,
        prob_given_true: entity.prob_given_true,
        prob_given_false: entity.prob_given_false,
        weight: entity.weight,
        previous_evidence: entity.previous_evidence,
        // Include decay state but not decay_start (it changes over time)
        decay: {
          is_decaying: entity.decay?.is_decaying ?? false,
        },
      })),
    };

    // Create a simple hash from the JSON string
    const jsonString = JSON.stringify(hashableData);
    // Simple hash function (not cryptographically secure, but sufficient for change detection)
    let hash = 0;
    for (let i = 0; i < jsonString.length; i++) {
      const char = jsonString.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString();
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
      // Store request hash after successful API call
      state.lastRequestHash = getRequestHash();
      // Store decay factors from API result for local decay calculation
      if (result.entity_decay) {
        state.lastDecayFactors = {};
        Object.keys(result.entity_decay).forEach((entityId) => {
          state.lastDecayFactors[entityId] =
            result.entity_decay[entityId].decay_factor ?? 1.0;
        });
      }
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
    // Add entity_id as data attribute for focus restoration
    if (entity?.entity_id) {
      card.setAttribute("data-entity-id", entity.entity_id);
    }
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

    // Preserve focus and value of any currently focused numeric input
    const activeElement = document.activeElement;
    let focusedEntityId = null;
    let focusedInputValue = null;
    let focusedInputSelectionStart = null;
    let focusedInputSelectionEnd = null;

    if (activeElement && sensorsContainer.contains(activeElement)) {
      const card = activeElement.closest(".sim-card");
      if (card) {
        const entityIdAttr = card.getAttribute("data-entity-id");
        if (entityIdAttr) {
          focusedEntityId = entityIdAttr;
          // If it's a numeric input, preserve its value and selection
          if (
            activeElement.tagName === "SL-INPUT" &&
            activeElement.type === "number"
          ) {
            focusedInputValue = activeElement.value;
            // Try to get selection from shadow DOM
            if (activeElement.shadowRoot) {
              const input = activeElement.shadowRoot.querySelector("input");
              if (input) {
                focusedInputSelectionStart = input.selectionStart;
                focusedInputSelectionEnd = input.selectionEnd;
              }
            }
          }
        }
      }
    }

    sensorsContainer.innerHTML = "";
    sensorsContainer.classList.toggle("empty", entities.length === 0);

    // Get request entities to merge user-modified states
    const requestEntities = state.simulation?.request?.entities ?? [];
    const requestEntityMap = new Map();
    requestEntities.forEach((reqEntity) => {
      if (reqEntity?.entity_id) {
        requestEntityMap.set(reqEntity.entity_id, reqEntity);
      }
    });

    const sorted = [...entities].sort((a, b) => {
      const aId = a.entity_id ?? "";
      const bId = b.entity_id ?? "";
      return aId.localeCompare(bId);
    });

    sorted.forEach((entity) => {
      // Merge state from request entity if it exists (preserves user modifications)
      const requestEntity = requestEntityMap.get(entity.entity_id);
      let entityToRender = { ...entity };

      if (requestEntity && requestEntity.state !== undefined) {
        // Update state from request (user's modification)
        // However, if this input is currently focused and being edited,
        // preserve the input's current value instead
        if (
          focusedEntityId === entity.entity_id &&
          focusedInputValue !== null &&
          entityToRender.is_numeric
        ) {
          // User is typing in this input, preserve their current input value
          entityToRender.state = focusedInputValue;
        } else {
          entityToRender.state = requestEntity.state;
        }

        // Update state_display to match the new state
        // Use the actual state value (which may be from focused input)
        const stateValue = entityToRender.state;
        if (entityToRender.is_numeric) {
          // For numeric sensors, format as number
          try {
            const numValue = Number.parseFloat(stateValue);
            if (Number.isFinite(numValue)) {
              entityToRender.state_display = numValue.toFixed(2);
            } else {
              entityToRender.state_display = String(stateValue);
            }
          } catch {
            entityToRender.state_display = String(stateValue);
          }
        } else {
          // For binary sensors, use the state as-is
          entityToRender.state_display = String(stateValue);
        }

        // Determine if the user's state represents an active or inactive entity
        // The user's state might be "on"/"off" (from toggle) or the actual entity state
        // We need to determine evidence based on the state value
        // Use the actual state value (which may be from focused input)
        const userStateStr = String(stateValue).toLowerCase();

        // Check if state represents active (common active states)
        const isActiveState =
          userStateStr === "on" ||
          userStateStr === "open" ||
          userStateStr === "true" ||
          userStateStr === "1";

        // Check if state represents inactive (common inactive states)
        const isInactiveState =
          userStateStr === "off" ||
          userStateStr === "closed" ||
          userStateStr === "false" ||
          userStateStr === "0";

        // Determine evidence: if we can clearly determine active/inactive, use that
        // Otherwise, preserve the original entity's evidence if state matches original
        let newEvidence = entityToRender.evidence; // Default to original
        if (isActiveState) {
          newEvidence = true;
        } else if (isInactiveState) {
          newEvidence = false;
        } else {
          // For unknown states, check if it matches the original entity's state
          const originalStateStr = String(entity.state || "").toLowerCase();
          if (userStateStr === originalStateStr) {
            // State matches original, preserve original evidence
            newEvidence = entity.evidence;
          }
          // Otherwise keep the default (original evidence)
        }

        entityToRender.evidence = newEvidence;

        // Update actions to reflect the user's state
        // Actions always have state "on" (active) and "off" (inactive)
        if (Array.isArray(entityToRender.actions)) {
          entityToRender.actions = entityToRender.actions.map((action) => {
            if (
              action.state === "on" ||
              action.label?.toLowerCase() === "active"
            ) {
              return { ...action, active: newEvidence === true };
            } else if (
              action.state === "off" ||
              action.label?.toLowerCase() === "inactive"
            ) {
              return { ...action, active: newEvidence === false };
            }
            return action;
          });
        }
      }

      // Attach decay data to entity for display
      const entityWithDecay = {
        ...entityToRender,
        decay: entityDecay[entity.entity_id] ?? null,
      };
      sensorsContainer.appendChild(createSensorCard(entityWithDecay));
    });

    // Restore focus to the previously focused input if it exists
    if (focusedEntityId) {
      const card = sensorsContainer.querySelector(
        `[data-entity-id="${focusedEntityId}"]`
      );
      if (card) {
        const input = card.querySelector('sl-input[type="number"]');
        if (input) {
          // Restore the value if it was being edited
          if (focusedInputValue !== null) {
            input.value = focusedInputValue;
          }
          // Use requestAnimationFrame to ensure DOM is ready
          requestAnimationFrame(() => {
            input.focus();
            // Try to restore selection in shadow DOM
            if (
              focusedInputSelectionStart !== null &&
              focusedInputSelectionEnd !== null &&
              input.shadowRoot
            ) {
              const shadowInput = input.shadowRoot.querySelector("input");
              if (shadowInput) {
                shadowInput.setSelectionRange(
                  focusedInputSelectionStart,
                  focusedInputSelectionEnd
                );
              }
            }
          });
        }
      }
    }
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

  function isNumericInputFocused() {
    const activeElement = document.activeElement;
    if (!activeElement) {
      return false;
    }

    // Check if active element is a sl-input with type="number"
    if (
      activeElement.tagName === "SL-INPUT" &&
      activeElement.type === "number"
    ) {
      // Verify it's within the sensors container
      if (sensorsContainer && sensorsContainer.contains(activeElement)) {
        return true;
      }
    }

    // Also check if focus is within a shadow DOM input
    if (activeElement.shadowRoot) {
      const shadowInput = activeElement.shadowRoot.querySelector(
        'input[type="number"]'
      );
      if (
        shadowInput &&
        sensorsContainer &&
        sensorsContainer.contains(activeElement)
      ) {
        return true;
      }
    }

    return false;
  }

  function renderSimulationResult(result, { skipSensors = false } = {}) {
    if (!simulationDisplay || !result) {
      return;
    }

    simulationDisplay.classList.remove("hidden");

    const area = result.area ?? {};
    if (areaName) {
      areaName.textContent = area.name ?? "Area";
    }

    setProbability(result.probability ?? 0);

    // Skip rendering sensors if a numeric input is focused (user is typing)
    // or if explicitly requested to skip
    if (!skipSensors && !isNumericInputFocused()) {
      renderSensors(result);
    }

    syncPriorControls(result);
  }

  function applySimulationResult(
    result,
    { resetHistory = false, skipSensors = false } = {}
  ) {
    if (resetHistory) {
      state.probabilityHistory = [];
      initChart();
    }

    renderSimulationResult(result, { skipSensors });
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

  function calculateLocalDecay() {
    if (!hasSimulation()) {
      return {};
    }

    const entities = state.simulation.request.entities ?? [];
    const areaHalfLife = state.simulation.request.area?.half_life ?? 300; // Default 5 minutes
    const now = new Date();
    const decayUpdates = {};

    entities.forEach((entity) => {
      if (!entity.decay?.is_decaying) {
        return;
      }

      const decayStart = entity.decay.decay_start;
      if (!decayStart) {
        return;
      }

      // Parse decay_start (ISO string) and calculate age in seconds
      const decayStartTime = new Date(decayStart);
      const ageSeconds = (now - decayStartTime) / 1000;

      if (ageSeconds <= 0) {
        return;
      }

      // Calculate decay_factor: 0.5^(age / half_life)
      const decayFactor = Math.pow(0.5, ageSeconds / areaHalfLife);

      if (decayFactor < 0.05) {
        // Practical zero - decay is complete
        decayUpdates[entity.entity_id] = {
          is_decaying: false,
          decay_factor: 0.0,
        };
      } else {
        decayUpdates[entity.entity_id] = {
          is_decaying: true,
          decay_factor: decayFactor,
        };
      }
    });

    return decayUpdates;
  }

  function updateProbabilityWithLocalDecay() {
    if (!hasSimulation() || !state.simulation.result) {
      return;
    }

    const decayUpdates = calculateLocalDecay();
    const result = state.simulation.result;

    // Update entity_decay in result
    if (!result.entity_decay) {
      result.entity_decay = {};
    }

    Object.keys(decayUpdates).forEach((entityId) => {
      const update = decayUpdates[entityId];
      if (result.entity_decay[entityId]) {
        result.entity_decay[entityId] = {
          ...result.entity_decay[entityId],
          ...update,
        };
      } else {
        result.entity_decay[entityId] = {
          is_decaying: update.is_decaying,
          decay_factor: update.decay_factor,
          decay_start: state.simulation.request.entities.find(
            (e) => e.entity_id === entityId
          )?.decay?.decay_start,
          evidence: false,
        };
      }
    });

    // Update entities in result to show updated decay factors
    if (Array.isArray(result.entities)) {
      result.entities.forEach((entity) => {
        const decayUpdate = decayUpdates[entity.entity_id];
        if (decayUpdate && entity.decay) {
          entity.decay = {
            ...entity.decay,
            ...decayUpdate,
          };
        }
      });
    }

    // Estimate probability change based on decay
    // When entities decay, their influence moves toward neutral (0.5),
    // which tends to move the probability toward the prior
    const lastProbability = result.probability ?? 0;
    const areaPriors = result.area?.priors ?? {};
    const prior = areaPriors.final ?? areaPriors.combined ?? 0.5;

    // Calculate average decay impact
    // Entities with higher weight and contribution have more impact when they decay
    let totalDecayImpact = 0;
    let totalWeight = 0;

    if (Array.isArray(result.entities)) {
      result.entities.forEach((entity) => {
        const decayUpdate = decayUpdates[entity.entity_id];
        if (decayUpdate) {
          // Use stored last decay factor, or current if not stored
          const originalDecayFactor =
            state.lastDecayFactors[entity.entity_id] ??
            entity.decay?.decay_factor ??
            1.0;
          const newDecayFactor = decayUpdate.decay_factor ?? 0.0;
          const decayChange = originalDecayFactor - newDecayFactor;

          if (decayChange > 0) {
            // Entity is decaying - calculate its impact
            const entityWeight = entity.weight ?? 0;
            const contribution = Math.abs(entity.contribution ?? 0);
            // Impact is proportional to weight and contribution
            // Decay reduces influence, so we weight by how much it's decaying
            const impact = decayChange * entityWeight * contribution;
            totalDecayImpact += impact;
            totalWeight += entityWeight * contribution;
          }
        }
      });
    }

    // Estimate new probability
    // As decay progresses, probability moves toward prior
    // The amount of movement depends on the decay impact
    let estimatedProbability = lastProbability;
    if (totalWeight > 0 && totalDecayImpact > 0) {
      // Normalize decay impact (0 to 1 scale)
      const normalizedImpact = Math.min(1.0, totalDecayImpact / totalWeight);
      // Interpolate between last probability and prior based on decay impact
      // More decay = closer to prior
      estimatedProbability =
        lastProbability * (1 - normalizedImpact * 0.3) +
        prior * (normalizedImpact * 0.3);
      // Clamp to valid range
      estimatedProbability = Math.max(0, Math.min(1, estimatedProbability));
    }

    // Update probability display and chart
    setProbability(estimatedProbability);
    // Note: We don't update lastDecayFactors here - they should only be updated
    // after API calls to preserve the baseline for calculating decay changes

    // Re-render sensors to show updated decay factors (but skip if input is focused)
    if (!isNumericInputFocused()) {
      renderSensors(result);
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

      // Check if inputs have changed since last API call
      const currentHash = getRequestHash();
      if (
        currentHash === state.lastRequestHash &&
        state.lastRequestHash !== null
      ) {
        // No changes - simulate decay locally instead of calling API
        updateProbabilityWithLocalDecay();
        return;
      }

      // Inputs have changed or this is the first call - call API
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
      // Reset request hash on new simulation load to ensure first auto-update calls API
      state.lastRequestHash = null;
      // Reset decay factors tracking
      state.lastDecayFactors = {};
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
      // Reset request hash when switching areas
      state.lastRequestHash = null;
      // Reset decay factors tracking
      state.lastDecayFactors = {};
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
