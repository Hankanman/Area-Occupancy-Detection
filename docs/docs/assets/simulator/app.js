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

  function formatPercent(value, digits = 2) {
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
      const response = await apiFetch("/api/get-purposes");
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

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
      const card = document.createElement("div");
      card.className = "md-card sim-card entity";

      const actionsContainer = document.createElement("div");
      actionsContainer.className = "sim-card-actions";

      const contentContainer = document.createElement("div");
      contentContainer.className = "sim-card-content";

      const header = document.createElement("div");
      header.className = "sim-card-header";

      const title = document.createElement("h3");
      title.textContent = entity.name;
      header.appendChild(title);

      const stateEl = document.createElement("div");
      stateEl.textContent = `State: ${entity.state_display}`;
      header.appendChild(stateEl);

      contentContainer.appendChild(header);

      if (entity.details) {
        const detailsRow = document.createElement("div");
        detailsRow.className = "sim-card-row";

        const segments = Array.isArray(entity.details)
          ? entity.details
          : String(entity.details)
              .split(/(?:\s*•\s*|\n+)/)
              .map((segment) => segment.trim())
              .filter(Boolean);

        if (segments.length === 0) {
          segments.push(String(entity.details));
        }

        segments.forEach((segment) => {
          const detailItem = document.createElement("div");
          detailItem.textContent = segment;
          detailsRow.appendChild(detailItem);
        });

        contentContainer.appendChild(detailsRow);
      }

      let hasActions = false;

      const supportsBinaryToggle =
        !entity.is_numeric &&
        entity.actions.length >= 2 &&
        entity.actions.every((action) => typeof action.state === "string");

      if (supportsBinaryToggle) {
        const onAction =
          entity.actions.find(
            (action) => action.state?.toLowerCase() === "on"
          ) ?? entity.actions.find((action) => action.active);
        const offAction = entity.actions.find((action) => action !== onAction);

        const toggle = document.createElement("sl-switch");
        toggle.textContent = onAction?.label ?? "Active";
        toggle.checked = Boolean(onAction?.active);
        toggle.disabled = !onAction || !offAction;

        let lastKnownChecked = toggle.checked;

        toggle.addEventListener("sl-change", async (event) => {
          if (!onAction || !offAction) {
            return;
          }

          const control = event.currentTarget;
          const isChecked = control.checked;
          const targetAction = isChecked ? onAction : offAction;

          control.disabled = true;

          try {
            const response = await apiFetch("/api/toggle", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                entity_id: entity.entity_id,
                state: targetAction.state,
              }),
            });

            const payload = await response.json();

            if (!response.ok) {
              throw new Error(payload.error || "Failed to toggle sensor");
            }

            state.simulationData = payload;
            renderSimulation(payload);
            lastKnownChecked = isChecked;
          } catch (error) {
            showError(error.message);
            control.checked = lastKnownChecked;
          } finally {
            control.disabled = false;
          }
        });

        actionsContainer.appendChild(toggle);
        hasActions = true;
      } else if (entity.is_numeric) {
        const currentValueAction = entity.actions.find(
          (action) => action.active
        );
        const input = document.createElement("sl-input");
        input.type = "number";
        input.size = "medium";
        input.value =
          currentValueAction?.value?.toString() ?? entity.state ?? "";
        input.placeholder = "--";
        input.step = "any";
        input.inputMode = "decimal";
        input.disabled = false;

        let lastKnownValue = input.value;
        let isUpdating = false;

        const setLoadingState = (loading) => {
          isUpdating = loading;
          input.loading = loading;
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
            const response = await apiFetch("/api/update", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                entity_id: entity.entity_id,
                value: numericValue,
              }),
            });

            const payload = await response.json();

            if (!response.ok) {
              throw new Error(payload.error || "Failed to update sensor");
            }

            lastKnownValue = numericValue.toString();
            input.value = lastKnownValue;
            state.simulationData = payload;
            renderSimulation(payload);
            hideError();
          } catch (error) {
            showError(error.message ?? error);
            input.value = lastKnownValue;
          } finally {
            setLoadingState(false);
          }
        };

        input.addEventListener("sl-change", (event) => {
          if (isUpdating) {
            return;
          }

          const target = event.currentTarget;
          submitValue(target.value);
        });

        input.addEventListener("keydown", (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            input.blur();
          }
        });

        actionsContainer.appendChild(input);
        hasActions = true;
      } else if (entity.actions.length > 0) {
        entity.actions.forEach((action) => {
          const button = document.createElement("sl-button");
          button.variant = action.active ? "primary" : "default";
          button.size = "small";
          button.textContent = action.label;

          button.addEventListener("click", async () => {
            try {
              const response = await apiFetch("/api/toggle", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({
                  entity_id: entity.entity_id,
                  state: action.state,
                }),
              });

              const payload = await response.json();

              if (!response.ok) {
                throw new Error(payload.error || "Failed to toggle sensor");
              }

              state.simulationData = payload;
              renderSimulation(payload);
            } catch (error) {
              showError(error.message);
            }
          });

          actionsContainer.appendChild(button);
        });

        hasActions = true;
      }

      if (hasActions) {
        card.appendChild(actionsContainer);
      }

      card.appendChild(contentContainer);
      sensorsContainer.appendChild(card);
    });
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
        const response = await apiFetch("/api/tick", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({}),
        });

        const payload = await response.json();

        if (!response.ok) {
          throw new Error(payload.error || "Failed to update simulation");
        }

        state.simulationData = payload;
        renderSimulation(payload);
      } catch (error) {
        // Silent failure keeps UI responsive, show status only if API offline
        updateApiStatus("offline", "API Offline");
      }
    }, 5000);
  }

  async function loadPurposes() {
    if (!purposeSelect) {
      return;
    }

    try {
      const response = await apiFetch("/api/get-purposes");
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.error || "Failed to load purposes");
      }

      purposeSelect.innerHTML = "";
      payload.purposes.forEach((purpose) => {
        const option = document.createElement("sl-option");
        option.value = purpose.value;
        option.textContent = purpose.label;
        purposeSelect.appendChild(option);
      });

      updateApiStatus("online", "API Online");
    } catch (error) {
      showError(error.message);
      updateApiStatus("offline", "API Offline");
    }
  }

  async function handleLoadSimulation() {
    const yamlText = (yamlInput?.value ?? "").trim();

    if (!yamlText) {
      showError("Please paste YAML analysis output");
      return;
    }

    try {
      const response = await apiFetch("/api/load", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ yaml: yamlText }),
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.error || "Failed to load simulation");
      }

      state.simulationData = payload;
      state.probabilityHistory = [];
      initChart();
      renderSimulation(payload);
      hideError();

      startAutoUpdate();
      updateApiStatus("online", "API Online");
    } catch (error) {
      showError(`Error loading simulation: ${error.message}`);
      updateApiStatus("offline", "API Offline");
    }
  }

  function addPriorListeners(control) {
    const slider = control.slider;
    const key = control.key;
    const baseLabel = control.baseLabel;

    if (!slider) {
      return;
    }

    const updateValue = (value) => {
      const numericValue = Number.parseFloat(value);
      const clamped = Math.min(
        Math.max(Number.isNaN(numericValue) ? 0 : numericValue, 0),
        1
      );
      slider.value = clamped;
      setRangeLabelWithValue(slider, clamped, baseLabel);
      return clamped;
    };

    const handleUpdate = async (value) => {
      const clamped = updateValue(value);

      if (!state.simulationData) {
        return;
      }

      try {
        const response = await apiFetch("/api/update-priors", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ [key]: clamped }),
        });

        const payload = await response.json();

        if (!response.ok) {
          throw new Error(payload.error || "Failed to update priors");
        }

        state.simulationData = payload;
        renderSimulation(payload);
      } catch (error) {
        showError(error.message);
        updateApiStatus("offline", "API Offline");
      }
    };

    slider.addEventListener("sl-input", (event) => {
      updateValue(getEventValue(event));
    });

    slider.addEventListener("sl-change", (event) => {
      handleUpdate(getEventValue(event));
    });

    updateValue(slider.value);
  }

  function addWeightListeners(type, control) {
    const slider = control.slider;
    const baseLabel = control.baseLabel;

    if (!slider) {
      return;
    }

    const updateValue = (value) => {
      const numericValue = Number.parseFloat(value);
      const clamped = Math.min(
        Math.max(Number.isNaN(numericValue) ? 0.01 : numericValue, 0.01),
        1
      );
      slider.value = clamped;
      setRangeLabelWithValue(slider, clamped, baseLabel, formatDecimal);
      return clamped;
    };

    const handleUpdate = async (value) => {
      const clamped = updateValue(value);

      if (!state.simulationData) {
        return;
      }

      try {
        const response = await apiFetch("/api/update-weights", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ weight_type: type, value: clamped }),
        });

        const payload = await response.json();

        if (!response.ok) {
          throw new Error(payload.error || "Failed to update weights");
        }

        state.simulationData = payload;
        renderSimulation(payload);
      } catch (error) {
        showError(error.message);
        updateApiStatus("offline", "API Offline");
      }
    };

    slider.addEventListener("sl-input", (event) => {
      updateValue(getEventValue(event));
    });

    slider.addEventListener("sl-change", (event) => {
      handleUpdate(getEventValue(event));
    });

    updateValue(slider.value);
  }

  async function handlePurposeChange(event) {
    const value = getEventValue(event) ?? "";

    if (!state.simulationData) {
      return;
    }

    try {
      const response = await apiFetch("/api/update-purpose", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ purpose: value }),
      });

      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.error || "Failed to update purpose");
      }

      state.simulationData = payload;
      renderSimulation(payload);
    } catch (error) {
      showError(error.message);
      updateApiStatus("offline", "API Offline");
    }
  }

  function initPriorControls() {
    addPriorListeners({
      slider: globalPriorSlider,
      key: "global_prior",
      baseLabel: "Global Prior",
    });

    addPriorListeners({
      slider: timePriorSlider,
      key: "time_prior",
      baseLabel: "Time Prior",
    });
  }

  function initWeightControls() {
    Object.entries(weightControls).forEach(([type, control]) => {
      addWeightListeners(type, control);
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
