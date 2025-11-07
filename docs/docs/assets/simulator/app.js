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
  const sensorsList = document.getElementById("sensors-list");
  const breakdownList = document.getElementById("breakdown-list");
  const globalPriorSlider = document.getElementById("global-prior-slider");
  const globalPriorInput = document.getElementById("global-prior-input");
  const globalPriorDisplay = document.getElementById("global-prior-display");
  const timePriorSlider = document.getElementById("time-prior-slider");
  const timePriorInput = document.getElementById("time-prior-input");
  const timePriorDisplay = document.getElementById("time-prior-display");
  const combinedPriorDisplay = document.getElementById(
    "combined-prior-display"
  );
  const finalPriorDisplay = document.getElementById("final-prior-display");
  const purposeSelect = document.getElementById("purpose-select");
  const halfLifeDisplay = document.getElementById("half-life-display");
  const apiBaseInput = document.getElementById("api-base-input");
  const apiBaseSave = document.getElementById("api-base-save");
  const apiBaseReset = document.getElementById("api-base-reset");
  const apiBaseDisplay = document.getElementById("api-base-display");
  const apiStatusBadge = document.getElementById("api-status-badge");

  const weightControls = {
    motion: {
      slider: document.getElementById("weight-motion-slider"),
      input: document.getElementById("weight-motion-input"),
      display: document.getElementById("weight-motion-display"),
    },
    media: {
      slider: document.getElementById("weight-media-slider"),
      input: document.getElementById("weight-media-input"),
      display: document.getElementById("weight-media-display"),
    },
    appliance: {
      slider: document.getElementById("weight-appliance-slider"),
      input: document.getElementById("weight-appliance-input"),
      display: document.getElementById("weight-appliance-display"),
    },
    door: {
      slider: document.getElementById("weight-door-slider"),
      input: document.getElementById("weight-door-input"),
      display: document.getElementById("weight-door-display"),
    },
    window: {
      slider: document.getElementById("weight-window-slider"),
      input: document.getElementById("weight-window-input"),
      display: document.getElementById("weight-window-display"),
    },
    illuminance: {
      slider: document.getElementById("weight-illuminance-slider"),
      input: document.getElementById("weight-illuminance-input"),
      display: document.getElementById("weight-illuminance-display"),
    },
    humidity: {
      slider: document.getElementById("weight-humidity-slider"),
      input: document.getElementById("weight-humidity-input"),
      display: document.getElementById("weight-humidity-display"),
    },
    temperature: {
      slider: document.getElementById("weight-temperature-slider"),
      input: document.getElementById("weight-temperature-input"),
      display: document.getElementById("weight-temperature-display"),
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
      probabilityValue.textContent = `${(probability * 100).toFixed(2)}%`;
    }

    if (probabilityFill) {
      probabilityFill.style.width = `${(probability * 100).toFixed(2)}%`;
    }

    addChartPoint(probability);
  }

  function renderSensors(data) {
    if (!sensorsList) {
      return;
    }

    sensorsList.innerHTML = "";

    data.entities.forEach((entity) => {
      const card = document.createElement("div");
      card.className = "sim-sensor-card";

      const title = document.createElement("h3");
      title.textContent = entity.name;
      card.appendChild(title);

      const stateEl = document.createElement("div");
      stateEl.className = "sim-sensor-state";
      stateEl.textContent = `State: ${entity.state_display}`;
      card.appendChild(stateEl);

      const details = document.createElement("div");
      details.className = "sim-sensor-details";
      details.textContent = entity.details;
      card.appendChild(details);

      if (entity.actions.length > 0) {
        const buttonsContainer = document.createElement("div");
        buttonsContainer.className = "sim-sensor-buttons";

        entity.actions.forEach((action) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "sim-sensor-button";
          button.textContent = action.label;
          if (action.active) {
            button.classList.add("active");
          }

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

          buttonsContainer.appendChild(button);
        });

        card.appendChild(buttonsContainer);
      }

      sensorsList.appendChild(card);
    });
  }

  function renderBreakdown(data) {
    if (!breakdownList) {
      return;
    }

    breakdownList.innerHTML = "";

    data.breakdown.forEach((item) => {
      const breakdownItem = document.createElement("div");
      breakdownItem.className = "sim-breakdown-item";

      const label = document.createElement("div");
      label.className = "sim-breakdown-label";
      label.innerHTML = `<strong>${item.name}</strong><span>${item.description}</span>`;

      const likelihood = document.createElement("div");
      likelihood.className = "sim-breakdown-metric";
      likelihood.innerHTML = `<strong>${(item.likelihood * 100).toFixed(
        2
      )}%</strong><span>Likelihood</span>`;

      const contribution = document.createElement("div");
      contribution.className = "sim-breakdown-metric";
      contribution.innerHTML = `<strong>${(item.contribution * 100).toFixed(
        2
      )}%</strong><span>Contribution</span>`;

      breakdownItem.append(label, likelihood, contribution);
      breakdownList.appendChild(breakdownItem);
    });
  }

  function syncPriorControls(data) {
    if (globalPriorSlider) {
      globalPriorSlider.value = data.global_prior.toFixed(2);
    }
    if (globalPriorInput) {
      globalPriorInput.value = data.global_prior.toFixed(2);
    }
    if (globalPriorDisplay) {
      globalPriorDisplay.textContent = `${(data.global_prior * 100).toFixed(
        2
      )}%`;
    }

    if (timePriorSlider) {
      timePriorSlider.value = data.time_prior.toFixed(2);
    }
    if (timePriorInput) {
      timePriorInput.value = data.time_prior.toFixed(2);
    }
    if (timePriorDisplay) {
      timePriorDisplay.textContent = `${(data.time_prior * 100).toFixed(2)}%`;
    }

    if (combinedPriorDisplay) {
      combinedPriorDisplay.textContent = `${(data.combined_prior * 100).toFixed(
        2
      )}%`;
    }
    if (finalPriorDisplay) {
      finalPriorDisplay.textContent = `${(data.final_prior * 100).toFixed(2)}%`;
    }

    if (halfLifeDisplay) {
      halfLifeDisplay.textContent = `Half-life: ${Math.round(data.half_life)}s`;
    }

    Object.entries(weightControls).forEach(([key, control]) => {
      if (!data.weights || typeof data.weights[key] !== "number") {
        return;
      }

      const value = data.weights[key].toFixed(2);
      if (control.slider) {
        control.slider.value = value;
      }
      if (control.input) {
        control.input.value = value;
      }
      if (control.display) {
        control.display.textContent = value;
      }
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
        const option = document.createElement("option");
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
    const input = control.input;
    const display = control.display;
    const key = control.key;

    if (!slider || !input || !display) {
      return;
    }

    const updateValue = (value) => {
      const clamped = Math.min(Math.max(parseFloat(value) || 0, 0), 1);
      slider.value = clamped.toFixed(2);
      input.value = clamped.toFixed(2);
      display.textContent = `${(clamped * 100).toFixed(2)}%`;
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

    slider.addEventListener("input", (event) => {
      updateValue(event.target.value);
    });

    slider.addEventListener("change", (event) => {
      handleUpdate(event.target.value);
    });

    input.addEventListener("change", (event) => {
      handleUpdate(event.target.value);
    });
  }

  function addWeightListeners(type, control) {
    const slider = control.slider;
    const input = control.input;
    const display = control.display;

    if (!slider || !input || !display) {
      return;
    }

    const updateValue = (value) => {
      const clamped = Math.min(Math.max(parseFloat(value) || 0.01, 0.01), 0.99);
      slider.value = clamped.toFixed(2);
      input.value = clamped.toFixed(2);
      display.textContent = clamped.toFixed(2);
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

    slider.addEventListener("input", (event) => {
      updateValue(event.target.value);
    });

    slider.addEventListener("change", (event) => {
      handleUpdate(event.target.value);
    });

    input.addEventListener("change", (event) => {
      handleUpdate(event.target.value);
    });
  }

  async function handlePurposeChange(event) {
    const value = event.target.value;

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
      input: globalPriorInput,
      display: globalPriorDisplay,
      key: "global_prior",
    });

    addPriorListeners({
      slider: timePriorSlider,
      input: timePriorInput,
      display: timePriorDisplay,
      key: "time_prior",
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

    apiBaseSave.addEventListener("click", () => {
      updateApiBaseUrl(apiBaseInput.value);
    });

    apiBaseReset.addEventListener("click", () => {
      updateApiBaseUrl(DEFAULT_API_BASE_URL);
    });

    apiBaseInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        updateApiBaseUrl(apiBaseInput.value);
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
      purposeSelect.addEventListener("change", handlePurposeChange);
    }
  }

  init();
})();
