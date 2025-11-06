// Global state
let simulationData = null;
let probabilityChart = null;
let probabilityHistory = [];

// DOM elements
const yamlInput = document.getElementById('yaml-input');
const loadBtn = document.getElementById('load-btn');
const errorMessage = document.getElementById('error-message');
const simulationDisplay = document.getElementById('simulation-display');
const areaName = document.getElementById('area-name');
const probabilityValue = document.getElementById('probability-value');
const probabilityFill = document.getElementById('probability-fill');
const sensorsList = document.getElementById('sensors-list');
const breakdownList = document.getElementById('breakdown-list');

// Prior control elements
const globalPriorSlider = document.getElementById('global-prior-slider');
const globalPriorInput = document.getElementById('global-prior-input');
const globalPriorDisplay = document.getElementById('global-prior-display');
const timePriorSlider = document.getElementById('time-prior-slider');
const timePriorInput = document.getElementById('time-prior-input');
const timePriorDisplay = document.getElementById('time-prior-display');
const combinedPriorDisplay = document.getElementById('combined-prior-display');
const purposeSelect = document.getElementById('purpose-select');
const halfLifeDisplay = document.getElementById('half-life-display');

// Auto-update interval
let tickInterval = null;

// Initialize chart
function initChart() {
    const ctx = document.getElementById('probability-chart');
    if (probabilityChart) {
        probabilityChart.destroy();
    }

    probabilityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Occupancy Probability',
                data: [],
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'Probability: ' + (context.parsed.y * 100).toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });
}

// Add data point to chart
function addChartPoint(probability) {
    const now = new Date();
    const timeLabel = now.toLocaleTimeString();

    probabilityHistory.push({
        time: timeLabel,
        probability: probability
    });

    // Keep only last 50 points
    if (probabilityHistory.length > 50) {
        probabilityHistory.shift();
    }

    if (probabilityChart) {
        probabilityChart.data.labels = probabilityHistory.map(p => p.time);
        probabilityChart.data.datasets[0].data = probabilityHistory.map(p => p.probability);
        probabilityChart.update('none');
    }
}

// Load simulation from YAML
loadBtn.addEventListener('click', async () => {
    const yamlText = yamlInput.value.trim();

    if (!yamlText) {
        showError('Please paste YAML analysis output');
        return;
    }

    try {
        const response = await fetch('/api/load', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ yaml: yamlText })
        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || 'Failed to load simulation');
            return;
        }

        simulationData = data;
        probabilityHistory = [];
        initChart();
        renderSimulation(data);
        hideError();

        // Start auto-update interval
        startAutoUpdate();

    } catch (error) {
        showError('Error loading simulation: ' + error.message);
    }
});

// Render simulation
function renderSimulation(data) {
    // Show simulation display
    simulationDisplay.classList.remove('hidden');

    // Update area info
    areaName.textContent = data.area_name;

    // Initialize prior controls
    globalPriorSlider.value = data.global_prior;
    globalPriorInput.value = data.global_prior;
    globalPriorDisplay.textContent = (data.global_prior * 100).toFixed(2) + '%';

    timePriorSlider.value = data.time_prior;
    timePriorInput.value = data.time_prior;
    timePriorDisplay.textContent = (data.time_prior * 100).toFixed(2) + '%';

    // Calculate and display combined prior
    updateCombinedPrior(data.global_prior, data.time_prior);

    // Set purpose
    if (data.area_purpose && purposeSelect) {
        purposeSelect.value = data.area_purpose;
        if (data.half_life) {
            halfLifeDisplay.textContent = `Half-life: ${data.half_life}s`;
        }
    }

    // Update probability
    updateProbability(data.probability);

    // Render sensors
    renderSensors(data.entities);

    // Render breakdown
    renderBreakdown(data.breakdown);

    // Add initial chart point
    addChartPoint(data.probability);
}

// Update combined prior display
function updateCombinedPrior(globalPrior, timePrior) {
    // Use the same combine_priors logic from backend
    // For now, use a simple weighted average (time_weight = 0.2)
    const timeWeight = 0.2;
    const areaWeight = 1.0 - timeWeight;

    // Convert to logit space for better interpolation
    function probToLogit(p) {
        if (p <= 0) return -10;
        if (p >= 1) return 10;
        return Math.log(p / (1 - p));
    }

    function logitToProb(logit) {
        return 1 / (1 + Math.exp(-logit));
    }

    const areaLogit = probToLogit(globalPrior);
    const timeLogit = probToLogit(timePrior);
    const combinedLogit = areaWeight * areaLogit + timeWeight * timeLogit;
    const combinedPrior = logitToProb(combinedLogit);

    combinedPriorDisplay.textContent = (combinedPrior * 100).toFixed(2) + '%';
}

// Update probability display
function updateProbability(probability) {
    const percentage = (probability * 100).toFixed(2);
    probabilityValue.textContent = percentage + '%';
    probabilityFill.style.width = percentage + '%';

    // Update color class
    probabilityValue.classList.remove('low', 'medium', 'high');
    if (probability < 0.3) {
        probabilityValue.classList.add('low');
    } else if (probability < 0.7) {
        probabilityValue.classList.add('medium');
    } else {
        probabilityValue.classList.add('high');
    }
}

// Render sensors
function renderSensors(entities) {
    sensorsList.innerHTML = '';

    entities.forEach(entity => {
        const sensorItem = document.createElement('div');
        sensorItem.className = 'sensor-item';
        sensorItem.dataset.entityId = entity.entity_id;

        const isActive = entity.evidence === true;
        const isNumeric = entity.is_numeric;

        let controlsHtml = '';
        if (isNumeric) {
            controlsHtml = `
                <div class="sensor-controls">
                    <input type="number"
                           class="numeric-input"
                           value="${entity.current_state || ''}"
                           step="0.01"
                           data-entity-id="${entity.entity_id}">
                    <span class="sensor-state ${isActive ? 'active' : 'inactive'}">
                        ${isActive ? 'Active' : 'Inactive'}
                    </span>
                </div>
            `;
        } else {
            // Use evidence to determine if sensor is on
            const isOn = entity.evidence === true;

            controlsHtml = `
                <div class="sensor-controls">
                    <label class="toggle-switch">
                        <input type="checkbox"
                               ${isOn ? 'checked' : ''}
                               data-entity-id="${entity.entity_id}">
                        <span class="toggle-slider"></span>
                    </label>
                    <span class="sensor-state ${isActive ? 'active' : 'inactive'}">
                        ${isActive ? 'Active' : 'Inactive'}
                    </span>
                </div>
            `;
        }

        sensorItem.innerHTML = `
            <div class="sensor-header">
                <span class="sensor-name">${entity.entity_id}</span>
                <span class="sensor-type">${entity.type}</span>
            </div>
            <div class="sensor-info">
                <div class="sensor-info-item">
                    <span class="sensor-info-label">Weight:</span>
                    <span>${entity.weight.toFixed(2)}</span>
                </div>
                <div class="sensor-info-item">
                    <span class="sensor-info-label">P(E|H):</span>
                    <span>${entity.prob_given_true.toFixed(3)}</span>
                </div>
                <div class="sensor-info-item">
                    <span class="sensor-info-label">P(E|¬H):</span>
                    <span>${entity.prob_given_false.toFixed(3)}</span>
                </div>
                <div class="sensor-info-item">
                    <span class="sensor-info-label">Current State:</span>
                    <span>${entity.current_state || 'N/A'}</span>
                </div>
            </div>
            ${controlsHtml}
        `;

        sensorsList.appendChild(sensorItem);

        // Add event listeners
        if (isNumeric) {
            const input = sensorItem.querySelector('.numeric-input');
            input.addEventListener('change', () => handleNumericUpdate(entity.entity_id, input.value));
        } else {
            const checkbox = sensorItem.querySelector('input[type="checkbox"]');
            checkbox.addEventListener('change', () => handleToggle(entity.entity_id, checkbox.checked));
        }
    });
}

// Render breakdown
function renderBreakdown(breakdown) {
    breakdownList.innerHTML = '';

    // Sort by absolute contribution
    const sorted = Object.entries(breakdown)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

    sorted.forEach(([entityId, contribution]) => {
        const breakdownItem = document.createElement('div');
        breakdownItem.className = 'breakdown-item';

        const percentage = (contribution * 100).toFixed(2);
        const isPositive = contribution >= 0;

        breakdownItem.innerHTML = `
            <span class="breakdown-name">${entityId}</span>
            <span class="breakdown-value ${isPositive ? 'positive' : 'negative'}">
                ${isPositive ? '+' : ''}${percentage}%
            </span>
        `;

        breakdownList.appendChild(breakdownItem);
    });
}

// Handle toggle
async function handleToggle(entityId, isOn) {
    try {
        const response = await fetch('/api/toggle', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                entity_id: entityId,
                state: isOn ? 'on' : 'off'
            })
        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || 'Failed to toggle sensor');
            return;
        }

        updateProbability(data.probability);
        renderBreakdown(data.breakdown);
        addChartPoint(data.probability);

        // Update sensor card display
        const sensorItem = document.querySelector(`[data-entity-id="${entityId}"]`);
        if (sensorItem) {
            // Update Active/Inactive badge
            const stateSpan = sensorItem.querySelector('.sensor-state');
            if (stateSpan) {
                const isActive = data.evidence === true;
                stateSpan.textContent = isActive ? 'Active' : 'Inactive';
                stateSpan.className = `sensor-state ${isActive ? 'active' : 'inactive'}`;
            }

            // Update Current State in sensor-info section
            const sensorInfoItems = sensorItem.querySelectorAll('.sensor-info-item');
            sensorInfoItems.forEach(item => {
                const label = item.querySelector('.sensor-info-label');
                if (label && label.textContent.trim() === 'Current State:') {
                    const valueSpan = item.querySelector('span:last-child');
                    if (valueSpan) {
                        valueSpan.textContent = data.entity_state || 'N/A';
                    }
                }
            });
        }

    } catch (error) {
        showError('Error toggling sensor: ' + error.message);
    }
}

// Handle numeric update
async function handleNumericUpdate(entityId, value) {
    try {
        const response = await fetch('/api/update', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                entity_id: entityId,
                value: parseFloat(value)
            })
        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || 'Failed to update sensor');
            return;
        }

        updateProbability(data.probability);
        renderBreakdown(data.breakdown);
        addChartPoint(data.probability);

        // Update sensor card display
        const sensorItem = document.querySelector(`[data-entity-id="${entityId}"]`);
        if (sensorItem) {
            // Update Active/Inactive badge
            const stateSpan = sensorItem.querySelector('.sensor-state');
            if (stateSpan) {
                const isActive = data.evidence === true;
                stateSpan.textContent = isActive ? 'Active' : 'Inactive';
                stateSpan.className = `sensor-state ${isActive ? 'active' : 'inactive'}`;
            }

            // Update Current State in sensor-info section
            const sensorInfoItems = sensorItem.querySelectorAll('.sensor-info-item');
            sensorInfoItems.forEach(item => {
                const label = item.querySelector('.sensor-info-label');
                if (label && label.textContent.trim() === 'Current State:') {
                    const valueSpan = item.querySelector('span:last-child');
                    if (valueSpan) {
                        valueSpan.textContent = data.entity_state || 'N/A';
                    }
                }
            });

            // Update numeric input value if it's a numeric sensor
            const numericInput = sensorItem.querySelector('.numeric-input');
            if (numericInput) {
                numericInput.value = data.entity_state || '';
            }
        }

    } catch (error) {
        showError('Error updating sensor: ' + error.message);
    }
}

// Error handling
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('show');
}

function hideError() {
    errorMessage.classList.remove('show');
}

// Prior adjustment handlers
function setupPriorHandlers() {
    // Global prior slider
    globalPriorSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        globalPriorInput.value = value;
        globalPriorDisplay.textContent = (value * 100).toFixed(2) + '%';
        updateCombinedPrior(value, parseFloat(timePriorSlider.value));
        handlePriorUpdate();
    });

    // Global prior input
    globalPriorInput.addEventListener('change', (e) => {
        let value = parseFloat(e.target.value);
        if (isNaN(value) || value < 0) value = 0;
        if (value > 1) value = 1;
        globalPriorSlider.value = value;
        globalPriorInput.value = value;
        globalPriorDisplay.textContent = (value * 100).toFixed(2) + '%';
        updateCombinedPrior(value, parseFloat(timePriorSlider.value));
        handlePriorUpdate();
    });

    // Time prior slider
    timePriorSlider.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        timePriorInput.value = value;
        timePriorDisplay.textContent = (value * 100).toFixed(2) + '%';
        updateCombinedPrior(parseFloat(globalPriorSlider.value), value);
        handlePriorUpdate();
    });

    // Time prior input
    timePriorInput.addEventListener('change', (e) => {
        let value = parseFloat(e.target.value);
        if (isNaN(value) || value < 0) value = 0;
        if (value > 1) value = 1;
        timePriorSlider.value = value;
        timePriorInput.value = value;
        timePriorDisplay.textContent = (value * 100).toFixed(2) + '%';
        updateCombinedPrior(parseFloat(globalPriorSlider.value), value);
        handlePriorUpdate();
    });
}

// Handle prior update
let priorUpdateTimeout = null;
async function handlePriorUpdate() {
    // Debounce API calls
    clearTimeout(priorUpdateTimeout);
    priorUpdateTimeout = setTimeout(async () => {
        try {
            const globalPrior = parseFloat(globalPriorSlider.value);
            const timePrior = parseFloat(timePriorSlider.value);

            const response = await fetch('/api/update-priors', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    global_prior: globalPrior,
                    time_prior: timePrior
                })
            });

            const data = await response.json();

            if (!response.ok) {
                showError(data.error || 'Failed to update priors');
                return;
            }

            updateProbability(data.probability);
            renderBreakdown(data.breakdown);
            addChartPoint(data.probability);

        } catch (error) {
            showError('Error updating priors: ' + error.message);
        }
    }, 300); // 300ms debounce
}

// Load purposes on page load
async function loadPurposes() {
    try {
        const response = await fetch('/api/get-purposes');
        const data = await response.json();

        if (response.ok && purposeSelect) {
            purposeSelect.innerHTML = '';
            data.purposes.forEach(purpose => {
                const option = document.createElement('option');
                option.value = purpose.value;
                option.textContent = `${purpose.name} (${purpose.half_life}s)`;
                purposeSelect.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading purposes:', error);
    }
}

// Start auto-update interval
function startAutoUpdate() {
    // Clear existing interval
    if (tickInterval) {
        clearInterval(tickInterval);
    }

    // Set up 1-second interval
    tickInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/tick', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const data = await response.json();

            if (response.ok) {
                updateProbability(data.probability);
                renderBreakdown(data.breakdown);
                addChartPoint(data.probability);

                // Update sensor decay states if provided
                if (data.entity_decay) {
                    updateSensorDecayStates(data.entity_decay);
                }
            }
        } catch (error) {
            console.error('Error in tick:', error);
        }
    }, 1000); // 1 second
}

// Stop auto-update interval
function stopAutoUpdate() {
    if (tickInterval) {
        clearInterval(tickInterval);
        tickInterval = null;
    }
}

// Update sensor decay states display
function updateSensorDecayStates(entityDecay) {
    Object.entries(entityDecay).forEach(([entityId, decayInfo]) => {
        const sensorItem = document.querySelector(`[data-entity-id="${entityId}"]`);
        if (sensorItem) {
            // Update decay indicator if we add one
            // For now, we can add a visual indicator
            const stateSpan = sensorItem.querySelector('.sensor-state');
            if (stateSpan && decayInfo.is_decaying) {
                // Add decay indicator
                if (!stateSpan.textContent.includes('(decaying)')) {
                    stateSpan.textContent += ' (decaying)';
                }
            }
        }
    });
}

// Handle purpose change
async function handlePurposeChange() {
    if (!purposeSelect || !purposeSelect.value) return;

    try {
        const response = await fetch('/api/update-purpose', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                purpose: purposeSelect.value
            })
        });

        const data = await response.json();

        if (!response.ok) {
            showError(data.error || 'Failed to update purpose');
            return;
        }

        // Update half-life display
        if (data.half_life) {
            halfLifeDisplay.textContent = `Half-life: ${data.half_life}s`;
        }

        // Update probability
        updateProbability(data.probability);
        renderBreakdown(data.breakdown);
        addChartPoint(data.probability);

    } catch (error) {
        showError('Error updating purpose: ' + error.message);
    }
}

// Initialize chart on page load
initChart();
setupPriorHandlers();
loadPurposes();

// Set up purpose change handler
if (purposeSelect) {
    purposeSelect.addEventListener('change', handlePurposeChange);
}

