---
title: Simulator
hide:
  - toc
  - navigation
  - path
  - content.action.edit
  - content.action.view
  - footer
---

<link rel="stylesheet" href="../assets/simulator/style.css">
<style>
  .md-content__button {
    display: none;
  }
  h1:first-of-type {
    display: none;
  }
</style>
<div class="aod-simulator">
  <div class="sim-container">
    <div class="sim-main">
      <section class="sim-api-status">
        <div class="sim-api-heading">
          <h2>API Connection</h2>
          <span id="api-status-badge" class="sim-api-badge">Checking</span>
        </div>
        <div class="sim-api-controls">
          <div class="sim-api-input">
            <input id="api-base-input" type="url" placeholder="https://your-simulator-host (e.g. http://127.0.0.1:5000)" aria-label="API base URL" />
            <button type="button" id="api-base-save" class="sim-btn sim-btn-secondary">Update</button>
            <button type="button" id="api-base-reset" class="sim-btn sim-btn-secondary">Reset</button>
          </div>
          <small>
            Defaults to the hosted backend at https://aod-simulator.onrender.com.
            When running MkDocs locally, the field auto-populates with http://127.0.0.1:5000 so it connects to the backend started via <code>python main.py</code>.
          </small>
        </div>
        <p class="sim-api-help">Requests will be sent to <span id="api-base-display"></span>. Update this if you are running the backend locally.</p>
      </section>
      <section class="sim-input-section">
        <h2>Analysis Output</h2>
        <textarea id="yaml-input" placeholder="Paste your YAML analysis output here..."></textarea>
        <button id="load-btn" class="sim-btn sim-btn-primary">Load Simulation</button>
        <div id="error-message" class="sim-error"></div>
      </section>
      <div id="simulation-display" class="sim-display hidden">
        <div class="sim-layout">
          <aside class="sim-sidebar">
            <section class="sim-probability">
              <div class="sim-probability-label">Occupancy Probability</div>
              <div id="probability-value" class="sim-probability-value">0.00%</div>
              <div class="sim-probability-bar">
                <div id="probability-fill" class="sim-probability-fill"></div>
              </div>
            </section>
            <section class="sim-chart">
              <h3>Probability Over Time</h3>
              <canvas id="probability-chart"></canvas>
            </section>
          </aside>
          <main class="sim-content">
            <section class="sim-area-info">
              <h2 id="area-name">Area Name</h2>
            </section>
            <section class="sim-prior-section">
              <h2>Prior Probabilities</h2>
              <div class="sim-prior-controls">
                <div class="sim-prior-item">
                  <label for="global-prior-slider">Global Prior:</label>
                  <div class="sim-prior-inputs">
                    <input type="range" id="global-prior-slider" min="0" max="1" step="0.01" value="0.5" class="sim-prior-slider">
                    <input type="number" id="global-prior-input" min="0" max="1" step="0.01" value="0.5" class="sim-prior-input">
                  </div>
                  <span class="sim-prior-display" id="global-prior-display">50.00%</span>
                </div>
                <div class="sim-prior-item">
                  <label for="time-prior-slider">Time Prior:</label>
                  <div class="sim-prior-inputs">
                    <input type="range" id="time-prior-slider" min="0" max="1" step="0.01" value="0.5" class="sim-prior-slider">
                    <input type="number" id="time-prior-input" min="0" max="1" step="0.01" value="0.5" class="sim-prior-input">
                  </div>
                  <span class="sim-prior-display" id="time-prior-display">50.00%</span>
                </div>
                <div class="sim-prior-item sim-combined">
                  <label>Combined Prior:</label>
                  <span class="sim-prior-display large" id="combined-prior-display">50.00%</span>
                </div>
                <div class="sim-prior-item sim-combined">
                  <label>Final Prior (after factor &amp; clamp):</label>
                  <span class="sim-prior-display large" id="final-prior-display">50.00%</span>
                </div>
                <div class="sim-prior-item">
                  <label for="purpose-select">Area Purpose:</label>
                  <select id="purpose-select" class="sim-purpose-select">
                    <option value="">Loading...</option>
                  </select>
                  <span class="sim-prior-display" id="half-life-display">Half-life: --</span>
                </div>
              </div>
            </section>
            <section class="sim-weight-section">
              <h2>Entity Type Weights</h2>
              <div class="sim-weight-controls">
                <div class="sim-weight-item">
                  <label for="weight-motion-slider">Motion:</label>
                  <div class="sim-weight-inputs">
                    <input type="range" id="weight-motion-slider" min="0.01" max="0.99" step="0.01" value="1.0" class="sim-weight-slider">
                    <input type="number" id="weight-motion-input" min="0.01" max="0.99" step="0.01" value="1.0" class="sim-weight-input">
                  </div>
                  <span class="sim-weight-display" id="weight-motion-display">1.00</span>
                </div>
                <div class="sim-weight-item">
                  <label for="weight-media-slider">Media:</label>
                  <div class="sim-weight-inputs">
                    <input type="range" id="weight-media-slider" min="0.01" max="0.99" step="0.01" value="0.85" class="sim-weight-slider">
                    <input type="number" id="weight-media-input" min="0.01" max="0.99" step="0.01" value="0.85" class="sim-weight-input">
                  </div>
                  <span class="sim-weight-display" id="weight-media-display">0.85</span>
                </div>
                <div class="sim-weight-item">
                  <label for="weight-appliance-slider">Appliance:</label>
                  <div class="sim-weight-inputs">
                    <input type="range" id="weight-appliance-slider" min="0.01" max="0.99" step="0.01" value="0.4" class="sim-weight-slider">
                    <input type="number" id="weight-appliance-input" min="0.01" max="0.99" step="0.01" value="0.4" class="sim-weight-input">
                  </div>
                  <span class="sim-weight-display" id="weight-appliance-display">0.40</span>
                </div>
                <div class="sim-weight-item">
                  <label for="weight-door-slider">Door:</label>
                  <div class="sim-weight-inputs">
                    <input type="range" id="weight-door-slider" min="0.01" max="0.99" step="0.01" value="0.3" class="sim-weight-slider">
                    <input type="number" id="weight-door-input" min="0.01" max="0.99" step="0.01" value="0.3" class="sim-weight-input">
                  </div>
                  <span class="sim-weight-display" id="weight-door-display">0.30</span>
                </div>
                <div class="sim-weight-item">
                  <label for="weight-window-slider">Window:</label>
                  <div class="sim-weight-inputs">
                    <input type="range" id="weight-window-slider" min="0.01" max="0.99" step="0.01" value="0.2" class="sim-weight-slider">
                    <input type="number" id="weight-window-input" min="0.01" max="0.99" step="0.01" value="0.2" class="sim-weight-input">
                  </div>
                  <span class="sim-weight-display" id="weight-window-display">0.20</span>
                </div>
                <div class="sim-weight-item">
                  <label for="weight-illuminance-slider">Illuminance:</label>
                  <div class="sim-weight-inputs">
                    <input type="range" id="weight-illuminance-slider" min="0.01" max="0.99" step="0.01" value="0.1" class="sim-weight-slider">
                    <input type="number" id="weight-illuminance-input" min="0.01" max="0.99" step="0.01" value="0.1" class="sim-weight-input">
                  </div>
                  <span class="sim-weight-display" id="weight-illuminance-display">0.10</span>
                </div>
                <div class="sim-weight-item">
                  <label for="weight-humidity-slider">Humidity:</label>
                  <div class="sim-weight-inputs">
                    <input type="range" id="weight-humidity-slider" min="0.01" max="0.99" step="0.01" value="0.1" class="sim-weight-slider">
                    <input type="number" id="weight-humidity-input" min="0.01" max="0.99" step="0.01" value="0.1" class="sim-weight-input">
                  </div>
                  <span class="sim-weight-display" id="weight-humidity-display">0.10</span>
                </div>
                <div class="sim-weight-item">
                  <label for="weight-temperature-slider">Temperature:</label>
                  <div class="sim-weight-inputs">
                    <input type="range" id="weight-temperature-slider" min="0.01" max="0.99" step="0.01" value="0.1" class="sim-weight-slider">
                    <input type="number" id="weight-temperature-input" min="0.01" max="0.99" step="0.01" value="0.1" class="sim-weight-input">
                  </div>
                  <span class="sim-weight-display" id="weight-temperature-display">0.10</span>
                </div>
              </div>
            </section>
            <section class="sim-sensors-section">
              <h2>Sensors</h2>
              <div id="sensors-list" class="sim-sensors-grid"></div>
            </section>
            <section class="sim-breakdown-section">
              <h2>Sensor Contributions</h2>
              <div id="breakdown-list" class="sim-breakdown-list"></div>
            </section>
          </main>
        </div>
      </div>
    </div>
  </div>
</div>

<script defer src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script defer src="../assets/simulator/app.js"></script>
