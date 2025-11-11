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
<link
  rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/@shoelace-style/shoelace@2.20.1/cdn/themes/light.css"
  id="shoelace-theme-light"
/>
<link
  rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/@shoelace-style/shoelace@2.20.1/cdn/themes/dark.css"
  id="shoelace-theme-dark"
  disabled
/>
<div class="aod-simulator sim-stack md-typeset">

  <div id="simulation-display" class="sim-stack">
    <div class="sim-layout">
      <div class="sim-header sim-sticky">
        <div class="sim-header-content">
          <div class="sim-prob-value">
            <h3 id="area-name">Area</h3>
            <h3 id="probability-value">0.00%</h3>
            <div id="probability-bar">
              <div id="probability-fill"></div>
            </div>
          </div>
          <div class="sim-prob-chart">
            <canvas id="probability-chart"></canvas>
          </div>
        </div>
      </div>
      <main class="sim-stack">
        <sl-details class="md-card sim-section" summary="Sensors" open="true">
          <sl-alert
            id="sensors-placeholder"
            variant="primary"
            open
            class="sim-placeholder"
          >
            <span slot="icon">📡</span>
            Add output yaml from the "Run Analysis" service below to populate sensors.
          </sl-alert>
          <div id="sensors-container" class="sim-card-grid">
          </div>
        </sl-details>
        <sl-details class="md-card sim-section" summary="Prior Probabilities">
          <div class="sim-controls">
            <sl-range
              label="Global Prior"
              data-label="Global Prior"
              min="0"
              max="1"
              step="0.01"
              value="0.5"
              id="global-prior-slider"
            ></sl-range>
            <sl-range
              label="Time Prior"
              data-label="Time Prior"
              min="0"
              max="1"
              step="0.01"
              value="0.5"
              id="time-prior-slider"
            ></sl-range>
            <sl-input
              label="Combined Prior"
              value="50.00%"
              disabled
              id="combined-prior-input"
            ></sl-input>
            <sl-input
              label="Final Prior (after factor &amp; clamp)"
              value="50.00%"
              disabled
              id="final-prior-input"
            ></sl-input>
            <sl-select
              id="purpose-select"
              label="Area Purpose"
              clearable
            >
              <sl-option value="">Loading...</sl-option>
            </sl-select>
            <sl-input
              id="half-life-display"
              label="Half-life"
              value="--"
              disabled
            ></sl-input>
          </div>
        </sl-details>
        <sl-details class="md-card sim-section" summary="Entity Type Weights">
          <div class="sim-controls sim-controls--autofit">
            <sl-range
              label="Motion"
              data-label="Motion"
              min="0.01"
              max="1"
              step="0.01"
              value="1"
              id="weight-motion-slider"
            ></sl-range>
            <sl-range
              label="Media"
              data-label="Media"
              min="0.01"
              max="1"
              step="0.01"
              value="0.85"
              id="weight-media-slider"
            ></sl-range>
            <sl-range
              label="Appliance"
              data-label="Appliance"
              min="0.01"
              max="1"
              step="0.01"
              value="0.4"
              id="weight-appliance-slider"
            ></sl-range>
            <sl-range
              label="Door"
              data-label="Door"
              min="0.01"
              max="1"
              step="0.01"
              value="0.3"
              id="weight-door-slider"
            ></sl-range>
            <sl-range
              label="Window"
              data-label="Window"
              min="0.01"
              max="1"
              step="0.01"
              value="0.2"
              id="weight-window-slider"
            ></sl-range>
            <sl-range
              label="Illuminance"
              data-label="Illuminance"
              min="0.01"
              max="1"
              step="0.01"
              value="0.1"
              id="weight-illuminance-slider"
            ></sl-range>
            <sl-range
              label="Humidity"
              data-label="Humidity"
              min="0.01"
              max="1"
              step="0.01"
              value="0.1"
              id="weight-humidity-slider"
            ></sl-range>
            <sl-range
              label="Temperature"
              data-label="Temperature"
              min="0.01"
              max="1"
              step="0.01"
              value="0.1"
              id="weight-temperature-slider"
            ></sl-range>
          </div>
        </sl-details>
      </main>
    </div>
  </div>
  <div class="md-card sim-section">
    <div class="sim-section-header">
    <h2>Analysis Output</h2>
      <span id="api-status-badge">Checking</span>
    </div>
    <sl-textarea id="yaml-input" placeholder="Paste your YAML analysis output here..."></sl-textarea>
    <div class="sim-actions">
      <sl-button id="load-btn" variant="primary">Load Simulation</sl-button>
    </div>
    <div id="error-message" role="alert"></div>
  </div>
</div>

<script defer src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script defer src="../assets/simulator/themeDetector.js"></script>
<script
  type="module"
  src="https://cdn.jsdelivr.net/npm/@shoelace-style/shoelace@2.20.1/cdn/shoelace.js"
></script>
<script defer src="../assets/simulator/app.js"></script>
