import { LitElement, html, css } from 'https://unpkg.com/lit?module';
import Chart from 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js';

class AreaOccupancyCard extends LitElement {
  static properties = {
    hass: {},
    config: {},
    _history: {},
  };

  constructor() {
    super();
    this._history = [];
    this._chart = null;
  }

  setConfig(config) {
    if (!config.entity) {
      throw new Error('The "entity" property is required.');
    }
    this.config = {
      name: 'Occupancy',
      status_entity: null,
      ...config,
    };
  }

  static async getConfigElement() {
    await import('./area-occupancy-card-editor.js');
    return document.createElement('area-occupancy-card-editor');
  }

  static getStubConfig(hass) {
    const [entity] = Object.keys(hass.states).filter((e) =>
      e.includes('occupancy_probability')
    );
    return { entity };
  }

  get probability() {
    const stateObj = this.hass.states[this.config.entity];
    return stateObj ? Number(stateObj.state) : 0;
  }

  updated(changedProps) {
    if (changedProps.has('hass')) {
      this._fetchHistory();
    }
  }

  async _fetchHistory() {
    if (!this.config.entity || !this.hass) return;
    const end = new Date();
    const start = new Date(end.getTime() - 3600 * 1000);
    const url = `history/period/${start.toISOString()}?filter_entity_id=${this.config.entity}&minimal_response&no_attributes&end_time=${end.toISOString()}`;
    const result = await this.hass.callApi('GET', url);
    if (Array.isArray(result) && result[0]) {
      this._history = result[0].map((it) => ({
        t: new Date(it.last_changed),
        v: Number(it.state),
      }));
      this._updateChart();
    }
  }

  _updateChart() {
    const canvas = this.renderRoot?.querySelector('#chart');
    if (!canvas) return;
    const labels = this._history.map((h) => h.t.toLocaleTimeString());
    const data = this._history.map((h) => h.v);
    if (!this._chart) {
      this._chart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              data,
              fill: false,
              borderColor: '#03a9f4',
              tension: 0.1,
            },
          ],
        },
        options: {
          animation: false,
          plugins: {legend: {display: false}},
          scales: {
            y: {min: 0, max: 100},
          },
        },
      });
    } else {
      this._chart.data.labels = labels;
      this._chart.data.datasets[0].data = data;
      this._chart.update();
    }
  }

  get occupied() {
    if (this.config.status_entity) {
      const status = this.hass.states[this.config.status_entity];
      return status && status.state === 'on';
    }
    return this.probability >= 50;
  }

  render() {
    const probability = this.probability.toFixed(0);
    const statusText = this.occupied ? 'Occupied' : 'Clear';
    return html`
      <ha-card .header=${this.config.name}>
        <div class="value">${probability}%</div>
        <div class="status">${statusText}</div>
        <ha-linear-progress .value=${this.probability / 100}></ha-linear-progress>
        <canvas id="chart" height="60"></canvas>
      </ha-card>
    `;
  }

  static styles = css`
    ha-card {
      padding: 16px;
      text-align: center;
    }
    .value {
      font-size: 2em;
      font-weight: bold;
      line-height: 1;
    }
    .status {
      margin-top: 4px;
      font-size: 1em;
      color: var(--secondary-text-color);
    }
    ha-linear-progress {
      width: 100%;
      margin-top: 8px;
    }
    canvas {
      width: 100%;
      margin-top: 8px;
    }
  `;
}

customElements.define('area-occupancy-card', AreaOccupancyCard);
