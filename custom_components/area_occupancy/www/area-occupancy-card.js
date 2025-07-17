import { LitElement, html, css } from 'https://unpkg.com/lit?module';

class AreaOccupancyCard extends LitElement {
  static properties = {
    hass: {},
    config: {},
  };

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

  get probability() {
    const stateObj = this.hass.states[this.config.entity];
    return stateObj ? Number(stateObj.state) : 0;
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
  `;
}

customElements.define('area-occupancy-card', AreaOccupancyCard);
