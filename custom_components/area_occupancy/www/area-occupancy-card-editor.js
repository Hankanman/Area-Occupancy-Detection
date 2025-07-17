import { LitElement, html } from 'https://unpkg.com/lit?module';

function fireEvent(node, type, detail, options) {
  options = options || {};
  options.bubbles = options.bubbles ?? true;
  options.cancelable = options.cancelable ?? false;
  options.composed = options.composed ?? true;
  const event = new CustomEvent(type, {
    detail,
    bubbles: options.bubbles,
    cancelable: options.cancelable,
    composed: options.composed,
  });
  node.dispatchEvent(event);
  return event;
}

class AreaOccupancyCardEditor extends LitElement {
  static properties = {
    hass: {},
    _config: {},
  };

  setConfig(config) {
    this._config = { ...config };
  }

  render() {
    if (!this.hass) return html``;
    return html`
      <div class="form">
        <ha-textfield
          label="Probability entity"
          .value=${this._config.entity || ''}
          @change=${this._valueChanged}
          .configValue=${'entity'}
        ></ha-textfield>
        <ha-textfield
          label="Status entity (optional)"
          .value=${this._config.status_entity || ''}
          @change=${this._valueChanged}
          .configValue=${'status_entity'}
        ></ha-textfield>
      </div>
    `;
  }

  _valueChanged(ev) {
    const target = ev.target;
    this._config = {
      ...this._config,
      [target.configValue]: target.value,
    };
    fireEvent(this, 'config-changed', { config: this._config });
  }
}

customElements.define('area-occupancy-card-editor', AreaOccupancyCardEditor);
