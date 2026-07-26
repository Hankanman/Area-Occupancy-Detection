/*!
 * Area Occupancy — Time Priors Heatmap (custom Lovelace card)
 * -----------------------------------------------------------
 * Visualises the learned weekly occupancy forecast (7×24 = 168 slots per area)
 * returned by the `area_occupancy.get_time_priors` service. A comfort-threshold
 * slider highlights the slots each area is habitually occupied — useful to see,
 * at a glance, when a predictive automation (e.g. climate pre-heating) would act.
 *
 * Install:
 *   1. Copy this file to  /config/www/area-occupancy-time-priors-card.js
 *   2. Settings → Dashboards → ⋮ → Resources → Add:
 *        URL  /local/area-occupancy-time-priors-card.js   Type: JavaScript Module
 *   3. Add a card:  type: custom:area-occupancy-time-priors-card
 *
 * Options (all optional):
 *   title:           string  (default "Occupancy forecast")
 *   threshold:       number  0–100, comfort cutoff (default 50)
 *   refresh_minutes: number  re-poll interval (default 10)
 *   area_id:         string  limit to one area
 *
 * Requires the Area Occupancy build that exposes get_time_priors
 * (SupportsResponse.ONLY).
 */

const RAMP = ["#2c3f5e", "#3f8aa0", "#e0a63a", "#d1491f"]; // thermal: empty → occupied
const RAMP_STOPS = [0, 0.34, 0.64, 1];

// Localised short weekday names, Monday-first (AOD day_of_week: 0=Monday).
const DAYS = (() => {
  const fmt = new Intl.DateTimeFormat(
    (typeof navigator !== "undefined" && navigator.language) || "en",
    { weekday: "short" }
  );
  // 2024-01-01 is a Monday.
  return Array.from({ length: 7 }, (_, d) => fmt.format(new Date(Date.UTC(2024, 0, 1 + d))));
})();

const hex2rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const lerp = (a, b, t) => a + (b - a) * t;

function thermal(p) {
  let i = 0;
  while (i < RAMP_STOPS.length - 1 && p > RAMP_STOPS[i + 1]) i++;
  const t = (p - RAMP_STOPS[i]) / (RAMP_STOPS[i + 1] - RAMP_STOPS[i] || 1);
  const c1 = hex2rgb(RAMP[i]);
  const c2 = hex2rgb(RAMP[i + 1]);
  const c = c1.map((v, k) => Math.round(lerp(v, c2[k], Math.max(0, Math.min(1, t)))));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

class AreaOccupancyTimePriorsCard extends HTMLElement {
  static getStubConfig() {
    return { threshold: 50 };
  }

  setConfig(config) {
    this._config = {
      title: "Occupancy forecast",
      threshold: 50,
      refresh_minutes: 10,
      area_id: null,
      ...config,
    };
    this._threshold = (Number(this._config.threshold) || 50) / 100;
    if (!this.shadowRoot) this.attachShadow({ mode: "open" });
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
    if (!this._fetchedOnce) {
      this._fetchedOnce = true;
      this._fetch();
    }
  }

  connectedCallback() {
    this._startTimer();
  }

  disconnectedCallback() {
    this._stopTimer();
  }

  _startTimer() {
    this._stopTimer();
    const mins = Number(this._config?.refresh_minutes) || 10;
    this._timer = window.setInterval(() => this._fetch(), mins * 60000);
  }

  _stopTimer() {
    if (this._timer) {
      window.clearInterval(this._timer);
      this._timer = null;
    }
  }

  async _fetch() {
    if (!this._hass) return;
    const data = this._config.area_id ? { area_id: this._config.area_id } : {};
    try {
      // callService(domain, service, data, target, notifyOnError, returnResponse)
      const res = await this._hass.callService(
        "area_occupancy",
        "get_time_priors",
        data,
        undefined,
        false,
        true
      );
      this._data = (res && res.response) || res || null;
      this._error = null;
    } catch (e) {
      this._error = (e && (e.message || e.error)) || String(e);
      this._data = null;
    }
    this._render();
  }

  getCardSize() {
    const n = this._data && this._data.areas ? Object.keys(this._data.areas).length : 1;
    return 2 + n * 4;
  }

  _render() {
    if (!this.shadowRoot) return;
    const cfg = this._config || {};
    const pct = Math.round(this._threshold * 100);

    const style = `
      <style>
        ha-card { padding: 12px 4px 16px; }
        .head { display:flex; flex-wrap:wrap; align-items:center; gap:10px 16px; padding: 4px 14px 12px; }
        .title { font-size: 1.25rem; font-weight: 600; color: var(--primary-text-color); }
        .ctl { display:flex; align-items:center; gap:10px; margin-left:auto; }
        .ctl label { font-size:.8rem; color: var(--secondary-text-color); }
        .ctl input[type=range]{ width:150px; accent-color: var(--primary-color); }
        .pctv { font-variant-numeric: tabular-nums; font-weight:600; color: var(--primary-text-color); min-width:3ch; }
        .msg { padding: 10px 16px; color: var(--secondary-text-color); font-size:.9rem; }
        .msg code { background: var(--secondary-background-color); padding:1px 5px; border-radius:5px; }
        .rooms { display:grid; gap: 4px 16px; padding-top: 4px;
                 grid-template-columns: repeat(auto-fill, minmax(min(100%, 560px), 1fr)); }
        .room { padding: 4px 14px 14px; min-width: 0; }
        .room-head { display:flex; align-items:baseline; gap:10px; margin-bottom:6px; }
        .room-name { font-weight:600; color: var(--primary-text-color); }
        .room-id { font-size:.75rem; color: var(--secondary-text-color); font-family: var(--code-font-family, monospace); }
        .room-stat { margin-left:auto; font-size:.8rem; color: var(--secondary-text-color); }
        .room-stat b { color: var(--primary-color); font-size:1rem; font-variant-numeric: tabular-nums; }
        .scroll { overflow-x:auto; }
        .grid { display:grid; gap:2px; min-width:600px;
                grid-template-columns: 34px repeat(var(--cols,24), minmax(0,1fr)); }
        .hh { font-size:9px; text-align:center; color: var(--secondary-text-color);
              font-variant-numeric: tabular-nums; padding-bottom:2px; }
        .dl { font-size:10px; color: var(--secondary-text-color); display:flex; align-items:center; height:20px; }
        .cell { height:20px; border-radius:3px; }
        .cell.comfort { box-shadow: inset 0 0 0 2px var(--primary-color); }
        .cell.eco { opacity:.28; }
        .legend { display:flex; align-items:center; gap:8px 16px; flex-wrap:wrap;
                  padding: 4px 16px 0; font-size:.75rem; color: var(--secondary-text-color); }
        .ramp { height:10px; width:150px; border-radius:6px;
                background: linear-gradient(90deg, ${RAMP[0]}, ${RAMP[1]} 34%, ${RAMP[2]} 64%, ${RAMP[3]}); }
        .sw { display:inline-block; width:13px; height:13px; border-radius:3px; vertical-align:-2px; }
        .sw.comfort { box-shadow: inset 0 0 0 2px var(--primary-color); }
      </style>`;

    let body = "";
    if (this._error) {
      body = `<div class="msg">Could not read <code>area_occupancy.get_time_priors</code>: ${esc(
        this._error
      )}<br>Make sure the Area Occupancy build that exposes this service is installed.</div>`;
    } else if (!this._data || !this._data.areas) {
      body = `<div class="msg">Loading occupancy forecast…</div>`;
    } else {
      const slotMin = this._data.slot_minutes || 60;
      const cols = Math.round(1440 / slotMin);
      const hoursPerSlot = slotMin / 60;
      body =
        `<div class="legend"><span>Probability</span><span class="ramp"></span>` +
        `<span>0–100%</span><span><span class="sw comfort"></span> comfort</span></div>` +
        `<div class="rooms">` +
        Object.entries(this._data.areas)
          .map(([name, area]) => this._room(name, area, cols, hoursPerSlot))
          .join("") +
        `</div>`;
    }

    this.shadowRoot.innerHTML = `${style}
      <ha-card>
        <div class="head">
          <span class="title">${esc(cfg.title || "Occupancy forecast")}</span>
          <span class="ctl">
            <label>Comfort threshold</label>
            <input id="thr" type="range" min="0" max="100" value="${pct}">
            <span class="pctv">${pct}%</span>
          </span>
        </div>
        ${body}
      </ha-card>`;

    const slider = this.shadowRoot.getElementById("thr");
    if (slider) {
      slider.addEventListener("input", (e) => {
        this._threshold = Number(e.target.value) / 100;
        this._render(); // re-render only, no re-fetch
      });
    }
  }

  _room(name, area, cols, hoursPerSlot) {
    const slots = area.slots || {};
    let climatized = 0;
    for (const v of Object.values(slots)) if (v >= this._threshold) climatized += hoursPerSlot;

    let head = "<div class='hh'></div>";
    for (let s = 0; s < cols; s++) {
      const hour = Math.round(s * hoursPerSlot);
      head += `<div class="hh">${cols <= 24 || s % 2 === 0 ? hour : ""}</div>`;
    }
    let rows = "";
    for (let d = 0; d < 7; d++) {
      rows += `<div class="dl">${esc(DAYS[d])}</div>`;
      for (let s = 0; s < cols; s++) {
        const p = slots[`${d},${s}`];
        const hour = Math.round(s * hoursPerSlot);
        if (p === undefined) {
          rows += `<div class="cell eco" style="background:var(--secondary-background-color)"></div>`;
          continue;
        }
        const comfort = p >= this._threshold;
        const title = `${DAYS[d]} ${String(hour).padStart(2, "0")}:00 · ${Math.round(
          p * 100
        )}% · ${comfort ? "comfort" : "eco/off"}`;
        rows += `<div class="cell ${comfort ? "comfort" : "eco"}" title="${esc(
          title
        )}" style="background:${thermal(p)}"></div>`;
      }
    }
    const hrs = Math.round(climatized * 10) / 10;
    return `<div class="room">
      <div class="room-head">
        <span class="room-name">${esc(name)}</span>
        <span class="room-id">${esc(area.area_id || "")}</span>
        <span class="room-stat"><b>${hrs}</b> h/week comfort</span>
      </div>
      <div class="scroll"><div class="grid" style="--cols:${cols}">${head}${rows}</div></div>
    </div>`;
  }
}

customElements.define("area-occupancy-time-priors-card", AreaOccupancyTimePriorsCard);

window.customCards = window.customCards || [];
window.customCards.push({
  type: "area-occupancy-time-priors-card",
  name: "Area Occupancy — Time Priors Heatmap",
  description: "Weekly learned-occupancy forecast (168 slots) from get_time_priors.",
});

console.info(
  "%c AREA-OCCUPANCY-TIME-PRIORS-CARD %c loaded",
  "background:#d9662c;color:#fff;padding:2px 4px;border-radius:3px",
  ""
);
