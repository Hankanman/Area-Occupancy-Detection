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
 *   columns:         "auto" | 1 | 2  area layout (default "auto": 2 columns
 *                    only when the card itself is wider than 1100px)
 *   metric:          "raw" | "combined"  which number to plot (default "raw").
 *                    "raw"      = the learned time prior — carries the weekly
 *                                 shape, best for reading habits.
 *                    "combined" = time prior blended with the area's global
 *                                 prior; comparable to the occupancy threshold
 *                                 but with a compressed dynamic range.
 *   scale:           "area" | "absolute"  colour ramp (default "area"):
 *                    "area" stretches the ramp over the area's own min..max,
 *                    "absolute" pins it to 0..100%.
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
      columns: "auto",
      metric: "raw",
      scale: "area",
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
        /* The card is its own query container: every size rule below reacts to
           the width Lovelace actually gives this card, not to the viewport. */
        ha-card { padding: 12px 4px 16px; container-type: inline-size;
                  --cell-h: clamp(13px, 2.6cqw, 24px); --dl-w: 34px; }
        .head { display:flex; flex-wrap:wrap; align-items:center; gap:10px 16px; padding: 4px 14px 12px; }
        .title { font-size: 1.25rem; font-weight: 600; color: var(--primary-text-color); }
        .ctl { display:flex; align-items:center; gap:10px; margin-left:auto; }
        .ctl label { font-size:.8rem; color: var(--secondary-text-color); }
        .ctl input[type=range]{ width: clamp(90px, 30cqw, 160px); accent-color: var(--primary-color); }
        .pctv { font-variant-numeric: tabular-nums; font-weight:600; color: var(--primary-text-color); min-width:3ch; }
        .msg { padding: 10px 16px; color: var(--secondary-text-color); font-size:.9rem; }
        .msg code { background: var(--secondary-background-color); padding:1px 5px; border-radius:5px; }
        .rooms { display:grid; gap: 4px 16px; padding-top: 4px;
                 grid-template-columns: minmax(0, 1fr); }
        @container (min-width: 1100px) {
          .rooms.cols-auto { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @container (min-width: 700px) {
          .rooms.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        .room { padding: 4px 14px 14px; min-width: 0; }
        .room-head { display:flex; align-items:baseline; gap:10px; margin-bottom:6px; }
        .room-name { font-weight:600; color: var(--primary-text-color); }
        .room-id { font-size:.75rem; color: var(--secondary-text-color); font-family: var(--code-font-family, monospace); }
        .room-stat { margin-left:auto; font-size:.8rem; color: var(--secondary-text-color); }
        .room-stat b { color: var(--primary-color); font-size:1rem; font-variant-numeric: tabular-nums; }
        .scroll { overflow-x:auto; }
        /* Cells compress with the card; the scrollbar is a fallback that only
           kicks in once a cell would drop below 8px, not the normal path. */
        .grid { display:grid; gap:2px;
                min-width: calc(var(--dl-w) + var(--cols,24) * 8px);
                grid-template-columns: var(--dl-w) repeat(var(--cols,24), minmax(0,1fr)); }
        .hh { font-size:9px; text-align:center; color: var(--secondary-text-color);
              font-variant-numeric: tabular-nums; padding-bottom:2px; }
        .dl { font-size:10px; color: var(--secondary-text-color); display:flex; align-items:center;
              height: var(--cell-h); }
        .cell { height: var(--cell-h); border-radius:3px; }
        .cell.comfort { box-shadow: inset 0 0 0 2px var(--primary-color); }
        .cell.eco { opacity:.28; }
        /* "Never observed" must not look like a low probability — it isn't a
           probability at all. Hatched neutral, distinct from every ramp colour. */
        .cell.nodata { opacity:.5;
          background: repeating-linear-gradient(45deg,
            var(--secondary-background-color) 0 3px, transparent 3px 6px); }
        .legend { display:flex; align-items:center; gap:8px 16px; flex-wrap:wrap;
                  padding: 4px 16px 0; font-size:.75rem; color: var(--secondary-text-color); }
        .ramp { height:10px; width: clamp(90px, 25cqw, 160px); border-radius:6px;
                background: linear-gradient(90deg, ${RAMP[0]}, ${RAMP[1]} 34%, ${RAMP[2]} 64%, ${RAMP[3]}); }
        .sw { display:inline-block; width:13px; height:13px; border-radius:3px; vertical-align:-2px; }
        .sw.comfort { box-shadow: inset 0 0 0 2px var(--primary-color); }
        .sw.nodata { opacity:.5;
          background: repeating-linear-gradient(45deg,
            var(--secondary-text-color) 0 3px, transparent 3px 6px); }
        /* Progressive thinning of the hour ruler: keep the slot, drop the label,
           so the header never collapses into unreadable digits. */
        @container (max-width: 640px) { .hh:not(.m2) { visibility: hidden; } }
        @container (max-width: 520px) { .hh:not(.m3) { visibility: hidden; } }
        @container (max-width: 380px) { .hh:not(.m6) { visibility: hidden; } }
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
        `<span>${
          (this._config?.scale ?? "area") === "area" ? "area min–max" : "0–100%"
        }</span><span><span class="sw comfort"></span> comfort</span>` +
        `<span><span class="sw nodata"></span> no data</span>` +
        `<span>metric: ${esc(this._config?.metric ?? "raw")}</span></div>` +
        `<div class="rooms ${this._roomsClass()}">` +
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

  _roomsClass() {
    const c = String(this._config?.columns ?? "auto");
    return c === "1" || c === "2" ? `cols-${c}` : "cols-auto";
  }

  /** Per-slot values for the configured metric, falling back to the combined
   *  map when an older integration build doesn't send slots_raw. */
  _slotsOf(area) {
    const wantRaw = (this._config?.metric ?? "raw") === "raw";
    return (wantRaw && area.slots_raw) || area.slots || {};
  }

  /** Slots with zero weeks of data behind them: filled with a neutral
   *  fallback, not observed. Rendered as "no data", never as a probability. */
  _unknownOf(area) {
    const dp = area.data_points;
    if (!dp) return new Set();
    return new Set(Object.keys(dp).filter((k) => !dp[k]));
  }

  _room(name, area, cols, hoursPerSlot) {
    const slots = this._slotsOf(area);
    const unknown = this._unknownOf(area);
    const known = Object.entries(slots)
      .filter(([k]) => !unknown.has(k))
      .map(([, v]) => v);
    // Stretch the ramp over what this area actually spans, otherwise a room
    // whose values all sit in 7..43% reads as uniformly cold and looks untrained.
    const perArea = (this._config?.scale ?? "area") === "area" && known.length > 1;
    const lo = perArea ? Math.min(...known) : 0;
    const hi = perArea ? Math.max(...known) : 1;
    const norm = (v) => (hi > lo ? (v - lo) / (hi - lo) : 0.5);
    // With a stretched ramp an absolute cutoff is meaningless, so the comfort
    // threshold becomes a position within the area's own range.
    const cutoff = perArea ? lo + (hi - lo) * this._threshold : this._threshold;
    let climatized = 0;
    for (const [k, v] of Object.entries(slots)) {
      if (!unknown.has(k) && v >= cutoff) climatized += hoursPerSlot;
    }

    let head = "<div class='hh'></div>";
    for (let s = 0; s < cols; s++) {
      const hour = Math.round(s * hoursPerSlot);
      // Sub-hourly slots already halve the ruler; CSS thins it further by width.
      const label = cols <= 24 || s % 2 === 0 ? hour : "";
      const cls = ["hh"];
      if (hour % 2 === 0) cls.push("m2");
      if (hour % 3 === 0) cls.push("m3");
      if (hour % 6 === 0) cls.push("m6");
      head += `<div class="${cls.join(" ")}">${label}</div>`;
    }
    let rows = "";
    for (let d = 0; d < 7; d++) {
      rows += `<div class="dl">${esc(DAYS[d])}</div>`;
      for (let s = 0; s < cols; s++) {
        const key = `${d},${s}`;
        const p = slots[key];
        const hour = Math.round(s * hoursPerSlot);
        const hh = String(hour).padStart(2, "0");
        if (p === undefined || unknown.has(key)) {
          const why = p === undefined ? "not returned" : "never observed";
          rows += `<div class="cell nodata" title="${esc(
            `${DAYS[d]} ${hh}:00 · no data (${why})`
          )}"></div>`;
          continue;
        }
        const comfort = p >= cutoff;
        const title = `${DAYS[d]} ${hh}:00 · ${Math.round(p * 100)}% · ${
          comfort ? "comfort" : "eco/off"
        }`;
        rows += `<div class="cell ${comfort ? "comfort" : "eco"}" title="${esc(
          title
        )}" style="background:${thermal(norm(p))}"></div>`;
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
