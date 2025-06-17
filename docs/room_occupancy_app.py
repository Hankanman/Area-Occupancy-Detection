# ruff: noqa: INP001
"""Room Occupancy Bayesian Fusion - Tkinter Demo (edge-triggered).

This GUI lets you configure three PIR sensors (entities), toggle their
raw motion state ON/OFF, and watch both the per-sensor posterior
probabilities **and** the overall room-occupancy probability update in
real-time.  It now performs a Bayesian update **only when the raw state
changes** (OFF→ON or ON→OFF). A continuous HIGH therefore pushes the
posterior once, after which belief decays according to the sensor's
half-life.

Run with:  python room_occupancy_app.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
import tkinter as tk

# ─────────────────────────────────────────────────────────────────────────────
#  Core Bayesian logic
# ─────────────────────────────────────────────────────────────────────────────

EPS = 1e-12
TICK_MS = 100  # milliseconds between updates
DEFAULT_HALF_LIFE = 10.0  # seconds until evidence halves (default 5 min)


def bayesian_probability(
    *,
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
    is_active: bool | None,
    weight: float,
    decay_factor: float,
) -> float:
    """Weighted, time-decaying single-step Bayesian update (fractional Bayes).

    This function implements a Bayesian update that:
    1. Weights the evidence based on sensor reliability
    2. Applies time decay to reduce evidence strength over time
    3. Handles both positive (ON) and negative (OFF) evidence

    Args:
        prior: Current probability estimate (0-1)
        prob_given_true: P(sensor=ON | room=occupied)
        prob_given_false: P(sensor=ON | room=empty)
        is_active: Current sensor state (True=ON, False=OFF, None=no change)
        weight: Sensor reliability weight (0-1)
        decay_factor: Time decay factor (0-1)

    Returns:
        Updated probability estimate (0-1)

    """
    # Return prior unchanged if no new evidence or zero weight/decay
    if is_active is None or weight == 0.0 or decay_factor == 0.0:
        return prior

    # Calculate likelihood ratio based on sensor state
    # For ON state: P(sensor=ON|occupied) / P(sensor=ON|empty)
    # For OFF state: P(sensor=OFF|occupied) / P(sensor=OFF|empty)
    likelihood_ratio = (
        (prob_given_true + EPS) / (prob_given_false + EPS)
        if is_active
        else (1.0 - prob_given_true + EPS) / (1.0 - prob_given_false + EPS)
    )

    # Apply sensor weight and time decay to likelihood ratio
    # This reduces evidence strength for less reliable sensors
    # and for evidence that is older
    likelihood_ratio **= weight * decay_factor

    # Convert probability to odds ratio
    prior_odds = prior / (1.0 - prior + EPS)

    # Update odds ratio using likelihood ratio
    post_odds = prior_odds * likelihood_ratio

    # Convert back to probability
    return post_odds / (1.0 + post_odds)


# ─────────────────────────────────────────────────────────────────────────────
#  Data classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Decay:
    """Decay class for the room occupancy Bayesian demo."""

    half_life: float = DEFAULT_HALF_LIFE  # seconds until evidence halves
    last_trigger_ts: float = field(default_factory=time.time)
    is_decaying: bool = False

    @property
    def decay_factor(self) -> float:
        """Decay factor for the entity."""

        if not self.is_decaying:
            return 1.0
        age = time.time() - self.last_trigger_ts
        if age <= 0:
            return 1.0
        factor = math.pow(0.5, age / self.half_life)
        if factor < 0.05:
            self.is_decaying = False
            return 0.0
        return factor


@dataclass
class Prior:
    """Prior class for the room occupancy Bayesian demo."""

    prob_given_true: float = 0.9
    prob_given_false: float = 0.1


@dataclass
class Entity:
    """Entity class for the room occupancy Bayesian demo."""

    entity_id: str
    prior: Prior
    weight: float = 1.0

    # live state from sensor --------------------------------------------------
    current_state: bool | None = None  # current hardware reading (True=HIGH)
    previous_state: bool | None = None  # previous reading to detect edges

    decay: Decay = field(default_factory=Decay)
    belief: float = 0.5  # per-sensor posterior

    # GUI widgets (filled later) ---------------------------------------------
    lbl_belief: tk.Label | None = field(default=None, repr=False)
    lbl_decay: tk.Label | None = field(default=None, repr=False)
    lbl_state: tk.Label | None = field(default=None, repr=False)
    entry_p_true: tk.Entry | None = field(default=None, repr=False)
    entry_p_false: tk.Entry | None = field(default=None, repr=False)
    entry_weight: tk.Entry | None = field(default=None, repr=False)
    entry_half_life: tk.Entry | None = field(default=None, repr=False)
    btn_toggle: tk.Button | None = field(default=None, repr=False)

    # ---------------------------------------------------------------------
    def toggle_state(self):
        """Simulate hardware level flip (OFF<->ON)."""
        self.current_state = (
            not self.current_state if self.current_state is not None else True
        )

    def state_edge(self) -> bool | None:
        """Return edge value (True, False) or None if no change."""
        if self.current_state == self.previous_state:
            return None
        self.previous_state = self.current_state
        return self.current_state

    def update_params_from_ui(self):
        """Update parameters from UI."""
        try:
            if self.entry_p_true is not None:
                self.prior.prob_given_true = float(self.entry_p_true.get())
            if self.entry_p_false is not None:
                self.prior.prob_given_false = float(self.entry_p_false.get())
            if self.entry_weight is not None:
                self.weight = float(self.entry_weight.get())
            if self.entry_half_life is not None:
                self.decay.half_life = float(self.entry_half_life.get())
        except ValueError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Tkinter GUI helpers
# ─────────────────────────────────────────────────────────────────────────────


def build_entity_frame(parent: tk.Tk, ent: Entity) -> tk.LabelFrame:
    """Build the entity frame for the entity."""
    f = tk.LabelFrame(parent, text=ent.entity_id, padx=5, pady=5)

    # Probabilities ---------------------------------------------------------
    tk.Label(f, text="P(E|H)").grid(row=0, column=0, sticky="e")
    ent.entry_p_true = tk.Entry(f, width=6)
    ent.entry_p_true.insert(0, str(ent.prior.prob_given_true))
    ent.entry_p_true.grid(row=0, column=1)

    tk.Label(f, text="P(E|¬H)").grid(row=0, column=2, sticky="e")
    ent.entry_p_false = tk.Entry(f, width=6)
    ent.entry_p_false.insert(0, str(ent.prior.prob_given_false))
    ent.entry_p_false.grid(row=0, column=3)

    # Weight & half-life ----------------------------------------------------
    tk.Label(f, text="Weight").grid(row=1, column=0, sticky="e")
    ent.entry_weight = tk.Entry(f, width=6)
    ent.entry_weight.insert(0, str(ent.weight))
    ent.entry_weight.grid(row=1, column=1)

    tk.Label(f, text="Half-life [s]").grid(row=1, column=2, sticky="e")
    ent.entry_half_life = tk.Entry(f, width=6)
    ent.entry_half_life.insert(0, str(ent.decay.half_life))
    ent.entry_half_life.grid(row=1, column=3)

    # Controls --------------------------------------------------------------
    ent.btn_toggle = tk.Button(f, text="Toggle", command=ent.toggle_state)
    ent.btn_toggle.grid(row=2, column=0, columnspan=2, pady=2)

    ent.lbl_belief = tk.Label(f, text="Belief: 0.50", width=12, bg="white")
    ent.lbl_belief.grid(row=2, column=2, columnspan=2)

    ent.lbl_decay = tk.Label(f, text="Decay: 1.00", width=12, bg="white")
    ent.lbl_decay.grid(row=3, column=2, columnspan=2)

    ent.lbl_state = tk.Label(f, text="State: OFF", width=12, bg="white")
    ent.lbl_state.grid(row=4, column=2, columnspan=2)

    return f


# ─────────────────────────────────────────────────────────────────────────────
#  Main application
# ─────────────────────────────────────────────────────────────────────────────


class OccupancyApp:
    """Main application class for the room occupancy Bayesian demo."""

    def __init__(self, master: tk.Tk):
        """Initialize the occupancy app."""
        self.master = master
        master.title("Room Occupancy - Bayesian Demo")

        # Sensors -----------------------------------------------------------
        self.entities: dict[str, Entity] = {
            f"PIR-{i + 1}": Entity(entity_id=f"PIR-{i + 1}", prior=Prior())
            for i in range(3)
        }
        for row, ent in enumerate(self.entities.values()):
            build_entity_frame(master, ent).grid(
                row=row, column=0, padx=5, pady=5, sticky="ew"
            )

        # Global controls ---------------------------------------------------
        g = tk.LabelFrame(master, text="Global", padx=5, pady=5)
        g.grid(row=0, column=1, rowspan=3, padx=5, pady=5, sticky="ns")

        tk.Label(g, text="Prior P(occupied)").grid(row=0, column=0, sticky="e")
        self.prior_occupied = tk.Entry(g, width=6)
        self.prior_occupied.insert(0, "0.2")
        self.prior_occupied.grid(row=0, column=1)

        self.lbl_overall = tk.Label(g, text="P(occupied): --", width=18, bg="white")
        self.lbl_overall.grid(row=1, column=0, columnspan=2, pady=6)

        self.master.after(TICK_MS, self.update_loop)

    # ------------------------------------------------------------------
    def update_loop(self):
        """Update the occupancy app."""
        # read GUI boxes --------------------------------------------------
        for ent in self.entities.values():
            ent.update_params_from_ui()

        # global prior ----------------------------------------------------
        try:
            prior_occupied = float(self.prior_occupied.get())
        except ValueError:
            prior_occupied = 0.2
        prob_occupied = prior_occupied  # Use the entered value directly

        # per-entity update ----------------------------------------------
        for entity in self.entities.values():
            is_active = entity.state_edge()  # None if no change

            # manage decay timer -----------------------------------------
            if is_active is True:  # rising edge
                entity.decay.is_decaying = False  # stop decay when ON
            if is_active is False:  # falling edge
                entity.decay.is_decaying = True
                entity.decay.last_trigger_ts = time.time()

            entity.belief = bayesian_probability(
                prior=entity.belief,
                prob_given_true=entity.prior.prob_given_true,
                prob_given_false=entity.prior.prob_given_false,
                is_active=is_active,
                weight=entity.weight,
                decay_factor=entity.decay.decay_factor
                if entity.decay.is_decaying
                else 1.0,
            )
            if entity.lbl_belief is not None:
                entity.lbl_belief.configure(text=f"Belief: {entity.belief:0.2f}")
            if entity.lbl_decay is not None:
                entity.lbl_decay.configure(
                    text=f"Decay: {entity.decay.decay_factor:0.2f}"
                )
            if entity.lbl_state is not None:
                state_text = "ON" if entity.current_state else "OFF"
                entity.lbl_state.configure(text=f"State: {state_text}")

        # room-level fusion ----------------------------------------------
        posterior = prob_occupied
        for entity in self.entities.values():
            # Only apply Bayesian update if sensor has active evidence
            if entity.current_state is True or entity.decay.is_decaying:
                posterior = bayesian_probability(
                    prior=posterior,
                    prob_given_true=entity.prior.prob_given_true,
                    prob_given_false=entity.prior.prob_given_false,
                    is_active=True,  # use weight * decay as fractional power
                    weight=entity.weight * entity.decay.decay_factor,
                    decay_factor=1.0,
                )
        self.lbl_overall.configure(text=f"P(occupied): {posterior:0.3f}")

        # schedule next tick ---------------------------------------------
        self.master.after(TICK_MS, self.update_loop)


# ─────────────────────────────────────────────────────────────────────────────
#  Entrypoint
# ─────────────────────────────────────────────────────────────────────────────


def main():
    """Room occupancy Bayesian demo."""

    tk_root = tk.Tk()
    OccupancyApp(tk_root)
    tk_root.mainloop()


if __name__ == "__main__":
    main()
