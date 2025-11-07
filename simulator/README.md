# Area Occupancy Simulator

A web-based simulator for the Area Occupancy Detection integration that allows you to paste analysis output and interactively explore how sensor states affect occupancy probability.

## Features

- **YAML Input**: Paste analysis output from the `area_occupancy.run_analysis` service
- **Interactive Sensor Controls**:
  - Toggle binary sensors (motion, doors, windows, etc.)
  - Adjust numeric sensor values (temperature, humidity, illuminance)
- **Real-time Probability Calculation**: Uses the actual integration code for accurate results
- **Probability Breakdown**: See how each sensor contributes to the overall probability
- **Time-series Chart**: Visualize how probability changes over time as you toggle sensors

## Local Development

1. Create and activate the project virtual environment (if it is not already available):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install simulator dependencies from the repository root:

```bash
pip install -r simulator/requirements.txt
```

3. Start the backend API by running the repository entry point (`main.py`):

```bash
python main.py
```

   The server listens on `0.0.0.0` and uses port `5000` by default. Set the `PORT` environment variable to override the port, and `FLASK_DEBUG=0` to disable debug mode.

4. (Optional) Start the MkDocs preview to load the simulator UI:

```bash
cd docs
mkdocs serve
```

   Then open `http://localhost:8000/Area-Occupancy-Detection/simulator/` and update the **API Connection** field to `http://127.0.0.1:5000`.

5. Paste your `area_occupancy.run_analysis` YAML output, click **Load Simulation**, and interact with the sensor controls to see live probability updates.

## Configuration

- `SIMULATOR_ALLOWED_ORIGINS`: Comma-separated list of origins allowed to access the API (defaults include GitHub Pages and `http://localhost:8000`). Use `*` during private testing.
- `PORT`: Overrides the HTTP port (`5000` default when launched via `main.py`).
- `FLASK_DEBUG`: Set to `0` to disable Flask debug mode (defaults to `1`).

## Example Analysis Output

```yaml
area_name: Toilet
current_prior: 0.1
global_prior: 0.02996960406235303
time_prior: 0.01
prior_entity_ids:
  - binary_sensor.toilet_motion_1_occupancy
total_entities: 6
entity_states:
  binary_sensor.toilet_motion_1_occupancy: "off"
  sensor.toilet_humidity: "75.68"
  binary_sensor.toilet_window_contact: "off"
  binary_sensor.toilet_door_contact: "off"
  sensor.toilet_temperature: "19.21"
  sensor.toilet_illuminance: "1.0"
likelihoods:
  binary_sensor.toilet_motion_1_occupancy:
    type: motion
    weight: 0.85
    prob_given_true: 0.03199172557922136
    prob_given_false: 0.01
  # ... more entities
update_timestamp: "2025-11-06T17:55:00.730200+00:00"
```

## How It Works

The simulator uses the real integration code:
- `bayesian_probability()` from `custom_components/area_occupancy/utils.py`
- `EntityType` definitions from `custom_components/area_occupancy/data/entity_type.py`
- Same probability calculation logic as the actual integration

This ensures that the simulator results match what you would see in Home Assistant.

## UI Components

- **Area Information**: Shows area name and prior probabilities
- **Probability Display**: Large, color-coded probability percentage
- **Sensor List**: All sensors with their current states and controls
- **Sensor Contributions**: Breakdown showing how each sensor affects probability
- **Probability Chart**: Time-series visualization of probability changes

## Notes

- The simulator does not include time-based decay (decay_factor is always 1.0)
- Sensor states are updated immediately when toggled
- The chart tracks the last 50 probability updates

