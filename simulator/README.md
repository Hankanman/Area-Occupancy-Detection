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

## Installation

1. Install dependencies:
```bash
cd simulator
pip install -r requirements.txt
```

## Running the Simulator

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Paste your analysis output (YAML format) from the `area_occupancy.run_analysis` service into the text area

4. Click "Load Simulation" to initialize the simulator

5. Toggle sensors and watch the probability update in real-time!

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

