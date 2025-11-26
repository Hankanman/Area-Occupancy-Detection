# Visualize Sensor Distributions

This script visualizes the learned distributions for numeric sensors, showing both the raw data histograms and the fitted Gaussian distributions used for occupancy detection.

## Features

- **Dual Visualization**: Shows both frequency histograms and probability density functions
- **Gaussian Overlay**: Displays the learned Gaussian distributions overlaid on the raw data
- **Statistics Summary**: Prints detailed statistics about the distributions
- **Entity Listing**: Can list all available entities with correlations

## Installation

The script requires `matplotlib` which is included in `requirements_test.txt`. Install it with:

```bash
pip install -r requirements_test.txt
```

Or just matplotlib:

```bash
pip install matplotlib
```

## Usage

### Visualize a Specific Sensor

```bash
python scripts/visualize_distributions.py <area_name> <entity_id>
```

**Example:**

```bash
python scripts/visualize_distributions.py "Living Room" sensor.temperature
```

### Custom Options

```bash
# Use custom database path
python scripts/visualize_distributions.py "Living Room" sensor.temperature \
    --db-path /path/to/area_occupancy.db

# Analyze different time period (default: 30 days)
python scripts/visualize_distributions.py "Living Room" sensor.temperature --days 60

# Adjust histogram bins (default: 50)
python scripts/visualize_distributions.py "Living Room" sensor.temperature --bins 100

# Save visualization to file instead of displaying
python scripts/visualize_distributions.py "Living Room" sensor.temperature --output plot.png
```

### List Available Entities

```bash
# List all entities with correlations
python scripts/visualize_distributions.py --list

# List entities for a specific area
python scripts/visualize_distributions.py --list --area "Living Room"
```

## Output

By default, the script displays the visualization in an interactive matplotlib window. You can also save it to a file using the `--output` option.

The script generates two plots:

1. **Frequency Histogram**: Shows the raw distribution of sensor values during occupied vs unoccupied periods, with Gaussian curves overlaid
2. **Probability Density Functions**: Shows the normalized probability densities, useful for understanding how the Bayesian likelihood calculation works

The script also prints:

- Total sample counts
- Mean and standard deviation for each state
- Learned Gaussian parameters (μ and σ)
- Correlation coefficient and type

### Output Modes

- **Interactive Display** (default): Uses `plt.show()` to display plots in a window
- **File Output**: Use `--output <filename>` to save as PNG, PDF, SVG, etc.
  - Example: `--output temperature_distribution.png`
  - The file format is determined by the extension (.png, .pdf, .svg, etc.)

## Understanding the Visualizations

### Histogram Plot (Top)

- **Red bars**: Sensor values when area is occupied
- **Blue bars**: Sensor values when area is unoccupied
- **Red dashed line**: Learned Gaussian distribution for occupied state
- **Blue dashed line**: Learned Gaussian distribution for unoccupied state

The Gaussian curves show how well the learned parameters fit the actual data distribution.

### PDF Plot (Bottom)

- **Red area**: Normalized probability density for occupied state
- **Blue area**: Normalized probability density for unoccupied state
- **Dashed lines**: Theoretical Gaussian PDFs

This plot shows the probability densities used in the Bayesian calculation. When a new sensor value arrives, the system calculates:

- `P(value | Occupied)` using the red distribution
- `P(value | Unoccupied)` using the blue distribution

These densities are then used in the Bayesian update formula to determine occupancy probability.

## Database Location

By default, the script looks for the database at:

```
config/.storage/area_occupancy.db
```

You can override this with the `--db-path` option.

## Requirements

- Python 3.13+
- matplotlib >= 3.7.0
- numpy >= 1.24.0 (already in project dependencies)
- Access to the Home Assistant database file
