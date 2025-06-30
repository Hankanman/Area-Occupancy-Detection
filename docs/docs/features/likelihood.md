# Sensor Likelihoods

Each sensor has two learned values that describe how reliable it is as evidence of occupancy:

* **P(Active \| Occupied)** – how often the sensor is active when the area really is occupied.
* **P(Active \| Not Occupied)** – how often the sensor is active when the area is not occupied.

These values are called *likelihoods* and are calculated from your history data during the [Prior Learning](prior-learning.md) process. They are weighted according to the sensor type and used by the Bayesian calculation to update the occupancy probability whenever that sensor is active.

If history based learning is disabled, or insufficient history is available, default likelihoods from the integration are used instead.

You can manually refresh the stored likelihoods by calling the `area_occupancy.update_likelihoods` service.
