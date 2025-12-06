# Area Purpose and Decay Behaviour

The integration lets you assign an **Area Purpose** to each instance. The purpose describes how the room is typically used and determines a sensible default for the probability decay speed.

Selecting a purpose automatically sets the **Decay Half Life** – the time it takes for the occupancy probability to halve when no new activity occurs. You can still override this value in the options flow if required.

| Purpose     | Description                                                                         | Default Half Life        |
| ----------- | ----------------------------------------------------------------------------------- | ------------------------ |
| Passageway  | Short transit spaces such as hallways or landings. Evidence fades very quickly.     | Very short (~45 sec)     |
| Driveway    | Parking and vehicle access area. Brief transit for entering/exiting vehicles.       | Very short (~1 min)      |
| Utility     | Functional rooms like laundries or boot rooms.                                      | Short (~90 sec)          |
| Garage      | Storage and workshop space. Visits range from quick retrieval to extended projects. | Moderate (~3 min)        |
| Kitchen     | Kitchen work zones.                                                                 | Moderate (~4 min)        |
| Garden      | Outdoor activity space for gardening, relaxing, or yard work.                       | Moderate-long (~6 min)   |
| Bathroom    | Showers, baths, getting ready.                                                      | Moderate-long (~7.5 min) |
| Dining Room | Dining or breakfast areas.                                                          | Moderate-long (~8 min)   |
| Living Room | Living rooms or play areas.                                                         | Long (~8 minutes)        |
| Office      | Offices or desks.                                                                   | Long (~10 min)           |
| Media Room  | Lounges or reading nooks.                                                           | Long (~10 minutes)       |
| Bedroom     | Bedrooms and similar spaces.                                                        | Very long (~20 min)      |

The chosen purpose does not directly alter the Bayesian calculation beyond this decay timing. A shorter half life causes the probability to drop faster after activity stops; a longer half life keeps the area marked as occupied for longer.

## Purpose Justifications

Each purpose's half-life is tuned to match typical usage patterns:

- **Passageway (45s):** People walk through quickly (5-30 seconds typically). After someone leaves, the area should be empty almost immediately. A short half-life ensures quick clearing after transit.

- **Driveway (60s / 1 min):** People typically spend 10-30 seconds entering or exiting vehicles, with occasional brief pauses for loading or unloading. A very short half-life ensures the area clears quickly after vehicles depart, similar to passageways but accounting for brief stops.

- **Utility (90s):** Quick functional visits like grabbing detergent (10-30s) or putting on shoes (30-60s). A short half-life matches these brief interactions without unnecessary delay.

- **Garage (180s / 3 min):** Visits range from quick item retrieval (30-60 seconds) to extended projects (15-30+ minutes). A moderate half-life accommodates both brief functional visits and longer work sessions without premature clearing during active use.

- **Kitchen (240s / 4 min):** Kitchen work involves moving between stations (stove, fridge, sink, counter). Residents step away and return frequently. A moderate half-life prevents flicker when moving between work zones while still clearing reasonably quickly after cooking ends.

- **Garden (360s / 6 min):** Outdoor activities like gardening, relaxing, or yard work can involve periods of stillness where motion sensors may not detect activity. A moderate-long half-life accounts for sparse motion detection while maintaining occupancy during active outdoor use.

- **Bathroom (450s / 7.5 min):** Showers typically last 5-15 minutes, but motion sensors may not detect much movement during showers. A moderate-long half-life ensures the area stays marked as occupied throughout a typical shower, preventing lights from turning off.

- **Dining Room (480s / 8 min):** Meals typically last 10-20 minutes, but people sit relatively still while eating. A moderate-long half-life matches typical meal duration, accounting for periods of stillness between bites.

- **Living Room (480s / 8 min):** Conversations and board games create sporadic motion with quiet pauses. A moderate-long half-life allows evidence to fade gently, riding out quiet pauses without flickering.

- **Office (600s / 10 min):** Long seated sessions with occasional trips for coffee or printer. People can be very still while focused on work. A longer half-life prevents premature "vacant" detection during focused work periods while still clearing after extended absence.

- **Media Room (600s / 10 min):** People can remain very still while watching TV or reading. A longer half-life keeps the room marked as "occupied" through extended stretches of calm activity, matching the description's "quarter-hour memory" concept.

- **Bedroom (1200s / 20 min):** Deep sleep has minimal motion. A very long half-life prevents false vacancy during deep sleep while still allowing the house to revert to "empty" within a couple of hours after everyone gets up. This is especially important for preventing lights from turning off during the night.

## Dynamic Sleeping Decay

Areas with the `Bedroom` purpose have a special dynamic behavior tied to your household's sleep schedule.

- **During Sleep Hours:** Uses the configured `Bedroom` half-life to maintain occupancy probability for long periods of inactivity while you sleep.
- **Outside Sleep Hours:** Automatically switches to behave like a `Living Room` area, recognizing that bedrooms are often used for reading or getting ready during the day where shorter memory is appropriate.

You can configure your global `Sleep Start` and `Sleep End` times in the integration's global settings.
