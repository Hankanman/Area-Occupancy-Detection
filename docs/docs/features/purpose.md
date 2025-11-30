# Area Purpose and Decay Behaviour

The integration lets you assign an **Area Purpose** to each instance. The purpose describes how the room is typically used and determines a sensible default for the probability decay speed.

Selecting a purpose automatically sets the **Decay Half Life** – the time it takes for the occupancy probability to halve when no new activity occurs. You can still override this value in the options flow if required.

| Purpose                  | Description                                                                     | Default Half Life        |
| ------------------------ | ------------------------------------------------------------------------------- | ------------------------ |
| Passageway               | Short transit spaces such as hallways or landings. Evidence fades very quickly. | Very short (~45 sec)     |
| Utility                  | Functional rooms like laundries or boot rooms.                                  | Short (~90 sec)          |
| Food-Prep                | Kitchen work zones.                                                             | Moderate (~4 min)        |
| Bathroom / Personal Care | Showers, baths, getting ready.                                                  | Moderate-long (~7.5 min) |
| Eating                   | Dining or breakfast areas.                                                      | Moderate-long (~8 min)   |
| Working / Studying       | Offices or desks.                                                               | Long (~10 min)           |
| Social                   | Living rooms or play areas.                                                     | Long (~8 minutes)        |
| Relaxing                 | Lounges or reading nooks.                                                       | Long (~10 minutes)       |
| Sleeping                 | Bedrooms and similar spaces.                                                    | Very long (~20 min)      |

The chosen purpose does not directly alter the Bayesian calculation beyond this decay timing. A shorter half life causes the probability to drop faster after activity stops; a longer half life keeps the area marked as occupied for longer.

## Purpose Justifications

Each purpose's half-life is tuned to match typical usage patterns:

- **Passageway (45s):** People walk through quickly (5-30 seconds typically). After someone leaves, the area should be empty almost immediately. A short half-life ensures quick clearing after transit.

- **Utility (90s):** Quick functional visits like grabbing detergent (10-30s) or putting on shoes (30-60s). A short half-life matches these brief interactions without unnecessary delay.

- **Food-Prep (240s / 4 min):** Kitchen work involves moving between stations (stove, fridge, sink, counter). Residents step away and return frequently. A moderate half-life prevents flicker when moving between work zones while still clearing reasonably quickly after cooking ends.

- **Bathroom (450s / 7.5 min):** Showers typically last 5-15 minutes, but motion sensors may not detect much movement during showers. A moderate-long half-life ensures the area stays marked as occupied throughout a typical shower, preventing lights from turning off.

- **Eating (480s / 8 min):** Meals typically last 10-20 minutes, but people sit relatively still while eating. A moderate-long half-life matches typical meal duration, accounting for periods of stillness between bites.

- **Working (600s / 10 min):** Long seated sessions with occasional trips for coffee or printer. People can be very still while focused on work. A longer half-life prevents premature "vacant" detection during focused work periods while still clearing after extended absence.

- **Social (480s / 8 min):** Conversations and board games create sporadic motion with quiet pauses. A moderate-long half-life allows evidence to fade gently, riding out quiet pauses without flickering.

- **Relaxing (600s / 10 min):** People can remain very still while watching TV or reading. A longer half-life keeps the room marked as "occupied" through extended stretches of calm activity, matching the description's "quarter-hour memory" concept.

- **Sleeping (1200s / 20 min):** Deep sleep has minimal motion. A very long half-life prevents false vacancy during deep sleep while still allowing the house to revert to "empty" within a couple of hours after everyone gets up. This is especially important for preventing lights from turning off during the night.

## Dynamic Sleeping Decay

Areas with the `Sleeping` purpose have a special dynamic behavior tied to your household's sleep schedule.

- **During Sleep Hours:** Uses the configured `Sleeping` half-life to maintain occupancy probability for long periods of inactivity while you sleep.
- **Outside Sleep Hours:** Automatically switches to behave like a `Relaxing` area, recognizing that bedrooms are often used for reading or getting ready during the day where shorter memory is appropriate.

You can configure your global `Sleep Start` and `Sleep End` times in the integration's global settings.
