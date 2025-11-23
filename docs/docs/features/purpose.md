# Area Purpose and Decay Behaviour

The integration lets you assign an **Area Purpose** to each instance. The purpose describes how the room is typically used and determines a sensible default for the probability decay speed.

Selecting a purpose automatically sets the **Decay Half Life** – the time it takes for the occupancy probability to halve when no new activity occurs. You can still override this value in the options flow if required.

| Purpose                  | Description                                                                     | Default Half Life (s) |
| ------------------------ | ------------------------------------------------------------------------------- | --------------------- |
| Passageway               | Short transit spaces such as hallways or landings. Evidence fades very quickly. | 60                    |
| Utility                  | Functional rooms like laundries or boot rooms.                                  | 120                   |
| Food‑Prep                | Kitchen work zones.                                                             | 300                   |
| Bathroom / Personal Care | Showers, baths, getting ready.                                                  | 600                   |
| Eating                   | Dining or breakfast areas.                                                      | 600                   |
| Working / Studying       | Offices or desks.                                                               | 600                   |
| Social                   | Living rooms or play areas.                                                     | 720                   |
| Relaxing                 | Lounges or reading nooks.                                                       | 900                   |
| Sleeping                 | Bedrooms and similar spaces.                                                    | 1800                  |

The chosen purpose does not directly alter the Bayesian calculation beyond this decay timing. A shorter half life causes the probability to drop faster after activity stops; a longer half life keeps the area marked as occupied for longer.

## Dynamic Sleeping Decay

Areas with the `Sleeping` purpose have a special dynamic behavior tied to your household's sleep schedule.

* **During Sleep Hours:** Uses the configured `Sleeping` half-life (default 1800s) to maintain occupancy probability for long periods of inactivity while you sleep.
* **Outside Sleep Hours:** Automatically switches to behave like a `Relaxing` area (900s half-life), recognizing that bedrooms are often used for reading or getting ready during the day where shorter memory is appropriate.

You can configure your global `Sleep Start` and `Sleep End` times in the integration's global settings.
