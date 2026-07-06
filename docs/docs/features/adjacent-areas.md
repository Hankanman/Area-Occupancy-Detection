---
description: Let occupancy in one room influence its physically-connected neighbours
---

# Adjacent Areas

Adjacent Areas lets rooms that are physically connected — a hallway and a bedroom, a kitchen and a dining room — influence each other's occupancy calculation. Once the integration has learned how your household actually moves between two adjacent rooms, it uses that pattern in two ways:

1. **Boost**: if your history shows that leaving room A usually means arriving in room B a few minutes later, B's probability is nudged up when that pattern is under way.
2. **Slower decay**: if room B's only usual exit has been quiet since B's last motion, B's probability decays more slowly — it's less likely the person actually left if there's nowhere for them to have gone.

Both effects come from **learned transition history**, not from a fixed "influence" setting you configure. See [Transition Learning](../technical/transition-learning.md) for how that history is built and the underlying maths.

## Configuring adjacency

Adjacency is configured per area, in the area's options:

1. Open the area you want to configure and find the **Adjacent Areas** field on the basics step.
2. Select the other configured areas that are physically connected to this one.

!!! info "Configuration is symmetric"
    You only need to set this from one side. Adding Bedroom as adjacent to Hall automatically adds Hall as adjacent to Bedroom too — the integration writes both directions when you save. You can also edit it from either area and the result is the same.

You aren't setting a strength or weight for the relationship — how strongly one area influences another is entirely learned from observed transitions between them (see below). There's no "influence" slider to tune, by design.

## What to expect

Adjacency only affects an area once there's history to learn from:

- **Newly configured pairs**: no effect. The integration needs to observe real transitions between the two rooms before anything changes.
- **After some days of normal use**: as the household's analysis pipeline runs (hourly), it records transitions between adjacent areas and starts to learn how often — and at what time of day — moving between them happens.
- **Once learned**: the boost and decay-slowdown described above start to apply automatically. No further configuration is needed.

There's nothing to watch for during the learning period — the feature is silent until it has enough data, then it starts contributing in the background.

### Example: a quiet hallway keeps the bedroom "occupied"

Say Hall and Bedroom are configured as adjacent, and the integration has learned that leaving Bedroom nearly always means passing through Hall. One evening, Bedroom's motion sensor goes quiet — normally its probability would start decaying right away. But if Hall has also been quiet (no one has walked through it) since Bedroom's last motion, the integration treats "no one used the only known exit" as evidence the person is probably still in the room, so it decays Bedroom's probability more slowly than it otherwise would.

If Hall lights up with motion shortly after, that's read as evidence of an actual Bedroom → Hall move, and Bedroom's probability follows its normal decay from that point.

## Observing it

The [Diagnostics export](../technical/diagnostics.md) includes an `adjacency` block under each area's `current` section whenever the boost or decay modifier has fired for that area on the current tick. It shows:

- The two-hop trajectory (the last two rooms someone was recently in) and the hour-of-week bucket used for the lookup
- The learned probability that was looked up, and which smoothing fallback level supplied it
- The logit-space contribution added to the area's probability (boost), or the silence score and resulting decay multiplier (decay modifier), including a breakdown per silent neighbour

This is the most direct way to confirm the feature is active for a given area and to see exactly what it's doing on any given tick.

## FAQ

**Does this work immediately after I configure two areas as adjacent?**
No. There's no effect until the integration has recorded and learned enough transitions between the pair — this can take some days of normal use, and depends on how often you actually move between the two rooms.

**Can I control how strongly one area influences another?**
Not currently. Influence comes entirely from what's learned from your transition history; there's no per-pair weight or global gain exposed in the UI. The underlying constants are fixed in code for now (see [Transition Learning](../technical/transition-learning.md#tunables)).

**Do I need to configure adjacency in both directions?**
No — configuring it on either area applies it to both, automatically.

**Does removing an adjacency delete the learned history?**
The learned transition data isn't actively used for a pair that's no longer configured as adjacent, since new transitions are only counted for currently-configured pairs. Re-adding the pair later resumes learning; already-recorded data is aged out over time by the normal recency decay, same as any other pair.

**Will this work for areas that aren't physically adjacent?**
It's designed for physically connected rooms — the transition detection assumes a person leaving one area and arriving at the other shortly after. Configuring unrelated or non-adjacent areas will just mean the learned patterns stay noisy and contribute little.
