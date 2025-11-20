# Data Flow Diagrams

This document provides visual flow diagrams showing how data moves through the area occupancy calculation system.

## Initialization Flow

The initialization sequence diagram shows how the system sets up when Home Assistant starts:

```mermaid
sequenceDiagram
    participant HA as Home Assistant
    participant Coord as Coordinator
    participant Area as Area
    participant DB as Database
    participant Prior as Prior
    participant Entities as EntityManager
    participant Analysis as Analysis

    HA->>Coord: async_config_entry_first_refresh()
    Coord->>Coord: setup()
    Coord->>Area: Load areas from config
    Area->>Area: Initialize components
    Coord->>DB: async_init_database()
    DB-->>Coord: Database ready
    Coord->>DB: load_data()
    DB->>Prior: Load global priors
    DB->>Entities: Load entity configs
    DB->>Entities: Load likelihoods
    Prior-->>Coord: Priors loaded
    Entities-->>Coord: Entities loaded
    Coord->>Coord: track_entity_state_changes()
    Coord->>Coord: Start timers (decay, save, analysis)
    Coord-->>HA: Setup complete
```

## Learning Phase Flow

The learning phase flowchart shows how priors and likelihoods are learned from historical data:

```mermaid
flowchart TD
    Start([Analysis Timer Fires]) --> Sync[Sync States from Recorder]
    Sync --> HealthCheck[Database Health Check]
    HealthCheck --> Prune[Prune Old Intervals]
    Prune --> PriorAnalysis[Run Prior Analysis]

    PriorAnalysis --> GetMotion[Get Motion Intervals]
    GetMotion --> CalcMotion[Calculate Motion Prior]
    CalcMotion --> CheckLow{Prior < 0.10?}
    CheckLow -->|Yes| GetMedia[Get Media/Appliance Intervals]
    CheckLow -->|No| SaveGlobal[Save Global Prior]
    GetMedia --> CalcAll[Calculate Combined Prior]
    CalcAll --> SaveGlobal

    SaveGlobal --> TimePrior[Calculate Time Priors]
    TimePrior --> Aggregate[Aggregate by Day/Slot]
    Aggregate --> CalcTime[Calculate Prior per Slot]
    CalcTime --> SaveTime[Save Time Priors]

    SaveTime --> LikelihoodAnalysis[Run Likelihood Analysis]
    LikelihoodAnalysis --> GetOccupied[Get Occupied Intervals]
    GetOccupied --> ForEach[For Each Entity]
    ForEach --> GetIntervals[Get Entity Intervals]
    GetIntervals --> Correlate[Correlate with Occupied]
    Correlate --> CalcLikelihood[Calculate Likelihoods]
    CalcLikelihood --> SaveLikelihood[Save Likelihoods]
    SaveLikelihood --> MoreEntities{More Entities?}
    MoreEntities -->|Yes| ForEach
    MoreEntities -->|No| Refresh[Refresh Coordinator]
    Refresh --> Save[Save to Database]
    Save --> End([Analysis Complete])
```

## Real-Time Update Flow

The real-time update sequence diagram shows what happens when a sensor state changes:

```mermaid
sequenceDiagram
    participant Sensor as Sensor Entity
    participant HA as Home Assistant
    participant Coord as Coordinator
    participant Entity as Entity Object
    participant Area as Area
    participant Prior as Prior
    participant Bayes as Bayesian Calc
    participant Output as Output Sensors

    Sensor->>HA: State Change Event
    HA->>Coord: State Change Callback
    Coord->>Entity: has_new_evidence()
    Entity->>HA: Get Current State
    HA-->>Entity: State Value
    Entity->>Entity: Determine Evidence
    Entity->>Entity: Check Decay Transition

    alt Evidence Changed
        Entity->>Entity: Update Decay State
        Entity-->>Coord: Transition Detected
        Coord->>Coord: async_refresh()
        Coord->>Area: probability()
        Area->>Prior: value (get combined prior)
        Prior->>Prior: Get Global Prior
        Prior->>Prior: Get Time Prior
        Prior->>Prior: combine_priors()
        Prior-->>Area: Combined Prior
        Area->>Bayes: bayesian_probability()
        Bayes->>Entity: Get Evidence & Likelihoods
        Entity-->>Bayes: Evidence Data
        Bayes->>Bayes: Calculate in Log Space
        Bayes-->>Area: Final Probability
        Area->>Output: Update Probability Sensor
        Area->>Output: Update Status Binary
        Area->>Output: Update Other Sensors
    end
```

## Entity State Change to Probability Update Flow

This flowchart shows the detailed flow from entity state change to final probability:

```mermaid
flowchart TD
    Start([Entity State Changes]) --> GetState[Get State from HA]
    GetState --> CheckAvailable{State Available?}
    CheckAvailable -->|No| Skip[Skip Entity]
    CheckAvailable -->|Yes| CheckActive{State Active?}

    CheckActive -->|Yes| EvidenceTrue[Evidence = True]
    CheckActive -->|No| EvidenceFalse[Evidence = False]

    EvidenceTrue --> CheckPrevious{Previous Evidence?}
    EvidenceFalse --> CheckPrevious

    CheckPrevious -->|Was False| Transition[Transition Detected]
    CheckPrevious -->|Was True| NoTransition[No Transition]
    CheckPrevious -->|Was None| NoTransition

    Transition --> CheckDirection{Which Direction?}
    CheckDirection -->|False→True| StopDecay[Stop Decay]
    CheckDirection -->|True→False| StartDecay[Start Decay]

    StopDecay --> UpdatePrevious[Update Previous Evidence]
    StartDecay --> UpdatePrevious
    NoTransition --> UpdatePrevious

    UpdatePrevious --> GetPrior[Get Combined Prior]
    GetPrior --> GetGlobal[Get Global Prior]
    GetPrior --> GetTime[Get Time Prior]
    GetGlobal --> Combine[Combine in Logit Space]
    GetTime --> Combine

    Combine --> InitLog[Initialize Log Probabilities]
    InitLog --> FilterEntities[Filter Entities]
    FilterEntities --> ForEach[For Each Entity]

    ForEach --> GetEvidence[Get Effective Evidence]
    GetEvidence --> CheckDecay{Is Decaying?}
    CheckDecay -->|Yes| Interpolate[Interpolate Likelihoods]
    CheckDecay -->|No| UseLearned[Use Learned Likelihoods]

    Interpolate --> Weight[Apply Entity Weight]
    UseLearned --> Weight
    Weight --> Accumulate[Accumulate Log Contributions]
    Accumulate --> More{More Entities?}
    More -->|Yes| ForEach
    More -->|No| Normalize[Normalize to Probability]

    Normalize --> Clamp[Clamp to Valid Range]
    Clamp --> UpdateOutput[Update Output Sensors]
    UpdateOutput --> End([Complete])

    Skip --> End
```

## Component Interaction Diagram

This graph shows how the main components interact:

```mermaid
graph TB
    subgraph "Home Assistant"
        HA[Home Assistant Core]
        States[State Registry]
        Recorder[Recorder Database]
    end

    subgraph "Area Occupancy Integration"
        Coord[AreaOccupancyCoordinator]

        subgraph "Area Components"
            Area[Area]
            Config[AreaConfig]
            Prior[Prior]
            Entities[EntityManager]
            Purpose[Purpose]
        end

        subgraph "Analysis"
            PriorAnalysis[PriorAnalyzer]
            LikelihoodAnalysis[LikelihoodAnalyzer]
        end

        subgraph "Database"
            DB[(AreaOccupancyDB)]
            Tables[Tables: Areas, Entities, Priors, Intervals]
        end

        subgraph "Calculation"
            Utils[Bayesian Utils]
            Decay[Decay Model]
        end

        subgraph "Output"
            Sensors[HA Sensors]
            Binary[Binary Sensors]
        end
    end

    HA -->|State Changes| Coord
    States -->|Get State| Entities
    Recorder -->|Sync History| DB

    Coord -->|Manages| Area
    Area -->|Contains| Config
    Area -->|Contains| Prior
    Area -->|Contains| Entities
    Area -->|Contains| Purpose

    Coord -->|Orchestrates| PriorAnalysis
    Coord -->|Orchestrates| LikelihoodAnalysis
    PriorAnalysis -->|Reads/Writes| DB
    LikelihoodAnalysis -->|Reads/Writes| DB

    Prior -->|Gets Data| DB
    Entities -->|Gets Data| DB
    Entities -->|Manages| Decay

    Area -->|Calls| Utils
    Utils -->|Uses| Prior
    Utils -->|Uses| Entities
    Entities -->|Uses| Decay

    Coord -->|Updates| Sensors
    Area -->|Provides Data| Binary

    DB -->|Stores| Tables
```

## Database Data Flow

This diagram shows how data flows through the database:

```mermaid
flowchart LR
    subgraph "Data Sources"
        Recorder[HA Recorder]
        Config[Configuration]
        Analysis[Analysis Results]
    end

    subgraph "Database Operations"
        Sync[Sync States]
        Save[Save Data]
        Load[Load Data]
    end

    subgraph "Database Tables"
        Intervals[(Intervals)]
        Entities[(Entities)]
        Areas[(Areas)]
        Priors[(Priors)]
        GlobalPriors[(GlobalPriors)]
    end

    subgraph "In-Memory Objects"
        EntityObjs[Entity Objects]
        PriorObj[Prior Object]
        AreaObj[Area Object]
    end

    Recorder -->|State History| Sync
    Sync -->|Create Intervals| Intervals

    Config -->|Entity Config| Save
    Save -->|Store Config| Entities
    Save -->|Store State| Entities

    Analysis -->|Prior Values| Save
    Save -->|Store Global| GlobalPriors
    Save -->|Store Time| Priors
    Analysis -->|Likelihoods| Save
    Save -->|Update| Entities

    Load -->|Read Config| Entities
    Load -->|Read Priors| GlobalPriors
    Load -->|Read Time| Priors
    Load -->|Read Likelihoods| Entities

    Entities -->|Create| EntityObjs
    GlobalPriors -->|Set| PriorObj
    Priors -->|Get| PriorObj
    Entities -->|Update| EntityObjs

    EntityObjs -->|Used By| AreaObj
    PriorObj -->|Used By| AreaObj
```

## Decay Flow

This flowchart shows how decay is managed:

```mermaid
flowchart TD
    Start([Entity State Check]) --> GetEvidence[Get Current Evidence]
    GetEvidence --> Compare{Compare with Previous}
    Compare -->|Same| NoChange[No Change]
    Compare -->|Different| Transition[Transition Detected]

    NoChange --> End([Continue])

    Transition --> CheckDirection{Which Direction?}
    CheckDirection -->|False → True| Active[Evidence Active]
    CheckDirection -->|True → False| Inactive[Evidence Inactive]

    Active --> StopDecay[Stop Decay if Active]
    StopDecay --> UpdatePrev[Update Previous Evidence]
    UpdatePrev --> End

    Inactive --> StartDecay[Start Decay]
    StartDecay --> SetStart[Set Decay Start Time]
    SetStart --> UpdatePrev

    UpdatePrev --> CalcDecay[Calculate Decay Factor]
    CalcDecay --> Formula[decay_factor = 0.5^age/half_life]
    Formula --> CheckExpired{Decay < 5%?}
    CheckExpired -->|Yes| Expire[Stop Decay]
    CheckExpired -->|No| ContinueDecay[Continue Decay]

    Expire --> End
    ContinueDecay --> UseDecay[Use Decay in Calculation]
    UseDecay --> Interpolate[Interpolate Likelihoods]
    Interpolate --> End
```

## See Also

- [Complete Calculation Flow](calculation-flow.md) - Detailed text explanation
- [Prior Calculation Deep Dive](prior-calculation.md) - Prior learning details
- [Likelihood Calculation Deep Dive](likelihood-calculation.md) - Likelihood learning details
- [Bayesian Calculation Deep Dive](bayesian-calculation.md) - Calculation details
- [Entity Evidence Collection](entity-evidence.md) - Evidence collection details

