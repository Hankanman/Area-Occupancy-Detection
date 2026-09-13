"""Declarative descriptions of the instances the harness can build.

A profile is the single source of truth for one throwaway instance: which
areas exist, which mock sensors each area has, and how those sensors behave
relative to occupancy. Every other module derives its output from it --
``mock_config`` renders the entities into YAML, ``storage`` writes the config
entry that points at them, and ``history`` synthesises intervals that match
the declared correlations. Change a profile and all three follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from custom_components.area_occupancy.const import (
    CONF_APPLIANCES,
    CONF_CO2_SENSORS,
    CONF_CUSTOM_BINARY_SENSORS,
    CONF_CUSTOM_NUMERIC_SENSORS,
    CONF_DOOR_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_ILLUMINANCE_SENSORS,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_TEMPERATURE_SENSORS,
    CONF_WINDOW_SENSORS,
)
from custom_components.area_occupancy.data.entity_type import InputType


@dataclass(frozen=True, slots=True)
class NumericModel:
    """How a numeric sensor's readings differ between empty and occupied.

    Attributes:
        empty_mean: Mean reading while the area is empty.
        occupied_mean: Mean reading while the area is occupied.
        sd: Standard deviation applied to both means.
        decimals: Rounding applied to generated readings.
    """

    empty_mean: float
    occupied_mean: float
    sd: float
    decimals: int = 1


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """One class of mock sensor: how it is exposed, and how it behaves.

    ``p_active_occupied`` and ``p_active_empty`` are the correlation the
    generator aims for. The realised correlation drifts from them -- a sensor
    with a dwell time cannot track occupancy exactly -- so history generation
    measures what it actually produced and uses that as ground truth.

    Attributes:
        conf_key: Area config key the generated entity ids are written to.
        input_type: Integration input type the channel maps to.
        domain: Entity domain the mock entity lives in.
        device_class: Device class for the generated template entity.
        unit: Unit of measurement, for numeric channels.
        active_states: States the integration counts as evidence.
        idle_state: State used when the channel is not active.
        p_active_occupied: Target P(active | area occupied).
        p_active_empty: Target P(active | area empty).
        mean_active_minutes: Average length of one active stretch, so that
            generated history has realistic dwell times rather than a state
            change every slot.
        numeric: Reading model for numeric channels, ``None`` for binary ones.
    """

    conf_key: str
    input_type: InputType
    domain: str
    device_class: str | None = None
    unit: str | None = None
    active_states: tuple[str, ...] = ("on",)
    idle_state: str = "off"
    p_active_occupied: float = 0.5
    p_active_empty: float = 0.02
    mean_active_minutes: float = 5.0
    numeric: NumericModel | None = None

    @property
    def is_numeric(self) -> bool:
        """Whether this channel produces numeric readings rather than states."""
        return self.numeric is not None


# The channels a profile can ask for. Deliberately a useful subset rather than
# every InputType: one channel per distinct behaviour the pipeline treats
# differently (ground truth, session-shaped, transient, numeric).
CHANNELS: dict[str, ChannelSpec] = {
    "motion": ChannelSpec(
        conf_key=CONF_MOTION_SENSORS,
        input_type=InputType.MOTION,
        domain="binary_sensor",
        device_class="motion",
        p_active_occupied=0.75,
        p_active_empty=0.01,
        mean_active_minutes=2.5,
    ),
    "media": ChannelSpec(
        conf_key=CONF_MEDIA_DEVICES,
        input_type=InputType.MEDIA,
        domain="media_player",
        active_states=("playing", "paused"),
        idle_state="off",
        p_active_occupied=0.35,
        p_active_empty=0.02,
        mean_active_minutes=50.0,
    ),
    "appliance": ChannelSpec(
        conf_key=CONF_APPLIANCES,
        input_type=InputType.APPLIANCE,
        domain="binary_sensor",
        device_class="power",
        p_active_occupied=0.3,
        p_active_empty=0.06,
        mean_active_minutes=25.0,
    ),
    "door": ChannelSpec(
        conf_key=CONF_DOOR_SENSORS,
        input_type=InputType.DOOR,
        domain="binary_sensor",
        device_class="door",
        p_active_occupied=0.12,
        p_active_empty=0.04,
        mean_active_minutes=1.5,
    ),
    "window": ChannelSpec(
        conf_key=CONF_WINDOW_SENSORS,
        input_type=InputType.WINDOW,
        domain="binary_sensor",
        device_class="window",
        p_active_occupied=0.1,
        p_active_empty=0.08,
        mean_active_minutes=45.0,
    ),
    # The escape hatch for entities no typed section accepts: a plain
    # ``binary_sensor`` with no device class, which every typed binary
    # channel filters out. Left on the default active state ("on") so the
    # seeds exercise the shipped default rather than an override.
    "custom_binary": ChannelSpec(
        conf_key=CONF_CUSTOM_BINARY_SENSORS,
        input_type=InputType.CUSTOM_BINARY,
        domain="binary_sensor",
        p_active_occupied=0.45,
        p_active_empty=0.03,
        mean_active_minutes=35.0,
    ),
    # The numeric half of the same escape hatch: an unclassed count that
    # straddles the shipped default active range of [1.0, 1000000], so an
    # empty area reads below 1 and an occupied one above it.
    "custom_numeric": ChannelSpec(
        conf_key=CONF_CUSTOM_NUMERIC_SENSORS,
        input_type=InputType.CUSTOM_NUMERIC,
        domain="sensor",
        numeric=NumericModel(empty_mean=0.2, occupied_mean=3.1, sd=0.5),
    ),
    "temperature": ChannelSpec(
        conf_key=CONF_TEMPERATURE_SENSORS,
        input_type=InputType.TEMPERATURE,
        domain="sensor",
        device_class="temperature",
        unit="°C",
        numeric=NumericModel(empty_mean=20.4, occupied_mean=21.6, sd=0.3),
    ),
    "humidity": ChannelSpec(
        conf_key=CONF_HUMIDITY_SENSORS,
        input_type=InputType.HUMIDITY,
        domain="sensor",
        device_class="humidity",
        unit="%",
        numeric=NumericModel(empty_mean=45.0, occupied_mean=52.0, sd=2.0),
    ),
    "illuminance": ChannelSpec(
        conf_key=CONF_ILLUMINANCE_SENSORS,
        input_type=InputType.ILLUMINANCE,
        domain="sensor",
        device_class="illuminance",
        unit="lx",
        numeric=NumericModel(empty_mean=6.0, occupied_mean=120.0, sd=25.0, decimals=0),
    ),
    "co2": ChannelSpec(
        conf_key=CONF_CO2_SENSORS,
        input_type=InputType.CO2,
        domain="sensor",
        device_class="carbon_dioxide",
        unit="ppm",
        numeric=NumericModel(
            empty_mean=450.0, occupied_mean=720.0, sd=45.0, decimals=0
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class OccupancySpec:
    """A daily occupancy routine, as the fraction of each hour spent occupied.

    History generation turns this into a two-state Markov chain per hour: the
    rate out of "occupied" comes from ``mean_visit_minutes`` and the rate into
    it is solved so the stationary occupancy matches ``hourly``. Visits
    therefore have realistic dwell times while the day still adds up to the
    routine asked for.

    Attributes:
        hourly: 24 occupancy fractions, index 0 = midnight local time.
        mean_visit_minutes: Average length of a single occupied stretch.
        weekend_scale: Multiplier applied to ``hourly`` on Saturday and Sunday.
    """

    hourly: tuple[float, ...]
    mean_visit_minutes: float = 35.0
    weekend_scale: float = 1.0

    def __post_init__(self) -> None:
        """Reject routines that are not a full day of valid fractions."""
        if len(self.hourly) != 24:
            raise ValueError(f"hourly needs 24 values, got {len(self.hourly)}")
        if any(not 0.0 <= value <= 1.0 for value in self.hourly):
            raise ValueError("hourly values must be fractions in [0, 1]")

    def fraction_at(self, hour: int, weekday: int) -> float:
        """Occupancy fraction for an hour, adjusted for the day of week.

        Args:
            hour: Local hour of day, 0-23.
            weekday: Python weekday, 0 = Monday.

        Returns:
            The occupancy fraction, clamped to [0, 1].
        """
        value = self.hourly[hour]
        if weekday >= 5:
            value *= self.weekend_scale
        return min(1.0, max(0.0, value))


def _routine(
    *, night: float, morning: float, day: float, evening: float
) -> tuple[float, ...]:
    """Build an hourly routine from four coarse parts of the day.

    Args:
        night: Fraction for 00:00-06:00.
        morning: Fraction for 06:00-09:00.
        day: Fraction for 09:00-17:00.
        evening: Fraction for 17:00-24:00.

    Returns:
        A 24-value tuple suitable for ``OccupancySpec.hourly``.
    """
    return (
        *(night,) * 6,
        *(morning,) * 3,
        *(day,) * 8,
        *(evening,) * 7,
    )


@dataclass(frozen=True, slots=True)
class AreaSpec:
    """One area in a profile.

    Attributes:
        slug: Identifier used for the HA area id and the mock entity prefix.
        name: Human-readable area name.
        purpose: Area purpose, which sets the decay defaults.
        channels: Channel name to how many sensors of it to create.
        occupancy: The routine synthetic history is generated from.
        adjacent: Slugs of adjacent areas, applied symmetrically.
        threshold: Occupancy threshold as a percentage.
        wasp_enabled: Whether to turn on wasp-in-box for this area.
    """

    slug: str
    name: str
    purpose: str
    channels: dict[str, int]
    occupancy: OccupancySpec
    adjacent: tuple[str, ...] = ()
    threshold: float = 50.0
    wasp_enabled: bool = False


@dataclass(frozen=True, slots=True)
class Profile:
    """A named set of areas to build an instance from.

    Attributes:
        name: Profile name, as passed to ``--profile``.
        description: One-line summary shown by ``scripts/harness profiles``.
        areas: The areas to create.
    """

    name: str
    description: str
    areas: tuple[AreaSpec, ...] = field(default_factory=tuple)

    def area(self, slug: str) -> AreaSpec:
        """Return the area with this slug.

        Args:
            slug: The area slug to look up.

        Returns:
            The matching area spec.

        Raises:
            KeyError: If no area in the profile has that slug.
        """
        for area in self.areas:
            if area.slug == slug:
                return area
        raise KeyError(f"{self.name} has no area {slug!r}")


_LIVING_ROOM = AreaSpec(
    slug="living_room",
    name="Living Room",
    purpose="social",
    channels={
        "motion": 2,
        "media": 1,
        "appliance": 2,
        "door": 1,
        "custom_binary": 1,
        "custom_numeric": 1,
        "temperature": 1,
        "humidity": 1,
        "illuminance": 1,
        "co2": 1,
    },
    occupancy=OccupancySpec(
        hourly=_routine(night=0.02, morning=0.25, day=0.2, evening=0.72),
        mean_visit_minutes=75.0,
        weekend_scale=1.4,
    ),
    adjacent=("hallway", "kitchen"),
)

_KITCHEN = AreaSpec(
    slug="kitchen",
    name="Kitchen",
    purpose="food_prep",
    channels={
        "motion": 2,
        "appliance": 2,
        "door": 1,
        "temperature": 1,
        "humidity": 1,
    },
    occupancy=OccupancySpec(
        hourly=_routine(night=0.01, morning=0.45, day=0.15, evening=0.4),
        mean_visit_minutes=20.0,
    ),
    adjacent=("hallway", "living_room"),
)

_BATHROOM = AreaSpec(
    slug="bathroom",
    name="Bathroom",
    purpose="bathroom",
    channels={"motion": 1, "door": 1, "humidity": 1},
    occupancy=OccupancySpec(
        hourly=_routine(night=0.02, morning=0.3, day=0.06, evening=0.18),
        mean_visit_minutes=11.0,
    ),
    adjacent=("hallway",),
    wasp_enabled=True,
)

_BEDROOM = AreaSpec(
    slug="bedroom",
    name="Bedroom",
    purpose="sleeping",
    channels={"motion": 1, "media": 1, "door": 1, "window": 1, "temperature": 1},
    occupancy=OccupancySpec(
        hourly=(
            *(0.95,) * 6,
            0.6,
            0.2,
            0.05,
            *(0.03,) * 8,
            *(0.05,) * 4,
            0.3,
            0.8,
            0.92,
        ),
        mean_visit_minutes=240.0,
    ),
    adjacent=("hallway",),
)

_HALLWAY = AreaSpec(
    slug="hallway",
    name="Hallway",
    purpose="passageway",
    channels={"motion": 1, "door": 2},
    occupancy=OccupancySpec(
        hourly=_routine(night=0.01, morning=0.12, day=0.05, evening=0.12),
        mean_visit_minutes=3.0,
    ),
    adjacent=("living_room", "kitchen", "bathroom", "bedroom"),
)


PROFILES: dict[str, Profile] = {
    "minimal": Profile(
        name="minimal",
        description="One social area with motion only -- fastest to boot.",
        areas=(
            AreaSpec(
                slug="living_room",
                name="Living Room",
                purpose="social",
                channels={"motion": 1},
                occupancy=OccupancySpec(
                    hourly=_routine(night=0.02, morning=0.25, day=0.2, evening=0.7),
                    mean_visit_minutes=60.0,
                ),
            ),
        ),
    ),
    "single": Profile(
        name="single",
        description="One area with every channel -- for sensor-group and preview work.",
        areas=(_LIVING_ROOM,),
    ),
    "house": Profile(
        name="house",
        description="Five adjacent areas covering every purpose -- the default.",
        areas=(_LIVING_ROOM, _KITCHEN, _BATHROOM, _BEDROOM, _HALLWAY),
    ),
}

DEFAULT_PROFILE = "house"


def get_profile(name: str) -> Profile:
    """Look up a profile by name.

    Args:
        name: Profile name.

    Returns:
        The matching profile.

    Raises:
        SystemExit: If the name is not a known profile, with the list of
            names that are.
    """
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise SystemExit(f"unknown profile {name!r}; available: {known}") from None


def entity_id(area: AreaSpec, channel: str, index: int) -> str:
    """Entity id of one mock sensor.

    Args:
        area: The area the sensor belongs to.
        channel: Channel name, a key of ``CHANNELS``.
        index: 1-based index within the channel.

    Returns:
        The full entity id, e.g. ``binary_sensor.living_room_motion_1``.
    """
    spec = CHANNELS[channel]
    return f"{spec.domain}.{area.slug}_{channel}_{index}"


def area_entities(area: AreaSpec) -> dict[str, list[str]]:
    """Every mock entity of an area, grouped by channel name.

    Args:
        area: The area to enumerate.

    Returns:
        Channel name to the entity ids created for it, in index order.
    """
    return {
        channel: [entity_id(area, channel, i + 1) for i in range(count)]
        for channel, count in area.channels.items()
        if count > 0
    }
