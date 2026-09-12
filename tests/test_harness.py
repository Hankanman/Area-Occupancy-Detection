"""Tests for the throwaway-instance harness (``harness/``).

Only the deterministic parts are covered here -- profiles, generated YAML,
seeded storage and synthesised history. Starting a real Home Assistant is
what ``scripts/harness verify`` is for; these tests guard the pieces that
decide what that instance will contain, so a broken generator is caught in
CI rather than in a five-minute boot.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json

import pytest
import sqlalchemy as sa
import yaml

from custom_components.area_occupancy.const import (
    CONF_AREA_ID,
    CONF_AREAS,
    CONF_DECAY_HALF_LIFE,
    CONF_MOTION_SENSORS,
    CONF_VERSION,
    DB_NAME,
    DB_SCHEMA_VERSION,
    DOMAIN,
    SUBENTRY_TYPE_AREA,
)
from harness import history, mock_config, storage
from harness.profiles import (
    CHANNELS,
    PROFILES,
    AreaSpec,
    OccupancySpec,
    Profile,
    area_entities,
    get_profile,
)

FIXED_END = datetime(2026, 3, 15, 12, 0, tzinfo=UTC)


@pytest.fixture
def profile() -> Profile:
    """A two-area profile small enough to generate quickly."""
    return Profile(
        name="test",
        description="test profile",
        areas=(
            AreaSpec(
                slug="living_room",
                name="Living Room",
                purpose="social",
                channels={"motion": 2, "media": 1, "temperature": 1},
                occupancy=OccupancySpec(hourly=(0.4,) * 24, mean_visit_minutes=45.0),
                adjacent=("kitchen", "not_in_profile"),
            ),
            AreaSpec(
                slug="kitchen",
                name="Kitchen",
                purpose="food_prep",
                channels={"motion": 1},
                occupancy=OccupancySpec(hourly=(0.1,) * 24, mean_visit_minutes=15.0),
                adjacent=("living_room",),
            ),
        ),
    )


class TestProfiles:
    def test_shipped_profiles_are_self_consistent(self) -> None:
        for name, shipped in PROFILES.items():
            assert shipped.name == name
            assert shipped.areas, f"{name} has no areas"
            slugs = [area.slug for area in shipped.areas]
            assert len(slugs) == len(set(slugs))
            for area in shipped.areas:
                assert set(area.channels) <= set(CHANNELS)

    def test_unknown_profile_names_the_alternatives(self) -> None:
        with pytest.raises(SystemExit, match="house"):
            get_profile("nope")

    def test_occupancy_needs_a_full_day(self) -> None:
        with pytest.raises(ValueError, match="24 values"):
            OccupancySpec(hourly=(0.5,) * 23)

    def test_weekend_scaling_clamps(self) -> None:
        spec = OccupancySpec(hourly=(0.8,) * 24, weekend_scale=2.0)
        assert spec.fraction_at(12, weekday=0) == 0.8
        assert spec.fraction_at(12, weekday=5) == 1.0

    def test_entity_ids_follow_the_channel_domain(self, profile: Profile) -> None:
        entities = area_entities(profile.area("living_room"))
        assert entities["motion"] == [
            "binary_sensor.living_room_motion_1",
            "binary_sensor.living_room_motion_2",
        ]
        assert entities["media"] == ["media_player.living_room_media_1"]
        assert entities["temperature"] == ["sensor.living_room_temperature_1"]


class TestMockConfig:
    def test_every_configured_sensor_has_a_backing_entity(
        self, profile: Profile
    ) -> None:
        # The point of the generated YAML: nothing an area points at may be
        # missing, or the instance loads with dead sensors.
        document = yaml.safe_load(
            mock_config.render(
                profile, time_zone="Europe/London", frontend=True, port=8123
            )
        )
        defined: set[str] = set()
        for block in document["template"]:
            for domain, entities in block.items():
                for entity in entities:
                    defined.add(f"{domain}.{entity['unique_id']}")
        for player in document.get("media_player", []):
            defined.add(f"media_player.{player['unique_id']}")

        for area in profile.areas:
            for entities in area_entities(area).values():
                for entity_id in entities:
                    domain, object_id = entity_id.split(".", 1)
                    assert f"{domain}.{object_id}" in defined

    def test_knobs_exist_for_every_template_entity(self, profile: Profile) -> None:
        document = yaml.safe_load(
            mock_config.render(
                profile, time_zone="Europe/London", frontend=True, port=8123
            )
        )
        helpers = (
            {f"input_boolean.{key}" for key in document.get("input_boolean", {})}
            | {f"input_select.{key}" for key in document.get("input_select", {})}
            | {f"input_number.{key}" for key in document.get("input_number", {})}
        )
        rendered = yaml.safe_dump(document)
        for helper in helpers:
            assert helper in rendered, f"{helper} is defined but never referenced"

    def test_port_is_pinned(self, profile: Profile) -> None:
        document = yaml.safe_load(
            mock_config.render(profile, time_zone="UTC", frontend=False, port=9999)
        )
        assert document["http"]["server_port"] == 9999
        # Without the frontend the API pieces have to be asked for explicitly.
        assert "onboarding" in document
        assert "frontend" not in document

    def test_recorder_is_always_configured(self, profile: Profile) -> None:
        # The analysis pipeline's first step goes through the recorder and
        # raises without it, so an instance without one is not usable.
        document = yaml.safe_load(
            mock_config.render(profile, time_zone="UTC", frontend=True, port=8123)
        )
        assert "recorder" in document


class TestStorage:
    def test_current_version_writes_one_subentry_per_area(
        self, tmp_path, profile: Profile
    ) -> None:
        entry_id = storage.write_config_entry(tmp_path, profile)
        entry = storage.read_config_entry(tmp_path)

        assert entry is not None
        assert entry["entry_id"] == entry_id
        assert entry["domain"] == DOMAIN
        assert entry["version"] == CONF_VERSION
        assert CONF_AREAS not in entry["data"]
        assert {sub["unique_id"] for sub in entry["subentries"]} == {
            "living_room",
            "kitchen",
        }
        assert all(
            sub["subentry_type"] == SUBENTRY_TYPE_AREA for sub in entry["subentries"]
        )

    def test_legacy_version_keeps_areas_in_the_old_list(
        self, tmp_path, profile: Profile
    ) -> None:
        storage.write_config_entry(
            tmp_path, profile, entry_version=storage.LEGACY_ENTRY_VERSION
        )
        entry = storage.read_config_entry(tmp_path)

        assert entry is not None
        assert entry["version"] == storage.LEGACY_ENTRY_VERSION
        assert entry["subentries"] == []
        assert [area[CONF_AREA_ID] for area in entry["data"][CONF_AREAS]] == [
            "living_room",
            "kitchen",
        ]

    def test_unsupported_version_is_rejected(self, tmp_path, profile: Profile) -> None:
        with pytest.raises(ValueError, match="entry_version"):
            storage.write_config_entry(tmp_path, profile, entry_version=17)

    def test_area_config_drops_unknown_adjacency_and_keeps_the_sentinel(
        self, profile: Profile
    ) -> None:
        config = storage.area_config(profile.area("living_room"), profile)

        assert config["adjacent_areas"] == ["kitchen"]
        assert config[CONF_MOTION_SENSORS] == [
            "binary_sensor.living_room_motion_1",
            "binary_sensor.living_room_motion_2",
        ]
        # 0 means "use the purpose default"; seeds keep that path in play.
        assert config[CONF_DECAY_HALF_LIFE] == 0

    def test_area_registry_names_every_area(self, tmp_path, profile: Profile) -> None:
        storage.write_area_registry(tmp_path, profile)
        document = json.loads(
            (tmp_path / ".storage" / "core.area_registry").read_text(encoding="utf-8")
        )
        areas = {area["id"]: area["name"] for area in document["data"]["areas"]}
        assert areas == {"living_room": "Living Room", "kitchen": "Kitchen"}


class TestHistory:
    @pytest.fixture
    def generated(self, profile: Profile) -> dict[str, history.AreaHistory]:
        return history.generate(
            profile, days=2, time_zone="UTC", seed=99, end=FIXED_END
        )

    def test_generation_is_reproducible(self, profile: Profile) -> None:
        first = history.generate(
            profile, days=1, time_zone="UTC", seed=5, end=FIXED_END
        )
        second = history.generate(
            profile, days=1, time_zone="UTC", seed=5, end=FIXED_END
        )
        assert [run.start for run in first["kitchen"].occupied] == [
            run.start for run in second["kitchen"].occupied
        ]

    def test_occupancy_tracks_the_routine(
        self, generated: dict[str, history.AreaHistory]
    ) -> None:
        # The chain is solved so its stationary occupancy matches the
        # routine; over two days it should land near it.
        assert generated["living_room"].global_prior == pytest.approx(0.4, abs=0.15)
        assert generated["kitchen"].global_prior == pytest.approx(0.1, abs=0.1)

    def test_runs_cover_the_window_without_gaps(
        self, generated: dict[str, history.AreaHistory]
    ) -> None:
        area = generated["living_room"]
        sensor = next(
            item for item in area.sensors if item.entity_id.endswith("motion_1")
        )
        assert sensor.runs[0].start == area.start
        assert sensor.runs[-1].end == area.end
        for earlier, later in zip(sensor.runs, sensor.runs[1:], strict=False):
            assert earlier.end == later.start
            assert earlier.active is not later.active

    def test_numeric_channels_produce_samples_not_runs(
        self, generated: dict[str, history.AreaHistory]
    ) -> None:
        sensor = next(
            item
            for item in generated["living_room"].sensors
            if item.channel == "temperature"
        )
        assert not sensor.runs
        assert sensor.samples
        values = [value for _, value in sensor.samples]
        assert min(values) > 15.0
        assert max(values) < 27.0

    def test_measured_correlation_is_reported(
        self, generated: dict[str, history.AreaHistory]
    ) -> None:
        sensor = next(
            item
            for item in generated["living_room"].sensors
            if item.channel == "motion"
        )
        # Measured, not declared: a sensor with a dwell time cannot track
        # occupancy exactly, and the measurement is what gets seeded.
        assert 0.0 < sensor.observed_prob_given_true <= 1.0
        assert 0.0 <= sensor.observed_prob_given_false < sensor.observed_prob_given_true

    def test_write_stamps_the_schema_and_fills_the_tables(
        self, tmp_path, profile: Profile, generated: dict[str, history.AreaHistory]
    ) -> None:
        counts = history.write(
            tmp_path, profile, generated, entry_id="entry-1", with_priors=True
        )

        assert counts["areas"] == 2
        assert counts["entities"] == 5
        assert counts["intervals"] > 0
        assert counts["numeric_samples"] > 0
        assert counts["global_priors"] == 2

        engine = sa.create_engine(f"sqlite:///{tmp_path / '.storage' / DB_NAME}")
        try:
            with engine.connect() as connection:
                version = connection.execute(
                    sa.text("SELECT value FROM metadata WHERE key = 'db_version'")
                ).scalar_one()
                # The stamp has to match or the integration wipes the database.
                assert version == str(DB_SCHEMA_VERSION)

                names = connection.execute(
                    sa.text("SELECT area_name FROM areas ORDER BY area_name")
                ).scalars()
                # Keyed by display name, which is how the integration reads it.
                assert list(names) == ["Kitchen", "Living Room"]

                threshold = connection.execute(
                    sa.text("SELECT threshold FROM areas WHERE area_name = 'Kitchen'")
                ).scalar_one()
                assert threshold == pytest.approx(0.5)

                slots = connection.execute(
                    sa.text(
                        "SELECT COUNT(DISTINCT time_slot) FROM priors"
                        " WHERE area_name = 'Kitchen'"
                    )
                ).scalar_one()
                assert 0 < slots <= history.SLOTS_PER_DAY
        finally:
            engine.dispose()

    def test_priors_can_be_left_out(
        self, tmp_path, profile: Profile, generated: dict[str, history.AreaHistory]
    ) -> None:
        counts = history.write(
            tmp_path, profile, generated, entry_id="entry-1", with_priors=False
        )
        assert counts["priors"] == 0
        assert counts["global_priors"] == 0
        assert counts["intervals"] > 0

    def test_intervals_are_stored_as_naive_utc(
        self, tmp_path, profile: Profile, generated: dict[str, history.AreaHistory]
    ) -> None:
        history.write(tmp_path, profile, generated, entry_id="entry-1")
        engine = sa.create_engine(f"sqlite:///{tmp_path / '.storage' / DB_NAME}")
        try:
            with engine.connect() as connection:
                start, end = connection.execute(
                    sa.text("SELECT MIN(start_time), MAX(end_time) FROM intervals")
                ).one()
        finally:
            engine.dispose()
        # SQLite hands back naive strings; what matters is that the window is
        # the one that was asked for, in UTC.
        assert start.startswith("2026-03-13")
        assert end.startswith("2026-03-15")
        assert FIXED_END - datetime.fromisoformat(end).replace(tzinfo=UTC) < timedelta(
            minutes=2
        )
