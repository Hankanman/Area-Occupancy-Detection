"""Tests for database area relationship functions."""

from typing import TYPE_CHECKING, Any

import pytest

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.db.relationships import (
    DEFAULT_INFLUENCE_WEIGHTS,
    get_adjacent_areas,
    get_influence_weight,
    save_area_relationship,
    sync_adjacent_areas_from_config,
)

if TYPE_CHECKING:
    from custom_components.area_occupancy.db.core import AreaOccupancyDB
else:
    # Import for runtime type checking in helper functions
    from custom_components.area_occupancy.db.core import AreaOccupancyDB


def _create_test_area(
    db: AreaOccupancyDB, area_name: str, area_id: str | None = None
) -> None:
    """Helper to create a test area in the database.

    Args:
        db: Database instance
        area_name: Name of the area to create
        area_id: Optional area ID (defaults to lowercase area_name)
    """
    if area_id is None:
        area_id = area_name.lower()

    with db.get_session() as session:
        area = db.Areas(
            entry_id=db.coordinator.entry_id,
            area_name=area_name,
            area_id=area_id,
            purpose="work",
            threshold=0.5,
        )
        session.add(area)
        session.commit()


def _verify_relationship(
    db: AreaOccupancyDB,
    area_name: str,
    related_area_name: str,
    **expected_fields: Any,
) -> None:
    """Helper to verify relationship fields match expected values.

    Args:
        db: Database instance
        area_name: Source area name
        related_area_name: Related area name
        **expected_fields: Expected field values (e.g., influence_weight=0.5)
    """
    with db.get_session() as session:
        relationship = (
            session.query(db.AreaRelationships)
            .filter_by(area_name=area_name, related_area_name=related_area_name)
            .first()
        )
        assert relationship is not None, (
            f"Relationship {area_name} -> {related_area_name} not found"
        )

        for field_name, expected_value in expected_fields.items():
            actual_value = getattr(relationship, field_name)
            assert actual_value == expected_value, (
                f"Field {field_name} mismatch: expected {expected_value}, got {actual_value}"
            )


class TestSaveAreaRelationship:
    """Test save_area_relationship function."""

    def test_save_area_relationship_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test saving area relationship successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        _create_test_area(db, "Kitchen")

        result = save_area_relationship(
            db, area_name, "Kitchen", "adjacent", influence_weight=0.5, distance=10.0
        )
        assert result is True

        # Verify relationship was saved with all fields
        _verify_relationship(
            db,
            area_name,
            "Kitchen",
            influence_weight=0.5,
            relationship_type="adjacent",
            distance=10.0,
        )

        # Verify timestamps are set
        with db.get_session() as session:
            relationship = (
                session.query(db.AreaRelationships)
                .filter_by(area_name=area_name, related_area_name="Kitchen")
                .first()
            )
            assert relationship.created_at is not None
            assert relationship.updated_at is not None

    def test_save_area_relationship_update_existing(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test updating existing relationship."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        _create_test_area(db, "Kitchen")

        # Save initial relationship
        save_area_relationship(
            db, area_name, "Kitchen", "adjacent", influence_weight=0.3, distance=5.0
        )

        # Get initial timestamps
        with db.get_session() as session:
            initial_relationship = (
                session.query(db.AreaRelationships)
                .filter_by(area_name=area_name, related_area_name="Kitchen")
                .first()
            )
            initial_updated_at = initial_relationship.updated_at
            initial_created_at = initial_relationship.created_at

        # Update relationship with all fields
        result = save_area_relationship(
            db,
            area_name,
            "Kitchen",
            "shared_wall",
            influence_weight=0.7,
            distance=8.0,
        )
        assert result is True

        # Verify relationship was updated with all fields
        _verify_relationship(
            db,
            area_name,
            "Kitchen",
            influence_weight=0.7,
            relationship_type="shared_wall",
            distance=8.0,
        )

        # Verify timestamps
        with db.get_session() as session:
            relationship = (
                session.query(db.AreaRelationships)
                .filter_by(area_name=area_name, related_area_name="Kitchen")
                .first()
            )
            assert (
                relationship.created_at == initial_created_at
            )  # Created_at shouldn't change
            assert (
                relationship.updated_at > initial_updated_at
            )  # Updated_at should change

    @pytest.mark.parametrize(
        ("input_weight", "expected_weight"),
        [
            (-0.5, 0.0),  # Clamp below 0.0
            (1.5, 1.0),  # Clamp above 1.0
        ],
    )
    def test_save_area_relationship_weight_clamping(
        self,
        coordinator: AreaOccupancyCoordinator,
        input_weight: float,
        expected_weight: float,
    ):
        """Test that influence weights are clamped to valid range."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        _create_test_area(db, "Kitchen")

        result = save_area_relationship(
            db, area_name, "Kitchen", "adjacent", influence_weight=input_weight
        )
        assert result is True

        _verify_relationship(db, area_name, "Kitchen", influence_weight=expected_weight)

    @pytest.mark.parametrize(
        ("relationship_type", "expected_weight"),
        [
            ("adjacent", DEFAULT_INFLUENCE_WEIGHTS["adjacent"]),
            ("shared_wall", DEFAULT_INFLUENCE_WEIGHTS["shared_wall"]),
            ("shared_entrance", DEFAULT_INFLUENCE_WEIGHTS["shared_entrance"]),
            ("open_connection", DEFAULT_INFLUENCE_WEIGHTS["open_connection"]),
        ],
    )
    def test_save_area_relationship_default_weights(
        self,
        coordinator: AreaOccupancyCoordinator,
        relationship_type: str,
        expected_weight: float,
    ):
        """Test default weights by relationship type."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        test_area_name = f"Area_{relationship_type}"
        _create_test_area(db, test_area_name)

        # Save relationship without specifying weight (should use default)
        result = save_area_relationship(
            db, area_name, test_area_name, relationship_type=relationship_type
        )
        assert result is True

        # Verify default weight was used
        _verify_relationship(
            db, area_name, test_area_name, influence_weight=expected_weight
        )

    def test_save_area_relationship_unknown_type_defaults(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that unknown relationship type falls back to adjacent default."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        _create_test_area(db, "Kitchen")

        # Save relationship with unknown type
        result = save_area_relationship(
            db, area_name, "Kitchen", relationship_type="unknown_type"
        )
        assert result is True

        # Verify it used adjacent default weight
        _verify_relationship(
            db,
            area_name,
            "Kitchen",
            relationship_type="unknown_type",
            influence_weight=DEFAULT_INFLUENCE_WEIGHTS["adjacent"],
        )

    @pytest.mark.parametrize(
        ("distance_value", "expected_distance"),
        [
            (15.5, 15.5),  # Normal distance value
            (None, None),  # None distance
        ],
    )
    def test_save_area_relationship_distance_parameter(
        self,
        coordinator: AreaOccupancyCoordinator,
        distance_value: float | None,
        expected_distance: float | None,
    ):
        """Test that distance parameter is saved correctly."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        _create_test_area(db, "Kitchen")

        result = save_area_relationship(
            db, area_name, "Kitchen", "adjacent", distance=distance_value
        )
        assert result is True

        _verify_relationship(db, area_name, "Kitchen", distance=expected_distance)


class TestGetAdjacentAreas:
    """Test get_adjacent_areas function."""

    def test_get_adjacent_areas_success(self, coordinator: AreaOccupancyCoordinator):
        """Test retrieving adjacent areas successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Create multiple areas and relationships with different types
        areas_to_create = [
            ("Kitchen", "adjacent", 0.3, 10.0),
            ("Bedroom", "shared_wall", 0.4, 5.0),
            ("Bathroom", "shared_entrance", 0.5, None),
        ]

        for adj_name, rel_type, weight, distance in areas_to_create:
            _create_test_area(db, adj_name)
            save_area_relationship(
                db,
                area_name,
                adj_name,
                rel_type,
                influence_weight=weight,
                distance=distance,
            )

        result = get_adjacent_areas(db, area_name)
        assert isinstance(result, list)
        assert len(result) == 3

        # Verify all expected keys are present in each result
        for rel in result:
            assert "related_area_name" in rel
            assert "relationship_type" in rel
            assert "influence_weight" in rel
            assert "distance" in rel

        # Verify specific relationships
        result_dict = {r["related_area_name"]: r for r in result}
        assert result_dict["Kitchen"]["relationship_type"] == "adjacent"
        assert result_dict["Kitchen"]["influence_weight"] == 0.3
        assert result_dict["Kitchen"]["distance"] == 10.0
        assert result_dict["Bedroom"]["relationship_type"] == "shared_wall"
        assert result_dict["Bedroom"]["influence_weight"] == 0.4
        assert result_dict["Bedroom"]["distance"] == 5.0
        assert result_dict["Bathroom"]["relationship_type"] == "shared_entrance"
        assert result_dict["Bathroom"]["influence_weight"] == 0.5
        assert result_dict["Bathroom"]["distance"] is None

    def test_get_adjacent_areas_empty(self, coordinator: AreaOccupancyCoordinator):
        """Test retrieving adjacent areas when none exist."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        result = get_adjacent_areas(db, area_name)
        assert result == []


class TestGetInfluenceWeight:
    """Test get_influence_weight function."""

    def test_get_influence_weight_success(self, coordinator: AreaOccupancyCoordinator):
        """Test retrieving influence weight successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        _create_test_area(db, "Kitchen")
        save_area_relationship(
            db, area_name, "Kitchen", "adjacent", influence_weight=0.6
        )

        weight = get_influence_weight(db, area_name, "Kitchen")
        assert weight == 0.6

    def test_get_influence_weight_default(self, coordinator: AreaOccupancyCoordinator):
        """Test retrieving influence weight when relationship doesn't exist."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        weight = get_influence_weight(db, area_name, "Nonexistent")
        assert weight == 0.0  # Default weight

    def test_get_influence_weight_different_types(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test retrieving influence weight for different relationship types."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Create areas with different relationship types
        relationships = [
            ("Kitchen", "adjacent", 0.3),
            ("Bedroom", "shared_wall", 0.4),
            ("Bathroom", "shared_entrance", 0.5),
        ]

        for adj_name, rel_type, weight in relationships:
            _create_test_area(db, adj_name)
            save_area_relationship(
                db, area_name, adj_name, rel_type, influence_weight=weight
            )

            # Verify weight retrieval
            retrieved_weight = get_influence_weight(db, area_name, adj_name)
            assert retrieved_weight == weight


class TestSyncAdjacentAreasFromConfig:
    """Test sync_adjacent_areas_from_config function."""

    def test_sync_adjacent_areas_from_config_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test syncing adjacent areas from config successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Ensure main area exists first (foreign key requirement)
        db.save_area_data(area_name)

        # Create adjacent areas
        for adj_name in ["Kitchen", "Bedroom"]:
            _create_test_area(db, adj_name)

        # Update the area record to include adjacent_areas in the database
        # sync_adjacent_areas_from_config reads from area_record.adjacent_areas
        with db.get_session() as session:
            area_record = session.query(db.Areas).filter_by(area_name=area_name).first()
            if area_record:
                area_record.adjacent_areas = ["Kitchen", "Bedroom"]
            session.commit()

        result = sync_adjacent_areas_from_config(db, area_name)
        assert result is True

        # Verify relationships were created with correct defaults
        adjacent = get_adjacent_areas(db, area_name)
        assert len(adjacent) == 2
        adjacent_names = {a["related_area_name"] for a in adjacent}
        assert adjacent_names == {"Kitchen", "Bedroom"}

        # Verify relationship_type is set to "adjacent" and default weight is used
        for rel in adjacent:
            assert rel["relationship_type"] == "adjacent"
            assert rel["influence_weight"] == DEFAULT_INFLUENCE_WEIGHTS["adjacent"]

    def test_sync_adjacent_areas_from_config_duplicate_updates(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that syncing with duplicate adjacent areas updates, not duplicates."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Ensure main area exists first
        db.save_area_data(area_name)

        # Create adjacent area
        _create_test_area(db, "Kitchen")
        with db.get_session() as session:
            area_record = session.query(db.Areas).filter_by(area_name=area_name).first()
            if area_record:
                area_record.adjacent_areas = ["Kitchen"]
            session.commit()

        # First sync
        result = sync_adjacent_areas_from_config(db, area_name)
        assert result is True

        # Verify one relationship exists
        adjacent = get_adjacent_areas(db, area_name)
        assert len(adjacent) == 1

        # Sync again with same area (should update, not duplicate)
        result = sync_adjacent_areas_from_config(db, area_name)
        assert result is True

        # Verify still only one relationship
        adjacent = get_adjacent_areas(db, area_name)
        assert len(adjacent) == 1
        assert adjacent[0]["related_area_name"] == "Kitchen"

    def test_sync_adjacent_areas_from_config_does_not_remove_old(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that syncing does NOT remove old relationships not in config."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Ensure main area exists first
        db.save_area_data(area_name)

        # Create old area and relationship
        _create_test_area(db, "Old Area", "old")

        # Create relationship manually (not from config)
        save_area_relationship(
            db, area_name, "Old Area", "adjacent", influence_weight=0.8
        )

        # Verify old relationship exists
        adjacent_before = get_adjacent_areas(db, area_name)
        assert len(adjacent_before) == 1
        assert adjacent_before[0]["related_area_name"] == "Old Area"
        assert adjacent_before[0]["influence_weight"] == 0.8

        # Create new area and update config to only include new area
        _create_test_area(db, "New Area", "new")
        with db.get_session() as session:
            area_record = session.query(db.Areas).filter_by(area_name=area_name).first()
            if area_record:
                area_record.adjacent_areas = ["New Area"]  # Only new area in config
            session.commit()

        # Sync with new config (should add New Area but NOT remove Old Area)
        result = sync_adjacent_areas_from_config(db, area_name)
        assert result is True

        # Verify both relationships exist (old one remains, new one added)
        adjacent = get_adjacent_areas(db, area_name)
        assert len(adjacent) == 2
        adjacent_names = {a["related_area_name"] for a in adjacent}
        assert adjacent_names == {"Old Area", "New Area"}

        # Verify old relationship's custom weight is preserved
        old_rel = next(r for r in adjacent if r["related_area_name"] == "Old Area")
        assert old_rel["influence_weight"] == 0.8

    @pytest.mark.parametrize(
        ("adjacent_areas_value", "expected_count"),
        [
            ([], 0),  # Empty list
            (None, 0),  # None value
        ],
    )
    def test_sync_adjacent_areas_from_config_empty_or_none(
        self,
        coordinator: AreaOccupancyCoordinator,
        adjacent_areas_value: list[str] | None,
        expected_count: int,
    ):
        """Test syncing with empty adjacent_areas list or None value."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Ensure main area exists first
        db.save_area_data(area_name)

        # Set adjacent_areas to empty list or None
        with db.get_session() as session:
            area_record = session.query(db.Areas).filter_by(area_name=area_name).first()
            if area_record:
                area_record.adjacent_areas = adjacent_areas_value
            session.commit()

        result = sync_adjacent_areas_from_config(db, area_name)
        assert result is True  # Empty list/None is valid, sync succeeds

        # Verify no relationships were created
        adjacent = get_adjacent_areas(db, area_name)
        assert len(adjacent) == expected_count

    def test_sync_adjacent_areas_from_config_area_not_found(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test syncing when area record doesn't exist."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Delete area record from database
        with db.get_session() as session:
            area_record = session.query(db.Areas).filter_by(area_name=area_name).first()
            if area_record:
                session.delete(area_record)
            session.commit()

        result = sync_adjacent_areas_from_config(db, area_name)
        assert result is False  # Should return False when area not found

    def test_sync_adjacent_areas_from_config_with_nonexistent_areas(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test syncing creates relationships even for non-existent related areas."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Ensure main area exists first
        db.save_area_data(area_name)

        # Create one valid area and include a non-existent area name
        _create_test_area(db, "Valid Area", "valid")
        with db.get_session() as session:
            area_record = session.query(db.Areas).filter_by(area_name=area_name).first()
            if area_record:
                # Include both valid and non-existent area names
                # Note: save_area_relationship doesn't validate related area exists
                area_record.adjacent_areas = ["Valid Area", "Nonexistent Area"]
            session.commit()

        # Sync should succeed for both (no validation of related area existence)
        result = sync_adjacent_areas_from_config(db, area_name)
        assert result is True  # Both relationships are created successfully

        # Verify both relationships were created
        adjacent = get_adjacent_areas(db, area_name)
        assert len(adjacent) == 2
        adjacent_names = {a["related_area_name"] for a in adjacent}
        assert adjacent_names == {"Valid Area", "Nonexistent Area"}
