"""Integration tests for AreaOccupancyStorage with real SQLite database."""

import asyncio
import os
import tempfile
from unittest.mock import Mock

from custom_components.area_occupancy.storage import AreaOccupancyStorage


# ruff: noqa: PLC0415
def test_real_db_area_occupancy_crud():
    """Integration test: CRUD for area occupancy with real SQLite DB."""

    # Create a temporary directory for the database
    with tempfile.TemporaryDirectory() as temp_dir:
        entry_id = "test_entry"

        # Create a comprehensive mock hass
        mock_hass = Mock()
        mock_hass.config = Mock()
        mock_hass.config.config_dir = temp_dir

        # Mock the async_add_executor_job to actually execute functions
        async def real_executor_job(func, *args, **kwargs):
            """Actually execute the function instead of mocking it."""
            return func(*args, **kwargs)

        mock_hass.async_add_executor_job = real_executor_job

        async def run_test():
            # Create storage
            storage = AreaOccupancyStorage(hass=mock_hass, entry_id=entry_id)

            # Initialize the storage (this creates the database)
            await storage.async_initialize()

            # Verify database was created in .storage subdirectory
            db_path = os.path.join(temp_dir, ".storage", "area_occupancy.db")
            assert os.path.exists(db_path), f"Database file not created at {db_path}"

            # Insert a record
            from homeassistant.util import dt as dt_util

            record = {
                "entry_id": entry_id,
                "area_name": "Test Area",
                "purpose": "test",
                "threshold": 0.5,
                "created_at": dt_util.utcnow(),
                "updated_at": dt_util.utcnow(),
            }

            saved = await storage.save_area_occupancy(record)
            assert saved["entry_id"] == entry_id
            assert saved["area_name"] == "Test Area"

            # Query the record
            fetched = await storage.get_area_occupancy(entry_id)
            assert fetched is not None
            assert fetched["area_name"] == "Test Area"
            assert fetched["entry_id"] == entry_id

            # Update the record
            record["area_name"] = "Updated Area"
            await storage.save_area_occupancy(record)
            updated = await storage.get_area_occupancy(entry_id)
            assert updated["area_name"] == "Updated Area"

            # Delete the record (via reset)
            await storage.reset_entry_data(entry_id)
            deleted = await storage.get_area_occupancy(entry_id)
            assert deleted is None

        # Run the async test
        asyncio.run(run_test())


def test_real_db_duplicate_area_update():
    """Integration test: Insert same entry_id twice should update, not error."""

    # Create a temporary directory for the database
    with tempfile.TemporaryDirectory() as temp_dir:
        entry_id = "test_entry"

        # Create a comprehensive mock hass
        mock_hass = Mock()
        mock_hass.config = Mock()
        mock_hass.config.config_dir = temp_dir

        # Mock the async_add_executor_job to actually execute functions
        async def real_executor_job(func, *args, **kwargs):
            """Actually execute the function instead of mocking it."""
            return func(*args, **kwargs)

        mock_hass.async_add_executor_job = real_executor_job

        async def run_test():
            # Create storage
            storage = AreaOccupancyStorage(hass=mock_hass, entry_id=entry_id)

            # Initialize the storage
            await storage.async_initialize()

            from homeassistant.util import dt as dt_util

            record = {
                "entry_id": entry_id,
                "area_name": "Test Area",
                "purpose": "test",
                "threshold": 0.5,
                "created_at": dt_util.utcnow(),
                "updated_at": dt_util.utcnow(),
            }

            # First insert
            await storage.save_area_occupancy(record)

            # Try to insert again with the same entry_id (should update, not error)
            record["area_name"] = "Test Area 2"
            await storage.save_area_occupancy(record)

            fetched = await storage.get_area_occupancy(entry_id)
            assert fetched["area_name"] == "Test Area 2"

        # Run the async test
        asyncio.run(run_test())


def test_real_db_basic_initialization():
    """Test basic storage initialization with real database."""

    with tempfile.TemporaryDirectory() as temp_dir:
        entry_id = "test_entry"

        mock_hass = Mock()
        mock_hass.config = Mock()
        mock_hass.config.config_dir = temp_dir

        async def real_executor_job(func, *args, **kwargs):
            return func(*args, **kwargs)

        mock_hass.async_add_executor_job = real_executor_job

        async def run_test():
            storage = AreaOccupancyStorage(hass=mock_hass, entry_id=entry_id)
            await storage.async_initialize()

            # Check that database file was created in .storage subdirectory
            db_path = os.path.join(temp_dir, ".storage", "area_occupancy.db")
            assert os.path.exists(db_path)

            # Check that storage has required attributes
            assert hasattr(storage, "db")
            assert hasattr(storage, "executor")
            assert hasattr(storage, "initializer")
            assert hasattr(storage, "queries")

        asyncio.run(run_test())
