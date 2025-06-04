"""Unit tests for StorageManager and related storage exception handling in the Area Occupancy Detection integration.

These tests cover:
- Creation of empty storage structure
- Saving and loading of instance prior state
- Removal of instance data
- Exception handling for load/save operations
"""

from unittest.mock import AsyncMock, patch

import pytest

from custom_components.area_occupancy.exceptions import (  # noqa: TID252
    StorageLoadError,
    StorageSaveError,
)
from custom_components.area_occupancy.storage import StorageManager  # noqa: TID252
from custom_components.area_occupancy.types import (  # noqa: TID252
    InstanceData,
    LoadedInstanceData,
    PriorState,
)

# Note: Using fixtures from conftest.py:
# - mock_hass
# - mock_config_entry


@pytest.fixture
def store(mock_hass):
    """Return an StorageManager instance using the mock HomeAssistant object."""
    return StorageManager(mock_hass)


@pytest.mark.asyncio
async def test_create_empty_storage(store):
    """Test that create_empty_storage returns a dict with an empty 'instances' key."""
    result = store.create_empty_storage()
    assert isinstance(result, dict)
    assert "instances" in result
    assert result["instances"] == {}


@pytest.mark.asyncio
async def test_async_save_and_load_instance_prior_state(store):
    """Test saving and loading of instance prior state using async methods."""
    entry_id = "test_entry"
    name = "Test Area"
    prior_state = PriorState()
    # Patch async_load and async_save
    with (
        patch.object(store, "async_load", new=AsyncMock(return_value=None)),
        patch.object(store, "async_save", new=AsyncMock()) as mock_save,
    ):
        await store.async_save_instance_prior_state(entry_id, name, prior_state)
        # Should call async_save with correct structure
        assert mock_save.call_count == 1
        args, kwargs = mock_save.call_args
        saved = args[0]
        assert entry_id in saved["instances"]
        assert saved["instances"][entry_id]["name"] == name
        assert "prior_state" in saved["instances"][entry_id]

    # Now test loading
    instance_data = InstanceData(
        name=name,
        prior_state=prior_state.to_dict(),
        last_updated="2024-01-01T00:00:00Z",
    )
    with patch.object(
        store,
        "async_load",
        new=AsyncMock(return_value={"instances": {entry_id: instance_data}}),
    ):
        loaded = await store.async_load_instance_prior_state(entry_id)
        assert isinstance(loaded, LoadedInstanceData)
        assert loaded.name == name
        assert isinstance(loaded.prior_state, PriorState)
        assert loaded.last_updated == "2024-01-01T00:00:00Z"


@pytest.mark.asyncio
async def test_async_remove_instance_removes_and_skips(store):
    """Test that async_remove_instance removes present entry and skips if not present."""
    entry_id = "test_entry"
    # Case: present
    with (
        patch.object(
            store,
            "async_load",
            new=AsyncMock(return_value={"instances": {entry_id: {}}}),
        ),
        patch.object(store, "async_save", new=AsyncMock()) as mock_save,
    ):
        result = await store.async_remove_instance(entry_id)
        assert result is True
        assert mock_save.call_count == 1
    # Case: not present
    with (
        patch.object(
            store, "async_load", new=AsyncMock(return_value={"instances": {}})
        ),
        patch.object(store, "async_save", new=AsyncMock()) as mock_save,
    ):
        result = await store.async_remove_instance(entry_id)
        assert result is False
        assert mock_save.call_count == 0


@pytest.mark.asyncio
async def test_async_remove_instance_handles_exception(store):
    """Test that async_remove_instance returns False if async_load raises an exception."""
    entry_id = "test_entry"
    with patch.object(
        store, "async_load", new=AsyncMock(side_effect=Exception("fail"))
    ):
        result = await store.async_remove_instance(entry_id)
        assert result is False


@pytest.mark.asyncio
async def test_async_load_instance_prior_state_error(store):
    """Test that StorageLoadError is raised if async_load fails during load."""
    entry_id = "test_entry"
    with (
        patch.object(store, "async_load", new=AsyncMock(side_effect=Exception("fail"))),
        pytest.raises(StorageLoadError),
    ):
        await store.async_load_instance_prior_state(entry_id)


@pytest.mark.asyncio
async def test_async_save_instance_prior_state_error(store):
    """Test that StorageSaveError is raised if async_save fails during save."""
    entry_id = "test_entry"
    name = "Test Area"
    prior_state = PriorState()
    with (
        patch.object(store, "async_load", new=AsyncMock(return_value=None)),
        patch.object(store, "async_save", new=AsyncMock(side_effect=Exception("fail"))),
        pytest.raises(StorageSaveError),
    ):
        await store.async_save_instance_prior_state(entry_id, name, prior_state)


@pytest.mark.asyncio
async def test_async_cleanup_orphaned_instances_removes_orphans(store):
    """Test that orphaned instances are removed and async_save is called."""
    active_ids = {"id1"}
    orphan_id = "id2"
    stored = {"instances": {"id1": {}, "id2": {}}}
    with (
        patch.object(store, "async_load", new=AsyncMock(return_value=stored)),
        patch.object(store, "async_save", new=AsyncMock()) as mock_save,
    ):
        result = await store.async_cleanup_orphaned_instances(active_ids)
        assert result is True
        assert mock_save.call_count == 1
        args, kwargs = mock_save.call_args
        saved = args[0]
        assert orphan_id not in saved["instances"]
        assert "id1" in saved["instances"]


@pytest.mark.asyncio
async def test_async_cleanup_orphaned_instances_no_orphans(store):
    """Test that no action is taken if there are no orphaned instances."""
    active_ids = {"id1"}
    stored = {"instances": {"id1": {}}}
    with (
        patch.object(store, "async_load", new=AsyncMock(return_value=stored)),
        patch.object(store, "async_save", new=AsyncMock()) as mock_save,
    ):
        result = await store.async_cleanup_orphaned_instances(active_ids)
        assert result is False
        assert mock_save.call_count == 0


@pytest.mark.asyncio
async def test_async_cleanup_orphaned_instances_missing_data(store):
    """Test that missing or malformed data is handled gracefully."""
    # No data at all
    with patch.object(store, "async_load", new=AsyncMock(return_value=None)):
        result = await store.async_cleanup_orphaned_instances({"id1"})
        assert result is False
    # No 'instances' key
    with (
        patch.object(store, "async_load", new=AsyncMock(return_value={})),
        patch.object(store, "async_save", new=AsyncMock()) as mock_save,
    ):
        result = await store.async_cleanup_orphaned_instances({"id1"})
        assert result is False
        assert mock_save.call_count == 0


@pytest.mark.asyncio
async def test_async_cleanup_orphaned_instances_handles_exception(store):
    """Test that exceptions during cleanup are handled and do not raise."""
    with patch.object(
        store, "async_load", new=AsyncMock(side_effect=Exception("fail"))
    ):
        result = await store.async_cleanup_orphaned_instances({"id1"})
        assert result is False
