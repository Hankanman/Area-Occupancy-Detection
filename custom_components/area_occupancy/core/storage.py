# custom_components/area_occupancy/core/storage.py

"""Storage provider for Area Occupancy Detection."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import json
import logging
import os
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from . import StorageProvider

_LOGGER = logging.getLogger(__name__)

STORAGE_VERSION = 1
STORAGE_KEY = "area_occupancy.{}"


class AreaStorageProvider(StorageProvider):
    """Manages persistent storage for area occupancy data."""

    def __init__(
        self,
        hass: HomeAssistant,
        area_id: str,
        max_history_entries: int = 1000,
    ) -> None:
        """Initialize the storage provider."""
        self.hass = hass
        self._area_id = area_id
        self._max_entries = max_history_entries
        self._storage_key = STORAGE_KEY.format(area_id)
        self._file_path = self.hass.config.path(f"{self._storage_key}.json")
        self._write_lock = asyncio.Lock()
        self._data: dict[str, Any] = {}

    async def async_load(self) -> dict[str, Any]:
        """Load stored data."""
        try:
            if not os.path.exists(self._file_path):
                self._data = self._get_default_data()
                return self._data.copy()

            async with self._write_lock:
                with open(self._file_path, "r", encoding="utf-8") as file:
                    raw_data = json.load(file)

                # Validate data
                if not self._validate_data(raw_data):
                    _LOGGER.warning(
                        "Invalid data format in %s, creating new", self._file_path
                    )
                    self._data = self._get_default_data()
                else:
                    self._data = raw_data

                return self._data.copy()

        except Exception as err:
            _LOGGER.error("Error loading storage data: %s", err)
            raise HomeAssistantError("Failed to load storage data") from err

    async def async_save(self, data: dict[str, Any]) -> None:
        """Save data to storage."""
        try:
            # Merge new data with existing
            self._merge_data(data)

            # Trim historical data if needed
            self._trim_history()

            # Update metadata
            self._data["metadata"]["last_updated"] = dt_util.utcnow().isoformat()

            # Convert datetime objects to a JSON-friendly format
            def default_converter(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(
                    f"Object of type {obj.__class__.__name__} is not JSON serializable"
                )

            # Save to file
            async with self._write_lock:
                temp_file = f"{self._file_path}.tmp"
                try:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(self._file_path), exist_ok=True)

                    # Write to temporary file first
                    with open(temp_file, "w", encoding="utf-8") as file:
                        json.dump(self._data, file, indent=2, default=default_converter)

                    # Rename temporary file to final location
                    os.replace(temp_file, self._file_path)

                finally:
                    # Clean up temporary file if it still exists
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)

        except Exception as err:
            _LOGGER.error("Error saving storage data: %s", err)
            raise HomeAssistantError("Failed to save storage data") from err

    async def async_remove(self) -> None:
        """Remove stored data."""
        try:
            async with self._write_lock:
                if os.path.exists(self._file_path):
                    os.unlink(self._file_path)
                self._data = self._get_default_data()
        except Exception as err:
            _LOGGER.error("Error removing storage data: %s", err)
            raise HomeAssistantError("Failed to remove storage data") from err

    def _get_default_data(self) -> dict[str, Any]:
        """Get default data structure."""
        return {
            "version": STORAGE_VERSION,
            "area_id": self._area_id,
            "metadata": {
                "created": dt_util.utcnow().isoformat(),
                "last_updated": dt_util.utcnow().isoformat(),
            },
            "history": [],
            "patterns": {
                "time_slots": {},
                "days": {},
            },
        }

    def _validate_data(self, data: dict[str, Any]) -> bool:
        """Validate data structure."""
        required_keys = ["version", "area_id", "metadata", "history", "patterns"]
        if not all(key in data for key in required_keys):
            return False

        if data["version"] != STORAGE_VERSION:
            return False

        if data["area_id"] != self._area_id:
            return False

        return True

    def _merge_data(self, new_data: dict[str, Any]) -> None:
        """Merge new data with existing data."""
        # Update history
        if "history" in new_data:
            self._data["history"].extend(new_data["history"])

        # Update patterns
        if "patterns" in new_data:
            for key in ["time_slots", "days"]:
                if key in new_data["patterns"]:
                    if key not in self._data["patterns"]:
                        self._data["patterns"][key] = {}
                    self._data["patterns"][key].update(new_data["patterns"][key])

        # Update other data
        for key, value in new_data.items():
            if key not in ["history", "patterns", "version", "area_id"]:
                self._data[key] = value

    def _trim_history(self) -> None:
        """Trim historical data to prevent excessive growth."""
        if len(self._data["history"]) > self._max_entries:
            excess = len(self._data["history"]) - self._max_entries
            self._data["history"] = self._data["history"][excess:]

        # Trim pattern data
        for pattern_type in ["time_slots", "days"]:
            patterns = self._data["patterns"].get(pattern_type, {})
            if len(patterns) > self._max_entries:
                # Sort patterns by last_updated and keep only most recent
                sorted_patterns = sorted(
                    patterns.items(),
                    key=lambda x: x[1].get("last_updated", ""),
                    reverse=True,
                )
                self._data["patterns"][pattern_type] = dict(
                    sorted_patterns[: self._max_entries]
                )

    async def async_migrate(self) -> None:
        """Migrate storage data to current version."""
        try:
            if not self._data:
                await self.async_load()

            current_version = self._data.get("version", 0)
            if current_version == STORAGE_VERSION:
                return

            _LOGGER.info(
                "Migrating storage data from version %s to %s",
                current_version,
                STORAGE_VERSION,
            )

            # Perform migrations based on version
            if current_version < 1:
                await self._migrate_to_v1()

            # Update version
            self._data["version"] = STORAGE_VERSION
            await self.async_save(self._data)

        except Exception as err:
            _LOGGER.error("Error migrating storage data: %s", err)
            raise HomeAssistantError("Failed to migrate storage data") from err

    async def _migrate_to_v1(self) -> None:
        """Migrate data to version 1 format."""
        # Add metadata if missing
        if "metadata" not in self._data:
            self._data["metadata"] = {
                "created": dt_util.utcnow().isoformat(),
                "last_updated": dt_util.utcnow().isoformat(),
            }

        # Ensure history is a list
        if "history" not in self._data or not isinstance(self._data["history"], list):
            self._data["history"] = []

        # Initialize patterns structure
        if "patterns" not in self._data:
            self._data["patterns"] = {
                "time_slots": {},
                "days": {},
            }

    async def async_backup(self) -> None:
        """Create a backup of the current storage file."""
        try:
            if not os.path.exists(self._file_path):
                return

            backup_path = f"{self._file_path}.backup"
            async with self._write_lock:
                with open(self._file_path, "r", encoding="utf-8") as source:
                    with open(backup_path, "w", encoding="utf-8") as backup:
                        backup.write(source.read())

        except Exception as err:
            _LOGGER.error("Error creating storage backup: %s", err)
            raise HomeAssistantError("Failed to create storage backup") from err

    async def async_restore_backup(self) -> None:
        """Restore from backup if main storage is corrupted."""
        backup_path = f"{self._file_path}.backup"
        try:
            if not os.path.exists(backup_path):
                _LOGGER.warning("No backup file found at %s", backup_path)
                return

            async with self._write_lock:
                with open(backup_path, "r", encoding="utf-8") as backup:
                    data = json.load(backup)

                if self._validate_data(data):
                    self._data = data
                    await self.async_save(self._data)
                    _LOGGER.info("Successfully restored from backup")
                else:
                    raise ValueError("Invalid backup data format")

        except Exception as err:
            _LOGGER.error("Error restoring from backup: %s", err)
            raise HomeAssistantError("Failed to restore from backup") from err

    def get_statistics(self) -> dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_entries": len(self._data.get("history", [])),
            "pattern_counts": {
                "time_slots": len(self._data.get("patterns", {}).get("time_slots", {})),
                "days": len(self._data.get("patterns", {}).get("days", {})),
            },
            "last_updated": self._data.get("metadata", {}).get("last_updated"),
            "created": self._data.get("metadata", {}).get("created"),
            "version": self._data.get("version"),
        }

    async def async_cleanup(self, max_age: int = 30) -> None:
        """Clean up old data beyond specified age in days."""
        try:
            if not self._data:
                return

            cutoff = dt_util.utcnow() - timedelta(days=max_age)

            # Clean up history
            self._data["history"] = [
                entry
                for entry in self._data["history"]
                if dt_util.parse_datetime(entry.get("timestamp", "")) > cutoff
            ]

            # Clean up patterns
            for pattern_type in ["time_slots", "days"]:
                patterns = self._data["patterns"].get(pattern_type, {})
                self._data["patterns"][pattern_type] = {
                    key: value
                    for key, value in patterns.items()
                    if dt_util.parse_datetime(value.get("last_updated", "")) > cutoff
                }

            await self.async_save(self._data)

        except Exception as err:
            _LOGGER.error("Error cleaning up storage data: %s", err)
            raise HomeAssistantError("Failed to clean up storage data") from err
