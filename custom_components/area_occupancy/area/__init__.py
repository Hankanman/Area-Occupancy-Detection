"""Area module for Area Occupancy Detection.

This module contains the Area class, which represents an individual
device area in the multi-area architecture, and the AllAreas class
for aggregating data across all areas.
"""

from .all_areas import AllAreas
from .area import Area, AreaDeviceHandle

__all__ = ["AllAreas", "Area", "AreaDeviceHandle"]
