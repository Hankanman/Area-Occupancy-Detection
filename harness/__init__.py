"""Throwaway Home Assistant instances for testing Area Occupancy Detection.

The harness builds a disposable HA config directory, seeds it with a config
entry, matching HA areas, mock sensor entities and synthetic learned history,
then starts a real Home Assistant against it. Everything it writes lives under
one directory that can be deleted afterwards; nothing touches the repo's own
``config/``.

See ``docs/docs/technical/dev-harness.md`` for the guided tour, or run
``scripts/harness --help``.
"""

from __future__ import annotations

__all__ = ["__doc__"]
