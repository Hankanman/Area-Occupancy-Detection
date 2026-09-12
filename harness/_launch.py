"""Home Assistant entry point used to start a harness instance.

Invoked as ``python -m harness._launch <config_dir> [args...]``.

It exists for one reason: an editable install of this repo puts a custom path
hook on ``sys.path`` (``__editable__.area_occupancy_detection...``) whose
finder file can go missing when the venv is rebuilt, and Home Assistant's
component importer then dies on a ``FileNotFoundError`` that has nothing to do
with the code under test. Dropping those entries is a no-op on a healthy
install and saves a confusing debugging detour on a broken one -- the
integration is loaded from the instance's ``custom_components`` symlink
either way.
"""

from __future__ import annotations

import sys


def main() -> int:
    """Start Home Assistant against the config directory in ``argv[1]``.

    Returns:
        Home Assistant's exit code.
    """
    config_dir = sys.argv[1]
    extra = sys.argv[2:]
    sys.path = [entry for entry in sys.path if "__editable__" not in entry]

    # Imported after the path fix-up, which is the entire reason this
    # launcher exists.
    from homeassistant.__main__ import main as hass_main  # noqa: PLC0415

    sys.argv = ["hass", "--config", config_dir, "--skip-pip", *extra]
    return hass_main()


if __name__ == "__main__":
    sys.exit(main())
