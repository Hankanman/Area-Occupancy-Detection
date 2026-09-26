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

The repo root is dropped from ``sys.path`` too, for a subtler reason.
``python -m`` puts the working directory (the repo root) on the path, and the
repo root has its own ``custom_components``. Home Assistant imports the
``custom_components`` namespace package with the config directory mounted,
then unmounts it -- and a namespace package's search path is recomputed from
``sys.path``, so from then on only the repo's copy is visible. With the usual
symlink that is the same code, but an instance running a *different* version
(``harness upgrade-base``) would silently run the working tree instead.
"""

from __future__ import annotations

from pathlib import Path
import sys

#: The checkout this launcher lives in.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_repo_root(entry: str) -> bool:
    """Whether a ``sys.path`` entry points at the repo root.

    Args:
        entry: A ``sys.path`` entry; empty means the working directory.

    Returns:
        True if it resolves to the repo root.
    """
    return Path(entry or ".").resolve() == REPO_ROOT


def main() -> int:
    """Start Home Assistant against the config directory in ``argv[1]``.

    Returns:
        Home Assistant's exit code.
    """
    config_dir = sys.argv[1]
    extra = sys.argv[2:]
    sys.path = [
        entry
        for entry in sys.path
        if "__editable__" not in entry and not _is_repo_root(entry)
    ]
    # The editable install's meta-path finder maps the integration package
    # straight to the checkout, whatever ``sys.path`` says.
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if not getattr(finder, "__module__", "").startswith("__editable__")
    ]

    # Imported after the path fix-up, which is the entire reason this
    # launcher exists.
    from homeassistant.__main__ import main as hass_main  # noqa: PLC0415

    sys.argv = ["hass", "--config", config_dir, "--skip-pip", *extra]
    return hass_main()


if __name__ == "__main__":
    sys.exit(main())
