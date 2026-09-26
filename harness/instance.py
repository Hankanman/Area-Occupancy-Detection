"""Lifecycle of a throwaway Home Assistant instance.

An instance is one directory. Everything about it -- generated config, seeded
storage, the integration database, logs, and the harness's own notes about
how it was built -- lives inside, so ``destroy`` is a single recursive delete
and a forgotten instance costs nothing but disk.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any

from custom_components.area_occupancy.const import DOMAIN

from . import history, mock_config, storage
from .client import ApiError, Client
from .profiles import Profile, get_profile

#: Marker file identifying a directory as a harness instance. ``destroy``
#: refuses to touch a directory without it.
MARKER = "harness.json"

#: Default credentials for the throwaway owner account. The instance listens
#: on localhost and is meant to be deleted, so these are fixed on purpose.
DEFAULT_USERNAME = "dev"
DEFAULT_PASSWORD = "devpassword"

REPO_ROOT = Path(__file__).resolve().parent.parent


class InstanceError(RuntimeError):
    """The instance could not be built, started, or inspected."""


def free_port() -> int:
    """Pick a free TCP port on localhost.

    Returns:
        A port number that was free a moment ago.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(slots=True)
class Instance:
    """A harness instance on disk.

    Attributes:
        path: The instance's configuration directory.
        meta: Contents of the marker file.
    """

    path: Path
    meta: dict[str, Any]

    # -- construction ---------------------------------------------------

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        profile_name: str,
        entry_version: int,
        days: int,
        seed: int,
        time_zone: str,
        port: int,
        frontend: bool = True,
        with_priors: bool = True,
        force: bool = False,
    ) -> Instance:
        """Build a new instance directory, seeded and ready to start.

        Args:
            path: Directory to create. Must not already exist unless
                ``force``.
            profile_name: Profile to build.
            entry_version: Config entry version to seed, current or legacy.
            days: Days of synthetic history to generate, 0 for none.
            seed: Random seed for history generation.
            time_zone: IANA time zone for the instance.
            port: Port to listen on, or 0 to pick a free one.
            frontend: Whether to load the frontend.
            with_priors: Whether to seed the priors the history implies.
            force: Replace an existing harness instance at this path.

        Returns:
            The created instance.

        Raises:
            InstanceError: If the path exists and cannot be replaced.
        """
        if path.exists():
            if not force:
                raise InstanceError(
                    f"{path} already exists; pass --force to replace it"
                )
            cls._remove_tree(path)

        profile = get_profile(profile_name)
        port = port or free_port()

        path.mkdir(parents=True)
        (path / "configuration.yaml").write_text(
            mock_config.render(
                profile, time_zone=time_zone, frontend=frontend, port=port
            ),
            encoding="utf-8",
        )
        # Symlinked rather than copied so edits to the integration show up on
        # the next restart without rebuilding the instance.
        (path / "custom_components").symlink_to(
            REPO_ROOT / "custom_components", target_is_directory=True
        )

        storage.write_http_config(path, port)
        storage.write_area_registry(path, profile)
        entry_id = storage.write_config_entry(
            path, profile, entry_version=entry_version
        )

        counts: dict[str, int] = {}
        if days > 0:
            histories = history.generate(
                profile, days=days, time_zone=time_zone, seed=seed
            )
            counts = history.write(
                path,
                profile,
                histories,
                entry_id=entry_id,
                with_priors=with_priors,
            )

        meta = {
            "profile": profile.name,
            "entry_id": entry_id,
            "entry_version": entry_version,
            "days": days,
            "seed": seed,
            "time_zone": time_zone,
            "port": port,
            "frontend": frontend,
            "rows": counts,
            "token": None,
            "refresh_token": None,
            "pid": None,
        }
        instance = cls(path=path, meta=meta)
        instance.save()
        return instance

    @classmethod
    def load(cls, path: Path) -> Instance:
        """Open an existing instance directory.

        Args:
            path: The instance directory.

        Returns:
            The instance.

        Raises:
            InstanceError: If the directory is not a harness instance.
        """
        marker = path / MARKER
        if not marker.is_file():
            raise InstanceError(f"{path} is not a harness instance (no {MARKER})")
        return cls(path=path, meta=json.loads(marker.read_text(encoding="utf-8")))

    def save(self) -> None:
        """Write the marker file back to disk."""
        (self.path / MARKER).write_text(
            json.dumps(self.meta, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _remove_tree(path: Path) -> None:
        """Delete an instance directory, refusing anything unmarked.

        Args:
            path: Directory to delete.

        Raises:
            InstanceError: If the directory has no marker file.
        """
        if not (path / MARKER).is_file():
            raise InstanceError(
                f"refusing to delete {path}: no {MARKER}, so it was not built by the harness"
            )
        shutil.rmtree(path)

    def destroy(self) -> None:
        """Stop the instance if running and delete its directory."""
        self.stop()
        self._remove_tree(self.path)

    # -- properties -----------------------------------------------------

    @property
    def profile(self) -> Profile:
        """The profile the instance was built from."""
        return get_profile(self.meta["profile"])

    @property
    def port(self) -> int:
        """The port the instance listens on."""
        return int(self.meta["port"])

    @property
    def base_url(self) -> str:
        """The instance's HTTP root."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def entry_id(self) -> str:
        """The seeded config entry id."""
        return str(self.meta["entry_id"])

    @property
    def log_path(self) -> Path:
        """Where the launcher's own output is written."""
        return self.path / "harness.log"

    def client(self) -> Client:
        """A client bound to this instance, carrying its saved token.

        Returns:
            The client.
        """
        return Client(
            base_url=self.base_url,
            token=self.meta.get("token"),
            refresh_token=self.meta.get("refresh_token"),
        )

    # -- process --------------------------------------------------------

    def is_running(self) -> bool:
        """Whether the recorded process is still alive.

        Returns:
            True if a pid is recorded and that process exists.
        """
        pid = self.meta.get("pid")
        if not pid:
            return False
        try:
            os.kill(int(pid), 0)
        except (OSError, ValueError):
            return False
        return True

    def start(self, *, timeout: float = 240.0, debug: bool = False) -> Client:
        """Start Home Assistant and return a ready, authenticated client.

        Args:
            timeout: Seconds to wait for the API to answer.
            debug: Pass ``--debug`` to Home Assistant.

        Returns:
            A client with a bearer token for the instance.

        Raises:
            InstanceError: If the instance is already running, the process
                exits during startup, or the API never answers.
        """
        if self.is_running():
            raise InstanceError(f"instance already running (pid {self.meta['pid']})")

        command = [
            sys.executable,
            "-m",
            "harness._launch",
            str(self.path),
            "--log-file",
            str(self.path / "home-assistant.log"),
        ]
        if debug:
            command.append("--debug")

        log = self.log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
        )
        self.meta["pid"] = process.pid
        self.save()

        try:
            self._wait_for_api(process, timeout)
            client = self.client()
            needs_token = not client.onboarding_done()
            if needs_token:
                self.meta["token"] = client.onboard(
                    username=DEFAULT_USERNAME, password=DEFAULT_PASSWORD
                )
                self.meta["refresh_token"] = client.refresh_token
                self.save()
        except Exception:
            self.stop()
            raise

        if not needs_token and not self.meta.get("token"):
            self.stop()
            raise InstanceError(
                "instance is already onboarded but the harness has no token; "
                "rebuild it with `scripts/harness new --force`"
            )

        return self.client()

    def _wait_for_api(self, process: subprocess.Popen[bytes], timeout: float) -> None:
        """Block until the HTTP API answers at all.

        Any HTTP response counts as ready, including an error one: what is
        being waited for is the server, not a particular endpoint. Probing a
        specific endpoint would be wrong here -- ``/api/onboarding`` stops
        existing once onboarding is done, so an instance being restarted
        would never look ready.

        Args:
            process: The launched Home Assistant process.
            timeout: Seconds to wait.

        Raises:
            InstanceError: If the process exits first or the wait times out,
                with the tail of the log to explain why.
        """
        client = Client(base_url=self.base_url, timeout=5.0)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (code := process.poll()) is not None:
                raise InstanceError(
                    f"Home Assistant exited with code {code} during startup:\n"
                    f"{self.log_tail()}"
                )
            try:
                client.request("GET", "/auth/providers", authenticated=False)
            except ApiError as err:
                if err.status:
                    return
                time.sleep(1.0)
                continue
            return
        raise InstanceError(
            f"Home Assistant did not answer on {self.base_url} within {timeout:.0f}s:\n"
            f"{self.log_tail()}"
        )

    def wait_for_running(self, *, timeout: float = 180.0) -> None:
        """Block until Home Assistant reports that it has finished starting.

        Worth waiting for before reading anything off disk: Home Assistant
        holds back store and registry writes until it is running, so
        ``.storage`` shows pre-startup state until then -- which reads as a
        migration that never happened.

        Args:
            timeout: Seconds to wait.

        Raises:
            InstanceError: If it never reports running.
        """
        client = self.client()
        deadline = time.monotonic() + timeout
        state = "unknown"
        while time.monotonic() < deadline:
            try:
                state = client.get("/api/core/state").get("state", "unknown")
            except ApiError:
                state = "unreachable"
            if state.upper() == "RUNNING":
                # The started event fires the deferred writes; give them a
                # moment to reach disk before anyone reads them.
                time.sleep(2.0)
                return
            time.sleep(1.0)
        raise InstanceError(
            f"Home Assistant stayed in state {state!r} for {timeout:.0f}s:\n{self.log_tail()}"
        )

    def wait_for_entry(self, *, timeout: float = 120.0) -> dict[str, Any]:
        """Block until the integration's config entry finishes setting up.

        Args:
            timeout: Seconds to wait.

        Returns:
            The entry as the API reports it.

        Raises:
            InstanceError: If the entry never leaves a transitional state, or
                is not present at all.
        """
        client = self.client()
        deadline = time.monotonic() + timeout
        last: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            entries = [
                entry
                for entry in client.config_entries(DOMAIN)
                if entry["entry_id"] == self.entry_id
            ]
            if entries:
                last = entries[0]
                if last.get("state") not in ("setup_in_progress", "not_loaded"):
                    return last
            time.sleep(1.0)
        state = last.get("state") if last else "missing"
        raise InstanceError(
            f"config entry did not finish setting up (state: {state}):\n{self.log_tail()}"
        )

    def stop(self, *, timeout: float = 90.0) -> None:
        """Stop the instance, waiting for a clean shutdown.

        Waiting matters: Home Assistant defers registry and store writes
        while it is still starting, so a killed instance can leave ``.storage``
        showing pre-migration state even though the migration ran. A clean
        shutdown flushes everything.

        Args:
            timeout: Seconds to wait before escalating to SIGKILL.
        """
        pid = self.meta.get("pid")
        if not pid:
            return
        pid = int(pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ValueError):
            self.meta["pid"] = None
            self.save()
            return

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            time.sleep(0.5)
        else:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)

        self.meta["pid"] = None
        self.save()

    def log_tail(self, lines: int = 25) -> str:
        """The tail of the instance's Home Assistant log.

        Args:
            lines: How many lines to return.

        Returns:
            The last lines of the log, or a note if there is none yet.
        """
        for candidate in (self.path / "home-assistant.log", self.log_path):
            if candidate.is_file():
                content = candidate.read_text(encoding="utf-8", errors="replace")
                if tail := content.splitlines()[-lines:]:
                    return "\n".join(tail)
        return "(no log output yet)"
