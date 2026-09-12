"""Command line interface for the test harness.

Run through ``scripts/harness`` so the repo's interpreter and working
directory are set up. ``scripts/harness --help`` lists the commands; each one
takes ``--help`` of its own.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from custom_components.area_occupancy.const import CONF_VERSION

from . import verify
from .instance import DEFAULT_PASSWORD, DEFAULT_USERNAME, Instance, InstanceError
from .profiles import DEFAULT_PROFILE, PROFILES
from .storage import LEGACY_ENTRY_VERSION, SUPPORTED_ENTRY_VERSIONS

#: Where instances live unless ``--dir`` says otherwise. Gitignored.
DEFAULT_ROOT = Path("instances")

GREEN = "\033[0;32m"
RED = "\033[0;31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _colour(text: str, code: str) -> str:
    """Colour text when stdout is a terminal.

    Args:
        text: The text to colour.
        code: The ANSI escape to wrap it in.

    Returns:
        The text, wrapped only if colour will be rendered.
    """
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{RESET}"


def _instance_path(args: argparse.Namespace) -> Path:
    """Resolve the instance directory an invocation refers to.

    Args:
        args: Parsed arguments.

    Returns:
        The directory path.
    """
    if args.dir:
        return Path(args.dir)
    return DEFAULT_ROOT / args.name


def _add_target_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments identifying which instance to act on.

    Args:
        parser: The subcommand parser to extend.
    """
    parser.add_argument(
        "--name",
        default="dev",
        help="instance name under ./instances (default: dev)",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="explicit instance directory, overriding --name",
    )


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments describing how to build an instance.

    Args:
        parser: The subcommand parser to extend.
    """
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=sorted(PROFILES),
        help=f"which profile to build (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--entry-version",
        type=int,
        default=CONF_VERSION,
        choices=SUPPORTED_ENTRY_VERSIONS,
        help=(
            f"config entry version to seed; {LEGACY_ENTRY_VERSION} seeds a "
            f"pre-subentry entry so startup runs the real migration "
            f"(default: {CONF_VERSION})"
        ),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="days of synthetic history to seed, 0 for none (default: 14)",
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="random seed for history (default: 1234)"
    )
    parser.add_argument(
        "--time-zone",
        default="Europe/London",
        help=(
            "instance time zone; prior slots are local time, so a zone with "
            "DST is the more honest default (default: Europe/London)"
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="port to listen on, 0 picks a free one (default: 0)",
    )
    parser.add_argument(
        "--no-frontend",
        action="store_true",
        help="skip the frontend for a faster boot (no clicking, API only)",
    )
    parser.add_argument(
        "--no-priors",
        action="store_true",
        help="seed intervals but not the priors they imply",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing instance"
    )


def _build(args: argparse.Namespace) -> Instance:
    """Create an instance from build arguments.

    Args:
        args: Parsed arguments.

    Returns:
        The created instance.
    """
    path = _instance_path(args)
    instance = Instance.create(
        path,
        profile_name=args.profile,
        entry_version=args.entry_version,
        days=args.days,
        seed=args.seed,
        time_zone=args.time_zone,
        port=args.port,
        frontend=not args.no_frontend,
        with_priors=not args.no_priors,
        force=args.force,
    )
    rows = instance.meta["rows"]
    print(f"{_colour('created', GREEN)} {path}")
    print(f"  profile       {args.profile} ({len(instance.profile.areas)} areas)")
    print(f"  entry version {args.entry_version}")
    print(f"  time zone     {args.time_zone}")
    if rows:
        summary = ", ".join(
            f"{count} {table}" for table, count in rows.items() if count
        )
        print(f"  seeded        {args.days} days -- {summary}")
    else:
        print("  seeded        no history")
    return instance


def cmd_new(args: argparse.Namespace) -> int:
    """Create an instance and, unless told not to, start it.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    instance = _build(args)
    if args.no_start:
        print(f"\nstart it with: scripts/harness start --dir {instance.path}")
        return 0
    return _start(instance, follow=not args.detach)


def cmd_start(args: argparse.Namespace) -> int:
    """Start an existing instance.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    instance = Instance.load(_instance_path(args))
    return _start(instance, follow=not args.detach)


def _start(instance: Instance, *, follow: bool) -> int:
    """Start an instance and report how to reach it.

    Args:
        instance: The instance to start.
        follow: Stay in the foreground and stop the instance on Ctrl-C.

    Returns:
        Process exit code.
    """
    print(f"\nstarting {instance.path} ...")
    instance.start()
    entry = instance.wait_for_entry()
    instance.wait_for_running()
    print(f"{_colour('running', GREEN)} {instance.base_url}")
    print(f"  login         {DEFAULT_USERNAME} / {DEFAULT_PASSWORD}")
    print(f"  token         {instance.meta['token']}")
    print(f"  config entry  {entry.get('state')}")
    print(f"  log           {instance.path / 'home-assistant.log'}")

    if not follow:
        print(f"\nstop it with:  scripts/harness stop --dir {instance.path}")
        return 0

    print("\nCtrl-C to stop the instance.")
    try:
        while instance.is_running():
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nstopping ...")
    instance.stop()
    print("stopped")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop a running instance.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    instance = Instance.load(_instance_path(args))
    instance.stop()
    print(f"stopped {instance.path}")
    return 0


def cmd_destroy(args: argparse.Namespace) -> int:
    """Stop and delete an instance.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    instance = Instance.load(_instance_path(args))
    path = instance.path
    instance.destroy()
    print(f"destroyed {path}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Print what an instance is and whether it is running.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    instance = Instance.load(_instance_path(args))
    running = instance.is_running()
    print(f"{_colour(str(instance.path), BOLD)}")
    print(f"  state         {'running' if running else 'stopped'}")
    print(f"  url           {instance.base_url}")
    for key in ("profile", "entry_version", "days", "seed", "time_zone", "entry_id"):
        print(f"  {key:13s} {instance.meta[key]}")
    if rows := instance.meta.get("rows"):
        for table, count in rows.items():
            if count:
                print(f"  {table:13s} {count} rows")
    return 0


def cmd_profiles(args: argparse.Namespace) -> int:
    """List the available profiles.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    for name in sorted(PROFILES):
        profile = PROFILES[name]
        marker = " (default)" if name == DEFAULT_PROFILE else ""
        print(f"{_colour(name, BOLD)}{marker}")
        print(f"  {profile.description}")
        for area in profile.areas:
            channels = ", ".join(
                f"{channel}x{count}" for channel, count in area.channels.items()
            )
            print(f"    {area.name:14s} {area.purpose:12s} {channels}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Build a throwaway instance, run every check, and report.

    Args:
        args: Parsed arguments.

    Returns:
        0 if every check passed, 1 otherwise.
    """
    if args.dir or args.reuse:
        instance = Instance.load(_instance_path(args))
        if not instance.is_running():
            instance.start()
            instance.wait_for_entry()
        instance.wait_for_running()
        client = instance.client()
    else:
        instance = _build(args)
        print(f"\nstarting {instance.path} ...")
        client = instance.start()
        instance.wait_for_entry()
        instance.wait_for_running()

    print()
    results = verify.run_all(instance, client)

    failures = [result for result in results if not result.passed]
    for result in results:
        mark = _colour("PASS", GREEN) if result.passed else _colour("FAIL", RED)
        print(f"{mark}  {result.name}: {result.detail}")
        if result.notes and (not result.passed or args.verbose):
            for note in result.notes:
                if note:
                    print(f"      {_colour(note, DIM)}")

    print()
    summary = f"{len(results) - len(failures)}/{len(results)} checks passed"
    print(_colour(summary, RED if failures else GREEN))

    if args.keep:
        print(f"\ninstance kept at {instance.path}")
    else:
        instance.destroy()

    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="scripts/harness",
        description=(
            "Build, run and check throwaway Home Assistant instances for "
            "Area Occupancy Detection."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    new = subparsers.add_parser("new", help="create an instance and start it")
    _add_target_arguments(new)
    _add_build_arguments(new)
    new.add_argument(
        "--no-start", action="store_true", help="create it but do not start"
    )
    new.add_argument(
        "--detach", action="store_true", help="leave it running in the background"
    )
    new.set_defaults(func=cmd_new)

    start = subparsers.add_parser("start", help="start an existing instance")
    _add_target_arguments(start)
    start.add_argument(
        "--detach", action="store_true", help="leave it running in the background"
    )
    start.set_defaults(func=cmd_start)

    stop = subparsers.add_parser("stop", help="stop a running instance")
    _add_target_arguments(stop)
    stop.set_defaults(func=cmd_stop)

    destroy = subparsers.add_parser("destroy", help="stop and delete an instance")
    _add_target_arguments(destroy)
    destroy.set_defaults(func=cmd_destroy)

    info = subparsers.add_parser("info", help="describe an instance")
    _add_target_arguments(info)
    info.set_defaults(func=cmd_info)

    profiles = subparsers.add_parser("profiles", help="list the available profiles")
    profiles.set_defaults(func=cmd_profiles)

    check = subparsers.add_parser(
        "verify", help="build an instance, run every check, and destroy it"
    )
    _add_target_arguments(check)
    _add_build_arguments(check)
    check.add_argument(
        "--reuse", action="store_true", help="check an existing instance instead"
    )
    check.add_argument(
        "--keep", action="store_true", help="do not destroy it afterwards"
    )
    check.add_argument(
        "--verbose", action="store_true", help="show the detail of passing checks too"
    )
    check.set_defaults(func=cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list, defaulting to ``sys.argv``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except InstanceError as err:
        print(f"{_colour('error', RED)}: {err}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
