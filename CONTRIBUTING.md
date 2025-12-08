# Contribution guidelines

Contributing to this project should be as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

## Github is used for everything

Github is used to host code, to track issues and feature requests, as well as accept pull requests.

Pull requests are the best way to propose changes to the codebase.

1. Fork the repo and create your branch from `dev`.
2. If you've changed something, update the documentation.
3. Make sure your code lints (using `scripts/lint`).
4. Test you contribution.
5. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](../../issues)

GitHub issues are used to track public bugs.
Report a bug by [opening a new issue](../../issues/new/choose); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

People *love* thorough bug reports. I'm not even kidding.

## Getting Started

### Development Environment Setup

This project uses a **devcontainer** for a consistent development environment. When you open this repository in VS Code (or another IDE that supports devcontainers), you'll be prompted to reopen the project in a container. **Accept this prompt** - it will automatically:

- Set up the development environment
- Install all dependencies (root project, simulator, and docs)
- Configure the development tools
- Set up pre-commit hooks

The devcontainer provides a standalone Home Assistant instance running with the included [`configuration.yaml`](./config/configuration.yaml) file, making it easy to test your changes.

### Manual Setup (if not using devcontainer)

If you prefer not to use the devcontainer, you can set up the environment manually:

1. Install [uv](https://github.com/astral-sh/uv) if not already installed
2. Run `scripts/bootstrap` to install all dependencies
3. Activate the virtual environment: `source .venv/bin/activate`

## Use a Consistent Coding Style

This project uses [ruff](https://github.com/astral-sh/ruff) for linting and formatting. Before submitting your changes, run:

```bash
scripts/lint
```

Or manually:
```bash
uv run ruff format .
uv run ruff check . --fix
```

This will automatically format your code and fix any linting issues.

## Test your code modification

This custom component is based on [integration_blueprint template](https://github.com/ludeeus/integration_blueprint).

Run tests using:
```bash
scripts/test
```

Or manually:
```bash
uv run pytest --cov=custom_components/area_occupancy --cov-report=xml --cov-report=term-missing
```

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
