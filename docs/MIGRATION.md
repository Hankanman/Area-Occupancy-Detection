# Documentation Migration Plan

## Current stack

- **MkDocs 1.x** + **Material for MkDocs 9.x** (pinned in [`pyproject.toml`](./pyproject.toml))
- Plugins: `mkdocs-minify-plugin`, `mkdocs-include-markdown-plugin`
- Deployed via `mkdocs gh-deploy` in [`.github/workflows/docs.yml`](../.github/workflows/docs.yml)

## Why this page exists

Two upstream announcements change the long-term picture:

1. [Zensical announcement (2025-11-05)](https://squidfunk.github.io/mkdocs-material/blog/2025/11/05/zensical/) — Material for MkDocs enters maintenance mode for ~12 months (through ~Nov 2026). Critical bug and security fixes only.
2. [MkDocs 2.0 announcement (2026-02-18)](https://squidfunk.github.io/mkdocs-material/blog/2026/02/18/mkdocs-2.0/) — MkDocs 2.0 is intentionally incompatible with Material for MkDocs (removes plugins, switches to TOML, pre-renders nav).

The squidfunk-authored successor is **[Zensical](https://zensical.org/)**, which reads `mkdocs.yml` natively and offers a `classic` theme variant that matches Material's look and feel.

## Decision: defensive pin + wait

We are **staying on Material for MkDocs** for now. MkDocs 2.0 cannot accidentally be installed because `pyproject.toml` pins `mkdocs>=1.6.0,<2.0.0` and `mkdocs-material>=9.5.0,<10.0.0`. CI sets `NO_MKDOCS_2_WARNING=1` to silence the upstream deprecation banner.

## Blockers for migrating to Zensical today

Tracked so we know when to revisit:

- `mkdocs-minify-plugin` — not yet in Zensical's committed plugin list.
- `mkdocs-include-markdown-plugin` — not yet in Zensical's committed plugin list.
- `mkdocs gh-deploy` equivalent — not documented in Zensical CLI (`build`, `serve`, `new` only).
- Zensical version is `0.0.x` (pre-1.0); public Material→Zensical migration guide does not yet exist.
- Navigation feature parity (`navigation.tabs`, `navigation.instant`, `navigation.footer`, etc.) is not confirmed.

## Revisit criteria

Re-evaluate when **all** of the following hold:

- Zensical ships a `gh-deploy` equivalent (or a documented GitHub Pages workflow).
- Both `mkdocs-minify-plugin` and `mkdocs-include-markdown-plugin` are supported, or acceptable replacements exist.
- Zensical reaches a `1.0` or otherwise stable release.
- A community migration guide from Material for MkDocs exists.

Otherwise, revisit no later than **2026-08** to leave runway before the Material maintenance window closes.

## Useful references

- [Zensical — Get started](https://zensical.org/docs/get-started/)
- [Zensical — Basics / config](https://zensical.org/docs/setup/basics/)
- [Zensical — FAQ](https://zensical.org/docs/community/faqs/)
- [Material for MkDocs discussion: Migration to Zensical](https://github.com/squidfunk/mkdocs-material/discussions/8524)
