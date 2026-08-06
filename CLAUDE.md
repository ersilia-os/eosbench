# eosbench conventions

A Python package (`src/eosbench/`) providing molecular activity datasets for benchmarking
Ersilia ML models, plus the `scripts/` tooling that builds those datasets. This file records
the conventions actually in force here — see `README.md` for what the package does.

## Layout

```
src/eosbench/
├── __init__.py       # public API: get_catalog, load_dataset, make_id, mirror_dataset
├── dataset.py         # catalog + dataset loading
├── cli/               # one module per command (catalog, info, fetch) + main.py entry point
└── utils/logging.py   # the logger singleton
scripts/                # dataset-preparation tooling; needs the prepare-* extras, not part
                        # of the installed package
tests/
docs/                   # long-form reference content the README defers to
```

Keep public APIs small — `eosbench` is thought of as a simple API and CLI. Favour submodules
over a single flat file for anything that grows past a few hundred lines; the current
exceptions (`dataset.py`, `cli/catalog.py`, `scripts/_prepare_common.py`) are grandfathered,
not a pattern to extend.

## Code style

- Run `ruff check` and `ruff format` before every commit; both must pass against `ruff.toml`.
- Docstrings: NumPy convention for every public class, function, and method — a one-line
  summary, then `Parameters`/`Returns` sections when the function takes or returns anything
  non-obvious. Private helpers (leading `_`) only need a docstring when the intent isn't clear
  from the name and signature.
- Keep code, docstrings, and `README.md`/`docs/` aligned — when behavior changes, update all
  three in the same change.
- Catch specific exceptions, not bare `except:`/`except Exception:` — if a failure is meant to
  be silently tolerated, say why in a comment.

## Logging

A module-level singleton on `loguru` + Rich's `RichHandler`, exposed as `eosbench.utils.logging.logger`
with `debug`/`info`/`warning`/`error`/`critical`/`success`. Import the singleton everywhere —
don't call `loguru.logger` or stdlib `logging.getLogger(...)` directly in feature code.
CLI modules (`cli/*.py`) are the one exception: they may `print(..., file=sys.stderr)` for
direct user-facing status lines and render output, since that *is* their job.

## CLI

Argparse, not Click — a deliberate, accepted deviation from the org default here (three
subcommands with a well-established, tested argparse interface; not planned to change).
Still follow the shared vocabulary:

- Multiword options are kebab-case (`--output-dir`, not `--output_dir`).
- File arguments use `-i`/`--input` and `-o`/`--output`. Categorical filters (`--source`,
  `--task`) are not file arguments and don't need to follow this.
- Document commands in the README as a compact two-column table (command → one-line
  description), not prose or a paste of `--help` output.

## Tests

Smoke-test the public API and CLI (`tests/test_dataset.py`, `tests/test_*_cli.py`); skip
exhaustive unit coverage of internals. Keep `tests/` lean.

## Dependencies

- Pin exact versions (`==X.Y.Z`) for every entry in `pyproject.toml`, runtime and optional —
  no floors (`>=`), no ranges. Exceptions require a comment explaining why (see the
  `prepare-tdcommons`/`prepare-polaris` extras for an example: their pins exist to keep two
  otherwise-conflicting dependency trees resolvable in one `pip install`).
- Evaluate every new dependency; prefer the standard library or an existing transitive one.

## README

Be brutally brief — a screen or two, oriented at "what is this, how do I use it." Long-form
material (full API/CLI reference, dataset-preparation internals, leaderboard-provenance
detail) belongs in `docs/`, linked from the README, not inlined into it.

## Releases

Semantic versioning (`vMAJOR.MINOR.PATCH`). The git tag, GitHub release name, and
`[project].version` in `pyproject.toml` must all agree.

## Data

`data/` is gitignored on purpose — never commit datasets, model artifacts, or large binaries.
Prepared datasets are published to the public S3 bucket with Ersilia's internal `eosvc` tool,
scoped to exactly what was prepared (see `README.md`'s "Preparing new datasets" section).
