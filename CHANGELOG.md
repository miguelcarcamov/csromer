# Changelog

All notable changes to csromer are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Release stages

Work reaches users in three stages, mirrored by CI:

| stage | trigger | what happens |
|---|---|---|
| development | push or PR to any branch | tests run on Python 3.9, 3.10 and 3.11 |
| pre-release | merge to `master` | package and container build, image published as `latest`, release published to TestPyPI |
| release | a published GitHub release | published to PyPI |

Entries accumulate under **Unreleased** and are promoted to a version heading when a
release is tagged.

> **Note:** the project version is derived from git tags by `setuptools_scm`, and the
> repository currently has **no tags**. Until one is created, builds carry a
> development version rather than a release number. The first tag should be created
> before the first PyPI publish.

## [Unreleased]

### Added

- `CLAUDE.md` recording repository conventions: naming, the Google-style docstring
  format used here, the factory and step-pipeline architecture, array handling, and a
  section of Faraday-domain constraints that have each caused real bugs.
- `tox.ini` with `py39`/`py310`/`py311` environments driven by `tox-uv`, plus a `lint`
  environment that runs the pre-commit hooks across the tree.
- CI test matrix running the suite on every supported interpreter, on every push and
  every pull request.
- Regression tests for conjugate-gradient monotonicity, covering all five CG variants
  against both smooth and L1-regularized objectives.
- A regression guard for model-peak flux recovery, marked `xfail` because it documents
  a known, unfixed deficit rather than a regression.

### Fixed

- **Conjugate-gradient line search could take a step that increased the objective.** On
  exhausting the Armijo backtracking budget, `_line_search` returned the final trial
  step regardless of whether it improved anything. With L1 active this collapsed to
  `1.0 * rho**30` on every iteration from roughly the sixth onward — a silent stall
  presenting as thousands of no-op iterations with a slowly drifting objective. It now
  falls back to the best trial point seen, or to no step at all.
- **Full-versus-nominal resolution comparisons in notebooks 03 and 07 could not show
  the effect they claimed.** They plotted `|F(φ)|`, which is invariant to `l2_ref` —
  the reference applies only a phase ramp — so both settings produced identical curves
  by construction. They now plot `Re[F(φ)]`, which is what `delta_phi_full` and
  `delta_phi_nom` describe. Notebook 07 additionally aliased one dataset object across
  both cases, so both reported whichever `l2_ref` was set last.
- Docker `latest` was published from a `pull_request` trigger, i.e. built from unmerged
  code. It now publishes on merge to `master` and on release.
- TestPyPI published at the same moment as PyPI, making it redundant. It now publishes
  at the pre-release stage, on merge to `master`.
- `README.md` link text, which failed markdownlint's `MD059` and was blocking every
  commit in the repository regardless of what was staged.

### Changed

- `requires-python` raised from `>=3.8` to `>=3.9`. 3.8 is end-of-life, and the real
  ceiling is 3.11: `scipy`, `astropy`, `matplotlib` and `PyWavelets` publish no wheels
  beyond cp311 at the versions pinned in `requirements.txt`.
- CI runs on every branch. It previously ran only on pushes to `development` behind a
  paths filter that would not have matched changes to `tox.ini` or to the workflows
  themselves.
- Container-based `build` and `test` jobs are limited to pull requests targeting
  `master`, since the test matrix already covers every supported interpreter elsewhere.

### Known issues

- Reconstructed model peaks recover substantially less flux than expected — measured
  `dirty_peak/model_peak` ratios of 15–21 against an expectation near 6.9. Tracked in
  [#19](https://github.com/miguelcarcamov/csromer/issues/19) and
  [#16](https://github.com/miguelcarcamov/csromer/issues/16); whether the expectation
  itself is correct is questioned in
  [#12](https://github.com/miguelcarcamov/csromer/issues/12).
- `TSV.calculate_gradient` is numerically incorrect, `TSV.is_differentiable`
  contradicts its own documentation, and `nu` is ignored by both `TV` and `TSV`
  proximal operators. Latent — neither term is reachable from any default path — and
  tracked in [#28](https://github.com/miguelcarcamov/csromer/issues/28).
- Conjugate gradient handles non-smooth priors (L1) less effectively than a proximal
  method. Monotone decrease is now guaranteed, but sparsity is under-induced relative
  to FISTA.

[Unreleased]: https://github.com/miguelcarcamov/csromer/compare/master...HEAD
