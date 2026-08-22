# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `CLAUDE.md` recording repository conventions: naming, the Google-style docstring
  format used here, the factory and step-pipeline architecture, array handling, the
  release stages, and a section of Faraday-domain constraints that have each caused
  real bugs.
- `CHANGELOG.md`.
- `tox.ini` with `py39`, `py310` and `py311` environments driven by `tox-uv`, plus a
  `lint` environment that runs the pre-commit hooks across the tree.
- Continuous integration test matrix running the suite on every supported interpreter,
  on every push and every pull request.
- Regression tests for conjugate-gradient monotonicity, covering all five CG variants
  against both smooth and L1-regularized objectives.
- Regression guard for model-peak flux recovery, marked `xfail` because it documents a
  known, unfixed deficit rather than a regression.
- Unit tests for `TV` and `TSV`, which previously had no coverage at all.

### Changed

- `requires-python` raised from `>=3.8` to `>=3.9`. 3.8 is end-of-life, and the
  effective ceiling is 3.11: `scipy`, `astropy`, `matplotlib` and `PyWavelets` publish
  no wheels beyond cp311 at the versions pinned in `requirements.txt`.
- Continuous integration runs on every branch. It previously ran only on pushes to
  `development`, behind a paths filter that would not have matched changes to `tox.ini`
  or to the workflows themselves.
- Packaging and container jobs are limited to pull requests targeting `master`, since
  the test matrix already covers every supported interpreter elsewhere.
- `master` is now the pre-release stage and a published release the final one, rather
  than both container publication and TestPyPI firing at release time.
- `TV.evaluate`, `TV.calculate_gradient`, `TSV.evaluate` and `TSV.calculate_gradient`
  are vectorized rather than looping element by element, and are dask-compatible.
- `TSV.is_differentiable` is now `True`, matching both its documentation and the fact
  that a sum of squared differences is smooth. This changes solver routing for
  objectives containing a `TSV` term.

### Fixed

- Conjugate-gradient line search could take a step that increased the objective. On
  exhausting the Armijo backtracking budget, `_line_search` returned the final trial
  step regardless of whether it improved anything. With L1 active this collapsed to
  `1.0 * rho**30` on every iteration from roughly the sixth onward, a silent stall
  presenting as thousands of no-op iterations with a slowly drifting objective. It now
  falls back to the best trial point seen, or takes no step at all.
- `TSV.calculate_gradient` returned an incorrect gradient. It applied the sign-based
  subgradient formula belonging to `TV`, so its output was quantized to multiples of
  two and disagreed with finite differences of its own `evaluate` by a wide margin. It
  also left both endpoints at zero.
- `TV.calculate_prox` and `TSV.calculate_prox` ignored their `nu` argument and always
  thresholded on `reg`. Since FISTA varies its step size, this silently solved the
  wrong subproblem.
- `TV` and `TSV` proximal operators passed complex input directly to the real-valued
  `prox_tv`. Complex input is now handled channel-wise pending the in-house
  replacement.
- Full-versus-nominal resolution comparisons in the multiple-components and
  resolution-comparison notebooks could not show the effect they claimed. They plotted
  the amplitude of the Faraday spectrum, which is invariant to the reference lambda
  squared because that reference applies only a phase ramp, so both settings produced
  identical curves by construction. They now plot the real part, which is what the two
  resolution definitions describe. The multiple-components notebook additionally
  aliased one dataset object across both cases, so both reported whichever reference
  was set last.
- The `latest` container image was published from a `pull_request` trigger, meaning it
  was built from unmerged code. It is now published on merge to `master` and on
  release.
- TestPyPI published at the same moment as PyPI, which made the test index redundant.
  It now publishes at the pre-release stage.
- `README.md` link text failed markdownlint's `MD059` rule, which blocked every commit
  in the repository regardless of what was staged.

[Unreleased]: https://github.com/miguelcarcamov/csromer/compare/master...HEAD
