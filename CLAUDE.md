# CLAUDE.md — csromer conventions

csromer (Compressed Sensing ROtation MEasure Reconstructor) reconstructs Faraday depth
spectra `F(φ)` from polarized radio spectra `P(λ²)` using regularized maximum likelihood
(RML) and compressed-sensing techniques. Python 3.9–3.11, OOP-first, dask-aware, built around
injectable factories and a step pipeline.

---

## ⚠️ Environment — read this first

**The system `python` cannot run this project.** It has neither `csromer` nor `pywt`
installed. Running `pytest` with it silently skips nearly every meaningful test
(`importorskip("pywt")`) and reports success, which looks identical to passing.

Always use the micromamba environment:

```bash
/home/miguel/micromamba/envs/csromer-env/bin/python -m pytest ...
```

or `micromamba activate csromer-env` first. A result of "N skipped" for the integration
suite means you used the wrong interpreter, not that the tests are fine.

---

## 🔍 General principles

- **Python 3.9–3.11.** The floor is 3.9 (3.8 is EOL); the ceiling is a hard constraint,
  not a preference — `scipy==1.10.0`, `astropy==5.2.1`, `matplotlib==3.6.3` and
  `PyWavelets==1.4.1` publish no wheels past cp311, so 3.12+ falls back to building from
  source. Supporting newer interpreters requires bumping those pins first. `prox_tv` and
  `pynufft` ship no wheels at all and always compile, so a cold environment needs a
  C/C++ toolchain.
- Readable, maintainable, as simple as the problem allows.
- PEP 8, enforced by `yapf` (pep8 base, **column limit 100**, `dedent_closing_brackets`)
  and `isort` (line length 100, trailing commas). Config lives in `setup.cfg`.
- Style is enforced by pre-commit hooks. **They rewrite files**, so a commit whose files
  need reformatting will abort on the first attempt — re-stage and commit again. This is
  normal, not an error.
- `pytest` for all tests. New functionality must come with tests.
- Avoid code duplication. Avoid global mutable state.
- Make decisions with long-term code health in mind.

---

## 📦 Naming

- **Classes**: CamelCase — `ConjugateGradient`, `DirectFourier1D`, `FaradayThinSource`.
- **Functions / methods**: snake_case, verb-first — `calculate_gradient()`,
  `dirty_spectrum()`, `build_parameter()`.
- **Modules**: snake_case — `direct_fourier.py`, `optimization_steps.py`.
- **Collections**: plural — `terms`, `priors`, `samples`.
- **Protected** (`_`): internal but subclass-accessible — `_line_search()`, `_grad()`.
- **Private** (`__`): not for external access.
- Domain aliases exist and are intentional: `NDFT1D` is `DirectFourier1D`. Keep both
  working when refactoring.

### Method order within a class

Private (`__`) → protected (`_`) → public. Consistent ordering makes class structure
legible at a glance.

---

## 📖 Docstrings — **diverges from pyralysis**

pyralysis mandates NumPy-style docstrings. **csromer uses Google-style throughout**
(`Args:` / `Returns:` / `Raises:`), and every method additionally declares its visibility
and contract on the first body line. Match the surrounding code — do not introduce
NumPy-style sections here.

```python
def calculate_gradient(self, x):
    """
    Calculate gradient of normalized chi-squared.

    Public method. F(x) = (1/2) sum(w * |residual|^2) / n_eff, so
    dF/dx = (1/n_eff) * (-A^H(w*r)).

    Args:
        x: Input array (Faraday depth or coefficients)

    Returns:
        Gradient array (same shape as x), normalized by n_eff

    Raises:
        ValueError: If the operator is not configured
    """
```

- Open with a one-line summary, then `Public method.` / `Protected method.` /
  `Private helper function.` / `Abstract method: subclasses must implement.`
- Document array dtypes and shapes — most arrays here are **complex**, and that is the
  single most common source of confusion.
- State the mathematical convention when one exists (sign of the exponent, whether
  `l2_ref` is applied, what is normalized by what). These have caused real bugs.
- Comments explain **why**, not what. Update docstrings when you change behavior.

---

## 🧩 Architecture

### Injectable factories

The reconstructor takes behavior by injection, not by subclassing. A factory is a callable
returning a configured object:

- `optimizer_factory(parameter, F_obj) -> optimizer` with `.run()` — see
  `make_cg_optimizer`, `make_fista_optimizer`.
- `objective_factory(measurement_operator, parameter) -> OFunction`.
- `measurement_operator` — pass an instance, or let the pipeline build one from
  `measurement_operator_kind` (`"direct"`, `"nufft"`, `"gridded"`).

Add new solvers and priors as new factories. Do not add branches to the reconstructor.

### Step pipeline

`CSROMERReconstructorWrapper` runs an ordered list of step objects, each a class with
`run(ctx)` where `ctx` is the reconstructor itself. Steps read and write attributes on
`ctx`. Keep each step to one responsibility; see `steps/optimization_steps.py`.

### Objective terms — `Fi` / `OFunction`

`OFunction` is `F = Σᵢ regᵢ · termᵢ(x)`. Every term subclasses `Fi` and implements
`evaluate`, `calculate_gradient`, `calculate_prox`.

**`is_differentiable` is load-bearing.** It determines whether a term can be handled by a
gradient method or requires a proximal one. Set it correctly on every new term — solver
routing depends on it.

### Measurement operators

`MeasurementOperator` defines `forward` (φ → λ²), `adjoint` / `backward` (λ² → φ), and
`dirty_spectrum` (weighted, normalized adjoint). Subclasses implement `_forward_impl` /
`_adjoint_impl`; the base class applies the `l2_ref` phase ramp and any wavelet transform
so all paths share one convention. **Do not apply `l2_ref` inside a subclass.**

---

## 🔢 Arrays

- Faraday-space arrays are **`complex64`**. Preserve dtype through operations.
- Code must work for both numpy and dask. Use the helpers rather than importing numpy
  directly for dispatch: `math_module(x)`, `asnumpy(x)`, `is_dask_array(x)`.
- Call `.compute()` only when control flow genuinely needs a concrete value. Guard with
  `hasattr(x, "compute")`.
- Prefer `xp.maximum(a, eps)` over `xp.where(...)` for division guards — smaller graphs.
- Use `np.finfo(np.float32).eps` for numerical tolerances. **Do not** use
  `np.finfo(np.float32).tiny` as a smoothing scale: at ~1.18e-38 it is a division-by-zero
  guard only, and `sqrt(x² + tiny)` is bit-identical to `|x|` in float64. Conflating the
  two produced a real, hard-to-find bug (see below).

---

## 🌀 Faraday domain — gotchas that have caused real bugs

These are not style preferences. Each one cost significant debugging time.

**`|F(φ)|` is invariant to `l2_ref`.** The reference λ²₀ applies only a phase ramp
`exp(2jφλ²₀)` after the adjoint, so the amplitude is mathematically unchanged on any grid.
**Never use `|F(φ)|` to demonstrate a resolution difference** — it cannot show one.
`delta_phi_full` / `delta_phi_nom` describe the FWHM of the **real** beam peak, so compare
`Re[F(φ)]`. Verified: `FWHM(Re[F])` is ~15.5 vs ~73.7 for the two settings, matching the
formulas, while `FWHM(|F|)` is identical for both.

**Flux requires coherent summation.** `Σ|F|` is not flux; `|ΣF|` is. The ratio between
them is a useful phase-coherence diagnostic — a large gap means the model has scattered
phases and is not adding constructively.

**Keep the likelihood in complex space.** Noise in `P = Q + iU` is Gaussian and symmetric;
noise in `|F|` is Rician — biased high at low SNR and non-Gaussian. Impose amplitude- or
angle-motivated *priors* freely, but do not reparameterize the χ² itself into
`(amplitude, angle)`.

**Do not parameterize pixel-wise `(A, χ)`.** It is non-convex, χ wraps mod π, and χ is
undefined wherever `A = 0` — which for a sparse solution is almost everywhere. Angle
belongs at the *component* level, where it is identifiable. The existing complex-modulus
L1 already gives "penalize amplitude, leave phase free", convexly.

**`build_measurement_operator` returns the same dataset object** for `kind="direct"` and
`"nufft"` — only `"gridded"` returns a new one. Mutating `dataset.l2_ref` between two
operator builds therefore changes *both*, since `delta_phi` is a live property. Use
`copy.deepcopy` when comparing configurations.

**Non-smooth priors need proximal methods.** CG's Armijo line search assumes
differentiability. It will not diverge (the line search now refuses any step that
increases the objective), but its smoothed-gradient handling of L1 induces measurably less
sparsity than FISTA's exact soft-threshold prox. Route non-smooth objectives to FISTA.

---

## 🧪 Testing

- `tests/unit/` — fast, no full pipeline. `tests/integration/` — simulate → reconstruct.
- Markers, registered in `pyproject.toml`:
  - `integration` — full pipeline, requires PyWavelets.
  - `slow` — long-running regression tests. Run with `pytest -m slow`.
- Guard optional dependencies with `pytest.importorskip("pywt")` at module level.
- Seed every random draw: `np.random.RandomState(42)`. Reconstruction tests must be
  deterministic.
- **Assert on physics, not on "did not crash."** A test that only checks `isfinite` and a
  loose tolerance cannot detect a regression in reconstruction quality. Prefer recovered
  RM, recovered flux, and peak concentration against known truth.
- Some tests are *deliberately failing* regression guards documenting known defects
  (`test_optimizer_monotonicity.py::test_model_peak_recovers_reasonable_fraction_of_dirty_peak`).
  Do not "fix" these by loosening tolerances — either fix the underlying defect or leave
  them red.

**CI** (`.github/workflows/build.yaml`) triggers only on pushes to `development` and pull
requests to `master` — feature branches do not run it. When it does run, it invokes
`pytest tests/` with **no marker filter**, so `slow` tests run too. If you add a slow or
intentionally-failing test, decide deliberately what CI should do with it: exclude the
marker, or mark the test `xfail` so it stays visible without blocking the build.

---

## 🗺️ Roadmap

Active work is tracked in
[issue #26](https://github.com/miguelcarcamov/csromer/issues/26) (pinned), with per-
workstream issues #11–#25 labelled `roadmap`, `effort:*` and `difficulty:*`. Issues
labelled `unblocked` have no dependencies.

Two principles from that plan that affect day-to-day work:

1. **Diagnostics before claims.** The metrics are themselves under revision (#12). Any
   claim that a change improved reconstruction quality needs a diagnostic that can
   distinguish improvement from a loosened tolerance.
2. **1D is the testbed for 3D.** Keep operators strictly linear-operator-shaped and
   composable (`A_spatial ⊗ A_faraday`), and avoid hardening interfaces around
   single-prior assumptions — FISTA takes one prox term, and 3D will need several
   (#24: SDMM vs primal-dual).

---

## 💡 Error handling

- Raise exceptions; do not return error codes or fail silently.
- Error messages should say what went wrong **and** how to fix it.
- Prefer a loud failure over a plausible-looking wrong number. Several bugs in this
  codebase's history were silent: a line search returning a degenerate step, an aliased
  dataset, an amplitude plot that could not show the effect it claimed. When a result
  looks suspiciously clean, verify the quantity actually depends on the thing you varied.
