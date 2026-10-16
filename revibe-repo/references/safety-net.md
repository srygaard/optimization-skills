# Safety net: the few tests to add before refactoring

Add these before the first refactor, not a broad test suite. They catch what cleanups of ML code
actually break: imports after moves, entry points, numerics, checkpoint keys and config paths. None of
them needs data, a GPU or a training job.

Put them under `tests/`, in their own commit before the first refactor. Tests are outside the measured
scope by default, so they don't move the scorecard.

**They must run in the project's real environment.** If its dependencies aren't installed on this
machine (torch missing, a package that only exists on the cluster), ask the user for an environment
before refactoring. Don't commit stubs of framework modules: a safety net that only runs against stubs
doesn't protect the real code.

## 1. Always

**Every module imports.** This catches broken imports after moves and deletions.

```python
import importlib, pkgutil, pytest, mypkg

MODULES = [m.name for m in pkgutil.walk_packages(mypkg.__path__, "mypkg.")]

@pytest.mark.parametrize("name", MODULES)
def test_imports(name):
    importlib.import_module(name)
```

**Every entry point starts.** Run each script's `--help` in a subprocess and assert exit code 0.

**A characterization test for each risky function you change:** one that is somewhat complex (several
branches or loops, roughly CC above 5) or has side effects (files, global state, randomness, devices).
Call it with small synthetic inputs, such as tiny tensors or a toy config, and pin today's output, bugs
included. Write it before the change, and make sure it passes before and after. A short or simple pure
function needs no test of its own: read the change, and rely on the import test and its callers' tests.

```python
def test_attention_block_unchanged():
    torch.manual_seed(0)
    block = AttnBlock(8).eval()
    x = torch.randn(2, 8, 4, 4)
    torch.testing.assert_close(block(x).sum(), torch.tensor(EXPECTED), rtol=1e-5, atol=1e-5)
```

Get `EXPECTED` by running the test once on the old code.

## 2. Only when it applies

- **Only if it's cheap:** a seeded 2-step CPU training run on synthetic data, compared with a recorded
  loss. Reordering calls changes how the random-number stream is consumed, and this catches it. Skip it
  if it needs real data, a GPU or a cluster.
- **Models change:** snapshot the `state_dict` keys and shapes for each model config, and load an old
  checkpoint.
  - Renaming an attribute, wrapping layers in `Sequential`, or switching `ModuleList` to `ModuleDict`
    all change the keys silently.
  - If a rename is needed, migrate the old keys with `_register_load_state_dict_pre_hook`.
- **Modules move:** instantiate every yaml or hydra config. `_target_`/`target:` strings, registry
  names, `python -m` lines in SLURM scripts and pickled checkpoints all name code by its import path.
  Leave an alias at the old path when moving a class that is referenced.
- **Public names:** treat wandb metric keys and CLI/hydra keys as public API.

## 3. Coverage is optional

Measuring coverage usually means running a real training job, so the skill never requires it. Two
cheap uses when the repo already has fast tests:
- `coverage run -m pytest` followed by `coverage json` writes `coverage.json`. When it sits at the
  repo root, `measure` dismisses dead-code candidates that ran.
- The coverage report shows whether a function you are about to change is exercised. If it isn't, write
  its characterization test first.

## 4. Commit discipline

- One issue per commit, with a [Conventional Commits](https://www.conventionalcommits.org/) message
  and a `Cleanup-Issue: <ID>` footer: `test:` for the safety net, `refactor:` for dead code,
  complexity and duplication, `fix:` for a bug, `chore:` or `build:` for dependencies, `docs:` for the
  README and license. A scope is optional (`refactor(data): ...`).
- Run the safety net after every commit; revert rather than debug forward.
- Keep behavior changes (bugs found on the way) in their own `fix:` commits.
- Leave scripts that reproduce published results alone: freeze or tag them.
