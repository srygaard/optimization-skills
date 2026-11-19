#!/usr/bin/env python3
"""A file-backed, cross-job torch.compile artifact cache (Mega-Cache).

Not a command -- copy this file into your trainer. Like compile_watch.py and
unlike the other scripts here, it imports torch; it also imports
torch.distributed, because it runs inside your training process.

Its reason to exist is multi-node: the local Inductor directory is node-local,
so every new job on every new node starts cold. This moves one portable blob
through a shared filesystem instead. Single-node runs already get the local
cache for free -- the part worth stealing is the keying (see MegaCache.__init__).

    from megacache import MegaCache

    cache = MegaCache(Path(os.environ["SCRATCH"]) / "torch_cache", "my_model",
                      dtype="bf16", compile_mode=mode, seq_len=seq_len)
    cache.load()                  # every rank, before the first compiled forward
    model = torch.compile(model)
    train(...)                    # compilation happens here
    cache.save()                  # in a finally block, so a crash still saves

See references/compile-time.md section 7 for the keying rules and the traps.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _world_size() -> int:
    """Number of ranks in the job; 1 for a single-process run.

    Raises if the launcher started several ranks (``WORLD_SIZE``, as set by
    torchrun/accelerate, or ``SLURM_NTASKS`` under plain srun) but the process
    group isn't initialized, rather than silently treating every rank as rank 0.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    launched = os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS") or "1"
    launched_world_size = int(launched)
    if launched_world_size > 1:
        raise RuntimeError(
            f"world size {launched_world_size} but torch.distributed is not initialized; "
            "call MegaCache.load()/save() after init_process_group()."
        )
    return 1


def _is_main_process() -> bool:
    """True on global rank 0, or in a single-process run."""
    return _world_size() == 1 or dist.get_rank() == 0


@contextmanager
def _atomic_open(path: Path) -> Iterator[IO[bytes]]:
    """Open a sibling temp file for binary writing, renamed onto ``path`` on success.

    Readers never see a partial file, so a concurrent job cannot load a torn
    blob. On error the temp file is removed and ``path`` is left as it was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    replaced = False
    try:
        with os.fdopen(fd, "wb") as f:
            yield f
        os.replace(tmp_name, path)
        replaced = True
    finally:
        if not replaced:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


class MegaCache:
    """
    File-backed ``torch.compile`` cache (Inductor + Triton) shared across jobs.

    The blob is named by a hash of ``**meta``, the full torch version (including
    the build suffix, e.g. ``+cu128``) and the GPU name, prefixed with the model
    name so caches for different model configs are distinguishable at a glance
    in the cache directory rather than only inside the hash's ``.json`` sidecar.

    With more than one rank, :meth:`load` has the first rank on each node copy the
    blob from ``cache_dir`` (typically a shared filesystem) to node-local scratch,
    then every rank loads it from there. :meth:`load` is collective, so every
    rank must call it, after the process group is initialized (and, for NCCL,
    after each rank has set its CUDA device). :meth:`save` writes from global
    rank 0 only and is not collective.

    Example::

        cache = MegaCache(
            Path("/shared/torch_cache"), "llama3_8b",
            model_id="meta-llama/Meta-Llama-3-8B", dtype="bf16",
        )
        cache.load()  # on every rank, before the first compiled forward pass

        model = torch.compile(model)
        for batch in loader:
            loss = model(batch)
            ...

        cache.save()  # after compilation has happened (e.g. after the first steps)
    """

    def __init__(self, cache_dir: Path, model_name: str, **meta: Any):
        """
        Args:
            cache_dir: Persistent (usually shared) directory holding the cache blobs.
            model_name: Human-readable prefix for the blob's filename.
                Construct this AFTER ``torch.cuda.set_device()``: reading the GPU
                name initializes CUDA, and every rank would do so on device 0.
            **meta: Anything that changes the compiled graphs (model config, dtype,
                compile mode, ...). Hashed into the cache key; non-JSON values
                are stringified. Adding, removing or renaming a key here orphans
                every existing blob and silently forces a cold run.
        """
        manifest = {
            **meta,
            "torch_version": torch.__version__,
            "model_name": model_name,
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        }
        # default=str so Paths, dtypes, enums etc. in **meta hash instead of raising.
        self._manifest_json = json.dumps(manifest, sort_keys=True, indent=2, default=str)
        self.key = hashlib.sha256(self._manifest_json.encode()).hexdigest()[:16]

        # Sanitize so e.g. "org/model" can't introduce a subdirectory.
        safe_name = re.sub(r"[^a-z0-9_-]+", "_", model_name.lower())
        filename = f"{safe_name}_{self.key}.bin"
        self.shared_path = Path(cache_dir) / filename

    def load(self) -> None:
        """Warm the torch compile cache from previous runs. Collective: call on every rank."""
        world_size = _world_size()
        if world_size > 1:
            # Stage onto node-local disk once per node so N ranks don't all read the shared FS.
            # The all_gather doubles as the barrier and hands each rank the copy its node's
            # first rank staged (None if that failed), so nobody loads a stale one left in the
            # local dir by an earlier job. A failure on one node doesn't stop the others loading.
            # torchrun/accelerate set LOCAL_RANK; plain srun sets SLURM_LOCALID instead.
            # With neither, every rank stages its own copy -- correct, just not shared.
            local_rank = int(os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID") or 0)
            staged = self._stage_local_copy() if local_rank == 0 else None
            gathered: list[Path | None] = [None] * world_size
            dist.all_gather_object(gathered, staged)
            # torchrun/accelerate give each node a contiguous block of global ranks.
            src = gathered[dist.get_rank() - local_rank]
            if src is None:
                logger.info("No torch compile cache staged for this node; save() will create it")
                return
        else:
            src = self.shared_path
            if not src.is_file():
                logger.info(f"No torch compile cache at {src} yet; save() will create it")
                return
        try:
            artifacts = src.read_bytes()
            torch.compiler.load_cache_artifacts(artifacts)
            logger.info(f"Loaded torch compile cache from {src} ({len(artifacts)} bytes)")
        except Exception:  # cache priming is best-effort; never fail the run
            logger.exception(f"Failed to load torch compile cache {src}")

    def _stage_local_copy(self) -> Path | None:
        """Copy the shared blob to node-local scratch ($TMPDIR, else /tmp).

        Returns the fresh copy's path, or None if there is no shared blob or the copy failed.
        Must not raise: the other ranks are about to wait in the collective in :meth:`load`.

        The staged copies are not cleaned up here -- they are the size of the blob
        (hundreds of MB) and assume $TMPDIR is per-job scratch that the scheduler
        reaps. On a persistent node-local disk, clean them up yourself.
        """
        # Per-user directory: a shared /tmp/megacache would be unwritable for the
        # second user on the node, and every one of its ranks would compile cold.
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or str(os.getuid())
        staged_path = Path(tempfile.gettempdir()) / f"megacache-{user}" / self.shared_path.name
        try:
            if not self.shared_path.is_file():
                logger.debug(f"No torch compile cache at {self.shared_path} to pre-stage.")
                return None
            with self.shared_path.open("rb") as src, _atomic_open(staged_path) as dst:
                shutil.copyfileobj(src, dst)
            return staged_path
        except Exception:
            logger.exception(f"Failed to stage {self.shared_path} to {staged_path}")
            return None

    def save(self) -> bool:
        """Persist global rank 0's compiled artifacts. Not collective.

        Returns True if this rank wrote the cache; False on other ranks, when there
        is nothing to save, or when saving failed.
        """
        try:
            if not _is_main_process():
                return False
        except RuntimeError:
            # Called from a finally block after destroy_process_group(), or after the
            # PG died. Returning False beats masking the exception that got us here.
            logger.warning("Cannot determine rank; skipping torch compile cache save")
            return False
        try:
            result = torch.compiler.save_cache_artifacts()
        except Exception:
            logger.exception("Failed to save cache artifacts")
            return False

        if result is None:
            logger.info("Nothing to save (no compiled artifacts)")
            return False
        artifact_bytes, cache_info = result
        logger.debug(f"Torch compile cache info: {cache_info}")
        try:
            with _atomic_open(self.shared_path) as f:
                f.write(artifact_bytes)
            with _atomic_open(self.shared_path.with_suffix(".json")) as f:
                f.write(self._manifest_json.encode())
        except OSError:
            logger.exception(f"Failed to save torch compile cache {self.shared_path}")
            return False
        logger.info(f"Saved torch compile cache to {self.shared_path} ({len(artifact_bytes)} bytes)")
        return True
