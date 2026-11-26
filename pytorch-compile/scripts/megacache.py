#!/usr/bin/env python3
"""A file-backed, cross-job torch.compile artifact cache (Mega-Cache).

https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html

For multi-node: the local Inductor directory is node-local, so every job on a new
node starts cold; this moves one portable blob through a shared filesystem
instead.

    from megacache import MegaCache

    cache = MegaCache(Path(os.environ["SCRATCH"]) / "torch_cache", "my_model",
                      dtype="bf16", compile_mode=mode, seq_len=seq_len)
    cache.load()                  # every rank, before the first compiled forward
    model.compile()
    train(...)                    # compilation happens here
    cache.save()                  # in a finally block, so a crash still saves

"""

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _torch_dist_ready() -> bool:
    """Whether torch.distributed is built in and its process group is initialized."""
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    """Number of ranks in the job; 1 for a single-process run.

    Falls back to the launcher's count (``WORLD_SIZE`` under torchrun/accelerate,
    ``SLURM_NTASKS`` under plain srun) when the process group isn't initialized.
    """
    if _torch_dist_ready():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS") or 1)


def _is_main_process() -> bool:
    """True on global rank 0, or in a single-process run."""
    if _torch_dist_ready():
        return dist.get_rank() == 0
    return _world_size() == 1


def _local_rank() -> int:
    """This rank's index on its own node.

    torchrun/accelerate set ``LOCAL_RANK``; plain srun sets ``SLURM_LOCALID``. With
    neither, every rank reports 0 and stages its own copy: correct, but one read of
    the shared filesystem per rank instead of per node.
    """
    return int(os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID") or 0)


@contextmanager
def _atomic_open(path: Path) -> Generator[IO[bytes], None, None]:
    """Open a sibling temp file for binary writing, renamed onto ``path`` on success.

    A concurrent job never reads a torn blob; on error ``path`` is left as it was.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            yield f
        os.replace(tmp_name, path)
    except BaseException:  # including GeneratorExit, so an abandoned write leaves no litter
        Path(tmp_name).unlink(missing_ok=True)
        raise


class MegaCache:
    """File-backed ``torch.compile`` cache (Inductor + Triton) shared across jobs.

    The blob is named by a hash of ``**meta``, the full torch version (build
    suffix included) and the GPU name, prefixed with the model name so different
    configs are unique and distinguishable in the cache directory.

    Construct the object after ``torch.cuda.set_device()``.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        model_name: str,
        *,
        staged_dir: Path | str | None = None,
        **meta: Any
    ) -> None:
        """
        Args:
            cache_dir: Persistent directory to save the compile cache to.
            model_name: Human-readable prefix for the blob's filename.
            staged_dir: Node-local directory each node's first rank copies the blob into.
                Defaults to a per-job directory under $TMPDIR, else /tmp. Not explicitly
                cleaned.
            **meta: Values hashed to uniquely identify the cache blob.
        """
        manifest = {
            **meta,
            "torch_version": torch.__version__,
            "model_name": model_name,
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        }
        # default=str so Paths, dtypes, enums etc. in **meta hash instead of raising.
        self._manifest_json = json.dumps(manifest, sort_keys=True, indent=2, default=str)
        self.cache_key = hashlib.shake_128(self._manifest_json.encode()).hexdigest(8)

        # Sanitize so e.g. "org/model" can't introduce a subdirectory.
        safe_name = re.sub(r"[^a-z0-9_-]+", "_", model_name.lower())
        self.shared_path = Path(cache_dir) / f"{safe_name}_{self.cache_key}.bin"

        if staged_dir is None:
            job = os.environ.get("SLURM_JOB_ID", "local")
            staged_dir = Path(tempfile.gettempdir()) / f"megacache-{job}"
        self.staged_dir = Path(staged_dir)
        self.staged_path = self.staged_dir / self.shared_path.name

    def load(self) -> None:
        """Warm the torch compile cache from previous runs. Collective: call on every rank.

        Raises if the launcher started several ranks but torch.distributed isn't
        initialized, rather than silently treating every rank as rank 0.
        """
        world_size = _world_size()
        if world_size > 1:
            if not _torch_dist_ready():
                raise RuntimeError(
                    f"world size {world_size} but torch.distributed is not initialized; "
                    "call MegaCache.load() after init_process_group()."
                )
            # Stage onto node-local disk once per node so N ranks don't all read the shared FS.
            # The all_gather shares the staged path, never the blob, and doubles as the barrier.
            # A rank whose node staged nothing gets None and skips loading, so nobody picks up a
            # stale copy left by an earlier job and one node's failure doesn't stop the others.
            local_rank = _local_rank()
            staged_path = self._stage_local_copy() if local_rank == 0 else None
            staged_paths: list[Path | None] = [None] * world_size
            dist.all_gather_object(staged_paths, staged_path)
            # torchrun/accelerate give each node a contiguous block of global ranks.
            src = staged_paths[dist.get_rank() - local_rank]
        else:
            src = self.shared_path
        if src is None or not src.is_file():
            logger.info(f"No torch compile cache for this run yet; save() will create {self.shared_path}")
            return
        try:
            artifacts = src.read_bytes()
            torch.compiler.load_cache_artifacts(artifacts)
            logger.info(f"Loaded torch compile cache from {src} ({len(artifacts)} bytes)")
        except Exception:  # cache priming is best-effort; never fail the run
            logger.exception(f"Failed to load torch compile cache {src}")

    def _stage_local_copy(self) -> Path | None:
        """Copy the shared blob to ``staged_path`` on node-local scratch."""
        try:
            if not self.shared_path.is_file():
                logger.debug(f"No torch compile cache at {self.shared_path} to pre-stage.")
                return None
            with self.shared_path.open("rb") as src, _atomic_open(self.staged_path) as dst:
                shutil.copyfileobj(src, dst)
            return self.staged_path
        except Exception:
            logger.exception(f"Failed to stage {self.shared_path} to {self.staged_path}")
            return None

    def save(self) -> Path | None:
        """Persist global rank 0's compiled artifacts. Safe to call on any rank.

        Returns the blob's path if this rank wrote it; None on other ranks, with
        nothing to save, or on failure.
        """
        if _world_size() > 1 and not _torch_dist_ready():
            # Called before init_process_group(), or from a finally block after the PG
            # died. Warn: otherwise every rank skips the save and nothing is written.
            logger.warning("Cannot determine rank; skipping torch compile cache save")
            return None
        if not _is_main_process():
            return None
        try:
            result = torch.compiler.save_cache_artifacts()
        except Exception:
            logger.exception("Failed to save cache artifacts")
            return None

        artifact_bytes, cache_info = result or (b"", None)
        if not artifact_bytes:
            logger.info("Nothing to save (no compiled artifacts)")
            return None
        logger.debug(f"Torch compile cache info: {cache_info}")
        try:
            with _atomic_open(self.shared_path) as f:
                f.write(artifact_bytes)
            with _atomic_open(self.shared_path.with_suffix(".json")) as f:
                f.write(self._manifest_json.encode())
        except OSError:
            logger.exception(f"Failed to save torch compile cache {self.shared_path}")
            return None
        logger.info(f"Saved torch compile cache to {self.shared_path} ({len(artifact_bytes)} bytes)")
        return self.shared_path
