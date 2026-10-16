# optimization-skills

Each top-level directory is one self-contained skill in the format agents load from `~/.claude/skills/`.

| skill | reach for it when |
|---|---|
| [revibe-repo](revibe-repo/SKILL.md) | measurable, prioritized cleanup of a Python ML research repo before open-sourcing it: leaked secrets and internal references in git history, a missing LICENSE or README, dependencies, dead code, complexity and duplication |
| [pytorch-compile](pytorch-compile/SKILL.md) | compile is slow, recompiles repeat, compiled is slower than eager, a mode or CUDA-graph error to settle |
| [pytorch-dataloader](pytorch-dataloader/SKILL.md) | the GPU waits on `next()`: utilization low or bursty, worker and storage tuning, a late DDP rank |
| [pytorch-profiler](pytorch-profiler/SKILL.md) / [-full](pytorch-profiler-full/SKILL.md) | you need the step's composition: idle gaps, host syncs, hotspots, exposed all-reduce |

To install a skill, copy its folder into your skills directory (for Claude Code, `~/.claude/skills/`).

Licensed under the MIT License: see [LICENSE](LICENSE).
