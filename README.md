# Skills

Agent skills for ML research code. Each folder is one skill; its `SKILL.md` says when to use it and how.

| skill | what it does |
|---|---|
| [revibe-repo](revibe-repo/SKILL.md) | measurable, prioritized cleanup of a Python ML research repo before open-sourcing it: leaked secrets and internal references in git history, a missing LICENSE or README, dependencies, dead code, complexity and duplication |
| [pytorch-compile](pytorch-compile/SKILL.md) | diagnose and fix `torch.compile` problems: graph breaks, recompilation, slow compile time, and compile-mode / runtime tradeoffs |

To install a skill, copy its folder into your skills directory (for Claude Code, `~/.claude/skills/`).

Licensed under the MIT License: see [LICENSE](LICENSE).
