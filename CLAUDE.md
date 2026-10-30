# Claude Code instructions

All rules live in `AGENTS.md`, imported below. Edit rules there, not here. This file only maps
them to Claude Code features.

@AGENTS.md

## Claude Code specifics

- Worktrees: do not use `EnterWorktree` or the Agent tool's `isolation: "worktree"`. They create
  worktrees under `.claude/worktrees/` with generated branch names. Use the `git worktree add`
  command from `AGENTS.md` instead
- Personal notes: `CLAUDE.local.md` (gitignored)
