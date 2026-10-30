# Pointer file for Claude Code

All rules live in `AGENTS.md`, loaded below. This file only maps them to Claude Code features.

@AGENTS.md

- Worktrees: do not use `EnterWorktree` or the Agent tool's `isolation: "worktree"`. They create
  worktrees under `.claude/worktrees/` with generated branch names. Use the `git worktree add`
  command from `AGENTS.md` instead
