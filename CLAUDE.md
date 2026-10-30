# Claude Code instructions

The shared, tool-agnostic rulebook lives in `AGENTS.md`. Claude Code pulls it in via the import
below. **Edit shared rules in `AGENTS.md`, not here.** This file holds only Claude-Code-specific
notes.

@AGENTS.md

## Claude-Code-specific notes

- **Worktrees:** follow the Worktree convention in `AGENTS.md` for every change, including small
  ones: create `worktrees/<type>/<N-desc>` with `git worktree add` from the main folder and work
  only there. Do not use `EnterWorktree` or the Agent tool's `isolation: "worktree"`; they create
  worktrees under `.claude/worktrees/` with generated branch names that break the convention.
- **Sub-agents:** a sub-agent that edits code gets its own issue, branch and worktree, created
  before it starts, and its prompt names that worktree path. Never point two sub-agents at the
  same worktree. Read-only sub-agents (search, review) need no worktree.
- **Maus dashboard MCP:** the `maus-dashboard` MCP tools act as the signed-in user. Use the read
  tools freely; never call `*_save` or `*_delete` unless the user asks in the current session.
  Bulk exports go through `pixi run export`, not through MCP calls in the chat.
- **Persistent memory:** recalled memories are background context. Verify any file, function or
  flag they name still exists before acting on it.
- **Personal notes:** put personal, non-shared instructions in `CLAUDE.local.md` (gitignored).
