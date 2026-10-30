# Claude Code instructions

The shared, tool-agnostic rulebook lives in `AGENTS.md`. Claude Code pulls it in via the import
below. **Edit shared rules in `AGENTS.md`, not here.** This file holds only Claude-Code-specific
notes.

@AGENTS.md

## Claude-Code-specific notes

- **Sub-agents and worktrees:** when spawning sub-agents that edit code, pass
  `isolation: "worktree"` to the Agent tool so each one gets its own worktree.
- **Maus dashboard MCP:** the `maus-dashboard` MCP tools act as the signed-in user. Use the read
  tools freely; never call `*_save` or `*_delete` unless the user asks in the current session.
  Bulk exports go through `pixi run export`, not through MCP calls in the chat.
- **Persistent memory:** recalled memories are background context. Verify any file, function or
  flag they name still exists before acting on it.
- **Personal notes:** put personal, non-shared instructions in `CLAUDE.local.md` (gitignored).
